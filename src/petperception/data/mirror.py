"""Mirrored dataset ingestion: fetch official dataset files from authorised mirrors.

Many official hosts are blocked here (results/data_access/network_probe.json).
The data owner can place the *same* files in an object store this workspace
can reach (Google Cloud Storage, S3, Azure Blob) and hand over download URLs,
often time-limited signed URLs. This module downloads them without changing
the dataset's identity:

* `manifests/mirror_files.json` (committed) is the catalogue: for every file
  the dataset id and exact version, the official source URL and filename,
  the expected size and official checksums where published, the target path
  under var/data, the annotations/metadata it carries, its usage terms and
  the capability it unblocks. A file is accepted only if it matches the
  catalogue (size, and every official checksum given); a file without an
  official checksum is pinned by SHA-256 on first retrieval
  (`manifests/mirror_pins.csv`) and must match on every later retrieval.
* URLs are not committed. They are read from the file named by
  PETPERCEPTION_MIRRORS (default var/secrets/mirrors.json, ignored by git;
  {"<dataset>/<file>": "<url>"}) or from an environment variable
  PETPERCEPTION_MIRROR_<DATASET>_<FILE> (non-alphanumerics -> "_",
  uppercase). Logs, errors and provenance records show the URL with its
  query string removed, so signatures and tokens never leave the process.
* Before downloading: expiry of signed URLs is parsed (GCS V4/V2, S3 V4/V2,
  Azure SAS) and reported; an expired URL is refused; free disk space is
  checked against the expected size plus a margin.
* Downloading is resumable (HTTP Range into `<target>.part`), retried with
  exponential backoff, checked against Content-Length, verified, and only
  then renamed into place. A provenance record `<target>.provenance.json`
  keeps the dataset identity, checksums, the mirror host (not the URL) and
  the time.
* Shards: split archives and multi-part datasets are separate catalogue
  entries, each verified on its own.
* Uploads (checkpoints) need write access that a read-only download link
  does not grant: `upload()` supports a PUT to a signed upload URL
  (GCS V4 with method PUT, S3 presigned PUT) and GCS resumable session URIs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..utils import USER_AGENT, utc_now

CHUNK = 1 << 20
DISK_MARGIN = 512 * 2**20


class MirrorError(RuntimeError):
    pass


def redact(url: str) -> str:
    """The URL without query string or fragment (signatures and tokens live there)."""
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, "", ""))


def url_expiry(url: str) -> datetime | None:
    """Expiry time encoded in a signed URL, or None when the URL carries none."""
    q = {k.lower(): v for k, v in urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query)}
    for date_k, exp_k in (("x-goog-date", "x-goog-expires"), ("x-amz-date", "x-amz-expires")):
        if date_k in q and exp_k in q:
            start = datetime.strptime(q[date_k], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            return start + timedelta(seconds=int(q[exp_k]))
    if "expires" in q and q["expires"].isdigit():  # GCS V2 / S3 V2: epoch seconds
        return datetime.fromtimestamp(int(q["expires"]), tz=UTC)
    if "se" in q and "sig" in q:  # Azure SAS
        return datetime.fromisoformat(q["se"].replace("Z", "+00:00"))
    return None


def expiry_report(url: str, now: datetime | None = None) -> dict:
    exp = url_expiry(url)
    now = now or utc_now()
    if exp is None:
        return {"signed": False, "expires_at": None, "seconds_left": None, "state": "no_expiry"}
    left = (exp - now).total_seconds()
    return {"signed": True, "expires_at": exp.isoformat(timespec="seconds"), "seconds_left": int(left),
            "state": "expired" if left <= 0 else "expires_within_1h" if left < 3600 else "valid"}


@dataclass
class CatalogFile:
    dataset: str
    version: str
    file: str
    official_url: str
    target: str
    size: int | None = None
    checksums: dict[str, str] = field(default_factory=dict)  # official: {"md5": ..., "sha256": ...}
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.file}"


def load_catalog(path: Path) -> list[CatalogFile]:
    d = json.loads(path.read_text())
    out = []
    for ds in d["datasets"]:
        for f in ds["files"]:
            out.append(CatalogFile(dataset=ds["id"], version=ds["version"], file=f["file"], official_url=f["official_url"], target=f["target"],
                                   size=f.get("size"), checksums=f.get("checksums", {}),
                                   info={k: ds.get(k) for k in ("name", "terms", "annotations", "unblocks", "priority")} | {"note": f.get("note")}))
    return out


def env_key(key: str) -> str:
    return "PETPERCEPTION_MIRROR_" + re.sub(r"[^A-Za-z0-9]", "_", key).upper()


def mirror_urls(path: Path | None = None) -> dict[str, str]:
    p = Path(os.environ.get("PETPERCEPTION_MIRRORS", path or "var/secrets/mirrors.json"))
    return json.loads(p.read_text()) if p.exists() else {}


def url_for(cf: CatalogFile, urls: dict[str, str]) -> str | None:
    return os.environ.get(env_key(cf.key)) or urls.get(cf.key)


def _hashers(names: list[str]) -> dict[str, Any]:
    return {n: hashlib.new(n) for n in names}


def file_digests(p: Path, names: tuple[str, ...] = ("sha256", "md5")) -> dict[str, str]:
    hs = _hashers(list(names))
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            for h in hs.values():
                h.update(chunk)
    return {k: h.hexdigest() for k, h in hs.items()}


def _pins(path: Path) -> dict[str, str]:
    return {r["key"]: r["sha256"] for r in csv.DictReader(open(path, newline=""))} if path.exists() else {}


def _pin(path: Path, key: str, sha: str) -> None:
    first = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if first:
            w.writerow(["key", "sha256"])
        w.writerow([key, sha])


def download(url: str, dest: Path, expected_size: int | None = None, attempts: int = 6, timeout: float = 120.0,
             sleep=time.sleep, log=print) -> dict:
    """Resumable download into dest (via dest.part). Returns transfer statistics; never logs the query string."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shown = redact(url)
    resumed_from = part.stat().st_size if part.exists() else 0
    for k in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if expected_size is not None and have == expected_size:
            break
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
                status = resp.status
                if have and status != 206:  # server ignored the range: start over
                    have = 0
                total = None
                cr = resp.headers.get("Content-Range")
                if cr and "/" in cr and cr.split("/")[-1].isdigit():
                    total = int(cr.split("/")[-1])
                elif resp.headers.get("Content-Length") and not resp.headers.get("Content-Encoding"):
                    total = have + int(resp.headers["Content-Length"])
                if expected_size is not None and total is not None and total != expected_size:
                    raise MirrorError(f"{shown}: server reports {total} bytes, catalogue expects {expected_size}")
                with open(part, "ab" if have else "wb") as fh:
                    for chunk in iter(lambda: resp.read(CHUNK), b""):
                        fh.write(chunk)
            got = part.stat().st_size
            if total is not None and got != total:
                raise urllib.error.URLError(f"short read: {got} of {total} bytes")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and expected_size is not None and part.exists() and part.stat().st_size == expected_size:
                break
            if exc.code in (401, 403):
                raise MirrorError(f"{shown}: HTTP {exc.code} (signature expired or not permitted for this method)") from None
            if exc.code == 416:
                part.unlink(missing_ok=True)
            if k == attempts - 1:
                raise MirrorError(f"{shown}: HTTP {exc.code} after {attempts} attempts") from None
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            if k == attempts - 1:
                raise MirrorError(f"{shown}: {type(exc).__name__} after {attempts} attempts") from None
        wait = min(60, 2 ** k)
        log(json.dumps({"retry": k + 1, "url": shown, "bytes_so_far": part.stat().st_size if part.exists() else 0, "wait_s": wait}))
        sleep(wait)
    return {"bytes": part.stat().st_size, "resumed_from": resumed_from, "url": shown}


def fetch_file(cf: CatalogFile, url: str, data_root: Path, pins_path: Path, log=print, sleep=time.sleep) -> dict:
    """Download, verify and place one catalogue file; writes a provenance record next to it."""
    ex = expiry_report(url)
    if ex["state"] == "expired":
        raise MirrorError(f"{cf.key}: the URL expired at {ex['expires_at']}")
    target = data_root / cf.target
    if target.exists():
        return {"key": cf.key, "state": "present", "path": str(target)}
    need = (cf.size or 0) + DISK_MARGIN
    free = shutil.disk_usage(data_root if data_root.exists() else data_root.parent).free
    if cf.size and free < need:
        raise MirrorError(f"{cf.key}: needs {need} bytes free, {free} available")
    t = time.time()
    stats = download(url, target, cf.size, log=log, sleep=sleep)
    part = target.with_name(target.name + ".part")
    dig = file_digests(part)
    problems = []
    if cf.size is not None and stats["bytes"] != cf.size:
        problems.append(f"size {stats['bytes']} != {cf.size}")
    for algo, want in cf.checksums.items():
        if algo not in dig:
            dig.update(file_digests(part, (algo,)))
        if dig[algo].lower() != want.lower():
            problems.append(f"{algo} mismatch")
    pins = _pins(pins_path)
    if not cf.checksums:
        if cf.key in pins and pins[cf.key] != dig["sha256"]:
            problems.append("sha256 differs from the value pinned at first retrieval")
    if problems:
        part.unlink(missing_ok=True)
        raise MirrorError(f"{cf.key}: rejected ({'; '.join(problems)})")
    if not cf.checksums and cf.key not in pins:
        _pin(pins_path, cf.key, dig["sha256"])
    os.replace(part, target)
    prov = {"dataset": cf.dataset, "version": cf.version, "file": cf.file, "official_url": cf.official_url,
            "retrieved_via": urllib.parse.urlsplit(url).netloc, "retrieved_at": utc_now().isoformat(timespec="seconds"),
            "bytes": stats["bytes"], "sha256": dig["sha256"], "md5": dig["md5"],
            "verified_against": sorted(cf.checksums) or ["sha256 pinned at first retrieval"], "url_expiry": ex,
            "terms": cf.info.get("terms"), "seconds": round(time.time() - t, 1), "resumed_from_bytes": stats["resumed_from"]}
    target.with_name(target.name + ".provenance.json").write_text(json.dumps(prov, indent=1))
    return {"key": cf.key, "state": "fetched", "path": str(target), **{k: prov[k] for k in ("bytes", "sha256", "seconds")}}


def status(catalog: list[CatalogFile], data_root: Path, urls: dict[str, str]) -> list[dict]:
    rows = []
    for cf in catalog:
        target = data_root / cf.target
        url = url_for(cf, urls)
        rows.append({"key": cf.key, "present": target.exists(), "partial_bytes": (target.with_name(target.name + ".part").stat().st_size
                                                                                   if target.with_name(target.name + ".part").exists() else 0),
                     "expected_size": cf.size, "official_checksums": sorted(cf.checksums), "url_configured": url is not None,
                     "url_host": urllib.parse.urlsplit(url).netloc if url else None, "expiry": expiry_report(url) if url else None,
                     "priority": cf.info.get("priority")})
    return rows


def upload(path: Path, url: str, content_type: str = "application/octet-stream", chunk: int = 8 * CHUNK, attempts: int = 5,
           sleep=time.sleep) -> dict:
    """Upload to a signed PUT URL, or in chunks to a GCS resumable session URI (detected by 'upload_id=')."""
    shown = redact(url)
    size = path.stat().st_size
    if "upload_id=" not in url:
        for k in range(attempts):
            try:
                req = urllib.request.Request(url, data=path.read_bytes(), method="PUT", headers={"Content-Type": content_type, "User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=600) as resp:
                    return {"url": shown, "bytes": size, "status": resp.status}
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise MirrorError(f"{shown}: HTTP {exc.code}: this URL does not grant upload (write) access") from None
                if k == attempts - 1:
                    raise MirrorError(f"{shown}: HTTP {exc.code}") from None
            except (urllib.error.URLError, OSError):
                if k == attempts - 1:
                    raise MirrorError(f"{shown}: upload failed after {attempts} attempts") from None
            sleep(min(60, 2 ** k))
    sent = 0
    with open(path, "rb") as fh:
        while sent < size:
            fh.seek(sent)
            data = fh.read(chunk)
            end = sent + len(data) - 1
            req = urllib.request.Request(url, data=data, method="PUT", headers={"Content-Range": f"bytes {sent}-{end}/{size}", "User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    if resp.status in (200, 201):
                        return {"url": shown, "bytes": size, "status": resp.status}
            except urllib.error.HTTPError as exc:
                if exc.code != 308:  # 308 = chunk accepted, continue at the reported offset
                    raise MirrorError(f"{shown}: HTTP {exc.code}") from None
                rng = exc.headers.get("Range")
                sent = int(rng.split("-")[1]) + 1 if rng else 0
                continue
            sent = end + 1
    return {"url": shown, "bytes": size, "status": "complete"}


def _range(url: str, start: int, end: int, attempts: int = 6, sleep=time.sleep) -> bytes:
    """Bytes [start, end] of a remote object (inclusive), retried; never logs the query string."""
    for k in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) == end - start + 1:
                return data
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise MirrorError(f"{redact(url)}: HTTP {exc.code}") from None
        except (urllib.error.URLError, OSError):
            pass
        sleep(min(60, 2 ** k))
    raise MirrorError(f"{redact(url)}: range {start}-{end} failed after {attempts} attempts")


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        cr = resp.headers.get("Content-Range")
        if cr and "/" in cr:
            return int(cr.split("/")[-1])
        return int(resp.headers["Content-Length"])


def stream_digests(url: str, names: tuple[str, ...] = ("sha256", "md5"), chunk: int = 8 * CHUNK) -> dict[str, str | int]:
    """Hash a remote object in one streaming pass without storing it (verifies archives larger than the disk)."""
    hs = _hashers(list(names))
    n = 0
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=600) as resp:
        for block in iter(lambda: resp.read(chunk), b""):
            n += len(block)
            for h in hs.values():
                h.update(block)
    return {"bytes": n, **{k: h.hexdigest() for k, h in hs.items()}}


@dataclass
class ZipEntry:
    name: str
    method: int
    crc: int
    csize: int
    usize: int
    disk: int
    offset: int


class RemoteZip:
    """Read members of a (possibly split, possibly ZIP64) archive over HTTP Range requests.

    `parts` are the URLs in disk order: for a split archive name.z01, name.z02, ...,
    name.zip (the last part holds the central directory). Only the central directory
    and the requested members are transferred; every member is checked against
    its CRC-32 before it is returned.
    """

    def __init__(self, parts: list[str], sleep=time.sleep):
        import struct

        self.parts, self.sleep, self._struct = parts, sleep, struct
        self.sizes = [self._size(u) for u in parts]
        last = parts[-1]
        tail_len = min(self.sizes[-1], 65557 + 20 + 56)
        tail = self._get(last, self.sizes[-1] - tail_len, self.sizes[-1] - 1)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise MirrorError(f"{redact(last)}: no end-of-central-directory record")
        _, disk_no, cd_disk, _, n_total, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
        loc = tail.rfind(b"PK\x06\x07", 0, i)
        if loc >= 0 and (cd_off == 0xFFFFFFFF or n_total == 0xFFFF or cd_size == 0xFFFFFFFF):
            _, z64_disk, z64_off, _ = struct.unpack("<IIQI", tail[loc:loc + 20])
            rec = self._get(parts[z64_disk], z64_off, z64_off + 55)
            (_, _, _, _, _, cd_disk, _, n_total, cd_size, cd_off) = struct.unpack("<IQHHIIQQQQ", rec)
        cd = self._get(parts[cd_disk], cd_off, cd_off + cd_size - 1)
        self.entries: dict[str, ZipEntry] = {}
        p = 0
        for _ in range(n_total):
            (sig, _, _, _, method, _, _, crc, csize, usize, nlen, xlen, clen, disk, _, _, off) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
            if sig != 0x02014B50:
                raise MirrorError("corrupt central directory")
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
            q = 0
            while q + 4 <= len(extra):  # ZIP64 extended information
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 0x0001:
                    vals = list(struct.unpack("<" + "Q" * (hlen // 8), extra[q + 4:q + 4 + (hlen // 8) * 8]))
                    if usize == 0xFFFFFFFF:
                        usize = vals.pop(0)
                    if csize == 0xFFFFFFFF:
                        csize = vals.pop(0)
                    if off == 0xFFFFFFFF:
                        off = vals.pop(0)
                    if disk == 0xFFFF and len(extra) >= q + 4 + hlen:
                        disk = struct.unpack("<I", extra[q + 4 + hlen - 4:q + 4 + hlen])[0]
                q += 4 + hlen
            self.entries[name] = ZipEntry(name, method, crc, csize, usize, disk, off)
            p += 46 + nlen + xlen + clen

    def _size(self, part: str) -> int:
        return remote_size(part)

    def _get(self, part: str, start: int, end: int) -> bytes:
        return _range(part, start, end, sleep=self.sleep)

    def names(self) -> list[str]:
        return list(self.entries)

    def _read_span(self, disk: int, offset: int, n: int) -> bytes:
        out = b""
        while n > 0:
            avail = self.sizes[disk] - offset
            take = min(n, avail)
            if take > 0:
                out += self._get(self.parts[disk], offset, offset + take - 1)
                n -= take
            disk, offset = disk + 1, 0
            if n > 0 and disk >= len(self.parts):
                raise MirrorError("member extends beyond the last part")
        return out

    def read(self, name: str) -> bytes:
        import zlib

        e = self.entries[name]
        hdr = self._read_span(e.disk, e.offset, 30)
        nlen, xlen = self._struct.unpack("<HH", hdr[26:30])
        disk, off = e.disk, e.offset + 30 + nlen + xlen
        while off >= self.sizes[disk]:
            off -= self.sizes[disk]
            disk += 1
        raw = self._read_span(disk, off, e.csize)
        if e.method == 0:
            data = raw
        elif e.method == 8:
            data = zlib.decompress(raw, -15)
        else:
            raise MirrorError(f"{name}: unsupported compression method {e.method}")
        if zlib.crc32(data) & 0xFFFFFFFF != e.crc or len(data) != e.usize:
            raise MirrorError(f"{name}: CRC or size mismatch")
        return data


class LocalSplitZip(RemoteZip):
    """The same reader over local files: for split archives (name.z01 ... name.zip) that
    Python's zipfile cannot open, from servers without Range support (downloaded whole)."""

    def _size(self, part: str) -> int:
        return Path(part).stat().st_size

    def _get(self, part: str, start: int, end: int) -> bytes:
        with open(part, "rb") as fh:
            fh.seek(start)
            return fh.read(end - start + 1)
