"""CatMeows importer (Zenodo record 4008297, v1.0.2).

Status in this workspace: BLOCKED - zenodo.org is denied by the session's
egress policy (see manifests/resources.json). The importer, grouping and E2
protocol are implemented and unit-tested on synthetic fixtures only; no
CatMeows result is reported until the real files are fetched and verified.

File naming (per the dataset description): ``C_NNNNN_BB_SS_OOOOO_RXX`` where
C = context (B brushing, F waiting for food, I isolation in an unfamiliar
environment), NNNNN = cat ID, BB = breed, SS = sex/neuter status,
OOOOO = owner ID, R = recording session, XX = vocalization counter.
The parser verifies this structure on the actual files and rejects anything
else rather than guessing. ``extra.zip`` (excluded / non-meow recordings and
uncut sequences) is kept separate and never used for context training.
Context labels keep their original semantics: ``isolation`` is a recording
context, not a fear label.
"""

from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from ..utils import DownloadError, download, read_json, sha256_file, write_json

RECORD_API = "https://zenodo.org/api/records/4008297"
CONTEXTS = {"B": "brushing", "F": "waiting_for_food", "I": "isolation"}
# RXX = one recording-session character followed by a two-digit counter.
_NAME_RE = re.compile(r"^(?P<ctx>[BFI])_(?P<cat>[A-Z0-9]+)_(?P<breed>[A-Z0-9]+)_(?P<sex>[A-Z0-9]+)_(?P<owner>[A-Z0-9]+)_(?P<session>[A-Z0-9])(?P<counter>\d{2})$")


@dataclass(frozen=True)
class CatMeowsRecording:
    path: str
    context: str
    cat_id: str
    breed: str
    sex: str
    owner_id: str
    session: str
    counter: str
    sample_rate_hz: int
    duration_s: float


def dataset_dir(data_root: Path) -> Path:
    return data_root / "catmeows"


def parse_name(stem: str) -> dict[str, str] | None:
    m = _NAME_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    d["context"] = CONTEXTS[d.pop("ctx")]
    return d


def fetch(data_root: Path) -> dict:
    """Fetch record metadata, then dataset.zip and extra.zip, verifying Zenodo MD5s."""
    import hashlib

    d = dataset_dir(data_root)
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / "zenodo_record.json"
    try:
        download(RECORD_API, meta_path, timeout=60)
    except DownloadError as exc:
        status = {"status": "blocked", "error": str(exc), "next_step": "allow zenodo.org in the environment network policy, then re-run"}
        write_json(d / "fetch_status.json", status)
        return status
    rec = read_json(meta_path)
    files = {f["key"]: f for f in rec.get("files", [])}
    out = {"status": "ok", "record_version": rec.get("metadata", {}).get("version"), "license": rec.get("metadata", {}).get("license"), "files": {}}
    for key in ("dataset.zip", "extra.zip"):
        if key not in files:
            out["files"][key] = "missing from record"
            continue
        f = files[key]
        url = f.get("links", {}).get("self") or f"{RECORD_API}/files/{key}/content"
        dest = d / key
        rec_dl = download(url, dest, timeout=600)
        md5 = hashlib.md5(dest.read_bytes()).hexdigest()
        want = str(f.get("checksum", "")).replace("md5:", "")
        if want and md5 != want:
            out["files"][key] = f"md5 mismatch (got {md5}, expected {want})"
            out["status"] = "error"
            continue
        target = d / key.replace(".zip", "")
        with zipfile.ZipFile(dest) as z:
            for member in z.namelist():
                p = (target / member).resolve()
                if not str(p).startswith(str(target.resolve())):
                    raise RuntimeError(f"unsafe path in archive: {member}")
            z.extractall(target)
        out["files"][key] = {"bytes": rec_dl["bytes"], "sha256": rec_dl["sha256"], "md5_verified": bool(want)}
    write_json(d / "fetch_status.json", out)
    return out


def scan(root: Path, manifest_csv: Path | None = None) -> tuple[list[CatMeowsRecording], list[str]]:
    """Index meow recordings under ``root/dataset``; unparseable names are reported, not guessed."""
    recs, rejected = [], []
    for p in sorted((root / "dataset").rglob("*.wav")):
        info = parse_name(p.stem)
        if info is None:
            rejected.append(str(p.relative_to(root)))
            continue
        si = sf.info(str(p))
        recs.append(CatMeowsRecording(str(p), info["context"], info["cat"], info["breed"], info["sex"], info["owner"],
                                      info["session"], info["counter"], int(si.samplerate), float(si.duration)))
    if manifest_csv is not None:
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "bytes", "sha256", "context", "cat_id", "owner_id", "session", "breed", "sex", "sample_rate_hz", "duration_s"])
            for r in recs:
                p = Path(r.path)
                w.writerow([p.name, p.stat().st_size, sha256_file(p), r.context, r.cat_id, r.owner_id, r.session, r.breed, r.sex, r.sample_rate_hz, f"{r.duration_s:.4f}"])
    return recs, rejected


def owner_cat_relationships(recs: list[CatMeowsRecording]) -> dict[str, list[str]]:
    rel: dict[str, set[str]] = {}
    for r in recs:
        rel.setdefault(r.owner_id, set()).add(r.cat_id)
    return {o: sorted(c) for o, c in sorted(rel.items())}
