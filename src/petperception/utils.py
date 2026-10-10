"""Small shared helpers: hashing, atomic downloads, JSON I/O, timing."""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

USER_AGENT = "petperception-fetch/0.1 (+research; explicit fetch only)"


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> datetime:
    return datetime.now(UTC)


def today_iso() -> str:
    return date.today().isoformat()


class DownloadError(RuntimeError):
    pass


def download(url: str, dest: Path, expected_sha256: str | None = None, timeout: float = 120.0) -> dict[str, Any]:
    """Download ``url`` to ``dest`` atomically and verify its SHA-256.

    Only called from explicit ``fetch-*`` commands; inference never downloads.
    Returns a record with byte count, checksum and selected response headers.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".part-")
    h = hashlib.sha256()
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, os.fdopen(fd, "wb") as out:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            while True:
                b = resp.read(1 << 20)
                if not b:
                    break
                out.write(b)
                h.update(b)
                n += len(b)
    except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException) as exc:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise DownloadError(f"{url}: {exc}") from exc
    digest = h.hexdigest()
    declared = headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) != n and "content-encoding" not in headers:
        # A connection cut mid-transfer can end without an error; never keep (or pin) a truncated file.
        os.unlink(tmp_name)
        raise DownloadError(f"{url}: truncated transfer ({n} of {declared} bytes)")
    if expected_sha256 and digest != expected_sha256:
        os.unlink(tmp_name)
        raise DownloadError(f"{url}: sha256 mismatch (got {digest}, expected {expected_sha256})")
    os.replace(tmp_name, dest)
    return {
        "url": url,
        "bytes": n,
        "sha256": digest,
        "etag": headers.get("etag"),
        "last_modified": headers.get("last-modified"),
        "x_goog_generation": headers.get("x-goog-generation"),
        "retrieved_at": utc_now().isoformat(timespec="seconds"),
    }


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, sort_keys=False, default=_json_default)
        fh.write("\n")
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


class StageTimer:
    """Collects wall-clock durations of named stages in milliseconds."""

    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = self.timings_ms.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0


def private_tempdir(root: Path | None = None, prefix: str = "pp-") -> Path:
    """Create a private (0700) temporary directory."""
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        os.chmod(root, 0o700)
    d = Path(tempfile.mkdtemp(prefix=prefix, dir=str(root) if root else None))
    os.chmod(d, 0o700)
    return d


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
