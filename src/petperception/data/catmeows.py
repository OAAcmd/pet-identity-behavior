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
else rather than guessing. ``extras.zip`` (excluded / non-meow recordings and
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
    """Fetch record metadata, then dataset.zip and extras.zip, verifying Zenodo MD5s."""
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
    for key in ("dataset.zip", "extras.zip"):  # record keys; the record's description calls the second "extra.zip"
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


# ---------------------------------------------------------------- unofficial GitHub mirror (zenodo.org is blocked here)

MIRROR_REPO = "https://github.com/jinjingleayi/meow-by-meow"
MIRROR_COMMIT = "0fe6d4c2edbc98c53f410d60a7654fbb1ae39a4e"  # 2023-12-04; README links the original as zenodo.org/records/4008297
MIRROR_DIR = "data/raw_data/zenodo.4008297"
MIRROR_PINS = "catmeows_mirror_sha256.csv"


def fetch_mirror(data_root: Path, manifest_dir: Path) -> dict:
    """Fetch the 440 CatMeows recordings from an unofficial GitHub mirror (sparse checkout at a
    pinned commit) and pin every file's SHA-256 on first retrieval. The files cannot be compared
    with Zenodo's checksums while zenodo.org is blocked; results carry that caveat."""
    import shutil
    import subprocess

    d = dataset_dir(data_root)
    clone = d / "_mirror"
    if not (clone / ".git").exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--sparse", MIRROR_REPO, str(clone)], check=True, timeout=600)
    subprocess.run(["git", "-C", str(clone), "sparse-checkout", "set", MIRROR_DIR], check=True, timeout=600)
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", MIRROR_COMMIT], check=True, timeout=600)
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if head != MIRROR_COMMIT:
        raise RuntimeError(f"mirror at {head}, expected {MIRROR_COMMIT}")
    src = clone / MIRROR_DIR
    files = sorted(p for p in src.glob("*.wav") if p.is_file() and not p.is_symlink())
    target = d / "dataset"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for p in files:
        shutil.copyfile(p, target / p.name)
    pin_path = manifest_dir / MIRROR_PINS
    current = {p.name: sha256_file(target / p.name) for p in files}
    if pin_path.exists():
        pinned = {r["file"]: r["sha256"] for r in csv.DictReader(open(pin_path, newline=""))}
        bad = sorted(k for k in pinned if current.get(k) != pinned[k]) + sorted(set(current) - set(pinned))
        if bad:
            raise RuntimeError(f"{len(bad)} mirror files differ from the pinned checksums, e.g. {bad[:3]}")
    else:
        with open(pin_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "sha256"])
            for k in sorted(current):
                w.writerow([k, current[k]])
    status = {"status": "ok", "source": "unofficial GitHub mirror", "repo": MIRROR_REPO, "commit": MIRROR_COMMIT, "path": MIRROR_DIR, "files": len(files),
              "integrity": "SHA-256 pinned at first retrieval (manifests/" + MIRROR_PINS + "); not verifiable against Zenodo (zenodo.org blocked)",
              "license": "CC BY 4.0 per the dataset paper (Ludovico et al., MMM 2021); the Zenodo record could not be read from this workspace"}
    write_json(d / "fetch_status.json", status)
    return status
