"""Kinetics-700-2020 validation clips of six pet-related human actions (E18).

Source: the CVDF mirror of Kinetics-700-2020 (official distribution),
https://kinetics.s3.amazonaws.com/700_2020/ - one tarball of 10-second
clips (video with audio) per class; the annotation CSV gives class and
YouTube id. Tarballs and the CSV are pinned by SHA-256 on first retrieval
(manifests/kinetics_pets_sha256.csv). Labels are human actions, not pet
behaviour: they say a cat or dog is involved and, for petting and
grooming, that a person touches it.

Terms: Kinetics annotations are CC BY 4.0 (Google / DeepMind, per secondary
sources; the primary page is blocked here); the clips are YouTube uploads
whose rights stay with their owners, so this is research-profile evaluation
only and no clip is committed.
"""

from __future__ import annotations

import csv
import tarfile
from pathlib import Path

from ..utils import download, sha256_file

BASE = "https://kinetics.s3.amazonaws.com/700_2020"
# class -> (tarball index = alphabetical class index + 1, species, person contact implied by the action)
CLASSES = {"petting cat": (363, "cat", True), "grooming cat": (231, "cat", True), "grooming dog": (232, "dog", True),
           "bathing dog": (26, "dog", True), "training dog": (634, "dog", False), "walking the dog": (670, "dog", False)}
PINS = "kinetics_pets_sha256.csv"


def root(data_root: Path) -> Path:
    return data_root / "kinetics_pets"


def fetch(data_root: Path, manifest_dir: Path) -> dict:
    d = root(data_root)
    pin_path = manifest_dir / PINS
    pins = {r["file"]: r["sha256"] for r in csv.DictReader(open(pin_path, newline=""))} if pin_path.exists() else {}
    files = {"val.csv": f"{BASE}/annotations/val.csv"}
    files.update({f"k700_val_{i:03d}.tar.gz": f"{BASE}/val/k700_val_{i:03d}.tar.gz" for i, _, _ in CLASSES.values()})
    got = {}
    for name, url in files.items():
        p = d / "_download" / name
        if not (p.exists() and (name not in pins or sha256_file(p) == pins[name])):
            download(url, p, expected_sha256=pins.get(name), timeout=3600)
        got[name] = sha256_file(p)
        if name in pins and pins[name] != got[name]:
            raise RuntimeError(f"{name}: SHA-256 differs from the pinned value")
    if not pins:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        with open(pin_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "url", "sha256"])
            for name in sorted(got):
                w.writerow([name, files[name], got[name]])
    clips = 0
    for cls, (i, _, _) in CLASSES.items():
        out = d / "clips"
        if not (out / cls).exists():
            with tarfile.open(d / "_download" / f"k700_val_{i:03d}.tar.gz") as tf:
                members = [m for m in tf.getmembers() if m.isfile() and m.name.startswith(f"{cls}/") and m.name.endswith(".mp4")]
                tf.extractall(out, members=members, filter="data")  # rejects absolute paths, '..' and links
        clips += len(list((out / cls).glob("*.mp4")))
    return {"files": got, "clips": clips}


TRAIN_PINS = "kinetics_pets_train_sample_sha256.csv"
HOLDOUT_PINS = "kinetics_pets_train_holdout_sha256.csv"
FINAL_PINS = "kinetics_pets_train_final_sha256.csv"
# Untouched final video evaluation set: train clips 101-160 of each class (after the E19 tuning
# sample, 1-50, and the E19b holdout, 51-100). Fetched once, pinned, and read only by the final
# evaluation (E23), which records its single use in results/e23/final_set_use.json.
FINAL_SKIP, FINAL_PER_CLASS, FINAL_SUBDIR = 100, 60, "train_final"


def fetch_train_sample(data_root: Path, manifest_dir: Path, per_class: int = 50, skip: int = 0, subdir: str = "train_sample",
                       pins_name: str = TRAIN_PINS) -> dict:
    """Tuning clips for E19: the first `per_class` clips of each class in the train tarballs.

    The tarballs are 0.8-1.6 GB each, so they are streamed and the transfer is
    stopped once enough clips of the class are extracted. A partial transfer
    has no tarball checksum, so every extracted clip is pinned by SHA-256 on
    first retrieval (manifests/kinetics_pets_train_sample_sha256.csv) and
    checked on later fetches. Train and validation are different YouTube
    videos by construction; the overlap of YouTube ids is checked anyway.
    """
    import io
    import urllib.request

    from ..utils import USER_AGENT

    d = root(data_root) / subdir
    pin_path = manifest_dir / pins_name
    pins = {r["file"]: r["sha256"] for r in csv.DictReader(open(pin_path, newline=""))} if pin_path.exists() else {}
    got: dict[str, str] = {}
    for cls, (i, _, _) in CLASSES.items():
        out = d / cls
        have = sorted(out.glob("*.mp4")) if out.exists() else []
        if len(have) < per_class:
            out.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(f"{BASE}/train/k700_train_{i:03d}.tar.gz", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=600) as resp, tarfile.open(fileobj=io.BufferedReader(resp, 1 << 20), mode="r|gz") as tf:
                n = seen = 0
                for m in tf:
                    if not (m.isfile() and m.name.startswith(f"{cls}/") and m.name.endswith(".mp4")):
                        continue
                    seen += 1
                    if seen <= skip:  # clips already used elsewhere (tuning sample) are passed over
                        continue
                    tf.extract(m, d, filter="data")  # rejects absolute paths, '..' and links
                    n += 1
                    if n >= per_class:
                        break
            have = sorted(out.glob("*.mp4"))
        for p in have[:per_class]:
            rel = f"{cls}/{p.name}"
            got[rel] = sha256_file(p)
            if rel in pins and pins[rel] != got[rel]:
                raise RuntimeError(f"{rel}: SHA-256 differs from the pinned value")
    if not pins:
        with open(pin_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "sha256"])
            for rel in sorted(got):
                w.writerow([rel, got[rel]])
    val_ids = {p.stem[:11] for p in (root(data_root) / "clips").glob("*/*.mp4")}
    overlap = sorted({rel.split("/")[1][:11] for rel in got} & val_ids)
    return {"clips": len(got), "youtube_ids_shared_with_validation": overlap}


def train_sample(data_root: Path, per_class: int = 50, subdir: str = "train_sample") -> list[dict]:
    rows = []
    for cls, (_, species, contact) in CLASSES.items():
        for p in sorted((root(data_root) / subdir / cls).glob("*.mp4"))[:per_class]:
            rows.append({"class": cls, "species": species, "person_contact_implied": contact, "path": str(p), "clip": p.stem})
    return rows


def clips(data_root: Path) -> list[dict]:
    rows = []
    for cls, (_, species, contact) in CLASSES.items():
        for p in sorted((root(data_root) / "clips" / cls).glob("*.mp4")):
            rows.append({"class": cls, "species": species, "person_contact_implied": contact, "path": str(p), "clip": p.stem})
    return rows
