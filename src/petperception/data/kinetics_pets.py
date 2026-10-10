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


def clips(data_root: Path) -> list[dict]:
    rows = []
    for cls, (_, species, contact) in CLASSES.items():
        for p in sorted((root(data_root) / "clips" / cls).glob("*.mp4")):
            rows.append({"class": cls, "species": species, "person_contact_implied": contact, "path": str(p), "clip": p.stem})
    return rows
