"""Weak posture labels for cats and dogs from Localized Narratives captions.

Localized Narratives (Pont-Tuset et al., 2020; CC BY 4.0) are spoken,
transcribed descriptions of Open Images photos. A caption is used only when
it mentions exactly one pet species, in the singular, with no counting or
plural words, and attaches exactly one posture word group to that animal in
the same sentence:

* ``lying``: lying, laying, lies, lays, sleeping, asleep, curled, resting
* ``sitting``: sitting, sits, seated, sat
* ``standing``: standing, stands, walking, walks, running, runs

These are WEAK labels (caption wording, not posture annotation). They are
used to fit and evaluate a coarse posture classifier, and every result built
on them is reported as agreement with caption-derived labels.

Splits follow Open Images: train (fit), test (development), validation
(final evaluation). The image box comes from the detector at analysis time,
as in the product.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from ..utils import download, sha256_file

CAPTIONS = {
    "train": ("open_images_train_v6_captions.jsonl", "a3ad02e46a9916f3fa513a9bdf8eeb4e3d6e4d108688ac3a6d7440e80b917f92"),
    "test": ("open_images_test_captions.jsonl", "382d1c59be20cddf336444166b69cd2e66dc16b57478de51744584d4c92370b1"),
    "validation": ("open_images_validation_captions.jsonl", "01c5eb64af111c546c8b0fec7ea1fee7f74ea9c0449c493e49c0bb749c2d618a"),
}
BASE_URL = "https://storage.googleapis.com/localized-narratives/annotations/"
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
POSTURES = ("lying", "sitting", "standing")
_CAT = r"(?:cat|kitten|kitty)"
_DOG = r"(?:dog|puppy|pup)"
_PLURAL = re.compile(r"\b(?:cats|kittens|dogs|puppies|two|three|four|five|group|couple|few|many|some)\b", re.I)
_POST = {
    "lying": r"(?:lying|laying|lies|lays|sleeping|asleep|curled|resting|lay down)",
    "sitting": r"(?:sitting|sits|seated|sat)",
    "standing": r"(?:standing|stands|walking|walks|running|runs)",
}


def label_caption(caption: str) -> tuple[str, str] | None:
    """(species, posture) or None when the caption is ambiguous."""
    c = caption.lower()
    sp = [s for s, pat in (("cat", _CAT), ("dog", _DOG)) if re.search(rf"\b{pat}\b", c)]
    if len(sp) != 1 or _PLURAL.search(c):
        return None
    pat = _CAT if sp[0] == "cat" else _DOG
    found = set()
    for sent in re.split(r"[.;]", c):
        if not re.search(rf"\b{pat}\b", sent):
            continue
        for k, p in _POST.items():
            if re.search(rf"\b{pat}\b[^.]{{0,40}}\b{p}\b", sent):
                found.add(k)
    if len(found) != 1:
        return None
    return sp[0], found.pop()


def root(data_root: Path) -> Path:
    return data_root / "localized_narratives"


def fetch_captions(data_root: Path) -> None:
    d = root(data_root)
    for name, sha in CAPTIONS.values():
        p = d / name
        if not (p.exists() and sha256_file(p) == sha):
            download(BASE_URL + name, p, expected_sha256=sha, timeout=900)


def build(data_root: Path, manifest_dir: Path) -> dict:
    """Write manifests/posture_weak_images.csv (one row per image with a usable caption)."""
    rows, counts = [], {}
    for split, (name, _) in CAPTIONS.items():
        oi_split = "validation" if split == "validation" else split
        seen = set()
        with open(root(data_root) / name, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                lab = label_caption(r["caption"])
                if lab is None or r["image_id"] in seen:
                    continue
                seen.add(r["image_id"])
                rows.append({"image_id": r["image_id"], "split": split, "species": lab[0], "posture": lab[1],
                             "annotator_id": r.get("annotator_id"), "download_url": IMAGE_URL.format(split=oi_split, image_id=r["image_id"])})
                counts[f"{split}/{lab[0]}/{lab[1]}"] = counts.get(f"{split}/{lab[0]}/{lab[1]}", 0) + 1
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "posture_weak_images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "split", "species", "posture", "annotator_id", "download_url"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["split"], r["image_id"])))
    return {"images": len(rows), "counts": dict(sorted(counts.items()))}


def fetch_images(data_root: Path, manifest_dir: Path, workers: int = 6) -> dict:
    from . import openimages as OI

    return OI.fetch_images(data_root, manifest_dir, workers=workers, image_dir=root(data_root) / "images",
                           images_csv="posture_weak_images.csv", sums_csv="posture_weak_sha256.csv")
