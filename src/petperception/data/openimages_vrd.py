"""Open Images V7 visual relationships: person-pet interaction pairs (E13).

Why this source: Open Images relationship annotations are exhaustive for the
pairs they cover. Per the Open Images V7 description, for an image that can
contain a triplet such as (Woman, interacts_with, Dog), every (woman, dog)
box pair in that relation is listed and all other (woman, dog) pairs are
reliable negatives (human-human relationships are the stated exception and
are not used here). That gives real positives *and* real negatives for
person-pet interaction in still images.

Pairs: every box of Man, Woman, Boy or Girl (the person classes used in
relationship triplets) with every box of Cat or Dog, in images that appear in
the relationship files with at least one relationship row involving one of
those classes (so the image was annotated for them). Group-of and depiction
boxes are excluded; images with a non-zero rotation are excluded.
Label: 1 if the (person, pet) pair carries any person-to-pet relationship
(``interacts_with``, ``holds``, ...), else 0. Relationship boxes are drawn
separately from the box annotations, so each relation is mapped to the
annotated boxes of the same classes with the highest IoU (>= 0.5); an image
with any relation that cannot be mapped is dropped. Pet-to-pet relationships
do not exist in the files.

Splits (image level): Open Images train (fit, with a seeded 10 % development
hold-out); Open Images validation and test (final evaluation, untouched until
the end). Train images are capped: every image with a positive pair plus an
equal-sized seeded sample of images with only negative pairs.

Terms: annotations CC BY 4.0 (Google); images CC BY 2.0 per the Open Images
metadata, with author attribution copied into the manifest.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..utils import download, sha256_file
from .openimages import _rows

PERSON = {"/m/04yx4": "man", "/m/03bt1vf": "woman", "/m/01bl7v": "boy", "/m/05r655": "girl"}
PET = {"/m/01yrx": "cat", "/m/0bt9lr": "dog"}
SEED = 20261013
DEV_FRACTION = 0.10
MATCH_IOU = 0.5
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
VRD = {"train": "oidv6-train-annotations-vrd.csv", "validation": "validation-annotations-vrd.csv", "test": "test-annotations-vrd.csv"}
SOURCES = {
    "oidv6-train-annotations-vrd.csv": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-vrd.csv",
    "validation-annotations-vrd.csv": "https://storage.googleapis.com/openimages/v5/validation-annotations-vrd.csv",
    "test-annotations-vrd.csv": "https://storage.googleapis.com/openimages/v5/test-annotations-vrd.csv",
    "oidv6-train-annotations-bbox.csv": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "train-images-boxable-with-rotation.csv": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
}
# Validation / test boxes and image metadata already pinned by the E4 and E7 builders.
LOCAL = {"validation": ("openimages/validation-annotations-bbox.csv", "openimages/validation-images-with-rotation.csv"),
         "test": ("openimages_test/test-annotations-bbox.csv", "openimages_test/test-images-with-rotation.csv")}


def root(data_root: Path) -> Path:
    return data_root / "openimages_vrd"


def fetch_annotations(data_root: Path, manifest_dir: Path) -> dict:
    """Download the relationship files and the train boxes / metadata; pin SHA-256 on first retrieval."""
    d = root(data_root) / "_download"
    pins_path = manifest_dir / "oi_vrd_sources_sha256.csv"
    pins = {r["file"]: r["sha256"] for r in _rows(pins_path)} if pins_path.exists() else {}
    out = {}
    for name, url in SOURCES.items():
        p = d / name
        if not p.exists():
            download(url, p, expected_sha256=pins.get(name), timeout=3600)
        sha = sha256_file(p)
        if name in pins and pins[name] != sha:
            raise RuntimeError(f"{name}: SHA-256 differs from the pinned value")
        out[name] = sha
    pins_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pins_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "url", "sha256"])
        for name in sorted(out):
            w.writerow([name, SOURCES[name], out[name]])
    return out


def _annotated_images(path: Path) -> tuple[set[str], dict[str, list[tuple]]]:
    """Images with a relationship row involving a person or pet class, and their person->pet relations."""
    images: set[str] = set()
    rel: dict[str, list[tuple]] = defaultdict(list)
    for r in _rows(path):
        a, b = r["LabelName1"], r["LabelName2"]
        if a in PERSON or a in PET or b in PERSON or b in PET:
            images.add(r["ImageID"])
        if a in PERSON and b in PET:
            rel[r["ImageID"]].append((a, tuple(float(r[k]) for k in ("XMin1", "XMax1", "YMin1", "YMax1")),
                                      b, tuple(float(r[k]) for k in ("XMin2", "XMax2", "YMin2", "YMax2")), r["RelationshipLabel"]))
    return images, rel


def _iou(a: tuple, b: tuple) -> float:
    """Boxes as (xmin, xmax, ymin, ymax)."""
    ix = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ix * iy
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def _coords(b: dict) -> tuple:
    return tuple(float(b[k]) for k in ("XMin", "XMax", "YMin", "YMax"))


def _match(rels: list[tuple], boxes: list[dict]) -> tuple[dict[tuple[int, int], set[str]], float] | None:
    """Map relation boxes to annotated boxes (same class, max IoU >= MATCH_IOU).

    Relationship boxes are drawn separately from the box annotations (about two thirds of
    them differ slightly), so exact coordinates cannot be used. Returns {(person box index,
    pet box index): relations} and the smallest matching IoU, or None when any relation
    cannot be matched (the image is then dropped: its negatives would not be reliable).
    """
    out: dict[tuple[int, int], set[str]] = defaultdict(set)
    worst = 1.0
    for pl, pbox, ql, qbox, lab in rels:
        best = []
        for lab_name, box in ((pl, pbox), (ql, qbox)):
            cands = [(_iou(box, _coords(b)), k) for k, b in enumerate(boxes) if b["LabelName"] == lab_name]
            if not cands or max(cands)[0] < MATCH_IOU:
                return None
            best.append(max(cands))
        worst = min(worst, best[0][0], best[1][0])
        out[(best[0][1], best[1][1])].add(lab)
    return out, worst


def _boxes(path: Path, ids: set[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in _rows(path):
        if r["ImageID"] in ids and (r["LabelName"] in PERSON or r["LabelName"] in PET):
            out[r["ImageID"]].append(r)
    return out


def build(data_root: Path, manifest_dir: Path) -> dict:
    d = root(data_root) / "_download"
    pairs: list[dict] = []
    images: dict[str, dict] = {}
    stats: dict = {}
    for split, vrd in VRD.items():
        ann, rel = _annotated_images(d / vrd)
        box_path, meta_path = (d / "oidv6-train-annotations-bbox.csv", d / "train-images-boxable-with-rotation.csv") if split == "train" \
            else tuple(data_root / p for p in LOCAL[split])
        boxes = _boxes(box_path, ann)
        meta = {r["ImageID"]: r for r in _rows(meta_path) if r["ImageID"] in boxes}
        by_img: dict[str, list[dict]] = {}
        dropped_unmatched = 0
        match_ious: list[float] = []
        for i, bs in boxes.items():
            m = meta.get(i)
            if m is None or m.get("Rotation", "").strip() not in ("", "0", "0.0"):
                continue
            matched = _match(rel.get(i, []), bs)
            if matched is None:
                dropped_unmatched += 1
                continue
            positive, worst = matched
            if rel.get(i):
                match_ious.append(worst)
            ok = [k for k, b in enumerate(bs) if b["IsGroupOf"] == "0" and b["IsDepiction"] == "0"]
            persons = [k for k in ok if bs[k]["LabelName"] in PERSON]
            pets = [k for k in ok if bs[k]["LabelName"] in PET]
            if not persons or not pets:
                continue
            rows = []
            for pk in persons:
                for qk in pets:
                    p, q = bs[pk], bs[qk]
                    labels = sorted(positive.get((pk, qk), ()))
                    rows.append({"image_id": i, "split": split, "person": PERSON[p["LabelName"]], "pet": PET[q["LabelName"]],
                                 **{f"person_{k.lower()}": p[k] for k in ("XMin", "XMax", "YMin", "YMax")},
                                 **{f"pet_{k.lower()}": q[k] for k in ("XMin", "XMax", "YMin", "YMax")},
                                 "relations": "|".join(labels), "interaction": int(bool(labels))})
            by_img[i] = rows
        stats[split] = {"annotated_images_with_person_and_pet": len(by_img), "pairs": sum(len(v) for v in by_img.values()),
                        "positive_pairs": sum(1 for rows in by_img.values() for r in rows if r["interaction"]),
                        "images_dropped_relation_not_matched": dropped_unmatched,
                        "min_match_iou_quantiles": [float(v) for v in np.percentile(match_ious, [0, 5, 50])] if match_ious else None}
        keep = sorted(by_img)
        if split == "train":
            pos = sorted(i for i in keep if any(r["interaction"] for r in by_img[i]))
            neg = sorted(i for i in keep if i not in set(pos))
            rng = random.Random(SEED)
            keep = sorted(pos + rng.sample(neg, min(len(neg), len(pos))))
            dev = set(rng.sample(keep, int(len(keep) * DEV_FRACTION)))
            stats[split]["kept_images"] = len(keep)
            stats[split]["development_images"] = len(dev)
        for i in keep:
            sp = "train_dev" if split == "train" and i in dev else split
            for r in by_img[i]:
                pairs.append({**r, "split": sp})
            m = meta[i]
            images[i] = {"image_id": i, "split": sp, "license": m.get("License", ""), "author": m.get("Author", ""),
                         "author_profile_url": m.get("AuthorProfileURL", ""), "original_landing_url": m.get("OriginalLandingURL", ""),
                         "download_url": IMAGE_URL.format(split=split, image_id=i)}
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "oi_vrd_interaction_images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(images.values()))))
        w.writeheader()
        w.writerows(images[i] for i in sorted(images))
    with open(manifest_dir / "oi_vrd_interaction_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(sorted(pairs, key=lambda r: (r["image_id"], r["person_xmin"], r["pet_xmin"])))
    return {"seed": SEED, "per_split": stats, "images": len(images), "pairs": len(pairs)}


def fetch_images(data_root: Path, manifest_dir: Path, workers: int = 6) -> dict:
    """Download the selected images and pin their SHA-256 (same fetcher as E4/E7)."""
    from . import openimages as OI

    img_dir = root(data_root) / "images"
    tmp = manifest_dir / ".oi_vrd_fetch"
    tmp.mkdir(exist_ok=True)
    with open(tmp / "openimages_catdog_images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "download_url"])
        for r in _rows(manifest_dir / "oi_vrd_interaction_images.csv"):
            w.writerow([r["image_id"], r["download_url"]])
    sums = manifest_dir / "oi_vrd_interaction_sha256.csv"
    if sums.exists():
        (tmp / "openimages_catdog_sha256.csv").write_bytes(sums.read_bytes())
    out = OI.fetch_images(img_dir.parent.parent, tmp, workers=workers, image_dir=img_dir)
    (tmp / "openimages_catdog_sha256.csv").replace(sums)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    return out
