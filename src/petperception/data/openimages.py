"""Open Images V7 validation subset for cat/dog detection evaluation (E4).

Open Images (https://storage.googleapis.com/openimages/web/index.html)
provides human-verified image-level labels and bounding boxes. Boxes are
drawn exhaustively for every verified-positive boxable label in an image, and
image-level labels can also be verified negative. That makes a
labelled cat/dog detection benchmark possible without inventing ground truth.

Terms: annotations CC BY 4.0 (Google); each image carries its own license
(the validation metadata lists CC BY 2.0 for all images) and author
attribution, which this module copies into the subset manifest.

Protocol choices (documented in docs/ALGORITHM_REPORT.md):
* images with a non-zero/unknown ``Rotation`` are excluded (box frame ambiguity);
* per class, evaluation uses only images where that class is verified
  (positive with boxes, or human-verified negative);
* ``IsDepiction`` boxes (drawings, toys, photos of animals) are ignore regions;
* ``IsGroupOf`` boxes are ignore regions for AP and count as present for
  image-level presence.
"""

from __future__ import annotations

import csv
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ..utils import DownloadError, download, sha256_file

LABELS = {"/m/01yrx": "cat", "/m/0bt9lr": "dog"}
ANNOTATION_FILES = {
    "validation-annotations-bbox.csv": {
        "url": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        "sha256": None,
    },
    "oidv7-val-annotations-human-imagelabels.csv": {
        "url": "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv",
        "sha256": None,
    },
    "oidv7-class-descriptions-boxable.csv": {
        "url": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
        "sha256": None,
    },
    "validation-images-with-rotation.csv": {
        "url": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
        "sha256": None,
    },
}
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
DEFAULT_SEED = 20261009


@dataclass
class GtBox:
    image_id: str
    label: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    is_occluded: int
    is_truncated: int
    is_group_of: int
    is_depiction: int
    is_inside: int


def dataset_dir(data_root: Path) -> Path:
    return data_root / "openimages"


def fetch_annotations(data_root: Path, expected: dict[str, str] | None = None) -> dict[str, dict]:
    d = dataset_dir(data_root)
    out = {}
    for name, spec in ANNOTATION_FILES.items():
        dest = d / name
        want = (expected or {}).get(name)
        if dest.exists() and (want is None or sha256_file(dest) == want):
            out[name] = {"path": str(dest), "sha256": sha256_file(dest), "bytes": dest.stat().st_size, "cached": True}
            continue
        out[name] = download(spec["url"], dest, expected_sha256=want, timeout=600)
    return out


def _rows(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(line.replace("\r", "") for line in fh)


def load_boxes(d: Path) -> list[GtBox]:
    boxes = []
    for r in _rows(d / "validation-annotations-bbox.csv"):
        lab = LABELS.get(r["LabelName"])
        if lab is None:
            continue
        boxes.append(
            GtBox(
                image_id=r["ImageID"], label=lab,
                xmin=float(r["XMin"]), xmax=float(r["XMax"]), ymin=float(r["YMin"]), ymax=float(r["YMax"]),
                is_occluded=int(r["IsOccluded"]), is_truncated=int(r["IsTruncated"]),
                is_group_of=int(r["IsGroupOf"]), is_depiction=int(r["IsDepiction"]), is_inside=int(r["IsInside"]),
            )
        )
    return boxes


def load_verified_labels(d: Path) -> dict[str, dict[str, str]]:
    """image_id -> {cat|dog: 'pos'|'neg'} from human-verified image labels."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for r in _rows(d / "oidv7-val-annotations-human-imagelabels.csv"):
        lab = LABELS.get(r["LabelName"])
        if lab is None:
            continue
        conf = float(r["Confidence"])
        out[r["ImageID"]][lab] = "pos" if conf >= 0.5 else "neg"
    return out


def load_image_meta(d: Path, ids: set[str] | None = None) -> dict[str, dict[str, str]]:
    meta = {}
    for r in _rows(d / "validation-images-with-rotation.csv"):
        if ids is None or r["ImageID"] in ids:
            meta[r["ImageID"]] = r
    return meta


def build_subset(data_root: Path, out_dir: Path, seed: int = DEFAULT_SEED, max_per_positive_class: int = 350) -> dict[str, object]:
    """Select a reproducible evaluation subset and write manifests.

    Writes ``openimages_catdog_images.csv`` and ``openimages_catdog_boxes.csv``.
    """
    d = dataset_dir(data_root)
    boxes = load_boxes(d)
    verified = load_verified_labels(d)
    boxed: dict[str, set[str]] = defaultdict(set)
    for b in boxes:
        boxed[b.label].add(b.image_id)
    all_ids = set(verified) | boxed["cat"] | boxed["dog"]
    meta = load_image_meta(d, all_ids)

    def rotation_ok(i: str) -> bool:
        m = meta.get(i)
        return m is not None and (m.get("Rotation", "").strip() in ("", "0", "0.0"))

    def status(i: str, lab: str) -> str:
        if i in boxed[lab]:
            return "pos"
        return verified.get(i, {}).get(lab, "unverified")

    rng = random.Random(seed)
    excluded_rotation = sorted(i for i in all_ids if not rotation_ok(i))
    cat_pos = sorted(i for i in boxed["cat"] if rotation_ok(i))
    dog_pos = sorted(i for i in boxed["dog"] if rotation_ok(i))
    negatives = sorted(
        i for i in verified
        if rotation_ok(i) and i not in boxed["cat"] and i not in boxed["dog"]
        and ("neg" in (verified[i].get("cat"), verified[i].get("dog")))
        and verified[i].get("cat") != "pos" and verified[i].get("dog") != "pos"
    )
    sel_cat = sorted(rng.sample(cat_pos, min(max_per_positive_class, len(cat_pos))))
    remaining_dog = [i for i in dog_pos if i not in set(sel_cat)]
    sel_dog = sorted(rng.sample(remaining_dog, min(max_per_positive_class, len(remaining_dog))))
    selected = sorted(set(sel_cat) | set(sel_dog) | set(negatives))

    out_dir.mkdir(parents=True, exist_ok=True)
    img_csv = out_dir / "openimages_catdog_images.csv"
    with open(img_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "selection_role", "cat_status", "dog_status", "license", "author", "author_profile_url", "original_landing_url", "title", "download_url"])
        for i in selected:
            role = "cat_positive_pool" if i in sel_cat else ("dog_positive_pool" if i in sel_dog else "verified_negative_pool")
            m = meta[i]
            w.writerow([i, role, status(i, "cat"), status(i, "dog"), m["License"], m["Author"], m["AuthorProfileURL"], m["OriginalLandingURL"], m["Title"], IMAGE_URL.format(image_id=i)])
    box_csv = out_dir / "openimages_catdog_boxes.csv"
    sel = set(selected)
    with open(box_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "label", "xmin", "xmax", "ymin", "ymax", "is_occluded", "is_truncated", "is_group_of", "is_depiction", "is_inside"])
        for b in boxes:
            if b.image_id in sel:
                w.writerow([b.image_id, b.label, b.xmin, b.xmax, b.ymin, b.ymax, b.is_occluded, b.is_truncated, b.is_group_of, b.is_depiction, b.is_inside])
    return {
        "seed": seed,
        "selected_images": len(selected),
        "cat_positive_pool_total": len(cat_pos),
        "dog_positive_pool_total": len(dog_pos),
        "selected_cat_positive": len(sel_cat),
        "selected_dog_positive": len(sel_dog),
        "selected_verified_negative": len(negatives),
        "excluded_for_rotation": len(excluded_rotation),
        "boxes_in_subset": sum(1 for b in boxes if b.image_id in sel),
    }


def read_subset(manifest_dir: Path) -> tuple[list[dict[str, str]], list[GtBox]]:
    images = list(_rows(manifest_dir / "openimages_catdog_images.csv"))
    boxes = []
    for r in _rows(manifest_dir / "openimages_catdog_boxes.csv"):
        boxes.append(GtBox(r["image_id"], r["label"], float(r["xmin"]), float(r["xmax"]), float(r["ymin"]), float(r["ymax"]),
                           int(r["is_occluded"]), int(r["is_truncated"]), int(r["is_group_of"]), int(r["is_depiction"]), int(r["is_inside"])))
    return images, boxes


def fetch_images(data_root: Path, manifest_dir: Path, workers: int = 4, retries: int = 4, image_dir: Path | None = None) -> dict[str, object]:
    """Download subset images and record their SHA-256 in a checksum manifest.

    The S3 copies are re-encoded relative to the Flickr originals, so the
    ``OriginalMD5`` field cannot be used; first retrieval pins our own digest.
    """
    images = list(_rows(manifest_dir / "openimages_catdog_images.csv"))
    img_dir = image_dir or dataset_dir(data_root) / "images"
    sums_path = manifest_dir / "openimages_catdog_sha256.csv"
    pinned: dict[str, str] = {}
    pinned_bytes: dict[str, int] = {}
    if sums_path.exists():
        for r in _rows(sums_path):
            pinned[r["image_id"]] = r["sha256"]
            pinned_bytes[r["image_id"]] = int(r["bytes"])
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}

    def one(row: dict[str, str]):
        i = row["image_id"]
        dest = img_dir / f"{i}.jpg"
        if dest.exists() and (i not in pinned or sha256_file(dest) == pinned[i]):
            return i, {"sha256": sha256_file(dest), "bytes": dest.stat().st_size}
        last: Exception | None = None
        for attempt in range(retries):
            try:
                rec = download(row["download_url"], dest, expected_sha256=pinned.get(i), timeout=120)
                return i, {"sha256": rec["sha256"], "bytes": rec["bytes"]}
            except DownloadError as exc:  # the S3 mirror returns sporadic 404s under load
                last = exc
                time.sleep(2.0 * (attempt + 1))
        raise last  # type: ignore[misc]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, r): r["image_id"] for r in images}
        for f in as_completed(futs):
            try:
                i, rec = f.result()
                results[i] = rec
            except DownloadError as exc:
                errors[futs[f]] = str(exc)
    # Pin digests on first retrieval; previously pinned digests never change.
    merged = {i: (pinned_bytes.get(i, results.get(i, {}).get("bytes")), d) for i, d in pinned.items()}
    for i, rec in results.items():
        merged.setdefault(i, (rec["bytes"], rec["sha256"]))
    with open(sums_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "bytes", "sha256"])
        for i in sorted(merged):
            w.writerow([i, merged[i][0], merged[i][1]])
    return {"downloaded_or_verified": len(results), "errors": errors, "bytes": sum(r["bytes"] for r in results.values())}
