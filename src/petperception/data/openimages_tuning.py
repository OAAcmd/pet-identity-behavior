"""Open Images V7 *test-split* tuning set for false-positive rejection.

Separation of data:
* tuning / model fitting: Open Images **test** split images selected here;
* final evaluation: the E4 **validation** subset (unchanged since Phase 1).
The two splits contain different images by construction; the builder also
asserts that no selected ID appears in the E4 manifest.

Selection (seed 20261010, rotation-safe images only):
* up to 500 cat-positive and 500 dog-positive images (boxed);
* all images with a human-verified negative for cat or dog and no positive;
* up to 700 images that contain boxes of visually confusable classes
  (foxes, big cats, bears, raccoons, rabbits, toys, ...) and no cat/dog
  boxes. Their confusable *boxes* are certain negatives for a crop-level
  verifier even when the image-level cat/dog label is unverified.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from ..utils import download
from .openimages import IMAGE_URL as _VAL_URL  # noqa: F401  (documentation: same S3 mirror layout)
from .openimages import _rows

TEST_IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/test/{image_id}.jpg"
ANNOTATIONS = {
    "test-annotations-bbox.csv": ("https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
                                  "fe22e579b9453875601576859d14ef3304058165de139c66b725e599202bd7b1"),
    "oidv7-test-annotations-human-imagelabels.csv": ("https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-human-imagelabels.csv",
                                                     "917d3ca8c0a2e09d652fb3148e9264df6117bfcaccf425a60a29b23269487953"),
    "test-images-with-rotation.csv": ("https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
                                      "de55c6b8cbda32a79f4a20f28572d54e9c49d527a4006725675764074a51a36d"),
}
PETS = {"/m/01yrx": "cat", "/m/0bt9lr": "dog"}
CONFUSABLE = {
    "/m/0306r": "fox", "/m/096mb": "lion", "/m/07dm6": "tiger", "/m/0c29q": "leopard", "/m/0cd4d": "cheetah", "/m/0449p": "jaguar",
    "/m/04g2r": "lynx", "/m/0dq75": "raccoon", "/m/06l9r": "red_panda", "/m/01dws": "bear", "/m/01dxs": "brown_bear", "/m/0633h": "polar_bear",
    "/m/03bj1": "panda", "/m/06mf6": "rabbit", "/m/071qp": "squirrel", "/m/0km7z": "skunk", "/m/07bgp": "sheep", "/m/03fwl": "goat",
    "/m/09kx5": "deer", "/m/0cn6p": "otter", "/m/03qrc": "hamster", "/m/04rmv": "mouse", "/m/04c0y": "kangaroo", "/m/08pbxl": "monkey",
    "/m/0kmg4": "teddy_bear", "/m/03k3r": "horse", "/m/01xq0k1": "cattle", "/m/068zj": "pig", "/m/0gd36": "sea_lion", "/m/0cl4p": "hedgehog",
    "/m/04cp_": "koala", "/m/0pcr": "alpaca", "/m/0138tl": "toy", "/m/06msq": "sculpture", "/m/0167gd": "doll",
}
SEED = 20261010


def root(data_root: Path) -> Path:
    return data_root / "openimages_test"


def fetch_annotations(data_root: Path) -> None:
    d = root(data_root)
    for name, (url, sha) in ANNOTATIONS.items():
        p = d / name
        from ..utils import sha256_file

        if not (p.exists() and sha256_file(p) == sha):
            download(url, p, expected_sha256=sha, timeout=900)


def build(data_root: Path, manifest_dir: Path, n_pos: int = 500, n_conf: int = 700) -> dict:
    d = root(data_root)
    boxes = defaultdict(list)
    for r in _rows(d / "test-annotations-bbox.csv"):
        lab = PETS.get(r["LabelName"]) or CONFUSABLE.get(r["LabelName"])
        if lab:
            boxes[r["ImageID"]].append((lab, r))
    verified: dict[str, dict[str, str]] = defaultdict(dict)
    for r in _rows(d / "oidv7-test-annotations-human-imagelabels.csv"):
        lab = PETS.get(r["LabelName"])
        if lab:
            verified[r["ImageID"]][lab] = "pos" if float(r["Confidence"]) >= 0.5 else "neg"
    ids = set(boxes) | set(verified)
    meta = {r["ImageID"]: r for r in _rows(d / "test-images-with-rotation.csv") if r["ImageID"] in ids}
    ok = {i for i in ids if i in meta and meta[i].get("Rotation", "").strip() in ("", "0", "0.0")}
    has = {i: {lab for lab, _ in boxes[i]} for i in boxes}
    cat_pos = sorted(i for i in ok if "cat" in has.get(i, ()))
    dog_pos = sorted(i for i in ok if "dog" in has.get(i, ()) and "cat" not in has.get(i, ()))
    negs = sorted(i for i in ok if not ({"cat", "dog"} & has.get(i, set())) and "neg" in verified.get(i, {}).values()
                  and "pos" not in verified.get(i, {}).values())
    conf = sorted(i for i in ok if has.get(i) and not ({"cat", "dog"} & has[i]) and verified.get(i, {}).get("cat") != "pos"
                  and verified.get(i, {}).get("dog") != "pos" and i not in set(negs))
    rng = random.Random(SEED)
    sel_cat = rng.sample(cat_pos, min(n_pos, len(cat_pos)))
    sel_dog = rng.sample(dog_pos, min(n_pos, len(dog_pos)))
    sel_conf = rng.sample(conf, min(n_conf, len(conf)))
    roles = {**{i: "cat_positive" for i in sel_cat}, **{i: "dog_positive" for i in sel_dog}, **{i: "verified_negative" for i in negs},
             **{i: "confusable_animal" for i in sel_conf}}
    with open(manifest_dir / "openimages_catdog_images.csv", newline="", encoding="utf-8") as fh:
        e4_ids = {r["image_id"] for r in csv.DictReader(fh)}
    overlap = set(roles) & e4_ids
    if overlap:
        raise RuntimeError(f"tuning set overlaps E4 evaluation images: {sorted(overlap)[:5]}")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "oi_test_tuning_images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "role", "cat_status", "dog_status", "license", "author", "author_profile_url", "original_landing_url", "download_url"])
        for i in sorted(roles):
            m = meta[i]
            st = {lab: ("pos" if lab in has.get(i, ()) else verified.get(i, {}).get(lab, "unverified")) for lab in ("cat", "dog")}
            w.writerow([i, roles[i], st["cat"], st["dog"], m["License"], m["Author"], m["AuthorProfileURL"], m["OriginalLandingURL"],
                        TEST_IMAGE_URL.format(image_id=i)])
    with open(manifest_dir / "oi_test_tuning_boxes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "label", "xmin", "xmax", "ymin", "ymax", "is_occluded", "is_truncated", "is_group_of", "is_depiction"])
        for i in sorted(roles):
            for lab, r in boxes.get(i, []):
                w.writerow([i, lab, r["XMin"], r["XMax"], r["YMin"], r["YMax"], r["IsOccluded"], r["IsTruncated"], r["IsGroupOf"], r["IsDepiction"]])
    return {"seed": SEED, "cat_positive": len(sel_cat), "dog_positive": len(sel_dog), "verified_negative": len(negs),
            "confusable_animal": len(sel_conf), "total": len(roles), "pools": {"cat": len(cat_pos), "dog": len(dog_pos), "confusable": len(conf)},
            "overlap_with_e4": 0}


def fetch_images(data_root: Path, manifest_dir: Path, workers: int = 4) -> dict:
    """Download tuning images; reuse the validation fetcher's retry and pinning logic."""
    from . import openimages as OI

    img_dir = root(data_root) / "images"
    rows = list(_rows(manifest_dir / "oi_test_tuning_images.csv"))
    # Adapt rows to the generic fetcher (same columns it needs).
    tmp_manifest = manifest_dir / ".oi_test_fetch"
    tmp_manifest.mkdir(exist_ok=True)
    with open(tmp_manifest / "openimages_catdog_images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "download_url"])
        for r in rows:
            w.writerow([r["image_id"], r["download_url"]])
    sums = manifest_dir / "oi_test_tuning_sha256.csv"
    if sums.exists():
        (tmp_manifest / "openimages_catdog_sha256.csv").write_bytes(sums.read_bytes())
    out = OI.fetch_images(img_dir.parent.parent, tmp_manifest, workers=workers, image_dir=img_dir)
    (tmp_manifest / "openimages_catdog_sha256.csv").replace(sums)
    for f in tmp_manifest.iterdir():
        f.unlink()
    tmp_manifest.rmdir()
    return out
