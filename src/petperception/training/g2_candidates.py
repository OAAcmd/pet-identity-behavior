"""G2 annotation candidates: cat and dog images for licensed keypoint labels.

Every public dog and cat keypoint set is research-only or has unclear terms
(docs/DATA_SOURCES.md), so ears, tail and body keypoints usable outside
research need new annotation. This job selects the images to annotate from
the G1 Open Images manifest (CC BY 2.0 images, CC BY 4.0 boxes):

* exactly one cat or dog box in the image, which is not group-of or a
  depiction, and covers at least MIN_AREA of the image (keypoints need
  pixels). Truncated and occluded animals stay in (their hidden or
  out-of-frame points get visibility 1 or 0) and are flagged, so a budget
  can start with fully visible animals;
* split by Flickr author into train / development / test (70 / 10 / 20),
  so one photographer's pets never span splits;
* per species, ordered by a seeded shuffle so any prefix is a random
  sample (annotate in that order and stop at the budget).

Output: var/training/g2/candidates.csv (image id, species, box, split,
author, licence, URL, order) and results/g2/candidates_summary.json. The
annotation schema and procedure are in docs/training/G2_ANNOTATION_SPEC.md.
"""

from __future__ import annotations

import csv
import gzip
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..utils import utc_now, write_json

MIN_AREA = 0.10
SPLITS = (("train", 0.7), ("development", 0.1), ("test", 0.2))
SEED = 20261017


def build(g1_dir: Path, out_dir: Path, results_dir: Path) -> dict:
    with gzip.open(g1_dir / "images.csv.gz", "rt", newline="", encoding="utf-8") as fh:
        images = {r["image_id"]: r for r in csv.DictReader(fh) if r["role"] == "pet"}
    boxes: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(g1_dir / "boxes.csv.gz", "rt", newline="", encoding="utf-8") as fh:
        for b in csv.DictReader(fh):
            if b["image_id"] in images and b["label"] in ("cat", "dog"):
                boxes[b["image_id"]].append(b)
    reasons = Counter()
    keep = []
    for iid, bs in boxes.items():
        if len(bs) != 1:
            reasons["more_than_one_pet"] += 1
            continue
        b = bs[0]
        if b["is_group_of"] == "1" or b["is_depiction"] == "1":
            reasons["group_of_or_depiction"] += 1
            continue
        area = (float(b["xmax"]) - float(b["xmin"])) * (float(b["ymax"]) - float(b["ymin"]))
        if area < MIN_AREA:
            reasons["box_under_min_area"] += 1
            continue
        keep.append((iid, b, area))
    authors = sorted({images[i]["author"] for i, _, _ in keep})
    rng = random.Random(SEED)
    rng.shuffle(authors)
    cut1, cut2 = int(len(authors) * SPLITS[0][1]), int(len(authors) * (SPLITS[0][1] + SPLITS[1][1]))
    split_of = {a: ("train" if k < cut1 else "development" if k < cut2 else "test") for k, a in enumerate(authors)}
    rows = []
    for species in ("cat", "dog"):
        sel = [(i, b, a) for i, b, a in keep if b["label"] == species]
        sel.sort(key=lambda x: x[0])
        random.Random(SEED + (1 if species == "cat" else 2)).shuffle(sel)
        for order, (i, b, area) in enumerate(sel):
            im = images[i]
            rows.append({"order": order, "species": species, "image_id": i, "split": split_of[im["author"]], "xmin": b["xmin"], "ymin": b["ymin"],
                         "xmax": b["xmax"], "ymax": b["ymax"], "box_area": round(area, 4), "truncated": b["is_truncated"], "occluded": b["is_occluded"],
                         "author": im["author"], "license": im["license"],
                         "download_url": im["download_url"], "original_landing_url": im["original_landing_url"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"job": "G2 annotation candidates", "created_at": utc_now().isoformat(timespec="seconds"), "protocol": __doc__.strip().splitlines(),
               "pet_images_in_g1": len(images), "excluded": dict(reasons), "min_box_area": MIN_AREA,
               "candidates": {sp: dict(Counter(r["split"] for r in rows if r["species"] == sp)) for sp in ("cat", "dog")},
               "fully_visible": {sp: dict(Counter(r["split"] for r in rows if r["species"] == sp and r["truncated"] == "0" and r["occluded"] == "0"))
                                 for sp in ("cat", "dog")},
               "authors": len(authors), "licences": dict(Counter(r["license"] for r in rows)), "seed": SEED,
               "file": str(out_dir / "candidates.csv")}
    write_json(results_dir / "candidates_summary.json", summary)
    return summary
