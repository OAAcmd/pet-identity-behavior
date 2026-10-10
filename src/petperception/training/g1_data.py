"""G1 training data: a pet-and-household detector set from Open Images V7 train.

Why Open Images train: annotations CC BY 4.0, images listed as CC BY 2.0, no
registration, reachable from here (storage.googleapis.com and the
open-images-dataset S3 bucket). A detector trained only on it removes the
COCO terms question for the detector (docs/ASSET_TERMS.md Q-T) when trained
from scratch, and adapts the class set to the product either way.

What this job does (CPU, streaming; nothing but compact manifests is stored):
1. streams the train box annotations, the train human-verified image labels
   and the train image metadata, hashing each source file as it streams;
2. keeps boxes of the product classes (CLASSES; Open Images leaf classes
   merged, for example Man/Woman/Boy/Girl -> person);
3. selects images: every image with a cat or dog box; a capped sample of
   person-only images; for each household class a quota of images
   containing it (rarest class first, so bed, couch, bowl or teddy bear are
   not swamped by windows and cars); a capped sample of confusable-animal
   images (hard negatives). Images whose cat/dog boxes are depictions
   (drawings, toys) are dropped from the positives;
4. marks for each non-pet image whether cat and dog are human-verified
   absent (unboxed pets in negatives are the main label-noise risk);
5. leakage control: drops train images whose Flickr author also took an
   image of any evaluation set used in this project (E4, E7, E13/E15 test
   and validation, E8 backgrounds), which also removes re-uploads of an
   evaluation photo by its author (identical photos uploaded by different
   authors are not checked); then holds out a development split grouped by
   author;
6. writes var/training/g1/{images,boxes}.csv.gz and
   results/g1/data_summary.json (counts, licences, bytes to download,
   source checksums).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from ..utils import USER_AGENT, utc_now, write_json

SOURCES = {
    "boxes": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "image_labels": "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv",
    "images": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
}
CLASSES = {
    "cat": ["/m/01yrx"], "dog": ["/m/0bt9lr"],
    "person": ["/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655"],
    "bed": ["/m/03ssj5"], "couch": ["/m/02crq1", "/m/03m3pdh", "/m/026qbn5"], "chair": ["/m/01mzpv"],
    "table": ["/m/04bcr3", "/m/0h8n5zk", "/m/078n6m", "/m/01y9k5"], "car": ["/m/0k4j"], "bowl": ["/m/04kkgm"],
    "toy": ["/m/0138tl"], "teddy_bear": ["/m/0kmg4"], "door": ["/m/02dgv"], "window": ["/m/0d4v4"], "stairs": ["/m/01lynh"],
}
# Confusable animals and dolls: images with these and no cat/dog are hard negatives (no box is trained for them).
CONFUSERS = {"/m/0306r": "fox", "/m/0dq75": "raccoon", "/m/071qp": "squirrel", "/m/06mf6": "rabbit", "/m/01dws": "bear", "/m/04g2r": "lynx",
             "/m/07dm6": "tiger", "/m/0c29q": "leopard", "/m/096mb": "lion", "/m/0cd4d": "cheetah", "/m/0449p": "jaguar", "/m/06l9r": "red_panda",
             "/m/0cn6p": "otter", "/m/0km7z": "skunk", "/m/08pbxl": "monkey", "/m/07bgp": "sheep", "/m/03fwl": "goat", "/m/03qrc": "hamster",
             "/m/04rmv": "mouse", "/m/0167gd": "doll"}
PETS = {"/m/01yrx": "cat", "/m/0bt9lr": "dog"}
MID2CLS = {m: c for c, ms in CLASSES.items() for m in ms}
CAPS = {"person_only": 15000, "household_per_class": 2500, "confuser": 10000}
HOUSEHOLD = ["bed", "couch", "chair", "table", "car", "bowl", "toy", "teddy_bear", "door", "window", "stairs"]
DEV_FRACTION = 0.05
SEED = 20261016
EVAL_MANIFESTS = [("openimages_catdog_images.csv", None), ("oi_test_tuning_images.csv", None),
                  ("oi_vrd_interaction_images.csv", {"validation", "test"}), ("oi_vrd_object_images.csv", {"test"}),
                  ("e8_backgrounds.csv", None)]


def _stream(url: str, digest: hashlib._Hash, chunk: int = 1 << 20):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    resp = urllib.request.urlopen(req, timeout=600)

    class _Hashing(io.RawIOBase):
        def readable(self) -> bool:
            return True

        def readinto(self, b) -> int:
            data = resp.read(min(len(b), chunk))
            digest.update(data)
            b[: len(data)] = data
            return len(data)

    return io.TextIOWrapper(io.BufferedReader(_Hashing(), buffer_size=chunk), encoding="utf-8", newline="")


def _eval_authors(manifest_dir: Path) -> tuple[set[str], set[str]]:
    authors, ids = set(), set()
    for name, splits in EVAL_MANIFESTS:
        p = manifest_dir / name
        if not p.exists():
            continue
        for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
            if splits and r.get("split") not in splits:
                continue
            ids.add(r["image_id"])
            if r.get("author"):
                authors.add(r["author"].strip())
    return authors, ids


def build(data_root: Path, manifest_dir: Path, out_dir: Path, results_dir: Path) -> dict:
    sums: dict[str, str] = {}
    # 1. boxes of product classes, confuser presence, depiction flags
    boxes: dict[str, list[tuple]] = defaultdict(list)
    confuser_imgs: dict[str, set[str]] = defaultdict(set)
    depicted_pet: set[str] = set()
    h = hashlib.sha256()
    n_rows = 0
    for r in csv.DictReader(_stream(SOURCES["boxes"], h)):
        n_rows += 1
        m = r["LabelName"]
        if m in MID2CLS:
            if m in PETS and r["IsDepiction"] == "1":
                depicted_pet.add(r["ImageID"])
            boxes[r["ImageID"]].append((MID2CLS[m], float(r["XMin"]), float(r["YMin"]), float(r["XMax"]), float(r["YMax"]),
                                        int(r["IsOccluded"]), int(r["IsTruncated"]), int(r["IsGroupOf"]), int(r["IsDepiction"])))
        elif m in CONFUSERS:
            confuser_imgs[r["ImageID"]].add(CONFUSERS[m])
    sums["boxes"] = h.hexdigest()
    # 2. human-verified absence of cat and dog
    h = hashlib.sha256()
    verified_absent: dict[str, set[str]] = defaultdict(set)
    for r in csv.DictReader(_stream(SOURCES["image_labels"], h)):
        if r["LabelName"] in PETS and float(r["Confidence"]) == 0.0:  # written as "0.0" in the v7 file
            verified_absent[r["ImageID"]].add(PETS[r["LabelName"]])
    sums["image_labels"] = h.hexdigest()
    # 3. candidate pools
    pets = {i for i, bs in boxes.items() if any(b[0] in ("cat", "dog") for b in bs)}
    pets_clean = pets - depicted_pet
    non_pet = [i for i in boxes if i not in pets]
    person_only = sorted(i for i in non_pet if {b[0] for b in boxes[i]} == {"person"})
    household = sorted(i for i in non_pet if {b[0] for b in boxes[i]} - {"person"} and i not in confuser_imgs)
    confuser = sorted(i for i in confuser_imgs if i not in pets)
    rng = random.Random(SEED)

    def sample(pool: list[str], k: int) -> list[str]:
        # prefer images where both pets are human-verified absent
        ver = [i for i in pool if verified_absent.get(i, set()) >= {"cat", "dog"}]
        vs = set(ver)
        rest = [i for i in pool if i not in vs]
        rng.shuffle(ver)
        rng.shuffle(rest)
        return (ver + rest)[:k]

    selected = {i: "pet" for i in pets_clean}
    for i in sample(person_only, CAPS["person_only"]):
        selected.setdefault(i, "person_only")
    by_class: dict[str, list[str]] = defaultdict(list)
    for i in household:
        for c in {b[0] for b in boxes[i]} - {"person"}:
            by_class[c].append(i)
    for c in sorted(HOUSEHOLD, key=lambda c: len(by_class[c])):
        for i in sample(by_class[c], CAPS["household_per_class"]):
            selected.setdefault(i, "household")
    for i in sample(confuser, CAPS["confuser"]):
        selected.setdefault(i, "confuser")
    # 4. metadata, leakage control
    eval_authors, eval_ids = _eval_authors(manifest_dir)
    h = hashlib.sha256()
    meta: dict[str, dict] = {}
    for r in csv.DictReader(_stream(SOURCES["images"], h)):
        if r["ImageID"] in selected:
            meta[r["ImageID"]] = {"author": r["Author"].strip(), "license": r["License"], "md5": r["OriginalMD5"], "bytes": int(r["OriginalSize"] or 0),
                                  "url": r["OriginalURL"], "landing": r["OriginalLandingURL"], "rotation": r["Rotation"]}
    sums["images"] = h.hexdigest()
    dropped = Counter()
    dropped_by_role = Counter()
    dropped_authors = Counter()
    for i in list(selected):
        mt = meta.get(i)
        if mt is None:
            dropped["no_metadata"] += 1
        elif i in eval_ids:
            dropped["is_evaluation_image"] += 1
        elif mt["author"] in eval_authors:
            dropped["author_in_evaluation_sets"] += 1
            dropped_authors[mt["author"]] += 1
        else:
            continue
        dropped_by_role[selected[i]] += 1
        del selected[i]
    # 5. development split grouped by author
    authors = sorted({meta[i]["author"] for i in selected})
    rng2 = random.Random(SEED + 1)
    dev_authors = set(rng2.sample(authors, max(1, int(len(authors) * DEV_FRACTION))))
    split = {i: ("development" if meta[i]["author"] in dev_authors else "train") for i in selected}
    # 6. write
    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_dir / "images.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "split", "role", "cat_verified_absent", "dog_verified_absent", "confusers", "license", "author", "original_md5",
                    "original_bytes", "rotation", "download_url", "original_landing_url"])
        for i in sorted(selected):
            mt, va = meta[i], verified_absent.get(i, set())
            w.writerow([i, split[i], selected[i], int("cat" in va), int("dog" in va), ";".join(sorted(confuser_imgs.get(i, ()))), mt["license"],
                        mt["author"], mt["md5"], mt["bytes"], mt["rotation"], f"https://open-images-dataset.s3.amazonaws.com/train/{i}.jpg", mt["landing"]])
    with gzip.open(out_dir / "boxes.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "label", "xmin", "ymin", "xmax", "ymax", "is_occluded", "is_truncated", "is_group_of", "is_depiction"])
        for i in sorted(selected):
            for b in boxes[i]:
                w.writerow([i, *b])
    box_counts = {s: Counter() for s in ("train", "development")}
    img_counts = {s: Counter() for s in ("train", "development")}
    for i in selected:
        for c in {b[0] for b in boxes[i]}:
            img_counts[split[i]][c] += 1
        for b in boxes[i]:
            box_counts[split[i]][b[0]] += 1
    small = Counter()
    for i in selected:
        for b in boxes[i]:
            if b[0] in ("cat", "dog") and (b[3] - b[1]) * (b[4] - b[2]) < 0.01:
                small[b[0]] += 1
    summary = {
        "job": "G1 data", "created_at": utc_now().isoformat(timespec="seconds"), "protocol": __doc__.strip().splitlines(),
        "sources": {k: {"url": SOURCES[k], "sha256": sums[k]} for k in SOURCES}, "box_rows_streamed": n_rows,
        "classes": CLASSES, "confusers": sorted(set(CONFUSERS.values())), "caps": CAPS, "seed": SEED,
        "pools": {"pet_images": len(pets), "pet_images_with_depicted_pets_removed": len(pets - pets_clean), "person_only": len(person_only),
                  "household": len(household), "household_by_class": {c: len(by_class[c]) for c in HOUSEHOLD}, "confuser": len(confuser)},
        "selected_by_role": dict(Counter(selected.values())), "dropped_for_leakage": dict(dropped), "dropped_by_role": dict(dropped_by_role),
        "dropped_authors": {"distinct": len(dropped_authors), "top_20_image_counts": [n for _, n in dropped_authors.most_common(20)]},
        "evaluation_authors_excluded": len(eval_authors), "development_authors": len(dev_authors),
        "images": {s: sum(1 for i in selected if split[i] == s) for s in ("train", "development")},
        "images_per_class": {s: dict(img_counts[s]) for s in img_counts}, "boxes_per_class": {s: dict(box_counts[s]) for s in box_counts},
        "small_pet_boxes_under_1pct_of_image": dict(small),
        "non_pet_images_with_both_pets_verified_absent": sum(1 for i in selected if selected[i] != "pet" and verified_absent.get(i, set()) >= {"cat", "dog"}),
        "licences": dict(Counter(meta[i]["license"] for i in selected)),
        "original_bytes_total": sum(meta[i]["bytes"] for i in selected),
        "manifests": {"images": str(out_dir / "images.csv.gz"), "boxes": str(out_dir / "boxes.csv.gz")},
    }
    write_json(results_dir / "data_summary.json", summary)
    (results_dir / "classes.json").write_text(json.dumps(list(CLASSES), indent=1) + "\n")
    return summary
