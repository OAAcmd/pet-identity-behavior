"""Open Images V7 visual relationships: pet-object pairs (E15).

Relations (product names): a cat or dog **on** a bed, couch, chair or table;
**under** a table or chair; **inside** a car. Object groups are the Open
Images classes that correspond to a class the product detector (COCO) reports:

    bed    <- Bed                                        (detector: bed)
    couch  <- Couch, Sofa bed, Loveseat, Studio couch    (detector: couch)
    chair  <- Chair                                      (detector: chair)
    table  <- Table, Coffee table, Desk,
              Kitchen & dining room table                (detector: dining table)
    car    <- Car                                        (detector: car)

Dog beds, cat furniture, flowerpots, benches and vehicles other than cars are
not used: the detector has no matching class (or, for benches, the relation
data are too few to evaluate).

Labels: Open Images relationship annotation is exhaustive for the triplets in
its vocabulary (see data/openimages_vrd.py), so for a (pet box, object box)
pair in an annotated image, a relation that is not listed is a negative - but
only if the triplet (pet class, relation, object class) is in the vocabulary.
The vocabulary is taken as every triplet that occurs in any of the three
relationship files; for other combinations the label is left empty (unknown)
and the pair is not used for that relation. An image is "annotated" when it
has at least one relationship row involving a cat or dog.

Duplicate objects: Open Images sometimes has two boxes for one object (for
example "Couch" and "Loveseat"). Object boxes of the same group with IoU >=
0.7 are merged into one object whose labels are the union, so a duplicate box
does not turn a positive into a negative.

Relation boxes are mapped to annotated boxes of the same class with the
highest IoU (>= 0.5); an image with a relation that cannot be mapped is
dropped. Group-of and depiction boxes and rotated images are excluded.

Splits: all three Open Images splits are pooled and split by **photographer**
(Flickr author profile; the image id when missing) with a fixed seed into
fit (about 70 % of images), development (10 %) and test (20 %), so no
photographer - and so no home or animal photographed by the same person -
is on both sides. Fit and development keep every image with a positive pair
plus an equal-sized seeded sample of images with only negative pairs; the
test split keeps every image (the natural mix of annotated images).

Terms: annotations CC BY 4.0 (Google); images CC BY 2.0 per the Open Images
metadata, with author attribution copied into the manifest.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from ..vision.object_relation import RELATIONS
from . import openimages_vrd as VR
from .openimages import _rows

PET = VR.PET
OBJECTS = {"/m/03ssj5": ("bed", "Bed"), "/m/02crq1": ("couch", "Couch"), "/m/03m3pdh": ("couch", "Sofa bed"),
           "/m/0703r8": ("couch", "Loveseat"), "/m/026qbn5": ("couch", "Studio couch"), "/m/01mzpv": ("chair", "Chair"),
           "/m/04bcr3": ("table", "Table"), "/m/078n6m": ("table", "Coffee table"), "/m/01y9k5": ("table", "Desk"),
           "/m/0h8n5zk": ("table", "Kitchen & dining room table"), "/m/0k4j": ("car", "Car")}
SEED = 20261015
FRACTIONS = {"fit": 0.70, "dev": 0.10, "test": 0.20}
DUPLICATE_IOU = 0.7
SPLIT_FILES = {"train": ("oidv6-train-annotations-bbox.csv", "train-images-boxable-with-rotation.csv"),
               **{k: v for k, v in VR.LOCAL.items()}}


def _vocabulary(d: Path) -> set[tuple[str, str, str]]:
    voc = set()
    for vrd in VR.VRD.values():
        for r in _rows(d / vrd):
            if r["LabelName1"] in PET and r["LabelName2"] in OBJECTS and r["RelationshipLabel"] in RELATIONS:
                voc.add((r["LabelName1"], r["RelationshipLabel"], r["LabelName2"]))
    return voc


def _relations(path: Path) -> tuple[set[str], dict[str, list[tuple]]]:
    images: set[str] = set()
    rel: dict[str, list[tuple]] = defaultdict(list)
    for r in _rows(path):
        a, b = r["LabelName1"], r["LabelName2"]
        if a in PET or b in PET:
            images.add(r["ImageID"])
        if a in PET and b in OBJECTS and r["RelationshipLabel"] in RELATIONS:
            rel[r["ImageID"]].append((a, tuple(float(r[k]) for k in ("XMin1", "XMax1", "YMin1", "YMax1")),
                                      b, tuple(float(r[k]) for k in ("XMin2", "XMax2", "YMin2", "YMax2")), r["RelationshipLabel"]))
    return images, rel


def _boxes(path: Path, ids: set[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in _rows(path):
        if r["ImageID"] in ids and (r["LabelName"] in PET or r["LabelName"] in OBJECTS):
            out[r["ImageID"]].append(r)
    return out


def _objects(bs: list[dict], ok: list[int]) -> list[dict]:
    """Merge same-group object boxes that overlap (IoU >= DUPLICATE_IOU) into one object."""
    objs: list[dict] = []
    for k in ok:
        b = bs[k]
        if b["LabelName"] not in OBJECTS:
            continue
        g = OBJECTS[b["LabelName"]][0]
        for o in objs:
            if o["group"] == g and VR._iou(VR._coords(bs[o["boxes"][0]]), VR._coords(b)) >= DUPLICATE_IOU:
                o["boxes"].append(k)
                break
        else:
            objs.append({"group": g, "boxes": [k]})
    return objs


def build(data_root: Path, manifest_dir: Path) -> dict:
    d = VR.root(data_root) / "_download"
    voc = _vocabulary(d)
    cand: dict[str, list[dict]] = {}
    meta_all: dict[str, dict] = {}
    stats: dict = {"vocabulary": sorted(f"{PET[a]} {r} {OBJECTS[b][1]}" for a, r, b in voc)}
    for split, vrd in VR.VRD.items():
        ann, rel = _relations(d / vrd)
        box_name, meta_name = SPLIT_FILES[split]
        base = d if split == "train" else data_root
        boxes = _boxes(base / box_name, ann)
        meta = {r["ImageID"]: r for r in _rows(base / meta_name) if r["ImageID"] in boxes}
        dropped = 0
        for i, bs in boxes.items():
            m = meta.get(i)
            if m is None or m.get("Rotation", "").strip() not in ("", "0", "0.0"):
                continue
            matched = VR._match(rel.get(i, []), bs)
            if matched is None:
                dropped += 1
                continue
            positive, _ = matched
            ok = [k for k, b in enumerate(bs) if b["IsGroupOf"] == "0" and b["IsDepiction"] == "0"]
            pets = [k for k in ok if bs[k]["LabelName"] in PET]
            objs = _objects(bs, ok)
            if not pets or not objs:
                continue
            rows, used = [], set()
            for qk in pets:
                q = bs[qk]
                for o in objs:
                    ob = bs[o["boxes"][0]]
                    labs = set().union(*(positive.get((qk, k), set()) for k in o["boxes"]))
                    used.update((qk, k) for k in o["boxes"])
                    row = {"image_id": i, "pet": PET[q["LabelName"]], "object": o["group"],
                           "object_classes": "|".join(sorted({OBJECTS[bs[k]["LabelName"]][1] for k in o["boxes"]})),
                           **{f"pet_{k.lower()}": q[k] for k in ("XMin", "XMax", "YMin", "YMax")},
                           **{f"object_{k.lower()}": ob[k] for k in ("XMin", "XMax", "YMin", "YMax")}}
                    for r_, groups in RELATIONS.items():
                        applicable = o["group"] in groups and any((q["LabelName"], r_, bs[k]["LabelName"]) in voc for k in o["boxes"])
                        row[r_] = (1 if r_ in labs else 0) if applicable else ""
                    rows.append(row)
            if any(pair not in used for pair in positive):  # a relation sits on a group-of / depiction box
                stats[f"{split}_images_dropped_relation_on_excluded_box"] = stats.get(f"{split}_images_dropped_relation_on_excluded_box", 0) + 1
                continue
            if any(r[r_] != "" for r in rows for r_ in RELATIONS):
                cand[i] = rows
                meta_all[i] = {**m, "_oi_split": split}
        stats[f"{split}_images_dropped_relation_not_matched"] = dropped
    author = {i: (m.get("AuthorProfileURL") or "").strip() or f"image:{i}" for i, m in meta_all.items()}
    groups: dict[str, list[str]] = defaultdict(list)
    for i in sorted(cand):
        groups[author[i]].append(i)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    total, assigned, split_of = len(cand), {k: 0 for k in FRACTIONS}, {}
    for g in keys:  # fill the split furthest below its target share
        sp = min(FRACTIONS, key=lambda s: assigned[s] / total - FRACTIONS[s])
        for i in groups[g]:
            split_of[i] = sp
        assigned[sp] += len(groups[g])
    rng = random.Random(SEED + 1)
    keep: set[str] = set()
    for sp in FRACTIONS:
        ids = sorted(i for i in cand if split_of[i] == sp)
        pos = [i for i in ids if any(r[r_] == 1 for r in cand[i] for r_ in RELATIONS)]
        if sp == "test":
            keep.update(ids)
            continue
        neg = [i for i in ids if i not in set(pos)]
        keep.update(pos + rng.sample(neg, min(len(neg), len(pos))))
    pairs, images = [], {}
    for i in sorted(keep):
        for r in cand[i]:
            pairs.append({**r, "split": split_of[i]})
        m = meta_all[i]
        images[i] = {"image_id": i, "split": split_of[i], "oi_split": m["_oi_split"], "license": m.get("License", ""),
                     "author": m.get("Author", ""), "author_profile_url": m.get("AuthorProfileURL", ""),
                     "original_landing_url": m.get("OriginalLandingURL", ""),
                     "download_url": VR.IMAGE_URL.format(split=m["_oi_split"], image_id=i)}
    for sp in FRACTIONS:
        ps = [p for p in pairs if p["split"] == sp]
        stats[sp] = {"images": sum(1 for v in images.values() if v["split"] == sp),
                     "photographers": len({author[i] for i, v in images.items() if v["split"] == sp}),
                     "pairs": len(ps),
                     **{f"{r_}_positive": sum(1 for p in ps if p[r_] == 1) for r_ in RELATIONS},
                     **{f"{r_}_negative": sum(1 for p in ps if p[r_] == 0) for r_ in RELATIONS}}
    stats["candidate_images_before_negative_sampling"] = total
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "oi_vrd_object_images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(images.values()))))
        w.writeheader()
        w.writerows(images[i] for i in sorted(images))
    with open(manifest_dir / "oi_vrd_object_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(sorted(pairs, key=lambda r: (r["image_id"], r["pet_xmin"], r["object_xmin"], r["object"])))
    return {"seed": SEED, "fractions": FRACTIONS, **stats, "images": len(images), "pairs": len(pairs)}


def fetch_images(data_root: Path, manifest_dir: Path, workers: int = 6) -> dict:
    """Download the selected images and pin their SHA-256 (same fetcher as E4/E7/E13)."""
    from . import openimages as OI

    img_dir = VR.root(data_root) / "images"
    tmp = manifest_dir / ".oi_vrd_object_fetch"
    tmp.mkdir(exist_ok=True)
    with open(tmp / "openimages_catdog_images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "download_url"])
        for r in _rows(manifest_dir / "oi_vrd_object_images.csv"):
            w.writerow([r["image_id"], r["download_url"]])
    sums = manifest_dir / "oi_vrd_object_sha256.csv"
    if sums.exists():
        (tmp / "openimages_catdog_sha256.csv").write_bytes(sums.read_bytes())
    out = OI.fetch_images(img_dir.parent.parent, tmp, workers=workers, image_dir=img_dir)
    (tmp / "openimages_catdog_sha256.csv").replace(sums)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    return out
