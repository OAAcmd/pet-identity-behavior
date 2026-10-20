"""G1 extra training views: cat oversampling and home-video frames (CoP3D), with contributor-disjoint splits.

Why. The full G1 run misses acceptance 1 only on cat AP50, and on home video
(E20) dogs are often reported as cats. Two data changes address that:
* `cat_repeat`: every training image that contains a cat appears that many
  times in the view (same file, new entry);
* `cop3d`: frames from CoP3D frame archives that E20 does not use
  (`cat_003`, `cat_004`, `dog_003`, `dog_004`), one box per frame from the
  foreground mask (pixels >= 128). The masks are model-generated, so these
  are pseudo-labels.

Leakage control. CoP3D sequence names start with a number shared by many
sequences (`1004_36_3`, `1004_40_7`, ...); it is treated as the contributor
(probably one household and pet). Every contributor of E20's archives
(`cat_001`, `dog_001`) is excluded. The remaining contributors are split
75 / 25 into training and development by a seeded shuffle. Development
frames select between candidates; E20 stays an untouched test set.

Terms: CoP3D is CC BY-NC 4.0, usable here for non-commercial research; a
model trained with it is non-commercial.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..data import cop3d
from ..data.mirror import RemoteZip

TRAIN_ARCHIVES = {"cat": ["cat_003.zip", "cat_004.zip"], "dog": ["dog_003.zip", "dog_004.zip"]}
EVAL_ARCHIVES = {"cat": ["cat_001.zip"], "dog": ["dog_001.zip"]}
SEED = 20261020
DEV_FRACTION = 0.25
MAX_SIDE = 320


def contributor(seq: str) -> str:
    return seq.split("_")[0]


def _sequences(archive: str) -> tuple[RemoteZip, dict[str, list[str]]]:
    z = RemoteZip([cop3d.BASE + archive])
    seqs: dict[str, list[str]] = {}
    for n in z.names():
        parts = n.split("/")
        if len(parts) == 4 and parts[2] == "images" and parts[3].endswith(".jpg"):
            seqs.setdefault(parts[1], []).append(n)
    return z, seqs


def cop3d_frames(store: Path, out_csv: Path, k_train: int = 8, k_dev: int = 10, workers: int = 16) -> list[dict]:
    """Sample frames, resize them into the G1 image store, and record mask boxes and the split."""
    if out_csv.exists():
        return list(csv.DictReader(open(out_csv, newline="", encoding="utf-8")))
    excluded = {sp: {contributor(s) for a in EVAL_ARCHIVES[sp] for s in _sequences(a)[1]} for sp in EVAL_ARCHIVES}
    jobs = []
    split_of: dict[str, str] = {}
    for sp, archives in TRAIN_ARCHIVES.items():
        opened = [_sequences(a) for a in archives]
        contribs = sorted({contributor(s) for _, seqs in opened for s in seqs} - excluded[sp])
        random.Random(f"{SEED}-{sp}").shuffle(contribs)
        n_dev = max(1, int(round(len(contribs) * DEV_FRACTION)))
        for i, c in enumerate(contribs):
            split_of[f"{sp}:{c}"] = "development" if i < n_dev else "train"
        for z, seqs in opened:
            for seq in sorted(seqs):
                if contributor(seq) in excluded[sp]:
                    continue
                split = split_of[f"{sp}:{contributor(seq)}"]
                names = sorted(seqs[seq])
                k = k_dev if split == "development" else k_train
                for n in sorted({names[int(round(i * (len(names) - 1) / max(1, k - 1)))] for i in range(k)}):
                    jobs.append((z, sp, seq, n, split))

    def get(job):
        z, sp, seq, n, split = job
        raw = z.read(n)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        mname = n.replace("/images/", "/masks/").replace(".jpg", ".png")
        if img is None or mname not in z.entries:
            return None
        box, area = cop3d.mask_box(cv2.imdecode(np.frombuffer(z.read(mname), np.uint8), cv2.IMREAD_GRAYSCALE))
        if box is None or not 0.005 <= area <= 0.95:
            return None
        h, w = img.shape[:2]
        s = min(1.0, MAX_SIDE / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        name = f"cop3d_{sp}_{seq}_{Path(n).stem}.jpg"
        cv2.imwrite(str(store / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return {"file_name": name, "species": sp, "sequence": seq, "contributor": contributor(seq), "split": split,
                "source_sha256": hashlib.sha256(raw).hexdigest(), "xmin": box[0], "ymin": box[1], "xmax": box[2], "ymax": box[3], "mask_area": area}

    store.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(get, jobs) if r is not None]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return rows


def build_view(base: Path, out: Path, store: Path, cat_repeat: int = 1, extra: list[dict] | None = None) -> dict:
    """A new COCO view from `base`: cat images repeated in train, extra frames added to their split."""
    summary = {}
    for split in ("train", "development"):
        coco = json.loads((base / split / "labels.json").read_text())
        cat_id = next(c["id"] for c in coco["categories"] if c["name"] == "cat")
        dog_id = next(c["id"] for c in coco["categories"] if c["name"] == "dog")
        by_img: dict[int, list] = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        images, anns = list(coco["images"]), list(coco["annotations"])
        if split == "train" and cat_repeat > 1:
            cat_imgs = [im for im in coco["images"] if any(a["category_id"] == cat_id for a in by_img.get(im["id"], []))]
            for _ in range(1, cat_repeat):
                for im in cat_imgs:
                    nid = len(images)
                    images.append({**im, "id": nid})
                    anns += [{**a, "id": len(anns) + j, "image_id": nid} for j, a in enumerate(by_img[im["id"]])]
        for row in extra or []:
            # Only training frames enter the view: its development split stays Open Images, so the trainer's
            # per-epoch AP50 is comparable across runs. CoP3D development frames are scored separately.
            if row["split"] != "train" or split != "train":
                continue
            p = store / row["file_name"]
            im = cv2.imread(str(p))
            if im is None:
                continue
            h, w = im.shape[:2]
            nid = len(images)
            images.append({"id": nid, "file_name": row["file_name"], "width": w, "height": h})
            x0, y0, x1, y1 = (float(row[c]) for c in ("xmin", "ymin", "xmax", "ymax"))
            bw, bh = min(x1 * w, w) - x0 * w, min(y1 * h, h) - y0 * h
            anns.append({"id": len(anns), "image_id": nid, "category_id": cat_id if row["species"] == "cat" else dog_id,
                         "bbox": [x0 * w, y0 * h, bw, bh], "area": bw * bh, "iscrowd": 0})
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        link = d / "images"
        if not link.exists():
            link.symlink_to(os.path.relpath(store, d), target_is_directory=True)
        (d / "labels.json").write_text(json.dumps({**coco, "images": images, "annotations": anns}))
        summary[split] = {"images": len(images), "boxes": len(anns), "base_images": len(coco["images"])}
    return summary
