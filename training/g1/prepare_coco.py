"""G1: turn the Open Images manifests into Model Maker COCO folders.

Input: var/training/g1/{images,boxes}.csv.gz written by
`uv run petperception train-prep g1` (main environment).
Output: <out>/{train,development}/{images/,labels.json} plus
<out>/download_log.csv (image id, bytes, SHA-256 of the original file).

Runs in the G1 training environment (training/g1/requirements.txt). Images
are downloaded from the open-images-dataset S3 bucket, decoded with Pillow,
EXIF/rotation-normalised by re-encoding, and resized so the longer side is at
most --max-side pixels (Model Maker trains at 256-384 px). Boxes are kept in
normalised coordinates and converted to pixels of the stored image.
Group-of boxes are kept; depiction boxes of non-pet classes are kept, pet
depictions were removed upstream. Downloaded files are untrusted: only
Pillow decodes them and nothing else is executed or unpacked.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

USER_AGENT = "petperception-g1/0.1 (+research)"


def load(manifest_dir: Path, limit: int | None, seed_order: bool = True) -> tuple[list[dict], dict[str, list[dict]]]:
    with gzip.open(manifest_dir / "images.csv.gz", "rt", newline="", encoding="utf-8") as fh:
        images = list(csv.DictReader(fh))
    if limit:
        # deterministic small subset for smoke tests: first `limit` pet images and limit/4 others per split
        keep = []
        for split in ("train", "development"):
            rows = [r for r in images if r["split"] == split]
            pets = [r for r in rows if r["role"] == "pet"][: limit if split == "train" else max(4, limit // 4)]
            other = [r for r in rows if r["role"] != "pet"][: max(1, (limit if split == "train" else max(4, limit // 4)) // 4)]
            keep += pets + other
        images = keep
    ids = {r["image_id"] for r in images}
    boxes: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(manifest_dir / "boxes.csv.gz", "rt", newline="", encoding="utf-8") as fh:
        for b in csv.DictReader(fh):
            if b["image_id"] in ids:
                boxes[b["image_id"]].append(b)
    return images, boxes


def fetch(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def store(row: dict, out_images: Path, max_side: int) -> tuple[str, int, str, int, int] | None:
    try:
        data = fetch(row["download_url"])
    except Exception:  # noqa: BLE001 - a missing image is logged and skipped
        return None
    sha = hashlib.sha256(data).hexdigest()
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    im.thumbnail((max_side, max_side))
    im.save(out_images / f"{row['image_id']}.jpg", quality=92)
    return row["image_id"], len(data), sha, im.width, im.height


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="var/training/g1")
    ap.add_argument("--out", default="var/training/g1/coco")
    ap.add_argument("--classes", default="results/g1/classes.json")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: number of pet images in train")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    classes = json.loads(Path(a.classes).read_text())
    cat_id = {c: i + 1 for i, c in enumerate(classes)}  # 0 = background (Model Maker convention)
    images, boxes = load(Path(a.manifests), a.limit)
    out = Path(a.out)
    log_rows = []
    for split in ("train", "development"):
        rows = [r for r in images if r["split"] == split]
        d = out / split / "images"
        d.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(a.workers) as ex:
            got = [g for g in ex.map(store, rows, [d] * len(rows), [a.max_side] * len(rows)) if g]
        coco = {"categories": [{"id": 0, "name": "background"}] + [{"id": cat_id[c], "name": c} for c in classes], "images": [], "annotations": []}
        for k, (iid, nbytes, sha, w, h) in enumerate(got):
            coco["images"].append({"id": k, "file_name": f"{iid}.jpg", "width": w, "height": h})
            for b in boxes[iid]:
                x0, y0, x1, y1 = (float(b[c]) for c in ("xmin", "ymin", "xmax", "ymax"))
                coco["annotations"].append({"id": len(coco["annotations"]), "image_id": k, "category_id": cat_id[b["label"]],
                                            "bbox": [x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h], "area": (x1 - x0) * w * (y1 - y0) * h,
                                            "iscrowd": int(b["is_group_of"])})
            log_rows.append((split, iid, nbytes, sha))
        (out / split / "labels.json").write_text(json.dumps(coco))
        print(json.dumps({"split": split, "requested": len(rows), "stored": len(got), "boxes": sum(len(boxes[g[0]]) for g in got)}))
    with open(out / "download_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "image_id", "original_bytes", "sha256"])
        w.writerows(log_rows)


if __name__ == "__main__":
    main()
