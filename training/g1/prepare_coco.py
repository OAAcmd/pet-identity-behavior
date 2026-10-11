"""G1: download Open Images images once and write Model Maker COCO views of them.

Input: var/training/g1/{images,boxes}.csv.gz written by
`uv run petperception train-prep g1` (main environment).

Layout (all under --root, default var/training/g1; nothing here is committed):
* images/<image_id>.jpg - one shared store; each image is downloaded once,
  decoded with Pillow, rotation-normalised by re-encoding and resized so its
  longer side is at most --max-side pixels;
* download_log.csv - image id, original bytes, SHA-256 of the original file,
  decode mode (merged across runs; rows without a decode mode, the first 4,600
  bench images, were decoded at full resolution, later ones with a JPEG draft
  decode to at least twice the target size, which is how Pillow's thumbnail
  reduces large JPEGs; both end in the same bicubic resize to --max-side);
* views/<name>/{train,development}/{images -> ../../../images, labels.json} -
  a COCO view per training run: the full set, or a seeded random sample for
  benchmarks (`--train-n`, `--dev-n`).

Re-running is idempotent: images already stored are not downloaded again,
failed downloads are retried with backoff, and a view lists only images that
exist. Boxes stay in normalised coordinates in the manifest and are converted
to pixels of the stored image. Downloaded files are untrusted: only Pillow
decodes them and nothing is executed or unpacked.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import random
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

USER_AGENT = "petperception-g1/0.2 (+research)"


def load(manifest_dir: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    with gzip.open(manifest_dir / "images.csv.gz", "rt", newline="", encoding="utf-8") as fh:
        images = list(csv.DictReader(fh))
    boxes: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(manifest_dir / "boxes.csv.gz", "rt", newline="", encoding="utf-8") as fh:
        for b in csv.DictReader(fh):
            boxes[b["image_id"]].append(b)
    return images, boxes


def fetch(url: str, timeout: float = 60.0, attempts: int = 4) -> bytes:
    for k in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception:  # noqa: BLE001 - retried, then reported by the caller
            if k == attempts - 1:
                raise
            time.sleep(2 ** k)
    raise RuntimeError("unreachable")


def store(row: dict, images_dir: Path, max_side: int, draft: bool = True) -> tuple[str, int, str, str] | None:
    """Download one image unless present; returns (image id, original bytes, sha256, decode mode) for new downloads."""
    dst = images_dir / f"{row['image_id']}.jpg"
    if dst.exists():
        return None
    try:
        data = fetch(row["download_url"])
        im = Image.open(io.BytesIO(data))
        if draft:  # JPEG DCT-domain downscale to >= 2x the target first (Pillow's own thumbnail reducing_gap)
            im.draft("RGB", (2 * max_side, 2 * max_side))
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:  # noqa: BLE001 - a missing or undecodable image is skipped
        return None
    im.thumbnail((max_side, max_side))
    tmp = dst.with_suffix(".part")
    im.save(tmp, format="JPEG", quality=90)
    os.replace(tmp, dst)
    return row["image_id"], len(data), hashlib.sha256(data).hexdigest(), "draft2x" if draft else "full"


def write_view(view: Path, split: str, rows: list[dict], boxes: dict[str, list[dict]], images_dir: Path, classes: list[str]) -> dict:
    cat_id = {c: i + 1 for i, c in enumerate(classes)}  # 0 = background (Model Maker convention)
    d = view / split
    d.mkdir(parents=True, exist_ok=True)
    link = d / "images"
    if not link.exists():
        link.symlink_to(os.path.relpath(images_dir, d), target_is_directory=True)
    coco = {"categories": [{"id": 0, "name": "background"}] + [{"id": cat_id[c], "name": c} for c in classes], "images": [], "annotations": []}
    for r in rows:
        p = images_dir / f"{r['image_id']}.jpg"
        if not p.exists():
            continue
        with Image.open(p) as im:
            w, h = im.size
        k = len(coco["images"])
        coco["images"].append({"id": k, "file_name": p.name, "width": w, "height": h})
        for b in boxes[r["image_id"]]:
            x0, y0, x1, y1 = (float(b[c]) for c in ("xmin", "ymin", "xmax", "ymax"))
            coco["annotations"].append({"id": len(coco["annotations"]), "image_id": k, "category_id": cat_id[b["label"]],
                                        "bbox": [x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h], "area": (x1 - x0) * w * (y1 - y0) * h,
                                        "iscrowd": int(b["is_group_of"])})
    (d / "labels.json").write_text(json.dumps(coco))
    return {"split": split, "images": len(coco["images"]), "boxes": len(coco["annotations"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="var/training/g1")
    ap.add_argument("--classes", default="results/g1/classes.json")
    ap.add_argument("--view", default="full")
    ap.add_argument("--train-n", type=int, default=0, help="random sample of train images (0 = all)")
    ap.add_argument("--dev-n", type=int, default=0, help="random sample of development images (0 = all)")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--max-side", type=int, default=320)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--full-decode", action="store_true", help="decode at full resolution before resizing (slower; the bench images were stored this way)")
    a = ap.parse_args()
    root = Path(a.root)
    classes = json.loads(Path(a.classes).read_text())
    images, boxes = load(root)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    sel: dict[str, list[dict]] = {}
    for split, n in (("train", a.train_n), ("development", a.dev_n)):
        rows = sorted((r for r in images if r["split"] == split), key=lambda r: r["image_id"])
        sel[split] = random.Random(f"{a.seed}-{split}").sample(rows, n) if n and n < len(rows) else rows
    todo = sel["train"] + sel["development"]
    t0 = time.time()
    log = root / "download_log.csv"
    first = not log.exists()
    new = 0
    with open(log, "a", newline="", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as ex:
        w = csv.writer(fh)
        if first:
            w.writerow(["image_id", "original_bytes", "sha256", "decode"])
        for g in ex.map(store, todo, [images_dir] * len(todo), [a.max_side] * len(todo), [not a.full_decode] * len(todo)):
            if g:  # logged as it completes, so an interrupted run keeps the hashes of what it stored
                w.writerow(g)
                new += 1
                if new % 500 == 0:
                    fh.flush()
                    print(json.dumps({"downloaded_now": new, "seconds": round(time.time() - t0, 1)}), flush=True)
    out = {"view": a.view, "downloaded_now": new, "seconds": round(time.time() - t0, 1),
           "stored_total": sum(1 for _ in images_dir.glob("*.jpg")),
           "store_bytes": sum(p.stat().st_size for p in images_dir.glob("*.jpg"))}
    out["splits"] = [write_view(root / "views" / a.view, s, sel[s], boxes, images_dir, classes) for s in ("train", "development")]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
