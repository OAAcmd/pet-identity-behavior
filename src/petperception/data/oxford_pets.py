"""Oxford-IIIT Pet: 37 breeds (12 cat, 25 dog), about 200 images each (E16).

Source: Parkhi et al., "Cats and Dogs", CVPR 2012. Official page
www.robots.ox.ac.uk/~vgg/data/pets/ (blocked from this workspace); the
identical archive is mirrored by fast.ai at
https://fast-ai-imageclas.s3.amazonaws.com/oxford-iiit-pet.tgz, pinned here by
SHA-256 on first retrieval (manifests/oxford_pets_sha256.csv).

Annotations: breed and species per image (from the file name and
annotations/list.txt), a head bounding box for the trainval images
(annotations/xmls) and a pixel trimap. Terms: "The dataset is available to
download for commercial/research purposes under a Creative Commons
Attribution-ShareAlike 4.0 International License. The copyright remains with
the original owners of the images." Used here only for evaluation; images
are never committed.
"""

from __future__ import annotations

import csv
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

from ..utils import download, sha256_file

URL = "https://fast-ai-imageclas.s3.amazonaws.com/oxford-iiit-pet.tgz"
PIN = "oxford_pets_sha256.csv"


def root(data_root: Path) -> Path:
    return data_root / "oxford_pets"


def fetch(data_root: Path, manifest_dir: Path) -> dict:
    d = root(data_root)
    tgz = d / "_download" / "oxford-iiit-pet.tgz"
    pin_path = manifest_dir / PIN
    pinned = next(csv.DictReader(open(pin_path, newline="")))["sha256"] if pin_path.exists() else None
    if not (tgz.exists() and (pinned is None or sha256_file(tgz) == pinned)):
        download(URL, tgz, expected_sha256=pinned, timeout=3600)
    sha = sha256_file(tgz)
    if pinned and sha != pinned:
        raise RuntimeError("oxford-iiit-pet.tgz: SHA-256 differs from the pinned value")
    if not pinned:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        with open(pin_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "url", "sha256"])
            w.writerow(["oxford-iiit-pet.tgz", URL, sha])
    x = d / "x"
    if not (x / "oxford-iiit-pet" / "annotations" / "list.txt").exists():
        with tarfile.open(tgz) as tf:
            tf.extractall(x, filter="data")  # rejects absolute paths, '..' and links outside the target
    return {"sha256": sha, "root": str(x / "oxford-iiit-pet"), "images": len(list((x / "oxford-iiit-pet" / "images").glob("*.jpg")))}


def records(data_root: Path) -> list[dict]:
    """One row per image: breed, species, head box (normalized x1, y1, x2, y2) when annotated."""
    base = root(data_root) / "x" / "oxford-iiit-pet"
    out = []
    for line in (base / "annotations" / "list.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, _cls, species, _breed = line.split()[:4]
        breed = name.rsplit("_", 1)[0]
        head = None
        xml = base / "annotations" / "xmls" / f"{name}.xml"
        if xml.exists():
            r = ET.parse(xml).getroot()
            w, h = float(r.findtext("size/width")), float(r.findtext("size/height"))
            b = r.find("object/bndbox")
            if b is not None and w > 0 and h > 0:
                head = [float(b.findtext("xmin")) / w, float(b.findtext("ymin")) / h, float(b.findtext("xmax")) / w, float(b.findtext("ymax")) / h]
        out.append({"image": f"{name}.jpg", "breed": breed, "species": "cat" if species == "1" else "dog", "head_box": head})
    return out
