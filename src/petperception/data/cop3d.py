"""CoP3D (Common Pets in 3D, v1): smartphone videos of people's own cats and dogs.

Layout (checked 2026-10-19 against facebookresearch/cop3d links.json and the
archives themselves): `<species>_000.zip` holds the annotations and LICENSE
(44 MB cat, 67 MB dog); `<species>_001.zip` onwards hold the frames as
`<species>/<sequence>/images/frameNNNNNN.jpg` with a foreground mask
`.../masks/frameNNNNNN.png` per frame (about 20 GB each). The masks are
model-generated, not human-verified.

What is read (never the whole archive): the frame archive's central
directory and, for every sequence in it, `k` frames evenly spaced over the
sequence plus their masks, by HTTP Range (`RemoteZip`, CRC-checked per
member). Each sampled frame is cached with its SHA-256, and the mask is
reduced to its bounding box (pixels >= 128) and area fraction.

Terms: CC BY-NC 4.0 (the LICENSE file in the archive). Research evaluation
only; nothing is trained on it.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .mirror import RemoteZip

BASE = "https://dl.fbaipublicfiles.com/cop3d/"
FRAME_ARCHIVES = {"cat": ["cat_001.zip"], "dog": ["dog_001.zip"]}
ANNOTATION_ARCHIVES = {"cat": "cat_000.zip", "dog": "dog_000.zip"}
MASK_THRESHOLD = 128


@dataclass(frozen=True)
class Frame:
    species: str
    sequence: str
    frame: str
    path: str
    sha256: str
    width: int
    height: int
    mask_box: tuple[float, float, float, float] | None  # normalised x1, y1, x2, y2 of mask pixels >= 128
    mask_area: float  # fraction of the image


def root(data_root: Path) -> Path:
    return data_root / "cop3d"


def mask_box(mask: np.ndarray) -> tuple[tuple[float, float, float, float] | None, float]:
    m = mask >= MASK_THRESHOLD
    if not m.any():
        return None, 0.0
    ys, xs = np.where(m)
    h, w = m.shape
    return (float(xs.min() / w), float(ys.min() / h), float((xs.max() + 1) / w), float((ys.max() + 1) / h)), float(m.mean())


def _fetch_frame(z: RemoteZip, species: str, seq: str, n: str, out_dir: Path) -> Frame | None:
    img = z.read(n)
    mname = n.replace("/images/", "/masks/").replace(".jpg", ".png")
    mask = cv2.imdecode(np.frombuffer(z.read(mname), np.uint8), cv2.IMREAD_GRAYSCALE) if mname in z.entries else None
    arr = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    dst = out_dir / seq / Path(n).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(img)
    box, area = mask_box(mask) if mask is not None else (None, 0.0)
    return Frame(species, seq, Path(n).stem, str(dst), hashlib.sha256(img).hexdigest(), arr.shape[1], arr.shape[0], box, area)


def sample(data_root: Path, species: str, k: int = 10, workers: int = 16) -> list[Frame]:
    """k evenly spaced frames per sequence of the species' frame archives (cached under cop3d/frames).

    Members are read concurrently: each costs two Range requests, and the CDN's latency, not
    its bandwidth, is the limit."""
    from concurrent.futures import ThreadPoolExecutor

    out_dir = root(data_root) / "frames" / species
    index = out_dir / f"index_k{k}.csv"
    if index.exists():
        with open(index, newline="", encoding="utf-8") as fh:
            return [Frame(r["species"], r["sequence"], r["frame"], r["path"], r["sha256"], int(r["width"]), int(r["height"]),
                          tuple(json.loads(r["mask_box"])) if r["mask_box"] else None, float(r["mask_area"])) for r in csv.DictReader(fh)]
    frames: list[Frame] = []
    for archive in FRAME_ARCHIVES[species]:
        z = RemoteZip([BASE + archive])
        seqs: dict[str, list[str]] = {}
        for n in z.names():
            parts = n.split("/")
            if len(parts) == 4 and parts[2] == "images" and parts[3].endswith(".jpg"):
                seqs.setdefault(parts[1], []).append(n)
        jobs = []
        for seq in sorted(seqs):
            names = sorted(seqs[seq])
            jobs += [(seq, n) for n in sorted({names[int(round(i * (len(names) - 1) / max(1, k - 1)))] for i in range(k)})]
        with ThreadPoolExecutor(workers) as ex:
            frames += [f for f in ex.map(lambda j, z=z: _fetch_frame(z, species, j[0], j[1], out_dir), jobs) if f is not None]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(index, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "sequence", "frame", "path", "sha256", "width", "height", "mask_box", "mask_area"])
        for f in frames:
            w.writerow([f.species, f.sequence, f.frame, f.path, f.sha256, f.width, f.height, json.dumps(f.mask_box) if f.mask_box else "", f.mask_area])
    return frames


def licence_text(species: str) -> str:
    return RemoteZip([BASE + ANNOTATION_ARCHIVES[species]]).read(f"{species}/LICENSE").decode("utf-8", "replace")
