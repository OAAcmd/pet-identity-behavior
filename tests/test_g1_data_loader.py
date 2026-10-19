"""G1 data path: boxes on the image edge survive, and images without boxes (hard negatives) reach training.

Model Maker 0.2.1.4 drops boxes with x + width > image width and skips every
image without a box (found by the G1 CPU benchmark, results/g1/cpu_training.json).
"""

from __future__ import annotations

import importlib.util
import random
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str, stub_pil: bool = False):
    if stub_pil and "PIL" not in sys.modules:  # prepare_coco imports Pillow at module level; these functions do not use it
        pil = types.ModuleType("PIL")
        pil.Image, pil.ImageOps = types.ModuleType("PIL.Image"), types.ModuleType("PIL.ImageOps")
        sys.modules.update({"PIL": pil, "PIL.Image": pil.Image, "PIL.ImageOps": pil.ImageOps})
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pixel_extent_never_crosses_the_image_edge():
    prep = _load("g1_prepare_coco", "training/g1/prepare_coco.py", stub_pil=True)
    rng = random.Random(0)
    for _ in range(20000):
        size = rng.randint(50, 640)
        x0 = rng.random() * 0.999
        x1 = rng.choice([1.0, rng.uniform(x0, 1.0)])
        start = x0 * size
        ext = prep.pixel_extent(start, x1 * size, size)
        assert ext >= 0.0
        assert start + ext <= size
        assert abs((start + ext) - min(x1 * size, size)) < 1e-6  # the box keeps its extent up to rounding


def test_pixel_extent_known_rounding_case():
    prep = _load("g1_prepare_coco", "training/g1/prepare_coco.py", stub_pil=True)
    # A case where the old formula x0*w + (x1 - x0)*w exceeded w = 320 by one rounding step.
    w = 320
    hits = [x0 for x0 in (i / 997 for i in range(997)) if x0 * w + (1.0 - x0) * w > w]
    assert hits, "expected at least one rounding case in the old formula"
    for x0 in hits:
        assert x0 * w + prep.pixel_extent(x0 * w, 1.0 * w, w) <= w


def test_coco_examples_keep_images_without_boxes_and_crowd_flags():
    tr = _load("g1_train_resumable", "training/g1/train_resumable.py")
    labels = {
        "images": [{"id": 0, "file_name": "a.jpg"}, {"id": 1, "file_name": "neg.jpg"}, {"id": 2, "file_name": "c.jpg"}],
        "annotations": [
            {"image_id": 0, "category_id": 1, "bbox": [10.0, 20.0, 30.0, 40.0], "iscrowd": 0},
            {"image_id": 0, "category_id": 3, "bbox": [0.0, 0.0, 100.0, 50.0], "iscrowd": 1},
            {"image_id": 2, "category_id": 2, "bbox": [90.0, 0.0, 20.0, 10.0], "iscrowd": 0},  # crosses the edge: dropped
            {"image_id": 2, "category_id": 2, "bbox": [5.0, 5.0, 0.0, 10.0], "iscrowd": 0},  # no area: dropped
        ],
    }
    out = list(tr.coco_examples(labels, lambda name: name.encode(), lambda enc: (100, 100)))
    assert [img["file_name"] for img, *_ in out] == ["a.jpg", "neg.jpg", "c.jpg"]  # nothing skipped
    _, _, h, w, a = out[0]
    assert (h, w) == (100, 100)
    assert a["category_id"] == [1, 3] and a["is_crowd"] == [0, 1]
    assert a["xmin"] == [0.1, 0.0] and a["xmax"] == [0.4, 1.0] and a["ymin"] == [0.2, 0.0] and a["ymax"] == [0.6, 0.5]
    for _, _, _, _, b in out[1:]:  # the hard negative, and an image whose boxes are all invalid, stay as negatives
        assert all(b[k] == [] for k in ("xmin", "xmax", "ymin", "ymax", "category_id", "is_crowd"))
