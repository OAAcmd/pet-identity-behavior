"""G2 candidate selection and the phone test page credits."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from petperception.ondevice.device_test import credits
from petperception.training.g2_candidates import build


def _write(g1: Path, images: list[dict], boxes: list[dict]) -> None:
    g1.mkdir(parents=True)
    with gzip.open(g1 / "images.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "split", "role", "license", "author", "download_url", "original_landing_url"])
        w.writeheader()
        w.writerows(images)
    with gzip.open(g1 / "boxes.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "label", "xmin", "ymin", "xmax", "ymax", "is_occluded", "is_truncated", "is_group_of", "is_depiction"])
        w.writeheader()
        w.writerows(boxes)


def test_g2_candidates_filters_and_splits_by_author(tmp_path: Path) -> None:
    images, boxes = [], []

    def add(iid: str, author: str, label: str, area_side: float, flags: dict | None = None, extra_pet: bool = False) -> None:
        images.append({"image_id": iid, "split": "train", "role": "pet", "license": "https://creativecommons.org/licenses/by/2.0/", "author": author,
                       "download_url": f"https://example.invalid/{iid}.jpg", "original_landing_url": f"https://example.invalid/{iid}"})
        f = {"is_occluded": "0", "is_truncated": "0", "is_group_of": "0", "is_depiction": "0", **(flags or {})}
        boxes.append({"image_id": iid, "label": label, "xmin": "0.1", "ymin": "0.1", "xmax": str(0.1 + area_side), "ymax": str(0.1 + area_side), **f})
        if extra_pet:
            boxes.append({"image_id": iid, "label": "dog", "xmin": "0.0", "ymin": "0.0", "xmax": "0.5", "ymax": "0.5", **f})

    for k in range(40):
        add(f"c{k}", f"author{k % 10}", "cat", 0.5)
        add(f"d{k}", f"author{k % 10}", "dog", 0.6)
    add("small", "a", "cat", 0.2)  # 4 % of the image
    add("drawing", "a", "dog", 0.5, {"is_depiction": "1"})
    add("occluded", "b", "dog", 0.5, {"is_occluded": "1"})  # kept, flagged
    add("two", "a", "cat", 0.5, extra_pet=True)
    _write(tmp_path / "g1", images, boxes)
    out = build(tmp_path / "g1", tmp_path / "g2", tmp_path / "res")
    assert out["excluded"] == {"box_under_min_area": 1, "group_of_or_depiction": 1, "more_than_one_pet": 1}
    rows = list(csv.DictReader(open(tmp_path / "g2" / "candidates.csv", newline="")))
    assert len(rows) == 81
    assert [r["occluded"] for r in rows if r["image_id"] == "occluded"] == ["1"]
    split_of_author: dict[str, set[str]] = {}
    for r in rows:
        split_of_author.setdefault(r["author"], set()).add(r["split"])
    assert all(len(s) == 1 for s in split_of_author.values())  # no photographer spans splits
    for sp in ("cat", "dog"):
        n = 41 if sp == "dog" else 40
        assert sorted(int(r["order"]) for r in rows if r["species"] == sp) == list(range(n))
    assert json.loads((tmp_path / "res" / "candidates_summary.json").read_text())["candidates"]["cat"]


def test_phone_page_credits_state_weight_licences(tmp_path: Path) -> None:
    (tmp_path / "device_check").mkdir()
    (tmp_path / "device_check" / "ATTRIBUTION.json").write_text(json.dumps(
        {"animal": {"image_id": "x", "author": "A. Author", "license": "https://creativecommons.org/licenses/by/2.0/", "source": "https://example.invalid/x"}}))
    manifest = {"files": [{"kind": "model", "resource_id": "yamnet_mediapipe_v1"}, {"kind": "model", "resource_id": "efficientdet_lite0_mediapipe_v1"},
                          {"kind": "model", "resource_id": "efficientdet_lite0_tfhub_rawdecode_v1"}]}
    c = {x["item"]: x for x in credits(tmp_path, manifest)}
    terms = [x["terms"] for x in c.values()]
    assert "weights licence: Apache-2.0" in terms
    assert "weights licence: Not stated" in terms
    assert any(x["credit"] == "A. Author" for x in c.values())
