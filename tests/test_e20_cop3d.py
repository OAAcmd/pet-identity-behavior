"""E20 scoring arithmetic and CoP3D mask boxes (synthetic inputs; not evidence of accuracy)."""

from __future__ import annotations

import numpy as np
import pytest

from petperception.data.cop3d import mask_box
from petperception.eval import e20_cop3d_home_video as E

pytestmark = pytest.mark.synthetic


def test_mask_box_from_soft_mask():
    m = np.zeros((100, 200), np.uint8)
    m[20:60, 50:150] = 200
    m[0:5, 0:5] = 100  # below the 128 threshold: ignored
    box, area = mask_box(m)
    assert box == pytest.approx((0.25, 0.2, 0.75, 0.6))
    assert area == pytest.approx(40 * 100 / (100 * 200))
    assert mask_box(np.zeros((10, 10), np.uint8)) == (None, 0.0)


def _rows(seq_found: dict[str, list[bool]], species="cat", wrong=None):
    out = []
    for s, fl in seq_found.items():
        for i, f in enumerate(fl):
            w = bool(wrong and wrong.get(s, [False] * len(fl))[i]) and not f
            out.append({"species": species, "sequence": s, "frame": str(i), "found": f, "wrong_only": w, "box_scored": True, "found_box": f,
                        "best_iou": 0.7 if f else 0.0})
    return out


def test_summary_and_sequence_rules():
    rows = _rows({"a": [True, True, False, False], "b": [False] * 4, "c": [True] * 4}, wrong={"b": [True, False, False, False]})
    r = E.summarize(rows, n_boot=200)["cat"]
    assert r["frames"] == 12 and r["sequences"] == 3
    assert r["frame_found"] == pytest.approx(6 / 12)
    assert r["frame_wrong_species_only"] == pytest.approx(1 / 12)
    assert r["sequence_found_half_frames"] == pytest.approx(2 / 3)  # a (0.5) and c count; b does not
    assert r["sequence_any_frame"] == pytest.approx(2 / 3)
    assert r["frame_found_ci95"]["n_groups"] == 3


def test_paired_difference_by_sequence():
    a = _rows({"a": [True, True], "b": [True, False]})
    b = _rows({"a": [True, False], "b": [True, False]})
    p = E.paired(a, b, n_boot=200)["cat"]
    assert p["mean_difference"] == pytest.approx(0.25) and p["sequences_better"] == 1 and p["sequences_worse"] == 0


def test_iou():
    assert E._iou((0, 0, 1, 1), (0, 0, 1, 1)) == pytest.approx(1.0)
    assert E._iou((0, 0, 0.5, 1), (0.5, 0, 1, 1)) == 0.0
    assert E._iou((0, 0, 1, 1), (0, 0, 0.5, 1)) == pytest.approx(0.5)
