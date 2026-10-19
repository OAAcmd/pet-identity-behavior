"""E25 species arbitration: margin, tau selection and relabelling (synthetic inputs; not evidence of accuracy)."""

from __future__ import annotations

import numpy as np
import pytest

from petperception.eval import e25_species_arbitration as E
from petperception.vision.detector import Detection

pytestmark = pytest.mark.synthetic


def test_margin_sign():
    assert E.margin({"cat": 0.1, "dog": 0.8}, "cat") > 0  # crop says dog: positive margin for a 'cat' label
    assert E.margin({"cat": 0.8, "dog": 0.1}, "cat") < 0


def test_tau_selection_prefers_fewer_relabels_on_ties():
    rows = [("cat", "dog", 3.0), ("cat", "dog", 2.6), ("cat", "cat", 1.0), ("dog", "dog", -2.0), ("cat", "cat", 0.2)]
    tau, curve = E.select_tau(rows)
    assert E.accuracy(rows, None) == pytest.approx(3 / 5)
    assert E.accuracy(rows, tau) == pytest.approx(1.0)
    assert 1.0 <= tau < 2.6  # any tau in [1.0, 2.5] is perfect; the largest of them is chosen
    assert tau == 2.5


def test_wrapper_relabels_only_above_tau():
    class Det:
        labels = ["cat", "dog"]

        def detect(self, rgb, classes, thr, iou=0.5, k=10):
            return [Detection("cat", 0.9, np.array([0.1, 0.1, 0.5, 0.5])), Detection("dog", 0.8, np.array([0.5, 0.5, 0.9, 0.9]))]

    class Clf:
        def scores(self, rgb, box):
            class S:
                groups = {"cat": 0.05, "dog": 0.9} if box[0] < 0.3 else {"cat": 0.3, "dog": 0.6}
            return S()

    out = E.Arbitrated(Det(), Clf(), tau=1.0).detect(np.zeros((8, 8, 3), np.uint8), ["cat", "dog"], 0.35)
    assert [d.label for d in out] == ["dog", "dog"]  # the first 'cat' crop is strongly dog; the real dog stays a dog
