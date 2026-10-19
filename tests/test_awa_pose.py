"""AwA-Pose loader: data-only unpickling, licence parsing, keypoint mapping, photographer splits (synthetic inputs)."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from petperception.data import awa_pose as A
from petperception.eval.e24_awa_keypoints import split_by_photographer
from petperception.vision import keypoints as KP

pytestmark = pytest.mark.synthetic


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


def test_unpickler_refuses_every_global():
    ok = pickle.dumps({"a1": {"nose": [1.0, 2.0], "bbox": [0.0, 0.0, 10.0, 10.0]}})
    assert A.load_annotation(ok)["a1"]["nose"] == [1.0, 2.0]
    with pytest.raises(pickle.UnpicklingError):
        A.load_annotation(pickle.dumps(_Evil()))
    with pytest.raises(pickle.UnpicklingError):
        A.load_annotation(pickle.dumps(np.zeros(2)))  # even harmless globals are refused


def test_licence_parsing():
    flickr = "Photographer : Seniju\nPhoto URL    : https://www.flickr.com/photos/seniju/1/\nLicense      : Creative Commons (http://creativecommons.org/licenses)\n"
    r = A.parse_licence(flickr)
    assert r["kind"] == "CC (variant not stated)" and r["photographer"] == "Seniju" and r["photo_url"].endswith("/1/")
    assert A.parse_licence("License : Creative Commons (x) Attribution 2.0 Generic (//creativecommons.org/licenses/by/2.0/deed.en)")["kind"] == "CC BY"
    assert A.parse_licence("License : Creative Commons (x) Attribution-Share Alike 3.0 Unported (//creativecommons.org/licenses/by-sa/3.0/)")["kind"] == "CC BY-SA"
    assert A.parse_licence("License : Public Domain (https://creativecommons.org/publicdomain/zero/1.0/)")["kind"] == "CC0"
    assert A.parse_licence("License : public domain")["kind"] == "PD"
    assert A.parse_licence("")["kind"] == "not stated"


def test_mapping_to_the_product_layout():
    pts = {"left_earbase": [10, 20], "left_earend": [12, 5], "tail_base": [80, 50], "tail_end": [-1, -1], "back_left_paw": [70, 90]}
    row = {"id": "x#a1", "image": "x.jpg", "class": "persian+cat", "species": "cat", "licence": "CC BY", "photographer": "p",
           "bbox": json.dumps([5, 2, 95, 98]), "points": json.dumps(pts)}
    rec = A.to_record(row, (100, 100))
    i = KP.DOG_KEYPOINTS.index
    assert rec["kp"][i("left_ear_base")].tolist() == [0.1, 0.2, 2.0]
    assert rec["kp"][i("left_ear_tip")].tolist() == [0.12, 0.05, 2.0]
    assert rec["kp"][i("rear_left_paw")].tolist() == [0.7, 0.9, 2.0]
    assert rec["kp"][i("tail_end")][2] == 0.0  # missing (-1) is unlabelled
    assert rec["box"].tolist() == pytest.approx([0.05, 0.02, 0.95, 0.98])
    assert set(A.TO_AWA) == set(KP.DOG_KEYPOINTS)


def test_photographer_split_is_disjoint_and_complete():
    recs = [{"id": str(i), "photographer": f"p{i % 37}"} for i in range(500)]
    s = split_by_photographer(recs, seed=1)
    ph = {k: {r["photographer"] for r in v} for k, v in s.items()}
    assert not (ph["fit"] & ph["dev"]) and not (ph["fit"] & ph["test"]) and not (ph["dev"] & ph["test"])
    assert sum(len(v) for v in s.values()) == 500
