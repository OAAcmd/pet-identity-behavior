"""Pet-on-furniture runtime pieces (E15): features, frame-run segments, shipped model.
The same feature values and segments are asserted in web/tests/unit/objectrelation.test.ts."""

from pathlib import Path

import numpy as np
import pytest

from petperception.vision.behavior import on_furniture_segments
from petperception.vision.object_relation import ObjectRelationModel, features

ON_BED = ([0.0, 0.3, 1.0, 1.0], [0.3, 0.35, 0.6, 0.6])
BESIDE_COUCH = ([0.0, 0.2, 0.5, 0.7], [0.6, 0.45, 0.95, 0.95])


def test_feature_layout():
    x = features(np.array(ON_BED[0]), np.array(ON_BED[1]), (480, 640), "cat", "bed")
    assert np.allclose(x, [0.107143, 1.0, 0.107143, -0.05, -0.25, -2.233592, 0.644357, 0.470004, 0.0, 1.0, 0.7, 0.075,
                           1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-6)
    x = features(np.array(BESIDE_COUCH[0]), np.array(BESIDE_COUCH[1]), (480, 640), "dog", "couch")
    assert np.allclose(x, [0.0, 0.0, 0.0, 1.05, 0.5, -0.356675, 0.287682, -0.068993, 0.2, 0.0, 0.25, 0.175,
                           0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_segments_split_on_object_change_and_gaps():
    t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    v = [(0.9, "couch"), (0.8, "couch"), (0.9, "bed"), (0.95, "bed"), None, (0.9, "chair"), (0.2, "chair"), (0.9, "chair")]
    segs = on_furniture_segments(t, v, 0.5, 2, 1.5, "t1")
    assert [(g.t_start_s, g.t_end_s, g.related_object, g.measurements["frames"]) for g in segs] == [(0.0, 0.5, "couch", 2.0), (1.0, 1.5, "bed", 2.0)]
    assert all(g.kind == "on_furniture_candidate" and g.validation == "per_frame_model_public_still_images" for g in segs)
    assert on_furniture_segments([0.0, 5.0], [(0.9, "bed"), (0.9, "bed")], 0.5, 2, 1.5, "t1") == []  # gap longer than the tracker gap
    # a frame without detected furniture neither extends nor ends a run; a low score ends it
    segs = on_furniture_segments([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [(0.9, "couch"), None, (0.9, "couch"), (0.1, "couch"), (0.9, "couch"), None],
                                 0.5, 2, 1.5, "t1")
    assert [(g.t_start_s, g.t_end_s, g.measurements["frames"]) for g in segs] == [(0.0, 1.0, 2.0)]


MODEL = Path(__file__).resolve().parents[1] / "artifacts/object_relation/pet_object_relation_v2.npz"
LEARNED = Path(__file__).resolve().parents[1] / "artifacts/object_relation/pet_object_relation_v1.npz"


@pytest.mark.skipif(not MODEL.exists(), reason="run petperception evaluate --experiment e15")
def test_shipped_model_contains_only_gated_relations_and_orders_cases():
    m = ObjectRelationModel.load(MODEL)
    assert set(m.heads) == {"on"} and m.detector_classes == ["bed", "chair", "couch", "dining table"]
    h = m.heads["on"]
    on = h.score(features(np.array(ON_BED[0]), np.array(ON_BED[1]), (480, 640), "cat", "bed"))
    beside = h.score(features(np.array(BESIDE_COUCH[0]), np.array(BESIDE_COUCH[1]), (480, 640), "dog", "couch"))
    assert on > h.threshold > beside
    best = m.best([("couch", np.array(BESIDE_COUCH[0])), ("bed", np.array(ON_BED[0])), ("tv", np.array([0.0, 0.0, 1.0, 1.0]))],
                  np.array(ON_BED[1]), (480, 640), "cat")
    assert best["on"][1] == "bed" and best["on"][0] == pytest.approx(on)


@pytest.mark.skipif(not MODEL.exists(), reason="run petperception evaluate --experiment e15-analysis")
def test_shipped_rule_is_half_of_the_pet_box_inside_the_furniture():
    m = ObjectRelationModel.load(MODEL)
    h = m.heads["on"]
    for frac, expect in ((0.49, False), (0.51, True), (1.0, True), (0.0, False)):
        x = np.zeros(19)
        x[1] = frac  # inter_over_pet
        assert (h.score(x) >= h.threshold) is expect
    with np.load(MODEL) as z:
        assert str(z["on_kind"]) == "containment_rule" and float(z["on_tau"]) == 0.5


@pytest.mark.skipif(not LEARNED.exists(), reason="run petperception evaluate --experiment e15")
def test_learned_head_is_kept_as_the_evaluated_alternative():
    assert set(ObjectRelationModel.load(LEARNED).heads) == {"on"}
