"""Unit tests for release gating, the Kalman tracker, keypoint decoding, posture
gating and verifier feature layout. Inputs are SYNTHETIC; accuracy is measured
by experiments E7-E10, not here."""

import numpy as np
import pytest

from petperception.registry import release_decision
from petperception.vision import keypoints as KP
from petperception.vision import verifier as V
from petperception.vision.detector import Detection
from petperception.vision.posture import PostureModel, posture_features
from petperception.vision.tracking import AnimalDetection, IouTracker, KalmanTracker


def test_release_profiles_gate_assets():
    assert release_decision("research", "efficientdet_lite0_mediapipe_v1").allowed
    d = release_decision("commercial", "efficientdet_lite0_mediapipe_v1")
    assert not d.allowed and d.status == "not_permitted"
    d = release_decision("commercial", "efficientdet_lite0_tfhub_v1", approvals={})
    assert not d.allowed and d.status == "requires_legal_review" and "release_approvals.yaml" in d.reason
    ok = release_decision("commercial", "efficientdet_lite0_tfhub_v1",
                          approvals={"efficientdet_lite0_tfhub_v1": {"commercial": {"approved_by": "Counsel", "reference": "MEMO-1"}}})
    assert ok.allowed and ok.status == "approved"
    # Research-only training data never becomes available outside the research profile.
    assert not release_decision("personal_noncommercial", "dog_keypoints_head_v1").allowed
    assert not release_decision("commercial", "audio_event_thresholds_esc50", approvals={}).allowed
    assert not release_decision("research", "no_such_asset").allowed


def _det(t, x, w=0.2):
    return AnimalDetection(t, int(t * 2), np.array([x, 0.4, x + w, 0.6]), {"cat": 0.8})


def test_kalman_tracker_keeps_identity_when_iou_association_breaks():
    # 0.15 normalized units per 0.5 s step with 0.2-wide boxes: IoU 0.14 < 0.3.
    xs = [0.05 + 0.15 * k for k in range(5)]
    iou, kal = IouTracker(0.3, 1.5), KalmanTracker(0.3, 1.5)
    for k, x in enumerate(xs):
        iou.update(k * 0.5, [_det(k * 0.5, x)])
        kal.update(k * 0.5, [_det(k * 0.5, x)])
    assert len(iou.tracks(1)) == 5
    assert len(kal.tracks(1)) == 1 and len(kal.tracks(1)[0].detections) == 5


def test_kalman_tracker_does_not_merge_distant_animals():
    kal = KalmanTracker(0.3, 1.5)
    for k in range(4):
        kal.update(k * 0.5, [_det(k * 0.5, 0.05), _det(k * 0.5, 0.75)])
    assert len(kal.tracks(1)) == 2


def test_keypoint_decode_places_points_with_offsets():
    g, k = KP.GRID, 2
    heat = np.full((1, g, g, k), -10.0)
    heat[0, 3, 5, 0] = 10.0  # row 3, col 5
    heat[0, 10, 1, 1] = 10.0
    offs = np.zeros((1, g, g, k, 2))
    offs[0, 3, 5, 0] = [0.25, -0.25]
    xy, peak, vis = KP.decode(heat, offs, np.array([[5.0, -5.0]]))
    assert xy[0, 0] == pytest.approx([(5 + 0.5 + 0.25) / g, (3 + 0.5 - 0.25) / g])
    assert xy[0, 1] == pytest.approx([(1 + 0.5) / g, (10 + 0.5) / g])
    assert peak[0, 0] > 0.99 and vis[0, 0] > 0.99 and vis[0, 1] < 0.01


def test_square_crop_maps_back_to_image():
    rgb = np.zeros((200, 400, 3), np.uint8)
    crop, (x0, y0, side) = KP.square_crop(rgb, np.array([0.25, 0.25, 0.5, 0.75]))
    assert crop.shape == (224, 224, 3)
    assert side == pytest.approx(100 * KP.CROP_SCALE) and x0 + side / 2 == pytest.approx(150) and y0 + side / 2 == pytest.approx(100)
    assert set(KP.REPORTED_KEYPOINTS).isdisjoint(KP.UNANNOTATED) and len(KP.REPORTED_KEYPOINTS) == 20


def test_posture_gate_reports_uncertain_below_gate():
    d = 1280 + 4
    m = PostureModel(np.zeros(d), np.ones(d), np.zeros((3, d)), np.array([0.0, 0.0, 0.1]), gate=0.6, version="t")
    x = posture_features(np.zeros(1280), np.array([0.1, 0.1, 0.5, 0.4]), (480, 640), "cat")
    assert x.shape == (d,) and x[-2:].tolist() == [1.0, 0.0]
    assert m.predict(np.zeros(1280), np.array([0.1, 0.1, 0.5, 0.4]), (480, 640), "cat")[0] == "uncertain"
    m.intercept = np.array([0.0, 0.0, 5.0])
    assert m.predict(np.zeros(1280), np.array([0.1, 0.1, 0.5, 0.4]), (480, 640), "dog")[0] == "standing"


def _row(anchor=True):
    groups = {g: 1.0 / len(V.GROUPS) for g in V.GROUPS}
    return {"label": "cat", "score": 0.6, "box": [0.1, 0.1, 0.5, 0.5], "hw": [480, 640], "groups": groups, "top1": 0.3,
            "redetect": [0.5, 0.1], "anchor": {k: 0.1 for k in V.ANCHOR_CLASSES} if anchor else None}


def test_verifier_feature_layout_and_runtime_scoring():
    sizes = {fs: V.features(_row(), fs).size for fs in V.FEATURE_SETS}
    assert sizes == {"crop": 12, "redetect": 5, "anchor": 7, "crop+redetect": 14, "all": 18}
    with pytest.raises(ValueError):
        V.features(_row(anchor=False), "anchor")

    class FakeClassifier:
        def scores(self, rgb, box):
            return V.CropScores({g: 1.0 / len(V.GROUPS) for g in V.GROUPS}, 0.3)

    params = {"schema": "petperception.verifier/1", "detector_id": "x",
              "classes": {"cat": {"feature_set": "crop", "mean": [0.0] * 12, "scale": [1.0] * 12, "coef": [0.0] * 12, "intercept": 2.0, "tau": 0.5}}}
    ver = V.DetectionVerifier(None, FakeClassifier(), params, ["cat", "dog"])
    rgb = np.zeros((480, 640, 3), np.uint8)
    v, ok = ver.verify(rgb, Detection("cat", 0.6, np.array([0.1, 0.1, 0.5, 0.5])))
    assert ok and v == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert ver.verify(rgb, Detection("dog", 0.6, np.array([0.1, 0.1, 0.5, 0.5]))) is None
    with pytest.raises(RuntimeError):
        V.DetectionVerifier(None, FakeClassifier(), {**params, "schema": "other"}, ["cat"])
