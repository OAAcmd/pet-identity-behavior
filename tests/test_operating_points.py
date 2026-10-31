"""Per-class detector operating points (a G1 export at its tuning-set points) and committed project models in the register."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from petperception.vision.detector import Detection, detect_at_operating_point


class _FakeDetector:
    def __init__(self, dets):
        self.dets, self.calls = dets, []

    def detect(self, rgb, classes, thr, iou=0.5, k=10):
        self.calls.append(thr)
        return [d for d in self.dets if d.label in classes and d.score >= thr][:k]


def _vision(**kw):
    base = {"score_threshold": 0.35, "class_score_thresholds": {}, "nms_iou": 0.5, "max_detections_per_frame": 10}
    return SimpleNamespace(**{**base, **kw})


def test_operating_point_scalar_and_per_class():
    box = np.array([0.1, 0.1, 0.5, 0.5])
    dets = [Detection("cat", 0.26, box), Detection("dog", 0.28, box), Detection("dog", 0.31, box), Detection("person", 0.30, box)]
    rgb = np.zeros((8, 8, 3), np.uint8)
    fake = _FakeDetector(dets)
    got = detect_at_operating_point(fake, rgb, ["cat", "dog", "person"], _vision())
    assert got == [] and fake.calls == [0.35]  # nothing reaches the scalar point
    fake = _FakeDetector(dets)
    got = detect_at_operating_point(fake, rgb, ["cat", "dog", "person"], _vision(class_score_thresholds={"cat": 0.24, "dog": 0.30}))
    assert fake.calls == [0.24]  # the detector runs at the lowest point
    assert [(d.label, d.score) for d in got] == [("cat", 0.26), ("dog", 0.31)]  # person keeps the scalar 0.35


def test_committed_project_model_resolves_from_the_repository(settings):
    from petperception.registry import check_model, release_decision

    r = check_model(settings, "pet_detector_oi_g1b_v1")
    assert r.ready and r.files["model_float32.tflite"].endswith("models/g1/g1b_warm_cat2_cop3d/export/model_float32.tflite")
    assert release_decision("research", "pet_detector_oi_g1b_v1").allowed
    assert not release_decision("commercial", "pet_detector_oi_g1b_v1").allowed


def test_phone_bundle_refuses_per_class_points(settings, tmp_path):
    from petperception.ondevice.bundle import export_bundle

    s = settings.model_copy(deep=True)
    s.vision.class_score_thresholds = {"cat": 0.24, "dog": 0.30}
    with pytest.raises(ValueError, match="per-class"):
        export_bundle(s, tmp_path / "bundle")


@pytest.mark.integration
def test_engine_loads_g1b_with_its_points_and_verifier(settings, models_ready):
    if not models_ready:
        pytest.skip("models not fetched")
    from petperception.config import REPO_ROOT, load_settings
    from petperception.pipeline import AnalysisEngine

    s = load_settings(REPO_ROOT / "configs" / "cpu.yaml", overrides={"release_profile": "research", "pose": {"provider": "none"},
                                                                     "vision": {"detector": "pet_detector_oi_g1b_v1",
                                                                                "class_score_thresholds": {"cat": 0.24, "dog": 0.30}}})
    eng = AnalysisEngine(s).load()
    st = {c.name: c.state for c in eng.status.values()}
    assert st["visual_detector"] == "ok" and st["detection_verifier"] == "ok"
    assert eng.verifier.detector_id == "pet_detector_oi_g1b_v1" and "cat" in eng.detector.labels and "person" in eng.detector.labels
