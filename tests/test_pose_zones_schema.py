"""Pose availability, zones and schema contract tests."""

import json
from dataclasses import dataclass

import numpy as np
import pytest
from pydantic import ValidationError

from petperception.config import REPO_ROOT
from petperception.context.zones import ZoneConfig, occupancy, point_in_polygon
from petperception.schemas.analysis import AnalysisResult, Hypothesis
from petperception.vision.pose import KeypointSet, UnavailablePoseProvider, compute_features

COCO_HUMAN = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_hip", "right_hip")


@dataclass
class FakeProvider:
    keypoint_vocabulary: tuple
    name: str = "fake"
    species: tuple = ("dog",)
    available: bool = True
    unavailable_reason: str | None = None

    def estimate(self, rgb, box, t_s):
        return None


@pytest.mark.synthetic
def test_unavailable_pose_provider_reports_every_feature_unavailable():
    feats = compute_features(UnavailablePoseProvider(), [])
    assert feats and all(not f.available and f.reason for f in feats)


@pytest.mark.synthetic
def test_human_pose_vocabulary_does_not_produce_animal_features():
    p = FakeProvider(COCO_HUMAN)
    ks = KeypointSet(0.0, list(COCO_HUMAN), np.full((len(COCO_HUMAN), 2), 0.5), np.ones(len(COCO_HUMAN)))
    feats = {f.name: f for f in compute_features(p, [ks])}
    assert not feats["tail_height"].available and "tail_base" in feats["tail_height"].reason
    assert not feats["ear_angle_deg"].available


@pytest.mark.synthetic
def test_pose_features_only_from_visible_declared_keypoints():
    vocab = ("tail_base", "tail_tip", "withers")
    p = FakeProvider(vocab)
    xy = np.array([[0.5, 0.5], [0.6, 0.3], [0.2, 0.5]])
    visible = KeypointSet(0.0, list(vocab), xy, np.ones(3))
    hidden = KeypointSet(0.0, list(vocab), xy, np.array([1.0, 0.1, 1.0]))
    f = {x.name: x for x in compute_features(p, [visible])}
    assert f["tail_height"].available and f["tail_height"].value == pytest.approx(0.2 / 0.3)
    f2 = {x.name: x for x in compute_features(p, [hidden])}
    assert not f2["tail_height"].available


@pytest.mark.synthetic
def test_zone_membership_and_dwell():
    cfg = ZoneConfig.model_validate({"schema": "petperception.zones/1", "zones": [
        {"zone_id": "bowl", "label": "bowl", "semantic_tag": "food_area", "polygon": [[0, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]]}]})
    assert point_in_polygon(0.25, 0.75, cfg.zones[0].polygon) and not point_in_polygon(0.75, 0.75, cfg.zones[0].polygon)
    boxes = [np.array([0.1, 0.4, 0.3, 0.8])] * 4 + [np.array([0.6, 0.4, 0.9, 0.8])]
    occ = occupancy([0.0, 0.5, 1.0, 1.5, 2.0], boxes, cfg, max_gap_s=1.0)
    assert occ[0]["dwell_s"] == pytest.approx(1.5) and occ[0]["fraction_of_track_detections"] == pytest.approx(0.8)


def test_zone_validation_rejects_bad_input():
    with pytest.raises(ValidationError):
        ZoneConfig.model_validate({"zones": [{"zone_id": "x", "label": "x", "semantic_tag": "food_area", "polygon": [[0, 0], [2, 0], [0, 1]]}]})
    with pytest.raises(ValidationError):
        ZoneConfig.model_validate({"zones": [{"zone_id": "x", "label": "x", "semantic_tag": "fear_zone", "polygon": [[0, 0], [1, 0], [0, 1]]}]})


def test_schema_forbids_extra_fields_and_fixes_version():
    schema = AnalysisResult.model_json_schema()
    assert schema["properties"]["schema_version"]["const"] == "1.0.0"
    with pytest.raises(ValidationError):
        Hypothesis.model_validate({"hypothesis_id": "h", "name": "fear", "semantic_type": "state", "status": "abstained",
                                   "supporting_evidence_ids": [], "conflicting_evidence_ids": [], "independent_provenance_roots": [],
                                   "score_kind": "none", "scorer": {"name": "x", "revision": "y", "kind": "rule_baseline"},
                                   "explanation": ""})


def test_committed_openapi_and_schema_are_current(tmp_path):
    from petperception.app.main import create_app
    from petperception.config import load_settings

    s = load_settings()
    s.app.storage.database_path = str(tmp_path / "db.sqlite3")
    s.app.storage.media_root = str(tmp_path / "media")
    app = create_app(s, engine_factory=lambda: None, start_workers=False)
    committed = json.loads((REPO_ROOT / "docs" / "openapi.json").read_text())
    assert committed == json.loads(json.dumps(app.openapi())), "run: uv run petperception export-openapi"
    assert json.loads((REPO_ROOT / "docs" / "analysis_result.schema.json").read_text()) == AnalysisResult.model_json_schema()
