"""Household evaluator: scoring logic on a hand-built fixture (software correctness, not accuracy)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from petperception.validation import household as H

MAN = {"schema": "petperception.household_recording/1", "recording_id": "r1", "household_id": "h1",
       "pets_in_household": [{"pet_id": "p1", "species": "cat", "breed_reported_by_owner": "Maine Coon"}],
       "device": {"model": "Pixel 7a", "os": "Android 14", "browser": "Chrome 126"}, "recorded_at": "2026-11-01T10:00:00Z", "duration_s": 10,
       "owner_context": {"situation": "before_feeding"},
       "consent": {"research_use": True, "annotation_by_project_staff": True, "people_in_recording_consented_or_blurred": True, "aggregate_publication": True}}
ANN = {"schema": "petperception.household_annotation/1", "recording_id": "r1", "annotator_id": "a1", "blind_to_app_output": True,
       "observed": {"presence": [{"pet_id": "p1", "species": "cat", "t_start_s": 0, "t_end_s": 4}],
                    "activities": [{"pet_id": "p1", "kind": "stationary", "t_start_s": 0, "t_end_s": 4}],
                    "vocalizations": [{"pet_id": "p1", "species": "cat", "type": "meow", "t_start_s": 1.0, "t_end_s": 1.6}],
                    "posture": [{"pet_id": "p1", "t_s": 1.0, "label": "sitting"}, {"pet_id": "p1", "t_s": 2.0, "label": "sitting"}],
                    "furniture": [{"pet_id": "p1", "object": "couch", "relation": "in_front_of", "t_start_s": 0, "t_end_s": 4}]},
       "interpretation": [{"pet_id": "p1", "hypothesis": "food", "label": "yes"}]}
RES = {"observations": {"animal_tracks": [{"species": "cat", "first_seen_s": 0.0, "last_seen_s": 3.0,
                                           "behavior_segments": [{"kind": "stationary", "t_start_s": 0.0, "t_end_s": 3.5},
                                                                 {"kind": "on_furniture_candidate", "related_object": "couch", "t_start_s": 0.5, "t_end_s": 3.0}],
                                           "posture": [{"t_s": 1.0, "label": "sitting"}, {"t_s": 2.0, "label": "lying"}]}],
                        "sound_events": [{"label": "cat_vocalization", "t_start_s": 0.96, "t_end_s": 1.92}]},
       "hypotheses": [{"name": "food", "status": "supported"}]}


def test_scores_one_recording():
    s = H.score_recording(H.RecordingManifest.model_validate(MAN), H.Annotation.model_validate(ANN), RES)
    assert s["presence"] == {"frames": 9, "found": 8}  # frames at 0, 0.5, ... 4.0; the track ends at 3.0 (+0.5 tolerance)
    assert s["segments"]["stationary"] == (1, 0, 0)
    assert s["vocal"]["cat"] == (1, 0, 0)
    assert s["posture"] == {"compared": 2, "agree": 1}
    assert s["near_furniture"] == {"intervals": 1, "reported_as_on": 1}  # the 'in front of' failure mode is counted
    assert s["hypotheses"] == [{"name": "food", "truth": "yes"}]


def test_schemas_reject_unknown_fields_and_values():
    with pytest.raises(ValidationError):
        H.Annotation.model_validate({**ANN, "mood": "happy"})
    with pytest.raises(ValidationError):
        H.RecordingManifest.model_validate({**MAN, "owner_context": {"situation": "hungry"}})


def test_evaluate_dataset_and_agreement(tmp_path: Path):
    for sub in ("recordings", "annotations", "adjudicated", "results"):
        (tmp_path / sub).mkdir()
    for h in ("h1", "h2"):
        rid = f"r_{h}"
        (tmp_path / "recordings" / f"{rid}.json").write_text(json.dumps({**MAN, "recording_id": rid, "household_id": h}))
        (tmp_path / "adjudicated" / f"{rid}.json").write_text(json.dumps({**ANN, "recording_id": rid, "annotator_id": "adj"}))
        (tmp_path / "results" / f"{rid}.json").write_text(json.dumps(RES))
        for a in ("a1", "a2"):
            (tmp_path / "annotations" / f"{rid}.{a}.json").write_text(json.dumps({**ANN, "recording_id": rid, "annotator_id": a}))
    out = H.evaluate(tmp_path)
    assert out["recordings_scored"] == 2 and out["households"] == 2
    m = out["metrics"]
    assert m["presence_recall"]["value"] == pytest.approx(8 / 9) and m["presence_recall"]["status"] == "met"
    assert m["false_on_rate_near_furniture"]["value"] == 1.0 and m["false_on_rate_near_furniture"]["status"] == "not_met"
    assert m["pet_free_false_alarm_rate"]["status"] == "not_evaluated"
    assert m["hypothesis_precision"]["food"]["precision"] == 1.0
    ag = H.agreement(tmp_path)
    assert ag["recordings_double_annotated"] == 2
    assert ag["posture_kappa"] is None  # both annotators used one label only: kappa is undefined, not 1
    assert ag["activity_interval_f1_between_annotators"] == 1.0


def test_intent_block_scoring_counts_only_supported_claims():
    res = {**RES, "intent": {"status": "interpretation", "hypotheses": [{"name": "food", "status": "supported"},
                                                                       {"name": "attention", "status": "context_only"},
                                                                       {"name": "play", "status": "tentative"}]}}
    s = H.score_recording(H.RecordingManifest.model_validate(MAN), H.Annotation.model_validate(ANN), res)
    assert s["intent"] == {"status": "interpretation", "claims": [{"name": "food", "truth": "yes"}]}
    assert "intent" not in H.score_recording(H.RecordingManifest.model_validate(MAN), H.Annotation.model_validate(ANN), RES)
