"""Intent stage of the analysis engine (settings.intent): explicit missing inputs, schema, phone parity inputs."""

from __future__ import annotations

import pytest

from petperception.config import REPO_ROOT
from petperception.schemas.analysis import AnalysisResult

ART = REPO_ROOT / "artifacts" / "intent" / "kinetics_situation_v1" / "phone_now" / "late_fusion.json"


@pytest.fixture(scope="module")
def intent_engine(settings, models_ready):
    if not models_ready:
        pytest.skip("models not fetched: uv run petperception fetch-models")
    if not ART.exists():
        pytest.skip("intent artifacts missing: petperception evaluate --experiment e28")
    from petperception.pipeline import AnalysisEngine

    s = settings.model_copy(deep=True)
    s.intent.enabled = True
    return AnalysisEngine(s).load()


def test_disabled_by_default_leaves_results_unchanged(engine, synth_media):
    r = engine.analyze(synth_media["noaudio"])
    assert r.intent is None and all(c.name != "intent" for c in r.components)


@pytest.mark.synthetic
@pytest.mark.integration
def test_no_animal_video_without_audio_is_insufficient_evidence(intent_engine, synth_media):
    r = intent_engine.analyze(synth_media["noaudio"])
    assert any(c.name == "intent" and c.state == "ok" for c in r.components)
    it = r.intent
    assert it is not None and it.status in ("insufficient_evidence", "situation_only")
    assert "audio" in it.missing_inputs and "animal_phone" in it.missing_inputs
    assert it.missing_inputs["animal_phone"] == "no cat or dog detected in any sampled frame"
    assert all(h.status != "supported" for h in it.hypotheses)
    AnalysisResult.model_validate_json(r.model_dump_json())  # schema round trip


@pytest.mark.synthetic
@pytest.mark.integration
def test_audio_only_input_reports_missing_video(intent_engine, esc50_root):
    r = intent_engine.analyze(esc50_root / "audio" / "1-34094-A-5.wav")  # ESC-50 'cat' clip
    it = r.intent
    assert it.inputs["environment"] == "missing" and it.missing_inputs["environment"].startswith("video")
    assert it.species in ("cat", "unknown")
    cat = [x for x in it.readings if x.channel == "cat_meow_context"]
    if cat:  # a meow was heard: the below-gate cat context head may speak, but never supports a hypothesis
        assert not cat[0].evidence_gate_passed
        assert not any(h.status.startswith("supported") for h in it.hypotheses)
