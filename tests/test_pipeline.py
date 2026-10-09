"""Pipeline tests: missing models (synthetic) and real-model integration runs."""

from __future__ import annotations

import pytest

from petperception.pipeline import AnalysisEngine


@pytest.mark.synthetic
def test_missing_models_report_unavailable_not_fabricated(settings, synth_media, tmp_path):
    s = settings.model_copy(deep=True)
    s.paths.models_root = str(tmp_path / "no-models")
    eng = AnalysisEngine(s).load()
    st = {c.name: c.state for c in eng.status.values()}
    assert st["audio_event"] == "unavailable" and st["visual_detector"] == "unavailable"
    r = eng.analyze(synth_media["sync_mp4"])
    assert r.observations.sound_events == [] and r.observations.animal_tracks == []
    assert "model_unavailable" in r.decision.reason_codes
    assert r.decision.state == "abstention" and r.hypotheses == []
    assert any(c.state == "unavailable" and "fetch-models" in (c.reason or "") for c in r.components)


@pytest.mark.synthetic
@pytest.mark.integration
def test_silent_and_no_audio_states_flow_into_result(engine, synth_media):
    r = engine.analyze(synth_media["silent"])
    assert r.modalities.audio_state == "silent"
    assert any(q.code == "audio_silent" for q in r.quality_flags)
    assert not any(h.name == "relaxed" and h.status == "experimental_hypothesis" for h in r.hypotheses)
    r2 = engine.analyze(synth_media["noaudio"])
    assert r2.modalities.audio_state == "absent" and "missing_modality_audio" in r2.decision.reason_codes


@pytest.mark.synthetic
@pytest.mark.integration
def test_bandwidth_limited_audio_flag(engine, synth_media):
    r = engine.analyze(synth_media["tone8k"])
    assert any(q.code == "bandwidth_limited_audio" for q in r.quality_flags)
    assert r.observations.audio_summary.effective_bandwidth_hz == 4000


@pytest.mark.integration
def test_real_yamnet_detects_esc50_cat_and_ignores_silence(engine, esc50_root, synth_media):
    r = engine.analyze(esc50_root / "audio" / "1-34094-A-5.wav")  # ESC-50 'cat' clip (CC BY, see manifest)
    labels = {e.label for e in r.observations.sound_events}
    assert "cat_vocalization" in labels and "dog_vocalization" not in labels
    ev = next(e for e in r.observations.sound_events if e.label == "cat_vocalization")
    assert ev.calibrated_probability is None and ev.attribution.status == "video_unavailable"
    assert engine.analyze(synth_media["silence_wav"]).observations.sound_events == []


@pytest.mark.integration
def test_yamnet_scores_are_independent_sigmoids(engine, esc50_root):
    from petperception.ingest.media import decode_audio, probe

    mp = probe(esc50_root / "audio" / "1-100032-A-0.wav", engine.settings.limits)
    ws = engine.audio_backend.score_track(decode_audio(mp, engine.settings.limits))
    sums = ws.scores.sum(axis=1)
    assert (sums > 1.05).any() or (sums < 0.95).any(), "scores must not be renormalized with softmax"
    assert ws.t_end[-1] >= mp.duration_s - 1e-6  # final partial window is covered (zero-padded)


@pytest.mark.integration
def test_real_detector_on_cat_photo_and_verified_negative(engine, oi_images):
    r = engine.analyze(oi_images / "d60e1c25e87d8b45.jpg")  # Open Images, CC BY 2.0 (see manifest)
    assert [t.species for t in r.observations.animal_tracks] == ["cat"]
    t = r.observations.animal_tracks[0]
    assert t.identity_note.startswith("per-analysis") and not t.movement.available
    neg = engine.analyze(oi_images / "35e7f8550ba0d7e7.jpg")  # human-verified: no cat, no dog
    assert neg.observations.animal_tracks == [] and neg.decision.state == "abstention"
    assert "no_pet_detected" in neg.decision.reason_codes


@pytest.mark.integration
def test_multi_pet_photo_gives_two_tracks_and_no_identity_claim(engine, oi_images):
    r = engine.analyze(oi_images / "bdf391a6f4b1840a.jpg")
    assert sorted(t.species for t in r.observations.animal_tracks) == ["cat", "dog"]
    assert "multiple_animals_visible" in r.decision.reason_codes
