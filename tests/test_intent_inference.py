from __future__ import annotations

import numpy as np

from petperception.audio.linear_model import LinearModel
from petperception.intent import heads as H
from petperception.intent import inference as I

SIT = ["bathing dog", "grooming cat", "grooming dog", "petting cat", "training dog", "walking the dog"]
ACT = sorted(I.DOG_ACTIVITY_HYP)
BARK = ["aggression", "contact", "play"]
YL = ["Dog", "Bark", "Cat", "Meow", "Water"]
A_DIM = 6 + len(ACT) + 3  # audio vector: [situation one-hot | dog activity one-hot | bark context one-hot]


def _head(name, classes, dim, offset=0, thr=0.6):
    """Linear head reading x[offset + j] -> class j."""
    k = len(classes)
    coef = np.zeros((k, dim))
    coef[np.arange(k), offset + np.arange(k)] = 8.0
    return H.Head(name, LinearModel(list(classes), np.zeros(dim), np.ones(dim), coef, np.zeros(k), 1.0), 1.0, thr)


def _poe(branches, classes):
    """Product of experts: sum of centred branch log-probabilities."""
    k, nb = len(classes), len(branches)
    coef = np.zeros((k, k * nb + nb))
    for b in range(nb):
        coef[np.arange(k), b * k + np.arange(k)] = 1.0
    return H.LateFusion(list(branches), list(classes), LinearModel(list(classes), np.zeros(k * nb + nb), np.ones(k * nb + nb), coef, np.zeros(k), 1.0),
                        1.0, 0.6)


class Models:
    def __init__(self):
        br = ["audio", "animal", "environment", "interaction"]
        self.situation_heads = {b: _head(b, SIT, A_DIM if b == "audio" else 6) for b in br}
        self.situation_fusion = _poe(br, SIT)
        self.situation_policy = {"visual_only_abstain_below": 0.6, "branches": br}
        self.dog_audio = _head("audio", ACT, A_DIM, 6)
        self.dog_env = _head("environment", ACT, 8)
        self.dog_audio_env = _poe(["audio", "environment"], ACT)
        self.dog_bark_context = _head("audio", BARK, A_DIM, 6 + len(ACT))
        self.cat_context = _head("audio", ["brushing", "isolation", "waiting_for_food"], A_DIM, 6 + len(ACT), thr=1.01)
        self.audio_branch = "audio"


def _audio(sit=None, act=None, bark=None):
    v = np.zeros((1, A_DIM))
    if sit:
        v[0, SIT.index(sit)] = 1.0
    if act:
        v[0, 6 + ACT.index(act)] = 1.0
    if bark:
        v[0, 6 + len(ACT) + BARK.index(bark)] = 1.0
    return v


def _video(sit):
    v = np.zeros((1, 6))
    v[0, SIT.index(sit)] = 1.0
    return v


def _branches(audio=None, sit_video=None):
    vid = None if sit_video is None else _video(sit_video)
    return {"audio": audio, "animal": vid, "environment": vid, "interaction": vid}


def _cues(dog_frac=0.9, pet_frames=10, sounds=None, missing=None):
    return {"missing": dict(missing or {}), "yamnet_max": np.array([(sounds or {}).get(n, 0.0) for n in YL]), "dog_fraction": dog_frac,
            "pet_frames": pet_frames, "frames": 20, "person_frames": 5, "interaction_max": 0.7}


def test_missing_audio_is_explicit_and_video_still_reads_situation():
    r = I.infer(Models(), _branches(None, "walking the dog"), _cues(missing={"audio": "audio absent"}), YL)
    assert r.inputs["audio"] == "missing" and r.missing_inputs["audio"] == "audio absent"
    assert r.missing_inputs["dog_vocal_activity"] == "audio missing"
    assert r.situation["label"] == "walking the dog" and r.situation["branches_used"] == ["animal", "environment", "interaction"]
    assert r.status == "situation_only"
    assert [h.name for h in r.hypotheses] == ["outside"] and r.hypotheses[0].status == "context_only"


def test_audio_video_disagreement_is_a_conflict_not_an_answer():
    r = I.infer(Models(), _branches(_audio("bathing dog"), "walking the dog"), _cues(sounds={"Water": 0.8}), YL)
    assert r.status == "conflict"
    assert any("audio suggests 'bathing dog'" in c for c in r.conflicts)
    assert r.situation["status"] == "conflict" and r.situation["qualifier"] is None


def test_species_contradiction_is_reported():
    r = I.infer(Models(), _branches(_audio("walking the dog"), "walking the dog"), _cues(dog_frac=0.0), YL)
    assert r.species == "cat"
    assert any("implies a dog" in c for c in r.conflicts)
    assert r.status == "conflict" and r.situation["status"] == "conflict" and r.situation["qualifier"] is None


def test_context_alone_never_supports_a_hypothesis():
    r = I.infer(Models(), _branches(_audio("petting cat"), "petting cat"), _cues(dog_frac=0.1), YL)
    assert r.status == "situation_only"
    assert {h.name for h in r.hypotheses} == {"attention", "relaxed"}
    assert all(h.status == "context_only" and h.support is None for h in r.hypotheses)


def test_agreeing_channels_corroborate_a_supported_hypothesis():
    a = _audio("training dog", "begging for food", "contact")
    r = I.infer(Models(), _branches(a, "training dog"), _cues(sounds={"Bark": 0.9}), YL)
    assert r.status == "interpretation"
    food = next(h for h in r.hypotheses if h.name == "food")
    assert food.status == "supported_corroborated" and food.support > 0.9
    assert any("annotated intent" in s for s in food.supporting) and any("training dog" in q for q in food.qualifying)
    att = next(h for h in r.hypotheses if h.name == "attention")
    assert att.status == "context_only"  # bark context + training context, but no direct evidence for attention


def test_handling_context_qualifies_a_vocal_reading():
    a = _audio("bathing dog", "seeking attention")
    r = I.infer(Models(), _branches(a, "bathing dog"), _cues(sounds={"Bark": 0.9, "Water": 0.5}), YL)
    att = next(h for h in r.hypotheses if h.name == "attention")
    assert att.status == "supported" and any("handled/bathed" in q for q in att.qualifying)
    assert "bathing" in r.qualifiers


def test_no_dog_vocalization_means_no_dog_audio_reading():
    r = I.infer(Models(), _branches(_audio("training dog", "begging for food"), "training dog"), _cues(sounds={}), YL)
    assert r.missing_inputs["dog_vocal_activity"] == "no dog vocalization heard"
    assert not any(h.status.startswith("supported") for h in r.hypotheses)


def test_nothing_available_gives_insufficient_evidence():
    r = I.infer(Models(), _branches(None, None), {"missing": {"audio": "audio absent", "animal": "video absent"}, "dog_fraction": None}, YL)
    assert r.status == "insufficient_evidence" and r.species == "unknown"
    assert "situation" in r.missing_inputs and r.hypotheses == []


def test_cat_meow_context_below_gate_never_supports():
    r = I.infer(Models(), _branches(_audio("petting cat", bark="contact"), "petting cat"), _cues(dog_frac=0.0, sounds={"Meow": 0.9}), YL)
    cat = [x for x in r.readings if x.channel == "cat_meow_context"]
    assert cat and not cat[0].confident and not cat[0].evidence_gate_passed
    assert not any(h.status.startswith("supported") for h in r.hypotheses)


def test_adopted_divergence_detector_flags_disagreement_when_audio_head_never_commits():
    m = Models()
    m.situation_heads["audio"] = _head("audio", SIT, A_DIM, thr=1.01)  # as in E28: the audio head never passes its threshold
    r = I.infer(m, _branches(_audio("bathing dog"), "walking the dog"), _cues(sounds={"Water": 0.8}), YL)
    assert r.status != "conflict"  # without an adopted detector nothing is flagged (E28's finding)
    m.situation_policy = {**m.situation_policy, "conflict_js_threshold": 0.2}
    r = I.infer(m, _branches(_audio("bathing dog"), "walking the dog"), _cues(sounds={"Water": 0.8}), YL)
    assert r.status == "conflict" and any("disagree about the situation" in c for c in r.conflicts)
    r = I.infer(m, _branches(_audio("walking the dog"), "walking the dog"), _cues(), YL)
    assert r.status != "conflict"  # agreeing channels are not flagged


def test_js_divergence():
    p = np.array([1.0, 0.0])
    assert I.js_divergence(p, p) == 0.0
    assert abs(I.js_divergence(p, np.array([0.0, 1.0])) - np.log(2)) < 1e-6
