from __future__ import annotations

from petperception.eval import e27_dog_audio_intent as E27
from petperception.intent import rules as R
from petperception.intent import taxonomy as T
from petperception.schemas.ontology import HYPOTHESES


def test_layers_relations_and_hypotheses_are_valid():
    for key, ds in T.DATASETS.items():
        assert ds.labels, key
        for lab in ds.labels:
            assert lab.layer in T.LAYERS
            for h, rel in lab.hypotheses.items():
                assert h in HYPOTHESES, (key, lab.raw, h)
                assert rel in T.RELATIONS
            if lab.qualifier is not None:
                assert lab.qualifier in T.QUALIFIERS


def test_context_is_never_an_intent_annotation():
    for ds in T.DATASETS.values():
        for lab in ds.labels:
            if lab.layer == T.CONTEXT:
                assert "annotated" not in lab.hypotheses.values(), lab.raw
            if "annotated" in lab.hypotheses.values():
                assert lab.layer == T.INTENT


def test_barkopedia_hypothesis_map_matches_taxonomy():
    em = T.evidence_map("barkopedia_act_env")
    assert set(em) == set(E27.HYPOTHESIS)
    for raw, hyp in E27.HYPOTHESIS.items():
        if hyp == "bathing":
            assert T.label("barkopedia_act_env", raw).qualifier == "bathing" and em[raw] == {}
        else:
            assert hyp in em[raw]


def test_coverage_and_missing_lists():
    cov = T.coverage()
    assert set(cov) == set(HYPOTHESES)
    assert any(s["relation"] == "annotated" for s in cov["food"]["dog"])
    assert not any(s["relation"] == "annotated" for s in cov["food"]["cat"])  # cats: contexts only
    assert all(m["exact_examples"] for m in T.MISSING)


def test_rules_are_deterministic_and_handle_missing_channels():
    lab, why = R.kinetics_situation({"Water": 0.5}, {"car": 5}, 0.9, 10)
    assert lab == "bathing dog" and why
    assert R.kinetics_situation(None, {"car": 3}, 0.8, 4)[0] == "walking the dog"
    assert R.kinetics_situation({"Meow": 0.6}, None, None, 0)[0] == "petting cat"
    assert R.kinetics_situation(None, None, None, 0)[0] == "training dog"
    assert R.barkopedia_activity({"Water tap, faucet": 0.3, "Bark": 0.9})[0] == "taking shower"
    assert R.barkopedia_activity({"Bark": 0.05})[0] == "rest"
    assert R.barkopedia_activity({"Bark": 0.9})[0] == "alerting to sounds"
    assert len(R.source_sha256()) == 64
