"""Behaviour understanding: knowledge base integrity, question parsing, answer composition, temporal pattern, trained head, E31 scoring.

All inputs are synthetic; real-recording results are in results/e31 (E31).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from petperception.eval import e31_behavior_questions as E31
from petperception.understand import explain as X
from petperception.understand import knowledge as K
from petperception.understand import question as Qn
from petperception.understand import recognize as R
from petperception.understand import vocab as V
from petperception.understand.head import BehaviorHead


class FakeZ(dict):
    @property
    def files(self):
        return list(self.keys())


def reading(b, score, bouts=((1.0, 3.0),), **ev):
    return R.BehaviorReading(b, score, [R.Bout(t0, t1, score) for t0, t1 in bouts], dict(ev))


def test_knowledge_base_is_consistent():
    assert K.validate() == []
    for (b, sp), e in K.ENTRIES.items():
        assert b in V.BEHAVIORS and sp in V.BEHAVIORS[b].species
        assert e.candidates
    # every vocabulary behaviour has an entry for each of its species
    missing = [(b, sp) for b, beh in V.BEHAVIORS.items() for sp in beh.species if K.entry(b, sp) is None]
    assert missing == []


def test_every_non_ordinary_candidate_cites_a_known_source():
    for e in K.ENTRIES.values():
        for c in e.candidates:
            assert all(s in K.SOURCES for s in c.sources)
            if c.kind != "play_or_normal":
                assert c.sources


def test_parse_user_example_and_pronoun_normalization():
    q = Qn.parse("cat", "", "In an open living room, my cat keeps turning in circles while biting its own tail. What could this behavior mean?")
    assert q.reported == ["tail_chasing", "tail_directed"]
    assert q.repetition_reported
    assert Qn.parse("dog", "", "My puppy chases her tail").reported == ["tail_chasing"]
    assert Qn.parse("dog", "", "My puppy chases her tail").age == "young"
    assert Qn.parse("cat", "", "she licks herself all the time").reported == ["self_grooming"]


def test_parse_respects_species_and_unknown_text():
    assert "kneading" not in Qn.parse("dog", "", "my dog is kneading the blanket").reported
    assert Qn.parse("cat", "", "what is this?").reported == []


def test_compose_separates_requests_and_never_claims_a_cause():
    q = Qn.parse("cat", "kitchen before breakfast", "My cat meows at me every morning near her food bowl and grooms herself afterwards.")
    rs = [reading("self_grooming", 0.6), reading("vocalizing", 0.7)]
    ans = X.compose(q, rs, {"available": False})
    ex = {e["behavior"]: e for e in ans.explained_behaviors}
    assert ex["vocalizing"]["request_or_intention_possible"] is True
    assert ex["vocalizing"]["kind_order_by_context"][0] == "request"  # food / bowl / morning cues
    assert ex["self_grooming"]["request_or_intention_possible"] is False
    assert ex["self_grooming"]["no_request_note"]
    text = " ".join([ans.summary] + [c["explanation"] for e in ans.explained_behaviors for g in e["candidates_by_kind"].values() for c in g]).lower()
    assert not any(p in text for p in E31.CERTAINTY_PHRASES)
    assert "cannot establish why" in ans.uncertainty[0]
    assert ans.species == "cat"


def test_reported_but_not_recognized_is_said_so():
    q = Qn.parse("dog", "", "My dog chases his tail all the time")
    ans = X.compose(q, [reading("tail_chasing", 0.1), reading("resting", 0.8)], {"available": True, "animal_found_fraction": 0.9})
    st = {u["behavior"]: u["status"] for u in ans.user_reported}
    assert st["tail_chasing"].startswith("not recognized")
    tc = next(e for e in ans.explained_behaviors if e["behavior"] == "tail_chasing")
    assert tc["basis"] == "user report only"
    assert {"request", "compulsive", "medical"} <= set(tc["candidates_by_kind"])


def test_species_from_user_is_kept_when_detector_disagrees():
    q = Qn.parse("cat", "", "my cat spins")
    ans = X.compose(q, [], {"available": True, "animal_found_fraction": 1.0, "detector_dog_vote": 0.9})
    assert ans.species == "cat"
    assert any("species vote disagrees" in u for u in ans.uncertainty)


def test_thresholds_from_config_decide_what_is_seen():
    q = Qn.parse("cat", "", "")
    rs = [reading("eating", 0.5), reading("kneading", 0.25)]
    assert {b["behavior"] for b in X.compose(q, rs, {}).behaviors_seen} == {"eating"}
    cfg = R.RecognizerConfig(thresholds={"eating": 0.6, "kneading": 0.2})
    assert {b["behavior"] for b in X.compose(q, rs, {}, cfg).behaviors_seen} == {"kneading"}


def test_temporal_pattern_turning_in_place():
    T = 60
    box = np.tile(np.array([[0.4, 0.4, 0.6, 0.6]], np.float32), (T, 1))
    flow = np.zeros((T, 5), np.float32)
    flow[:, 0] = 0.3
    flow[:, 1] = 2.0  # rad/s, one direction
    z = FakeZ(box=box, flow=flow, app=np.random.default_rng(0).normal(size=(T, 576)).astype(np.float16), det=np.zeros((T, 3), np.float32))
    p = R.temporal_pattern(z)
    assert p["available"] and p["stays_in_place"]
    assert p["turned_rad"] == pytest.approx(2.0 * T / 6.0, rel=1e-3)
    assert p["turning_consistency"] == pytest.approx(1.0)


def test_head_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X_ = rng.normal(size=(60, 8))
    y = (X_[:, 0] > 0).astype(int)
    groups = np.repeat(np.arange(12), 5)
    h = BehaviorHead.fit(X_, {"eating": y}, groups, {"note": "synthetic"})
    h.save(tmp_path / "head")
    h2 = BehaviorHead.load(tmp_path / "head")
    assert h2.behaviors == ["eating"]
    a, b = h.scores(X_[0])["eating"], h2.scores(X_[0])["eating"]
    assert a == pytest.approx(b, abs=1e-5)
    assert h.scores(X_[y == 1].mean(0))["eating"] > h.scores(X_[y == 0].mean(0))["eating"]


def test_recognizer_config_roundtrip(tmp_path):
    c = R.RecognizerConfig(method="head", head_behaviors=["eating"], thresholds={"eating": 0.4})
    c.save(tmp_path / "r.json")
    c2 = R.RecognizerConfig.load(tmp_path / "r.json")
    assert c2 == c and c2.threshold("kneading") == R.DEFAULT_SEEN and c2.threshold("vocalizing") == R.VOCAL_HEARD


def test_e31_scoring_on_a_synthetic_answer():
    item = {"id": "t1", "species": "dog", "category": "tail_chasing", "asked": ["tail_chasing"], "asked_ok": [], "reference_present": ["tail_chasing"],
            "must_not_recognize": ["resting"], "request_expected": {"tail_chasing": True},
            "source_interpretation": {"kinds": ["compulsive"], "authority": "expert", "text": ""}, "gap": None}
    q = Qn.parse("dog", "", "My dog chases his tail every day and can't stop")
    ans = X.compose(q, [reading("tail_chasing", 0.6, ((2.0, 6.0),)), reading("resting", 0.1)], {"available": False})
    seen = {b["behavior"] for b in ans.behaviors_seen}
    r = E31.score_question(item, ans, q.reported, seen, {"bouts": [{"behavior": "tail_chasing", "t0": 0, "t1": 4}]})
    assert r["asked_present_seen"] == ["tail_chasing"] and r["must_not_seen"] == []
    assert r["temporal"] == {"tail_chasing": True}
    assert all(r[k] for k in ("C1_parse", "C2_entry", "C3_grounded", "C4_separation", "C5_coverage", "C6_observations", "C7_no_certainty",
                              "C8_report_status", "C9_species"))
    assert r["C5_ranking"] is True  # "every day", "can't stop" put the compulsive candidate first
    s = E31.summarize_b([r])
    assert s["B1_tail_chasing"]["rate"] == 1.0 and s["B4_confuser_rejection"]["rate"] == 1.0


def test_benchmark_file_is_well_formed():
    b = json.loads(E31.BENCHMARK.read_text())
    ids = [q["id"] for q in b["questions"]]
    assert len(ids) == len(set(ids)) == 37
    for q in b["questions"]:
        for name in q["asked"] + q["asked_ok"] + q["reference_present"] + q["must_not_recognize"] + list(q["request_expected"]):
            assert name in V.BEHAVIORS, (q["id"], name)
        if q["source_interpretation"]:
            assert set(q["source_interpretation"]["kinds"]) <= set(K.KINDS)
    assert sum(q["question_origin"] == "user_example_verbatim" for q in b["questions"]) == 1


def test_source_label_mapping():
    from petperception.understand.dataset import Clip

    c = Clip("k", None, "cat", "commons", "g", "t", "", {"source": ["hiss_growl", "fighting", "kneading"], "assistant": ["eating"]})
    assert E31.human_labels(c) == {"threat_display", "kneading"}  # assistant labels never enter the human reference
