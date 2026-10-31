"""Behaviour explanation v2 (E32): context phrases in questions, turning-in-place model, recording context, events, E32 scoring.

All inputs are synthetic; real-recording results are in results/e32.
"""

from __future__ import annotations

import json

import numpy as np

from petperception.eval import e32_behavior_v2 as E32
from petperception.understand import context as CX
from petperception.understand import explain as X
from petperception.understand import knowledge as K
from petperception.understand import question as Qn
from petperception.understand import recognize as R
from petperception.understand import temporal as TM
from petperception.understand import vocab as V
from petperception.understand.features import CONTEXT_CLASSES, MOTION_FPS


class FakeZ(dict):
    @property
    def files(self):
        return list(self.keys())


def test_after_phrase_is_context_not_the_question():
    main, ctx = Qn.split_context("After eating she grooms herself for a long time.")
    assert ctx == ["eating"] and "grooms" in main
    q = Qn.parse_v2("cat", "", "After eating she grooms herself for a long time.")
    assert q.reported == ["self_grooming"] and q.context_behaviors == ["eating"]
    # a phrase stops at a new subject: "until" + "I open the door" keeps barking as the question
    q = Qn.parse_v2("dog", "", "He barks until I open the door")
    assert q.reported == ["vocalizing"] and q.context_behaviors == []


def test_circling_model_roundtrip_clip_score_and_bouts(tmp_path):
    k = len(TM.FEATURES)
    m = TM.CirclingModel(np.zeros(k), np.ones(k), np.r_[1.0, np.zeros(k - 1)], 0.0, 0.6, {"clip_score": "mean"})
    m.save(tmp_path / "c.json")
    m2 = TM.CirclingModel.load(tmp_path / "c.json")
    X_ = np.zeros((8, k))
    X_[2:4, 0] = 3.0  # two confident windows
    p = m2.window_probs(X_)
    assert np.allclose(p[[0, 1, 4, 5, 6, 7]], 0.5) and (p[2:4] > 0.9).all()
    assert m2.clip_score(p) == np.sort(p)[-2:].mean()  # top quarter of 8 windows = 2
    starts = np.arange(8) * TM.HOP / MOTION_FPS
    b = m2.bouts(starts, p)
    assert len(b) == 1 and b[0][0] == starts[2] and b[0][1] == starts[3] + TM.WIN / MOTION_FPS
    assert m2.meta["clip_score"] == "mean"


def test_window_features_see_box_shape_oscillation():
    T = 36
    t = np.arange(T)
    spin = np.stack([0.4 - 0.1 * (1 + np.sin(t)), np.full(T, 0.4), 0.6 + 0.1 * (1 + np.sin(t)), np.full(T, 0.6)], 1)
    still = np.tile([0.4, 0.4, 0.6, 0.6], (T, 1)).astype(float)
    common = {"det": np.ones((T, 1)), "flow": np.zeros((T, 3)), "app": np.random.default_rng(0).normal(size=(T, 8))}
    st_s, F_s = TM.window_features(FakeZ(box=spin, **common))
    st_n, F_n = TM.window_features(FakeZ(box=still, **common))
    j = TM.FEATURES.index("aspect_cv")
    assert F_s.shape == F_n.shape == ((T - TM.WIN) // TM.HOP + 1, len(TM.FEATURES))
    assert (F_s[:, j] > 0.1).all() and np.allclose(F_n[:, j], 0.0)
    assert np.allclose(F_s[:, TM.FEATURES.index("displacement")], 0.0, atol=1e-6)  # stays in place


def test_recording_context_and_events_around():
    T = 30
    ctx = np.zeros((T, len(CONTEXT_CLASSES)), np.float32)
    ctx[:, CONTEXT_CLASSES.index("person")] = 0.9
    ctx[:, CONTEXT_CLASSES.index("bowl")] = 0.8
    near = np.zeros(T, np.float32)
    near[12:18] = 0.5  # 2.0-3.0 s
    labels = ["Speech", "Laughter", "Meow"]
    a_scores = np.array([[0.1, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.5]], np.float32)
    z = FakeZ(ctx=ctx, ctx_person_near=near, a_scores=a_scores, a_t=np.array([0.0, 0.96, 1.92]))
    items, cues = CX.recording_context(z, labels, None)
    what = [i["what"] for i in items]
    assert what == ["a person close to the animal", "a food or water bowl", "people laughing"]
    assert items[0]["times_s"] == [[2.0, 3.0]] and "person near" in cues and "laugh" in cues and "bowl" in cues
    assert CX.vocal_types(z, "cat", labels) == [{"type": "Meow", "score": 0.5, "first_s": 1.9}]
    seen = [{"behavior": "tail_chasing", "bouts_s": [[4.0, 6.0]]}, {"behavior": "resting", "bouts_s": [[7.0, 9.0]]}]
    ev = CX.events_around(seen, items)
    assert ev[0]["behavior"] == "tail_chasing" and ev[0]["after"] == ["resting (7.0-9.0 s)"]
    assert "a person close to the animal (2.0-3.0 s)" in ev[0]["before"]
    assert T / MOTION_FPS == 5.0


def test_recording_cues_count_half_and_are_listed_apart():
    q = Qn.parse_v2("dog", "", "My dog chases his tail every day")
    reading = R.BehaviorReading("tail_chasing", 0.6, [R.Bout(1.0, 3.0, 0.6)], {})
    ctx = [{"what": "people laughing", "from": "sound model (YAMNet)", "score": 0.6, "times_s": [[0.0, 1.0]]}]
    ans = X.compose(q, [reading], {"available": False}, recording_context=ctx, recording_cues="laugh")
    e = ans.explained_behaviors[0]
    cands = {c["id"]: c for g in e["candidates_by_kind"].values() for c in g}
    w = next(c for c in K.entry("tail_chasing", "dog").candidates if c.id == "dog_tc_attention").context_cues["laugh"]
    assert cands["dog_tc_attention"]["recording_match"] == ["laugh"] and cands["dog_tc_attention"]["context_score"] == round(X.RECORDING_CUE_WEIGHT * w, 2)
    assert cands["dog_tc_compulsive"]["context_score"] > cands["dog_tc_attention"]["context_score"] and "recording_match" not in cands["dog_tc_compulsive"]
    assert e["kind_order_by_context"].index("compulsive") < e["kind_order_by_context"].index("request")  # the user's words weigh more
    assert ans.recording_context == ctx and any("half as much" in u for u in ans.uncertainty)


def test_benchmark_v2_is_well_formed_and_disjoint_from_development():
    b = json.loads(E32.BENCHMARK_V2.read_text())
    qs = b["questions"]
    assert len({q["id"] for q in qs}) == len(qs) == 37 and len({q["clip"] for q in qs}) == 37
    for q in qs:
        for name in q["asked"] + q["asked_ok"] + q["reference_present"] + q["must_not_recognize"] + list(q["request_expected"]):
            assert name in V.BEHAVIORS, (q["id"], name)
        assert not set(q["must_not_recognize"]) & set(q["reference_present"]), q["id"]
        assert bool(q["gap"]) == (q["asked"] == []), q["id"]
    from petperception.config import REPO_ROOT

    sets = {}
    for v in json.loads((REPO_ROOT / "manifests" / "paper_videos.json").read_text())["videos"]:
        sets[f"paper_{v['id']}"] = v.get("set", "dev")
    for r in json.loads((REPO_ROOT / "manifests" / "archive_pets.json").read_text())["records"]:
        sets[f"archive_{r['id']}"] = r.get("set", "dev")
    for r in json.loads((REPO_ROOT / "manifests" / "stock_pets.json").read_text())["records"]:
        sets[f"stock_{r['id']}"] = r.get("set", "e32")
    assert all(sets.get(q["clip"]) == "e32" for q in qs)  # never part of the development collection
    e31_clips = {q["clip"] for q in json.loads((REPO_ROOT / "benchmarks" / "behavior_questions.json").read_text())["questions"]}
    assert not e31_clips & {q["clip"] for q in qs}


def test_e32_outcome_and_decision_rules():
    base = {"gap": False, "parsed": ["tail_chasing"], "must_not_seen": [], "asked_present": ["tail_chasing"], "asked_present_seen": [],
            **{c: True for c in E32.EXPLAINED_CHECKS}}
    assert E32.outcome(base, []) == "explained_from_report"
    assert E32.outcome({**base, "asked_present_seen": ["tail_chasing"]}, []) == "explained_and_seen"
    assert E32.outcome({**base, "C5_coverage": False}, []) == "not_explained"
    assert E32.outcome({**base, "gap": True, "parsed": ["eating"]}, ["eating"]) == "gap_handled"
    assert E32.outcome({**base, "gap": True, "parsed": ["tail_chasing"]}, ["eating"]) == "gap_forced"

    def c(rate, k=1, n=1):
        return {"rate": rate, "pass": k, "n": n}

    expl = {**{k: c(1.0) for k in E32.EXPLAINED_CHECKS}, "C5_coverage_expert": c(1.0, 3, 3)}
    res = {"paired": {"C1_parse": {"v1": 20, "v2": 30}, "asked_present_recognized": {"v1": 10, "v2": 9},
                      "confusers_rejected": {"v1": 3, "v2": 3}},
           "v1": {"explanation": {**expl, "C5_coverage_expert": c(0.5, 2, 3)}},
           "v2": {"explanation": expl, "recognition": {"B1_behavior_tail_chasing": {"rate": 0.5, "n": 2}, "B4_tail_chasing_rejection": {"rate": 1.0, "n": 2}}},
           "context_v2": {"person": {"agree": 8, "n": 10}}}
    d = E32.decide(res)
    assert d["v2_stays_default"] and not d["tail_chasing_recognized_from_recordings_v2"] and d["context_person_agreement_at_least_0.8"]
    res["paired"]["asked_present_recognized"]["v2"] = 8  # two fewer than v1
    assert not E32.decide(res)["v2_stays_default"]


def test_default_is_the_frozen_e31_system():
    """E32's pre-registered decision keeps v1 as the default; its code must stay exactly as tested (understand/e31)."""
    import hashlib
    from pathlib import Path

    from petperception.cli import build_parser
    from petperception.understand import e31
    from petperception.understand.e31 import explain as X1
    from petperception.understand.e31 import knowledge as K1
    from petperception.understand.e31 import vocab as V1
    from petperception.understand.run import DEFAULT_VERSION

    assert DEFAULT_VERSION == 1 and build_parser().parse_args(["explain", "x.mp4", "--species", "cat"]).system_version == 1
    assert K1.KNOWLEDGE_VERSION == "behavior-kb-v1" and X1.ANSWER_VERSION == "behavior-answer-v1" and len(V1.BEHAVIORS) == 17
    pinned = {"explain.py": "92efe868debdbd9b62465b299a201e9db514475954a336790b05c605f8d9a5c6",
              "knowledge.py": "b65136ea4a70db846aea320e4073eab2c0c3fd4e3a2fa0cdc5294b3b22a86bc3",
              "question.py": "33dccbf0bc08b6ba686e696d341c9bd23b9da76c02004b62d154576ac9e62a74",
              "recognize.py": "af6fa7e4f663853d76e41b05996cac69297cc059c3ad0c46cfe35b203054110b",
              "vocab.py": "e7ea371b7f5c2c1e00147b16d2481036e9cc362ec52c171fdbcd7c1286668159"}
    root = Path(e31.__file__).parent
    assert {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in pinned} == pinned
