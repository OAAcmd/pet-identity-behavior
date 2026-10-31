"""E32: behaviour explanation v2 - development on the E31 collection, then one test on fresh recordings.

Development (this module's ``develop``; nothing here touches E32 test clips):
* turning-in-place recognizer (``understand/temporal.py``): window features
  from feature version 2 plus the zero-shot circling and neutral
  probabilities; window labels from the assistant frame review where it
  exists (inside a tail-chasing bout = positive, outside every bout =
  negative, partial overlap ignored) and, for the other development clips,
  every window negative unless a human label says tail chasing (then the clip
  is left out); logistic regression, C by grouped 5-fold CV (uploader/paper),
  out-of-fold clip scores (mean of the top quarter of windows; the first
  development run used the two best windows, which let one noisy window in a
  long negative clip decide); threshold = best
  clip-level F1 on out-of-fold scores (clip positives: frame-reviewed
  tail-chasing clips; negatives: development clips with no tail-chasing label
  of any kind). The zero-shot clip score of E31 is the comparator.
* question parser v2 (``understand/question.py``): keyword parse, after /
  before / until context phrases, sentence-embedding matching; threshold and
  margin chosen on ``benchmarks/parser_dev.json`` plus the E31 questions.
* fixed (not fitted): sound evidence for threat displays (YAMNet Hiss or
  Growling >= 0.3), recording context (detector scores >= 0.5, person overlap
  >= 0.2, place prompts on three scene frames, YAMNet laughter / speech /
  doorbell / knock >= 0.3).
Writes ``results/e32/development.json`` and ``artifacts/understand/circling_v1.json``.

Test (``test``; pre-registered: this text, the benchmark and the scoring code
were committed before any E32 test feature was extracted or any test answer
was produced; nothing below may be changed after scoring)
* Systems: v1 = the frozen E31 system (commit 0c1f7c5, run from a git worktree
  by ``e32_v1_runner.py`` on feature version 1); v2 = this milestone's system
  (frozen at commit 3f8eae9; feature version 2, ``Explainer.load(version=2)``).
  Both see the same question text, scene text and species, once.
* Benchmark: ``benchmarks/behavior_questions_v2.json`` (37 questions on 37
  clips that neither system was developed on: set "e32" in the manifests;
  references from the source caption / title and the assistant frame review
  done after the freeze, ``annotations/behavior_clips.json`` method_e32).
* Measures: the E31 measures (B1-B4, C1-C9, same code) for both systems, plus
  B1 per category and per behaviour for every category / behaviour in the
  benchmark, paired v1/v2 counts, and for v2 the recording-context checks
  X_person (a person reported iff the frame review saw one) and X_place (the
  reported place is one of the frame-review places; no place reported fails).
* Targets and decisions (fixed now):
  - Tail chasing counts as recognized from recordings (v2) only if B1 for
    tail chasing is at least 0.7 and B4 tail-chasing rejection at least 0.8
    (n = 2 and 2: reported with Wilson intervals; meeting it would not show
    general reliability).
  - Parser: C1 of v2 at least 0.9 on non-gap questions.
  - v2 stays the default system only if, on the same questions, (a) its C1
    passes are at least v1's, (b) its asked-and-present behaviours recognized
    are at least v1's minus 1, (c) its confuser rejections are at least v1's
    minus 1, (d) its C5 expert coverage is at least v1's, and (e) C2, C3, C4,
    C6, C7, C8, C9 each pass on at least 90 % of non-gap questions;
    otherwise the default goes back to version 1 and the failure is reported.
  - Recording context (descriptive): X_person agreement target 0.8; X_place
    reported without a target.
  - Per question (descriptive, both systems): "explained" when it is not a
    gap question, C1, C2, C3, C4, C6-C9 pass (and C5 coverage when an expert
    interpretation exists) and no must-not behaviour is recognized;
    "explained and seen" when, in addition, every asked-and-present behaviour
    is recognized (otherwise the answer rests on the user's report, which C8
    requires it to say). Gap questions count as "handled" when the parse adds
    no behaviour outside ``asked_ok`` (no unrelated explanation is forced).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT, Settings
from ..understand import dataset as D
from ..understand import recognize as R
from ..understand import temporal as TM
from ..utils import utc_now, write_json
from .e31_behavior_questions import ap, best_f1_threshold, folds_for, human_labels

C_GRID = (0.01, 0.1, 1.0)


def zs_frames(z, species: str, zsm: R.ZeroShot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "emb" not in z.files or not len(z["emb"]):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    p, names = zsm.frame_probs(z["emb"].astype(np.float32), species)
    return z["emb_t"], p[:, names.index("tail_chasing")], p[:, names.index("neutral")]


def dev_clips(settings: Settings, subdir: str = "features_v2"):
    data_root = settings.paths.resolve("data_root")
    fdir = data_root.parent / "understand" / subdir
    out = []
    for c in D.clips(data_root):
        if c.set != "dev" or ("pone.0041684" in c.title and c.source == "commons"):
            continue
        f = fdir / f"{c.key}.npz"
        if f.exists():
            out.append((c, np.load(f, allow_pickle=False)))
    return out


def window_labels(c: D.Clip, starts: np.ndarray, ann: dict) -> np.ndarray | None:
    """1 / 0 / -1 (ignore) per window; None = leave the clip out."""
    a = ann.get(c.key)
    if a is not None:
        bouts = [b for b in a.get("bouts", []) if b["behavior"] == "tail_chasing"]
        if not a.get("behaviors") and "tail_chasing" in human_labels(c):
            return None  # uploader says spinning, frame review could not verify
        lab = np.zeros(len(starts), int)
        for i, s in enumerate(starts):
            e = s + TM.WIN / 6.0
            ov = sum(max(0.0, min(e, b["t1"]) - max(s, b["t0"])) for b in bouts) / (e - s)
            lab[i] = 1 if ov >= 0.5 else (0 if ov == 0 else -1)
        return lab
    if "tail_chasing" in human_labels(c):
        return None
    return np.zeros(len(starts), int)


def develop_circling(settings: Settings, zsm: R.ZeroShot) -> tuple[dict, TM.CirclingModel]:
    ann = D.load_assistant()
    rows = []
    for c, z in dev_clips(settings):
        st, X = TM.window_features(z, zs_frames(z, c.species, zsm))
        if not len(X):
            continue
        lab = window_labels(c, st, ann)
        if lab is None:
            continue
        a = ann.get(c.key, {})
        clip_pos = any(b["behavior"] == "tail_chasing" for b in a.get("bouts", []))
        clip_neg = not clip_pos and "tail_chasing" not in human_labels(c) and "tail_chasing" not in a.get("behaviors", [])
        zt, zc, _ = zs_frames(z, c.species, zsm)
        zs_clip = float(np.mean(np.sort(zc)[-max(1, len(zc) // 4):])) if len(zc) else 0.0
        rows.append({"clip": c, "starts": st, "X": X, "y": lab, "group": c.group, "clip_pos": clip_pos, "clip_neg": clip_neg, "zs_clip": zs_clip})
    groups = [r["group"] for r in rows]
    fold = folds_for(groups)

    def fit_rows(idx, C):
        X = np.concatenate([rows[i]["X"][rows[i]["y"] >= 0] for i in idx])
        y = np.concatenate([rows[i]["y"][rows[i]["y"] >= 0] for i in idx])
        if y.sum() == 0:
            return None
        return TM.fit(X, y, C)

    def oof_scores(C):
        sc = np.zeros(len(rows))
        for k in range(max(fold) + 1):
            tr = [i for i in range(len(rows)) if fold[i] != k]
            prm = fit_rows(tr, C)
            for i in np.where(fold == k)[0]:
                if prm is None:
                    continue
                m = TM.CirclingModel(*prm, threshold=0.5)
                sc[i] = m.clip_score(m.window_probs(rows[i]["X"]))
        return sc

    evalmask = np.array([r["clip_pos"] or r["clip_neg"] for r in rows])
    y_clip = np.array([int(r["clip_pos"]) for r in rows])
    per_c = {}
    for C in C_GRID:
        s = oof_scores(C)
        per_c[C] = (ap(y_clip[evalmask], s[evalmask]), s)
    bestC = max(C_GRID, key=lambda c: per_c[c][0])
    oof = per_c[bestC][1]
    zs = np.array([r["zs_clip"] for r in rows])
    # which clip score to use is decided here, on development data: zero-shot alone, the window model alone, or their mean
    options = {"zero_shot": zs, "window_model": oof, "mean": 0.5 * (zs + oof)}
    option_ap = {k: ap(y_clip[evalmask], v[evalmask]) for k, v in options.items()}
    use = max(option_ap, key=option_ap.get)
    oof_window = oof
    oof = options[use]
    thr = best_f1_threshold(y_clip[evalmask], oof[evalmask])
    pred = oof >= thr
    tp = int(np.sum(pred & (y_clip == 1) & evalmask))
    res = {"clips": len(rows), "groups": len(set(groups)), "clips_scored": int(evalmask.sum()), "clip_positives": int(y_clip.sum()),
           "windows": int(sum(len(r["y"]) for r in rows)), "positive_windows": int(sum((r["y"] == 1).sum() for r in rows)),
           "C_grid_oof_clip_ap": {str(c): per_c[c][0] for c in C_GRID}, "C": bestC,
           "oof_clip_ap": per_c[bestC][0], "zero_shot_clip_ap": ap(y_clip[evalmask], zs[evalmask]), "clip_score_options_ap": option_ap,
           "clip_score_used": use,
           "base_rate": float(y_clip[evalmask].mean()), "threshold": thr,
           "oof_at_threshold": {"tp": tp, "positives": int(y_clip.sum()), "fp": int(np.sum(pred & (y_clip == 0) & evalmask)),
                                "negatives": int((evalmask & (y_clip == 0)).sum())},
           "oof_scores_positive_clips": {r["clip"].key: round(float(oof[i]), 3) for i, r in enumerate(rows) if r["clip_pos"]},
           "oof_scores_frame_review_confusers": {r["clip"].key: round(float(oof[i]), 3) for i, r in enumerate(rows)
                                                 if "tail_chasing" in ann.get(r["clip"].key, {}).get("absent", [])}}
    X = np.concatenate([r["X"][r["y"] >= 0] for r in rows])
    y = np.concatenate([r["y"][r["y"] >= 0] for r in rows])
    mean, scale, coef, b = TM.fit(X, y, bestC)
    model = TM.CirclingModel(mean, scale, coef, b, thr, {"C": bestC, "clip_score": use, "fitted_on": "E31 collection (development), feature version 2",
                                                       "created_at": utc_now().isoformat(timespec="seconds")})
    res["window_model_oof_scores_positive_clips"] = {r["clip"].key: round(float(oof_window[i]), 3) for i, r in enumerate(rows) if r["clip_pos"]}
    res["coefficients"] = dict(zip(TM.FEATURES, [round(float(v), 3) for v in coef]))
    return res, model


def develop_thresholds(settings: Settings, zsm: R.ZeroShot) -> dict:
    """E31 part A rule re-applied with the v2 vocabulary and feature version 2 on the development pool: zero-shot vs trained head, and
    per-behaviour thresholds (best F1 on out-of-fold scores)."""
    from ..intent import av_features as AF
    from . import e31_behavior_questions as E31

    pairs = dev_clips(settings)
    clips = [c for c, _ in pairs]
    zs = [z for _, z in pairs]
    yam = AF.load_components(1, with_audio=True).yamnet.labels
    a, _ = E31.part_a(clips, zs, zsm, yam)
    cfg = E31.config_for(a["adopted"], a["eligible"], a["thresholds_deployed"],
                         {"experiment": "E32 development", "vocabulary": "v2", "feature_version": 2, "pool_clips": a["clips"],
                          "adopted": a["adopted"], "created_at": utc_now().isoformat(timespec="seconds")})
    cfg.version = cfg.version.replace("-v1", "-v2")
    cfg.save(REPO_ROOT / "artifacts" / "understand" / "recognizer_v2.json")
    return {k: a[k] for k in ("clips", "groups", "eligible", "adopted", "gains_over_R0", "thresholds_deployed", "chance_macro_ap")} | {
        "macro_ap": {n: v["macro_ap"] for n, v in a["methods"].items()}, "per_behavior_ap_R0": a["methods"]["R0_zero_shot"]["per_behavior_ap"]}


def develop_parser() -> dict:
    from ..understand import question as Qn

    dev = json.loads((REPO_ROOT / "benchmarks" / "parser_dev.json").read_text())["items"]
    e31 = json.loads((REPO_ROOT / "benchmarks" / "behavior_questions.json").read_text())["questions"]
    items = [(d["species"], "", d["question"], set(d["gold"]), set()) for d in dev] + \
        [(q["species"], q["scene"], q["question"], set(q["asked"]), set(q["asked_ok"])) for q in e31 if not q.get("gap")]
    m = Qn.SemanticMatcher()
    v1 = sum(int(g <= set(Qn.parse(sp, sc, qq).reported) <= (g | o)) for sp, sc, qq, g, o in items)
    v2 = sum(int(g <= set(Qn.parse_v2(sp, sc, qq, m).reported) <= (g | o)) for sp, sc, qq, g, o in items)
    return {"items": len(items), "keyword_parser_v1_pass": v1, "parser_v2_pass": v2, "threshold": m.threshold, "margin": m.margin,
            "selection": "grid threshold 0.5-0.7 x margin 0-0.12 on these items (E32 development); best pass rate"}


def develop(settings: Settings, out_dir: Path) -> dict:
    from ..understand.features import SiglipEncoder

    zsm = R.ZeroShot(SiglipEncoder(threads=settings.runtime.num_threads))
    circ, model = develop_circling(settings, zsm)
    model.save(REPO_ROOT / "artifacts" / "understand" / "circling_v1.json")
    res = {"experiment": "E32 development", "created_at": utc_now().isoformat(timespec="seconds"), "circling": circ,
           "thresholds_v2": develop_thresholds(settings, zsm), "parser": develop_parser()}
    write_json(out_dir / "development.json", res)
    return res


BENCHMARK_V2 = REPO_ROOT / "benchmarks" / "behavior_questions_v2.json"
V1_VOCAB = {"tail_chasing", "tail_directed", "self_grooming", "self_biting", "tail_flicking", "staring_snapping", "scratching_self", "kneading",
            "playing_object", "social_play", "threat_display", "eating", "drinking", "resting", "locomotion", "being_petted", "vocalizing"}


def score_items(items: list[dict], answers: dict[str, dict], system: str) -> list[dict]:
    from ..understand import vocab as V
    from . import e31_behavior_questions as E31

    ann = D.load_assistant()
    rows = []
    for it in items:
        a = answers[it["id"]]
        ans = a["answer"]

        class _A:  # attribute view of the answer dict for E31.score_question
            pass

        A = _A()
        for k, v in ans.items():
            setattr(A, k, v)
        seen = {b["behavior"] for b in ans["behaviors_seen"]}
        r = E31.score_question(it, A, a["parsed"], seen, ann.get(it["clip"], {}))
        if system == "v1":  # knowledge entries of the E31 system: its 17 behaviours (all with entries for their species)
            r["C2_entry"] = all(b in V1_VOCAB and it["species"] in V.BEHAVIORS[b].species for b in it["asked"])
        r["parsed"], r["seen"] = a["parsed"], sorted(seen)
        if system == "v2":
            ctx = ans.get("recording_context", [])
            if "frame_person" in it:
                r["X_person"] = any("person" in c["what"] for c in ctx) == bool(it["frame_person"])
            if it.get("frame_place"):
                got = [c["what"].removeprefix("place: ") for c in ctx if c["what"].startswith("place: ")]
                r["X_place"] = (got[0] if got else None) in it["frame_place"]
                r["X_place_said"] = got[0] if got else None
        rows.append(r)
    return rows


def summarize_extra(rows: list[dict]) -> dict:
    """B1 per category and per asked behaviour for every one in the benchmark (E31's summary lists only its own)."""
    from .e31_behavior_questions import wilson

    def rate(k, n):
        return {"k": k, "n": n, "rate": k / n if n else float("nan"), "wilson95": wilson(k, n)}

    out = {"by_category": {}, "by_behavior": {}}
    for cat in sorted({r["category"] for r in rows}):
        rs = [r for r in rows if r["category"] == cat]
        out["by_category"][cat] = rate(sum(len(r["asked_present_seen"]) for r in rs), sum(len(r["asked_present"]) for r in rs))
    for b in sorted({b for r in rows for b in r["asked_present"]}):
        out["by_behavior"][b] = rate(sum(b in r["asked_present_seen"] for r in rows), sum(b in r["asked_present"] for r in rows))
    return out


EXPLAINED_CHECKS = ("C1_parse", "C2_entry", "C3_grounded", "C4_separation", "C6_observations", "C7_no_certainty", "C8_report_status", "C9_species")


def outcome(r: dict, asked_ok: list[str]) -> str:
    if r["gap"]:
        return "gap_handled" if set(r["parsed"]) <= set(asked_ok) else "gap_forced"
    ok = all(r[c] for c in EXPLAINED_CHECKS) and r.get("C5_coverage", True) and not r["must_not_seen"]
    if not ok:
        return "not_explained"
    return "explained_and_seen" if set(r["asked_present_seen"]) == set(r["asked_present"]) else "explained_from_report"


def decide(res: dict) -> dict:
    """The pre-registered decisions (module docstring)."""
    p, c2 = res["paired"], res["v2"]["explanation"]
    cov1, cov2 = res["v1"]["explanation"]["C5_coverage_expert"]["pass"], c2["C5_coverage_expert"]["pass"]
    rules = {"a_C1_not_below_v1": p["C1_parse"]["v2"] >= p["C1_parse"]["v1"],
             "b_recognized_at_least_v1_minus_1": p["asked_present_recognized"]["v2"] >= p["asked_present_recognized"]["v1"] - 1,
             "c_confusers_at_least_v1_minus_1": p["confusers_rejected"]["v2"] >= p["confusers_rejected"]["v1"] - 1,
             "d_C5_expert_coverage_not_below_v1": cov2 >= cov1,
             "e_C2_C3_C4_C6_to_C9_at_least_0.9": all(c2[c]["rate"] >= 0.9 for c in EXPLAINED_CHECKS if c != "C1_parse")}
    tc = res["v2"]["recognition"].get("B1_behavior_tail_chasing")
    rej = res["v2"]["recognition"]["B4_tail_chasing_rejection"]
    return {"v2_stays_default": all(rules.values()), "rules": rules,
            "parser_C1_v2_at_least_0.9": c2["C1_parse"]["rate"] >= 0.9,
            "tail_chasing_recognized_from_recordings_v2": bool(tc and tc["n"] and tc["rate"] >= 0.7 and rej["n"] and rej["rate"] >= 0.8),
            "context_person_agreement_at_least_0.8": (res["context_v2"]["person"]["agree"] >= 0.8 * res["context_v2"]["person"]["n"])
            if res["context_v2"]["person"]["n"] else None}


def paired(rows1: list[dict], rows2: list[dict], key) -> dict:
    a = [key(r) for r in rows1]
    b = [key(r) for r in rows2]
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return {"n": len(pairs), "v1": sum(x for x, _ in pairs), "v2": sum(y for _, y in pairs), "v2_only": sum(1 for x, y in pairs if y and not x),
            "v1_only": sum(1 for x, y in pairs if x and not y)}


def test(settings: Settings, out_dir: Path, v1_answers: Path) -> dict:
    """Score the frozen v1 answers (``e32_v1_runner.py``) and the v2 system once on the E32 benchmark."""
    import subprocess

    from ..understand.run import Explainer
    from . import e31_behavior_questions as E31

    bench = json.loads(BENCHMARK_V2.read_text())
    items = bench["questions"]
    fdir = settings.paths.resolve("data_root").parent / "understand" / "features_v2"
    ex = Explainer.load(settings.runtime.num_threads, version=2)
    ans2 = {}
    for it in items:
        z = np.load(fdir / f"{it['clip']}.npz", allow_pickle=False)
        q_ans = ex.answer(z, it["species"], it["scene"], it["question"]).to_dict()
        from ..understand import question as Qn

        q = Qn.parse_v2(it["species"], it["scene"], it["question"], ex.matcher)
        ans2[it["id"]] = {"answer": q_ans, "parsed": q.reported}
        write_json(out_dir / "answers" / f"{it['id']}.json", {"benchmark_item": it, "answer": q_ans})
    ans1 = json.loads(Path(v1_answers).read_text())["answers"]
    r1, r2 = score_items(items, ans1, "v1"), score_items(items, ans2, "v2")
    b1, b2 = E31.summarize_b(r1), E31.summarize_b(r2)
    c1, c2 = E31.summarize_c(r1), E31.summarize_c(r2)
    res = {"experiment": "E32", "created_at": utc_now().isoformat(timespec="seconds"),
           "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip(),
           "benchmark": bench["version"], "questions": len(items),
           "v1": {"recognition": b1, "explanation": c1}, "v2": {"recognition": b2, "explanation": c2},
           "paired": {"asked_present_recognized": paired(r1, r2, lambda r: len(r["asked_present_seen"]) if r["asked_present"] else None),
                      "confusers_rejected": paired(r1, r2, lambda r: len(r["must_not"]) - len(r["must_not_seen"]) if r["must_not"] else None),
                      "C1_parse": paired(r1, r2, lambda r: int(r["C1_parse"]) if not r["gap"] else None),
                      "C2_entry": paired(r1, r2, lambda r: int(r["C2_entry"]) if not r["gap"] else None),
                      "C5_coverage": paired(r1, r2, lambda r: int(r["C5_coverage"]) if "C5_coverage" in r else None),
                      "C5_ranking": paired(r1, r2, lambda r: int(r["C5_ranking"]) if r.get("C5_ranking") is not None else None)},
           "context_v2": {"person": {"agree": sum(bool(r.get("X_person")) for r in r2 if "X_person" in r), "n": sum("X_person" in r for r in r2)},
                          "place": {"agree": sum(bool(r.get("X_place")) for r in r2 if "X_place" in r), "n": sum("X_place" in r for r in r2)}},
           "per_question": {"v1": r1, "v2": r2}}
    res["v1"]["recognition_extra"], res["v2"]["recognition_extra"] = summarize_extra(r1), summarize_extra(r2)
    ok_by_id = {it["id"]: it["asked_ok"] for it in items}
    for sysname, rows in (("v1", r1), ("v2", r2)):
        for r in rows:
            r["outcome"] = outcome(r, ok_by_id[r["id"]])
        res[sysname]["outcomes"] = {o: [r["id"] for r in rows if r["outcome"] == o]
                                    for o in ("explained_and_seen", "explained_from_report", "not_explained", "gap_handled", "gap_forced")}
    res["decision"] = decide(res)
    write_json(out_dir / "metrics.json", res)
    md = ["# E32 answers of the v2 system (all benchmark questions)", ""]
    from ..understand.explain import render_markdown

    md += [render_markdown(ans2[it["id"]]["answer"], f"{it['id']} - {it['clip']} ({it['category']})") for it in items]
    (out_dir / "answers.md").write_text("\n".join(md), encoding="utf-8")
    return res


def posthoc(settings: Settings, out_dir: Path) -> dict:
    """Descriptive analyses added after the single scoring run (not pre-registered; nothing is chosen from them).

    References: the benchmark's reference_present plus the assistant frame review's behaviour list (machine-assisted labels)."""
    from collections import Counter

    from ..understand.features import CONTEXT_CLASSES

    m = json.loads((out_dir / "metrics.json").read_text())
    bench = {q["id"]: q for q in json.loads(BENCHMARK_V2.read_text())["questions"]}
    ann = D.load_assistant()
    out: dict = {"note": "post hoc, descriptive; added after the E32 scoring run; not used to change either system", "systems": {}}
    for s in ("v1", "v2"):
        fa, seen_n = Counter(), Counter()
        tc_fp = []
        parse = {"missed_all": [], "missed_some": [], "extra": []}
        for r in m["per_question"][s]:
            q = bench[r["id"]]
            ref = set(q["reference_present"]) | set(ann.get(q["clip"], {}).get("behaviors", []))
            for b in r["seen"]:
                seen_n[b] += 1
                if b not in ref:
                    fa[b] += 1
            if "tail_chasing" in r["seen"] and "tail_chasing" not in ref:
                tc_fp.append(r["id"])
            asked, parsed = set(q["asked"]), set(r["parsed"])
            if asked and not parsed & asked:
                parse["missed_all"].append(r["id"])
            elif asked - parsed:
                parse["missed_some"].append(r["id"])
            if parsed - asked - set(q["asked_ok"]):
                parse["extra"].append({r["id"]: sorted(parsed - asked - set(q["asked_ok"]))})
        n_tc_neg = sum("tail_chasing" not in (set(q["reference_present"]) | set(ann.get(q["clip"], {}).get("behaviors", []))) for q in bench.values())
        out["systems"][s] = {"seen_total": sum(seen_n.values()), "seen_not_in_reference": sum(fa.values()),
                             "seen_not_in_reference_by_behavior": dict(fa.most_common()),
                             "tail_chasing_seen_without_it": {"ids": tc_fp, "clips_without_tail_chasing": n_tc_neg}, "parse_errors": parse}
    fdir = settings.paths.resolve("data_root").parent / "understand" / "features_v2"
    j = CONTEXT_CLASSES.index("person")
    person, place = [], Counter()
    for r in m["per_question"]["v2"]:
        q = bench[r["id"]]
        if "X_person" in r:
            z = np.load(fdir / f"{q['clip']}.npz", allow_pickle=False)
            reported = bool(q["frame_person"]) == bool(r["X_person"])  # X_person: reported == frame review
            person.append({"id": r["id"], "frame_review": bool(q["frame_person"]), "reported": reported,
                           "max_person_score": round(float(z["ctx"][:, j].max()), 2), "share_frames_person_ge_0.5": round(float((z["ctx"][:, j] >= 0.5).mean()), 2),
                           "share_frames_near_ge_0.2": round(float((z["ctx_person_near"] >= 0.2).mean()), 2)})
        if q.get("frame_place"):
            place[f"{q['frame_place'][0]} -> {r.get('X_place_said')}"] += 1
    out["person_v2"] = {"false_reports": [p for p in person if p["reported"] and not p["frame_review"]],
                        "missed": [p for p in person if p["frame_review"] and not p["reported"]],
                        "counts": dict(Counter(f"frame {p['frame_review']} / reported {p['reported']}" for p in person))}
    out["place_v2"] = dict(sorted(place.items()))
    circ = json.loads((REPO_ROOT / "artifacts" / "understand" / "circling_v1.json").read_text())
    tc = []
    for it in bench.values():
        a = json.loads((out_dir / "answers" / f"{it['id']}.json").read_text())["answer"]
        tc += [{"id": it["id"], "score": b["score"], "threshold_applied": b["threshold"], "evidence": b["evidence"]}
               for b in a["behaviors_seen"] if b["behavior"] == "tail_chasing"]
    out["tail_chasing_v2_threshold"] = {
        "note": "the clip decision compared the combined score (mean of zero-shot and window model) with the tail-chasing threshold of "
                "recognizer_v2.json, which the part-A rule chose for zero-shot scores; development chose a different threshold for the "
                "combined score (circling_v1.json); counterfactual only, nothing is changed",
        "seen": tc, "development_threshold_for_combined_score": circ["threshold"],
        "seen_at_development_threshold": [t["id"] for t in tc if t["score"] >= circ["threshold"]]}
    # Knowledge base v2 added behaviours and candidates for conditions that the chosen test papers cover (the test material and the
    # new entries were registered together in 300eab5, before development): those questions favour v2 by construction.
    topic = {"t03": "neurological circling candidate", "t06": "head_tremor", "t08": "sleep_movements", "t09": "collapse", "t12": "rolling_rubbing"}
    sens = {"questions_excluded": topic, "note": "the same measures on the questions whose behaviour or candidate was not added for the test topics"}
    for s in ("v1", "v2"):
        rows = [r for r in m["per_question"][s] if r["id"] not in topic]
        ng = [r for r in rows if not r["gap"]]
        ex = [r for r in rows if r.get("C5_authority") == "expert"]
        sens[s] = {**{c: f"{sum(bool(r[c]) for r in ng)}/{len(ng)}" for c in EXPLAINED_CHECKS},
                   "asked_present_recognized": f"{sum(len(r['asked_present_seen']) for r in rows)}/{sum(len(r['asked_present']) for r in rows)}",
                   "confusers_rejected": f"{sum(len(r['must_not']) - len(r['must_not_seen']) for r in rows)}/{sum(len(r['must_not']) for r in rows)}",
                   "C5_coverage_expert": f"{sum(bool(r['C5_coverage']) for r in ex)}/{len(ex)}",
                   "C5_ranking_expert": f"{sum(bool(r.get('C5_ranking')) for r in ex)}/{len(ex)}",
                   "outcomes": dict(Counter(r["outcome"] for r in rows))}
    out["without_topic_matched_questions"] = sens
    write_json(out_dir / "posthoc.json", out)
    return out


def default_check(settings: Settings, out_dir: Path, v1_answers: Path) -> dict:
    """After the decision: the default system of this code (``Explainer.load()``, version 1 = ``understand/e31``) must give the
    answers the frozen v1 system gave in the test (same version-1 features). The E31 ``Explainer`` adds recording-quality notes
    (no sound, no video) to the uncertainty list; the E31 evaluation and the E32 v1 runner call ``compose`` without them, so those
    notes are the one expected difference (no check reads them)."""
    from ..understand.run import DEFAULT_VERSION, Explainer

    items = json.loads(BENCHMARK_V2.read_text())["questions"]
    fdir = settings.paths.resolve("data_root").parent / "understand" / "features"
    ex = Explainer.load(settings.runtime.num_threads)
    ref = json.loads(Path(v1_answers).read_text())["answers"]
    quality = ("No video frames could be decoded", "No usable sound in the recording", "The recording is silent")
    exact, other = [], []
    for it in items:
        a = json.loads(json.dumps(ex.answer(np.load(fdir / f"{it['clip']}.npz", allow_pickle=False), it["species"], it["scene"],
                                            it["question"]).to_dict()))
        if a == ref[it["id"]]["answer"]:
            exact.append(it["id"])
        elif {**a, "uncertainty": [u for u in a["uncertainty"] if not u.startswith(quality)]} != ref[it["id"]]["answer"]:
            other.append(it["id"])
    res = {"created_at": utc_now().isoformat(timespec="seconds"), "default_version": DEFAULT_VERSION, "questions": len(items),
           "identical_answers": len(exact), "identical_except_quality_notes": len(items) - len(exact) - len(other), "different": other}
    write_json(out_dir / "default_v1_check.json", res)
    return res


def main() -> None:
    import argparse

    from ..config import load_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["develop", "test", "posthoc", "default-check"])
    ap.add_argument("--v1-answers", default="var/understand/e32_v1_answers.json")
    a = ap.parse_args()
    s = load_settings()
    if a.step == "default-check":
        print(json.dumps(default_check(s, s.paths.resolve("results_root") / "e32", Path(a.v1_answers)), indent=1))
        return
    if a.step == "posthoc":
        print(json.dumps(posthoc(s, s.paths.resolve("results_root") / "e32"), indent=1))
        return
    if a.step == "test":
        r = test(s, s.paths.resolve("results_root") / "e32", Path(a.v1_answers))
        print(json.dumps({k: r[k] for k in ("paired", "context_v2", "decision")}, indent=1, default=str))
        print(json.dumps({s: r[s]["outcomes"] for s in ("v1", "v2")}, indent=1))
        return
    r = develop(s, s.paths.resolve("results_root") / "e32")
    print(json.dumps({"circling": {k: v for k, v in r["circling"].items() if k != "coefficients"}, "thresholds_v2": r["thresholds_v2"],
                      "parser": r["parser"]}, indent=1, default=str))


if __name__ == "__main__":
    main()

