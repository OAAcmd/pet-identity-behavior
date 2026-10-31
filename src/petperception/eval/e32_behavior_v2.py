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
    tc = b2.get("B1_behavior_tail_chasing", {"rate": float("nan")})
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
           "tail_chasing_recognized_from_recordings_v2": bool(tc["rate"] >= 0.7 and b2["B4_tail_chasing_rejection"]["rate"] >= 0.8)
           if tc.get("n") else False,
           "per_question": {"v1": r1, "v2": r2}}
    write_json(out_dir / "metrics.json", res)
    md = ["# E32 answers of the v2 system (all benchmark questions)", ""]
    from ..understand.explain import render_markdown

    md += [render_markdown(ans2[it["id"]]["answer"], f"{it['id']} - {it['clip']} ({it['category']})") for it in items]
    (out_dir / "answers.md").write_text("\n".join(md), encoding="utf-8")
    return res


def main() -> None:
    import argparse

    from ..config import load_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["develop", "test"])
    ap.add_argument("--v1-answers", default="var/understand/e32_v1_answers.json")
    a = ap.parse_args()
    s = load_settings()
    if a.step == "test":
        r = test(s, s.paths.resolve("results_root") / "e32", Path(a.v1_answers))
        print(json.dumps({k: r[k] for k in ("v1", "v2", "paired", "context_v2", "tail_chasing_recognized_from_recordings_v2")}, indent=1, default=str)[:6000])
        return
    r = develop(s, s.paths.resolve("results_root") / "e32")
    print(json.dumps({"circling": {k: v for k, v in r["circling"].items() if k != "coefficients"}, "thresholds_v2": r["thresholds_v2"],
                      "parser": r["parser"]}, indent=1, default=str))


if __name__ == "__main__":
    main()

