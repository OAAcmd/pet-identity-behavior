"""E31: explaining a pet's behaviour in context - recognition and explanation support on real recordings.

Protocol fixed before any E31 number was computed (this file and
``benchmarks/behavior_questions.json`` are committed first; the results record
the commit).

Data
* Pool: every clip of the behaviour collection (Wikimedia Commons, CC BY paper
  supplements, Internet Archive; ``understand/dataset.py``) whose features hold
  at least ``MIN_CROPS`` located animal crops. The Commons copy of Tiira et al.
  Movie S2 is a duplicate of ``paper_tiira2012_s2`` and is removed. Labels:
  human labels only - uploader title / description / categories (*source*) and
  paper captions (*expert*, case level) - mapped by ``SOURCE_MAP``; the
  assistant frame-review labels are never used in part A. Groups: uploader or
  paper (no group is split across folds).
* Benchmark: ``benchmarks/behavior_questions.json`` (37 questions on 37 clips;
  references in the file: the assistant frame review where it agrees with the
  source or expert label; sound from source labels).

Part A - recognition on the pool (human labels)
* Eligible behaviours: visual, at least 8 positive clips from at least 3 groups.
* R0 zero-shot SigLIP (vocabulary prompts; clip score = mean of the top
  quarter of per-frame probabilities); R1 trained head (``understand/head.py``,
  C by inner grouped CV on training folds); R2 mean of R0 and R1.
* 5-fold grouped cross-validation (GroupKFold); out-of-fold scores.
* Metric: average precision per behaviour, macro AP over eligible behaviours;
  95 % intervals by cluster bootstrap over groups (1,000 resamples, seed 0).
* Adoption: R1 or R2 replaces R0 for the eligible behaviours if its macro-AP
  gain over R0 is at least 0.03 and the bootstrap interval of the gain excludes
  0 (the larger gain if both qualify). Otherwise R0 stays.
* Thresholds: eligible behaviours use the score that maximizes F1 on
  out-of-fold scores (cross-fitted: a fold's clips get the threshold chosen on
  the other folds; the deployed threshold uses all clips); every other visual
  behaviour keeps the fixed zero-shot default 0.30, vocalizing the fixed YAMNet
  default 0.30.
* Sound (descriptive): YAMNet vocal score against the source 'vocalizing' label.

Part B - benchmark questions (adopted recognizer, cross-fitted as above)
Recognition, on ``reference_present`` / ``must_not_recognize``:
* B1 asked-and-present behaviours recognized (seen), by category; tail chasing
  separately (Wilson intervals);
* B2 every present behaviour (asked or not);
* B3 temporal: for B1 hits with frame-review bouts, a recognized bout overlaps
  one (clip-level detections without bouts do not count);
* B4 confusers: must-not behaviours not recognized.
Explanation support (per question, independent of recognition):
* C1 parse: asked <= parsed <= asked + asked_ok;
* C2 a knowledge entry for every asked behaviour (gap questions reported apart);
* C3 grounding: every candidate outside play_or_normal cites a source, every
  cited id resolves, every explained behaviour has candidates;
* C4 separation: request_or_intention_possible equals request_expected, and the
  no-request note appears exactly when no request candidate is offered (end to
  end: an asked behaviour the answer does not explain fails; the share among
  explained behaviours is reported as a secondary number);
* C5 source interpretation: coverage - each kind the source states is offered
  for the asked behaviours (expert and lay apart); ranking - the first
  candidate of the first asked behaviour has one of the expert's kinds;
* C6 distinguishing observations whenever two or more kinds are offered; urgent
  signs whenever a medical candidate is offered;
* C7 no certainty phrase (``CERTAINTY_PHRASES``) in the summary or candidate
  texts, and the first uncertainty statement says the recording cannot
  establish why;
* C8 an asked behaviour is reported as seen only when it was recognized;
* C9 the answer keeps the user's species.
Targets (fixed): C1-C4 and C6-C9 each pass on at least 90 % of non-gap
questions; C5 coverage at least 0.9 of questions with an expert
interpretation. Tail chasing counts as recognized from recordings only if B1
for tail chasing is at least 0.7 and B4 rejection at least 0.8; otherwise
answers about tail chasing rest on the user's report and say so.

Part C - downstream effect of the G1b detector (run when its export exists;
descriptive, no adoption decision here; G1b deployment follows
docs/TRAINING_PROGRAM.md): animal-located fraction per clip, part A R0 macro
AP and part B B1/B4 with features re-extracted using the G1b detector at the
current detector's score threshold.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT, Settings
from ..understand import dataset as D
from ..understand import explain as X
from ..understand import knowledge as K
from ..understand import question as Qn
from ..understand import recognize as R
from ..understand import vocab as V
from ..understand.head import BehaviorHead, clip_vector
from ..utils import utc_now, write_json
from .e16_breed_robustness import wilson

MIN_CROPS = 2
MIN_POS, MIN_GROUPS = 8, 3
FOLDS = 5
N_BOOT = 1000
ADOPT_GAIN = 0.03
SOURCE_MAP = {"hiss_growl": "threat_display", "being_handled": "being_petted", "self_directed_biting": "self_biting", "fly_snapping": "staring_snapping",
              "staring": "staring_snapping"}
UNMAPPED = ("fighting", "hunting_chasing", "jumping", "allogrooming")
DUPLICATES = ("pone.0041684",)  # Commons copies of paper supplements already in the collection
CERTAINTY_PHRASES = ("is caused by", "definitely", "certainly", "for sure", "clearly means", "this proves", "the reason is", "is due to", "your cat has ",
                     "your dog has ", "suffers from", "is diagnosed", "we can confirm")
TARGET_PASS = 0.9
TC_HIT_MIN, TC_REJECT_MIN = 0.7, 0.8
BENCHMARK = REPO_ROOT / "benchmarks" / "behavior_questions.json"


def human_labels(c: D.Clip) -> set[str]:
    out = set()
    for layer in ("source", "expert"):
        for lab in c.labels.get(layer, []):
            if lab in UNMAPPED:
                continue
            lab = SOURCE_MAP.get(lab, lab)
            if lab in V.BEHAVIORS and c.species in V.BEHAVIORS[lab].species:
                out.add(lab)
    return out


def load_pool(settings: Settings, features_dir: Path | None = None) -> tuple[list[D.Clip], list, dict]:
    data_root = settings.paths.resolve("data_root")
    fdir = features_dir or (data_root.parent / "understand" / "features")
    clips, zs, info = [], [], Counter()
    for c in D.clips(data_root):
        info["collection"] += 1
        if any(d in c.title for d in DUPLICATES) and c.source == "commons":
            info["duplicate_removed"] += 1
            continue
        f = fdir / f"{c.key}.npz"
        if not f.exists():
            info["no_features"] += 1
            continue
        z = np.load(f, allow_pickle=False)
        if "emb" not in z.files or len(z["emb"]) < MIN_CROPS:
            info["animal_not_located"] += 1
            clips.append(c)
            zs.append(z)  # kept for the benchmark (scored as not recognized); excluded from part A
            continue
        clips.append(c)
        zs.append(z)
        info["in_pool"] += 1
    return clips, zs, dict(info)


def folds_for(groups: list[str], k: int = FOLDS) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    g = np.array(groups)
    fold = np.zeros(len(g), int)
    for i, (_, te) in enumerate(GroupKFold(k).split(np.zeros(len(g)), groups=g)):
        fold[te] = i
    return fold


def ap(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    best, thr = -1.0, float(np.max(s)) + 1e-6
    for t in np.unique(s):
        pred = s >= t
        tp = float(np.sum(pred & (y == 1)))
        f1 = 2 * tp / max(1.0, float(pred.sum() + y.sum()))
        if f1 > best:
            best, thr = f1, float(t)
    return thr


def cluster_boot(groups: np.ndarray, fn, n: int = N_BOOT, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        v = fn(idx)
        if v == v:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


def part_a(clips, zs, zsm: R.ZeroShot, yam_labels: list[str]) -> dict:
    keep = [i for i, z in enumerate(zs) if "emb" in z.files and len(z["emb"]) >= MIN_CROPS]
    C = [clips[i] for i in keep]
    Z = [zs[i] for i in keep]
    groups = np.array([c.group for c in C])
    labels = [human_labels(c) for c in C]
    visual = sorted({b for b in V.BEHAVIORS if V.BEHAVIORS[b].channel == "visual"})
    counts = {b: (sum(b in l for l in labels), len({c.group for c, l in zip(C, labels) if b in l})) for b in visual}
    eligible = [b for b in visual if counts[b][0] >= MIN_POS and counts[b][1] >= MIN_GROUPS]
    Y = {b: np.array([b in l for l in labels], int) for b in visual}
    # R0
    r0 = np.zeros((len(C), len(visual)))
    for i, (c, z) in enumerate(zip(C, Z)):
        sc = R.zero_shot_scores(z, c.species, zsm)
        r0[i] = [sc.get(b, 0.0) for b in visual]
    # R1 out of fold
    Xv = np.stack([clip_vector(z, c.species) for c, z in zip(C, Z)])
    fold = folds_for(list(groups))
    r1 = np.zeros_like(r0)
    fold_heads = {}
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        h = BehaviorHead.fit(Xv[tr], {b: Y[b][tr] for b in eligible}, groups[tr])
        fold_heads[k] = h
        for i in np.where(te)[0]:
            s = h.scores(Xv[i])
            r1[i] = [s.get(b, 0.0) for b in visual]
    r2 = 0.5 * (r0 + r1)
    col = {b: j for j, b in enumerate(visual)}

    def macro(S, idx=None):
        idx = np.arange(len(C)) if idx is None else idx
        v = [ap(Y[b][idx], S[idx, col[b]]) for b in eligible]
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    res = {"clips": len(C), "groups": int(len(set(groups))), "label_counts": {b: {"clips": counts[b][0], "groups": counts[b][1]} for b in visual},
           "eligible": eligible, "methods": {}}
    for name, S in (("R0_zero_shot", r0), ("R1_head", r1), ("R2_mean", r2)):
        res["methods"][name] = {"macro_ap": macro(S), "macro_ap_ci95": cluster_boot(groups, lambda idx, S=S: macro(S, idx)),
                                "per_behavior_ap": {b: ap(Y[b], S[:, col[b]]) for b in eligible}}
    prev = {b: float(Y[b].mean()) for b in eligible}
    res["chance_macro_ap"] = float(np.mean(list(prev.values()))) if prev else float("nan")
    gains = {}
    for name, S in (("R1_head", r1), ("R2_mean", r2)):
        g = macro(S) - macro(r0)
        ci = cluster_boot(groups, lambda idx, S=S: macro(S, idx) - macro(r0, idx))
        gains[name] = {"gain": g, "ci95": ci, "qualifies": bool(g >= ADOPT_GAIN and ci[0] > 0)}
    res["gains_over_R0"] = gains
    q = [n for n, v in gains.items() if v["qualifies"]]
    adopted = max(q, key=lambda n: gains[n]["gain"]) if q else "R0_zero_shot"
    res["adopted"] = adopted
    S_ad = {"R0_zero_shot": r0, "R1_head": r1, "R2_mean": r2}[adopted]
    # thresholds: cross-fitted per fold, and deployed on all clips
    th_fold = {k: {b: best_f1_threshold(Y[b][fold != k], S_ad[fold != k, col[b]]) for b in eligible} for k in range(FOLDS)}
    th_all = {b: best_f1_threshold(Y[b], S_ad[:, col[b]]) for b in eligible}
    res["thresholds_deployed"] = th_all
    res["f1_at_crossfit_thresholds"] = {}
    for b in eligible:
        pred = np.array([S_ad[i, col[b]] >= th_fold[fold[i]][b] for i in range(len(C))])
        tp = float(np.sum(pred & (Y[b] == 1)))
        res["f1_at_crossfit_thresholds"][b] = {"f1": 2 * tp / max(1.0, float(pred.sum() + Y[b].sum())), "precision": tp / max(1.0, float(pred.sum())),
                                               "recall": tp / max(1.0, float(Y[b].sum()))}
    # sound (descriptive)
    ya, ys = [], []
    for c, z in zip(C, Z):
        if "a_scores" in z.files:
            v, _ = R.vocal_bouts(z, c.species, yam_labels)
            ya.append(v)
            ys.append(int("vocalizing" in human_labels(c)))
    res["sound_vocalizing"] = {"clips_with_sound": len(ya), "positives": int(sum(ys)), "ap": ap(np.array(ys), np.array(ya)) if ya else float("nan"),
                               "base_rate": float(np.mean(ys)) if ys else float("nan")}
    state = {"keys": [c.key for c in C], "fold": fold, "fold_heads": fold_heads, "th_fold": th_fold, "th_all": th_all, "adopted": adopted,
             "eligible": eligible, "Xv": Xv, "Y": Y, "groups": groups}
    return res, state


def config_for(adopted: str, eligible: list[str], thresholds: dict[str, float], provenance: dict) -> R.RecognizerConfig:
    method = {"R0_zero_shot": "zero_shot", "R1_head": "head", "R2_mean": "mean"}[adopted]
    return R.RecognizerConfig(version=f"recognizer-{method}-v1", method=method, head_behaviors=list(eligible) if method != "zero_shot" else [],
                              thresholds={b: round(float(t), 6) for b, t in thresholds.items()}, provenance=provenance)


def overlaps(bouts: list[list[float]], ref: list[dict]) -> bool:
    return any(min(b[1], r["t1"]) > max(b[0], r["t0"]) for b in bouts for r in ref)


def score_question(item: dict, ans: X.Answer, parsed: list[str], seen: set[str], ann: dict) -> dict:
    asked, ok = set(item["asked"]), set(item["asked_ok"])
    explained = {e["behavior"]: e for e in ans.explained_behaviors}
    r: dict = {"id": item["id"], "category": item["category"], "gap": bool(item.get("gap"))}
    # recognition
    present = set(item["reference_present"])
    r["asked_present"] = sorted(asked & present)
    r["asked_present_seen"] = sorted(asked & present & seen)
    r["present"] = sorted(present)
    r["present_seen"] = sorted(present & seen)
    r["must_not"] = sorted(item["must_not_recognize"])
    r["must_not_seen"] = sorted(set(item["must_not_recognize"]) & seen)
    bouts_by = {b["behavior"]: b["bouts_s"] for b in ans.behaviors_seen}
    ref_bouts = defaultdict(list)
    for b in ann.get("bouts", []):
        ref_bouts[b["behavior"]].append(b)
    r["temporal"] = {b: overlaps(bouts_by.get(b, []), ref_bouts[b]) for b in r["asked_present_seen"] if ref_bouts.get(b)}
    # explanation support
    r["C1_parse"] = asked <= set(parsed) <= (asked | ok)
    r["C2_entry"] = all(K.entry(b, item["species"]) is not None for b in asked)
    c3 = bool(ans.explained_behaviors) or not asked
    for e in ans.explained_behaviors:
        cands = [c for g in e["candidates_by_kind"].values() for c in g]
        if not cands:
            c3 = False
        for kind, g in e["candidates_by_kind"].items():
            for c in g:
                if kind != "play_or_normal" and not c["sources"]:
                    c3 = False
                if any(s not in K.SOURCES for s in c["sources"]):
                    c3 = False
    r["C3_grounded"] = c3
    c4 = True
    for b, want in item.get("request_expected", {}).items():
        e = explained.get(b)
        if e is None:
            c4 = False
            continue
        if e["request_or_intention_possible"] != want or (e["no_request_note"] is None) != e["request_or_intention_possible"]:
            c4 = False
    r["C4_separation"] = c4
    given = [(b, w) for b, w in item.get("request_expected", {}).items() if b in explained]
    r["C4_given_explained"] = (all(explained[b]["request_or_intention_possible"] == w for b, w in given) if given else None)
    interp = item.get("source_interpretation")
    if interp and interp["kinds"]:
        offered = {k for b in item["asked"] if b in explained for k in explained[b]["candidates_by_kind"]}
        r["C5_coverage"] = set(interp["kinds"]) <= offered
        r["C5_authority"] = interp["authority"]
        first = explained.get(item["asked"][0]) if item["asked"] else None
        r["C5_first_kind"] = first["kind_order_by_context"][0] if first else None
        r["C5_ranking"] = bool(first and first["kind_order_by_context"][0] in interp["kinds"]) if interp["authority"] == "expert" else None
    c6 = True
    for e in ans.explained_behaviors:
        kinds = set(e["candidates_by_kind"])
        if len(kinds) >= 2 and not ans.observations_needed:
            c6 = False
        if "medical" in kinds and not ans.urgent_signs:
            c6 = False
    r["C6_observations"] = c6
    texts = [ans.summary] + [c["explanation"] for e in ans.explained_behaviors for g in e["candidates_by_kind"].values() for c in g]
    bad = [p for p in CERTAINTY_PHRASES for t in texts if p in t.lower()]
    r["C7_no_certainty"] = not bad and "cannot establish why" in ans.uncertainty[0]
    r["C7_phrases_found"] = sorted(set(bad))
    r["C8_report_status"] = all((u["status"].startswith("also seen")) == (u["behavior"] in seen) for u in ans.user_reported)
    r["C9_species"] = ans.species == item["species"]
    return r


def summarize_b(rows: list[dict]) -> dict:
    def rate(k, n):
        return {"k": k, "n": n, "rate": k / n if n else float("nan"), "wilson95": wilson(k, n)}

    out = {}
    for cat in ("all", "tail_chasing", "tail_directed", "self_directed", "confuser", "other", "gap"):
        rs = rows if cat == "all" else [r for r in rows if r["category"] == cat]
        k = sum(len(r["asked_present_seen"]) for r in rs)
        n = sum(len(r["asked_present"]) for r in rs)
        out[f"B1_{cat}"] = rate(k, n)
    for b in ("tail_chasing", "tail_directed", "self_grooming", "kneading", "vocalizing", "playing_object", "social_play", "threat_display"):
        k = sum(b in r["asked_present_seen"] for r in rows)
        n = sum(b in r["asked_present"] for r in rows)
        if n:
            out[f"B1_behavior_{b}"] = rate(k, n)
    out["B2_present_seen"] = rate(sum(len(r["present_seen"]) for r in rows), sum(len(r["present"]) for r in rows))
    tk = [v for r in rows for v in r["temporal"].values()]
    out["B3_temporal_overlap"] = rate(int(sum(tk)), len(tk))
    n = sum(len(r["must_not"]) for r in rows)
    out["B4_confuser_rejection"] = rate(n - sum(len(r["must_not_seen"]) for r in rows), n)
    nt = sum(("tail_chasing" in r["must_not"]) for r in rows)
    out["B4_tail_chasing_rejection"] = rate(nt - sum(("tail_chasing" in r["must_not_seen"]) for r in rows), nt)
    return out


def summarize_c(rows: list[dict]) -> dict:
    ng = [r for r in rows if not r["gap"]]
    out = {}
    for c in ("C1_parse", "C2_entry", "C3_grounded", "C4_separation", "C6_observations", "C7_no_certainty", "C8_report_status", "C9_species"):
        k = sum(bool(r[c]) for r in ng)
        out[c] = {"pass": k, "n": len(ng), "rate": k / len(ng), "wilson95": wilson(k, len(ng)), "meets_target": k / len(ng) >= TARGET_PASS,
                  "failed": [r["id"] for r in ng if not r[c]]}
        g = [r for r in rows if r["gap"]]
        out[c]["gap_questions_pass"] = f"{sum(bool(r[c]) for r in g)}/{len(g)}"
    g4 = [r for r in ng if r["C4_given_explained"] is not None]
    out["C4_given_explained"] = {"pass": sum(bool(r["C4_given_explained"]) for r in g4), "n": len(g4), "note": "secondary: only behaviours the answer explained"}
    for auth in ("expert", "lay"):
        rs = [r for r in rows if r.get("C5_authority") == auth]
        k = sum(bool(r["C5_coverage"]) for r in rs)
        out[f"C5_coverage_{auth}"] = {"pass": k, "n": len(rs), "rate": k / len(rs) if rs else float("nan"),
                                      "failed": [r["id"] for r in rs if not r["C5_coverage"]]}
    out["C5_coverage_expert"]["meets_target"] = out["C5_coverage_expert"]["rate"] >= TARGET_PASS
    rs = [r for r in rows if r.get("C5_ranking") is not None]
    out["C5_ranking_expert"] = {"pass": sum(r["C5_ranking"] for r in rs), "n": len(rs),
                                "detail": {r["id"]: r["C5_first_kind"] for r in rs}}
    return out


def part_b(settings: Settings, clips, zs, state: dict, zsm: R.ZeroShot, yam_labels: list[str], out_dir: Path | None) -> tuple[list[dict], list[dict]]:
    items = json.loads(BENCHMARK.read_text())["questions"]
    by_key = {c.key: (c, z) for c, z in zip(clips, zs)}
    ann = D.load_assistant()
    pos = {k: i for i, k in enumerate(state["keys"])}
    full_head = BehaviorHead.fit(state["Xv"], {b: state["Y"][b] for b in state["eligible"]}, state["groups"]) if state["adopted"] != "R0_zero_shot" else None
    rows, answers = [], []
    for it in items:
        c, z = by_key[it["clip"]]
        if it["clip"] in pos:
            k = int(state["fold"][pos[it["clip"]]])
            head, th, fit_on = state["fold_heads"][k], state["th_fold"][k], f"fold {k} (clip's group held out)"
        else:
            head, th, fit_on = full_head, state["th_all"], "all pool clips (clip not in the pool)"
        cfg = config_for(state["adopted"], state["eligible"], th, {})
        q = Qn.parse(it["species"], it["scene"], it["question"])
        readings, pattern = R.recognize(z, it["species"], zsm, yam_labels if "a_scores" in z.files else None, head if cfg.method != "zero_shot" else None, cfg)
        ans = X.compose(q, readings, pattern, cfg)
        seen = {b["behavior"] for b in ans.behaviors_seen}
        row = score_question(it, ans, q.reported, seen, ann.get(it["clip"], {}))
        row["parsed"] = q.reported
        row["seen"] = sorted(seen)
        row["scorer"] = fit_on
        rows.append(row)
        a = {"benchmark_item": it, "answer": ans.to_dict(), "scores": {r.behavior: r.score for r in readings}}
        answers.append(a)
        if out_dir is not None:
            write_json(out_dir / "answers" / f"{it['id']}.json", a)
    return rows, answers


def render_example(a: dict) -> str:
    it = a["benchmark_item"]
    return X.render_markdown(a["answer"], f"{it['id']} - {it['clip']} ({it['category']})")


def run(settings: Settings, out_dir: Path) -> dict:
    from ..intent import av_features as AF
    from ..understand.features import SIGLIP_REVISION, SiglipEncoder

    t0 = utc_now()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    clips, zs, info = load_pool(settings)
    enc = SiglipEncoder(threads=settings.runtime.num_threads)
    zsm = R.ZeroShot(enc)
    yam_labels = AF.load_components(1, with_audio=True).yamnet.labels
    a, state = part_a(clips, zs, zsm, yam_labels)
    rows, answers = part_b(settings, clips, zs, state, zsm, yam_labels, out_dir)
    b, c = summarize_b(rows), summarize_c(rows)
    tc_ok = b["B1_behavior_tail_chasing"]["rate"] >= TC_HIT_MIN and b["B4_tail_chasing_rejection"]["rate"] >= TC_REJECT_MIN \
        if "B1_behavior_tail_chasing" in b else False
    res = {"experiment": "E31", "created_at": t0.isoformat(timespec="seconds"), "code_commit": commit, "encoder": SIGLIP_REVISION,
           "knowledge_version": K.KNOWLEDGE_VERSION, "benchmark": json.loads(BENCHMARK.read_text())["version"], "pool": info, "part_a": a,
           "part_b_recognition": b, "part_b_explanation": c, "tail_chasing_recognized_from_recordings": tc_ok, "per_question": rows}
    write_json(out_dir / "metrics.json", res)
    by_key = {c.key: c for c in clips}
    write_json(out_dir / "splits.json", {"grouping": "uploader (Commons, Internet Archive) or paper; GroupKFold(5) over the part A pool",
                                         "clips": [{"key": k, "source": by_key[k].source, "group": by_key[k].group, "fold": int(f),
                                                    "human_labels": sorted(human_labels(by_key[k]))} for k, f in zip(state["keys"], state["fold"])],
                                         "not_in_pool": sorted(c.key for c in clips if c.key not in set(state["keys"]))})
    # deployed recognizer and head (artifacts/understand/)
    art = REPO_ROOT / "artifacts" / "understand"
    prov = {"experiment": "E31", "code_commit": commit, "pool_clips": a["clips"], "pool_groups": a["groups"], "adopted": a["adopted"],
            "eligible": a["eligible"], "created_at": t0.isoformat(timespec="seconds")}
    cfg = config_for(a["adopted"], a["eligible"], a["thresholds_deployed"], prov)
    cfg.save(art / "recognizer.json")
    if a["adopted"] != "R0_zero_shot":
        BehaviorHead.fit(state["Xv"], {x: state["Y"][x] for x in a["eligible"]}, state["groups"],
                         {"encoder": SIGLIP_REVISION, **prov}).save(art / "behavior_head")
    md = ["# E31 benchmark answers (all 37 questions)", "", "Generated by `petperception evaluate --experiment e31`; scores cross-fitted.", ""]
    md += [render_example(x) for x in answers]
    (out_dir / "answers.md").write_text("\n".join(md), encoding="utf-8")
    return res


def run_downstream(settings: Settings, out_dir: Path, subdir: str, detector_label: str) -> dict:
    """Part C: the same protocol on features extracted with another detector (``var/understand/<subdir>``), next to the default ones."""
    from ..intent import av_features as AF

    base_clips, base_zs, base_info = load_pool(settings)
    alt_clips, alt_zs, alt_info = load_pool(settings, settings.paths.resolve("data_root").parent / "understand" / subdir)
    common = sorted({c.key for c in base_clips} & {c.key for c in alt_clips})
    bz = {c.key: z for c, z in zip(base_clips, base_zs)}
    az = {c.key: z for c, z in zip(alt_clips, alt_zs)}

    def found(z):
        return float(np.mean(~np.isnan(z["box"][:, 0]))) if "box" in z.files and len(z["box"]) else 0.0

    fb = np.array([found(bz[k]) for k in common])
    fa = np.array([found(az[k]) for k in common])
    from ..understand.features import SiglipEncoder

    zsm = R.ZeroShot(SiglipEncoder(threads=settings.runtime.num_threads))
    yam = AF.load_components(1, with_audio=True).yamnet.labels
    keep = set(common)
    res = {"detector": detector_label, "clips_compared": len(common), "base_pool": base_info, "alt_pool": alt_info,
           "located_fraction_mean": {"default": float(fb.mean()), "alternative": float(fa.mean())},
           "clips_located_in_more_frames": int(np.sum(fa > fb + 0.05)), "clips_located_in_fewer_frames": int(np.sum(fa < fb - 0.05))}
    for name, (cl, zz) in (("default", (base_clips, base_zs)), ("alternative", (alt_clips, alt_zs))):
        idx = [i for i, c in enumerate(cl) if c.key in keep]
        a, state = part_a([cl[i] for i in idx], [zz[i] for i in idx], zsm, yam)
        rows, _ = part_b(settings, [cl[i] for i in idx], [zz[i] for i in idx], state, zsm, yam, None)
        res[name] = {"part_a_R0_macro_ap": a["methods"]["R0_zero_shot"]["macro_ap"], "part_a_R0_ci95": a["methods"]["R0_zero_shot"]["macro_ap_ci95"],
                     "part_a_adopted": a["adopted"], "part_a_adopted_macro_ap": a["methods"][a["adopted"]]["macro_ap"],
                     "B": {k: v for k, v in summarize_b(rows).items() if k in ("B1_all", "B1_behavior_tail_chasing", "B2_present_seen",
                                                                            "B4_confuser_rejection", "B4_tail_chasing_rejection")}}
    write_json(out_dir / f"downstream_{subdir}.json", res)
    return res


def main() -> None:
    import argparse

    from ..config import load_settings

    ap = argparse.ArgumentParser(description="E31 part C: downstream effect of another detector (features in var/understand/<subdir>).")
    ap.add_argument("--subdir", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    s = load_settings()
    print(json.dumps(run_downstream(s, s.paths.resolve("results_root") / "e31", a.subdir, a.label), indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
