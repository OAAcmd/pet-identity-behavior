"""E1: cat/dog sound-event evaluation on ESC-50 with source-preserving folds.

Task: clip-level 3-way decision {cat, dog, other} plus binary cat-vs-rest and
dog-vs-rest detection. ESC-50 has clip-level labels only (5 s clips that often
contain padded silence), so window-level event timing cannot be scored here.

Protocol
* Outer CV: the 5 official ESC-50 folds (clips from one Freesound source are
  kept in one fold by the dataset authors; 4 non-pet sources were verified to
  span two folds and are reported, see manifests/resources.json).
* Everything fitted (feature scaling, C, decision thresholds, Platt scaling)
  uses only the 4 training folds of each outer split; inner model selection
  is leave-one-fold-out over those 4 folds.
* Bootstrap CIs resample Freesound source recordings, not clips.

Methods: majority, stratified-random, MFCC+logistic regression (CPU
baseline), frozen YAMNet tags with nested thresholds, and frozen YAMNet
embeddings + logistic regression (optional comparison).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import linear_model
from ..audio.events import build_groups, group_scores
from ..audio.features import FeatureConfig, extract, feature_names
from ..audio.yamnet import YamnetBackend
from ..config import Settings
from ..data import esc50
from ..ingest.media import AudioTrack
from ..registry import check_model, model_dir
from ..utils import utc_now, write_json
from . import metrics as M

LABELS = ["cat", "dog", "other"]
log = logging.getLogger("petperception.e1")


def _cache_key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def compute_inputs(settings: Settings, root: Path, clips: list[esc50.Esc50Clip], cache_dir: Path) -> dict[str, np.ndarray]:
    manifest = settings.paths.resolve("manifests_root") / "esc50_files.csv"
    mhash = hashlib.sha256(manifest.read_bytes()).hexdigest()[:16]
    fcfg = FeatureConfig()
    key = _cache_key(mhash, fcfg.version, json.dumps(fcfg.to_dict(), sort_keys=True), settings.audio.yamnet.model_id, str(settings.audio.yamnet.hop_s))
    cache = cache_dir / f"e1_inputs_{key}.npz"
    if cache.exists():
        log.info("using cached inputs %s", cache)
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}
    ready = check_model(settings, settings.audio.yamnet.model_id)
    if not ready.ready:
        raise RuntimeError(f"YAMNet not ready: {ready.reason}")
    backend = YamnetBackend(model_dir(settings, settings.audio.yamnet.model_id), settings.runtime.num_threads, with_embeddings=True)
    groups = build_groups(backend)
    n = len(clips)
    mfcc = np.zeros((n, len(feature_names(fcfg))), np.float32)
    cat_s = np.zeros(n, np.float32)
    dog_s = np.zeros(n, np.float32)
    growl_s = np.zeros(n, np.float32)
    emb = np.zeros((n, 2048), np.float32)
    top1 = []
    t0 = time.perf_counter()
    gi = backend.mid_to_index["/m/0ghcn6"]
    for i, c in enumerate(clips):
        x, sr = sf.read(root / "audio" / c.filename, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        mfcc[i] = extract(x, sr, fcfg)
        ws = backend.score_track(AudioTrack("present", x, sr, 0.0), hop_s=settings.audio.yamnet.hop_s)
        gs = group_scores(ws, groups)
        cat_s[i], dog_s[i] = gs["cat_vocalization"].max(), gs["dog_vocalization"].max()
        growl_s[i] = ws.scores[:, gi].max()
        emb[i] = np.concatenate([ws.embeddings.mean(axis=0), ws.embeddings.max(axis=0)])
        top1.append(backend.labels[int(np.argmax(ws.scores.max(axis=0)))])
        if (i + 1) % 200 == 0:
            log.info("E1 inputs %d/%d (%.1fs)", i + 1, n, time.perf_counter() - t0)
    out = {"mfcc": mfcc, "cat_score": cat_s, "dog_score": dog_s, "growl_score": growl_s, "embedding": emb,
           "yamnet_top1": np.array(top1), "elapsed_s": np.array([time.perf_counter() - t0])}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def _decide_tags(cs: np.ndarray, ds: np.ndarray, tc: float, td: float) -> list[str]:
    out = []
    for c, d in zip(cs, ds):
        rc, rd = c / max(tc, 1e-6), d / max(td, 1e-6)
        if c >= tc and (d < td or rc >= rd):
            out.append("cat")
        elif d >= td:
            out.append("dog")
        else:
            out.append("other")
    return out


def _platt(train_s: np.ndarray, train_y: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    z = np.log(np.clip(train_s, 1e-4, 1 - 1e-4) / (1 - np.clip(train_s, 1e-4, 1 - 1e-4)))
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(z[:, None], train_y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def run(settings: Settings, out_dir: Path, n_boot: int = 1000, seed: int = 0) -> dict:
    data_root = settings.paths.resolve("data_root")
    root = esc50.dataset_dir(data_root)
    if not (root / "meta" / "esc50.csv").exists():
        raise FileNotFoundError("ESC-50 not found; run: uv run petperception fetch-data esc50")
    t_start = time.perf_counter()
    clips = sorted(esc50.load_meta(root), key=lambda c: c.filename)
    verify = esc50.verify_against_manifest(root, settings.paths.resolve("manifests_root") / "esc50_files.csv")
    if verify["mismatched_or_missing"]:
        raise RuntimeError(f"ESC-50 checksum mismatch: {verify['mismatched_or_missing'][:5]}")
    X = compute_inputs(settings, root, clips, settings.paths.resolve("data_root").parent / "cache")
    y = np.array([c.pet_label for c in clips])
    folds = np.array([c.fold for c in clips])
    src = np.array([c.source_group for c in clips])
    cats = np.array([c.category for c in clips])
    preds: dict[str, np.ndarray] = {m: np.empty(len(clips), dtype=object) for m in
                                    ["majority", "mfcc_logreg", "yamnet_tags_nested_threshold", "yamnet_embedding_logreg"]}
    fold_info = []
    platt_prob = {"cat": np.zeros(len(clips)), "dog": np.zeros(len(clips))}
    rng = np.random.default_rng(seed)
    strat_preds = np.empty((200, len(clips)), dtype=object)
    for k in sorted(set(folds.tolist())):
        tr, te = folds != k, folds == k
        info: dict = {"test_fold": int(k), "n_train": int(tr.sum()), "n_test": int(te.sum()),
                      "test_counts": {l: int(np.sum(y[te] == l)) for l in LABELS}}
        # Majority and stratified-random baselines use training-fold label frequencies only.
        vals, cnts = np.unique(y[tr], return_counts=True)
        preds["majority"][te] = vals[np.argmax(cnts)]
        pri = cnts / cnts.sum()
        for r in range(strat_preds.shape[0]):
            strat_preds[r, te] = rng.choice(vals, size=int(te.sum()), p=pri)
        # MFCC + logistic regression (C by inner leave-one-fold-out).
        m, inf = linear_model.fit(X["mfcc"][tr], y[tr], folds[tr].astype(str), LABELS, seed=seed)
        preds["mfcc_logreg"][te] = m.predict(X["mfcc"][te])
        info["mfcc_logreg"] = inf
        # YAMNet tags with thresholds chosen on training folds.
        tc, f1c = M.best_f1_threshold(X["cat_score"][tr], y[tr] == "cat")
        td, f1d = M.best_f1_threshold(X["dog_score"][tr], y[tr] == "dog")
        preds["yamnet_tags_nested_threshold"][te] = _decide_tags(X["cat_score"][te], X["dog_score"][te], tc, td)
        info["yamnet_thresholds"] = {"cat": tc, "dog": td, "train_f1_cat": f1c, "train_f1_dog": f1d}
        # Platt scaling fitted on training folds (calibration diagnostic only).
        for lab, key in (("cat", "cat_score"), ("dog", "dog_score")):
            a, b = _platt(X[key][tr], (y[tr] == lab).astype(int))
            s = np.clip(X[key][te], 1e-4, 1 - 1e-4)
            platt_prob[lab][te] = 1 / (1 + np.exp(-(a * np.log(s / (1 - s)) + b)))
        # Frozen YAMNet embeddings + logistic regression.
        me, infe = linear_model.fit(X["embedding"][tr], y[tr], folds[tr].astype(str), LABELS, seed=seed)
        preds["yamnet_embedding_logreg"][te] = me.predict(X["embedding"][te])
        info["yamnet_embedding_logreg"] = infe
        fold_info.append(info)
        log.info("fold %d done", k)

    results: dict = {"methods": {}}
    for name, p in preds.items():
        rep = M.classification_report(y.tolist(), p.tolist(), LABELS)
        rep["per_fold_macro_f1"] = {int(k): M.macro_f1(y[folds == k].tolist(), p[folds == k].tolist(), LABELS) for k in sorted(set(folds.tolist()))}
        rep["macro_f1_ci95"] = M.group_bootstrap(lambda idx: M.macro_f1(y[idx].tolist(), p[idx].tolist(), LABELS), src.tolist(), n_boot, seed)
        for lab in ("cat", "dog"):
            rep[f"f1_{lab}_ci95"] = M.group_bootstrap(lambda idx, lab=lab: M.per_class(y[idx].tolist(), p[idx].tolist(), [lab])[lab]["f1"], src.tolist(), n_boot, seed)
        fp_other = {}
        for lab in ("cat", "dog"):
            mask = (y == "other") & (p == lab)
            u, c = np.unique(cats[mask], return_counts=True)
            fp_other[lab] = {str(a): int(b) for a, b in sorted(zip(u, c), key=lambda t: -t[1])}
        rep["false_positive_non_pet_categories"] = fp_other
        results["methods"][name] = rep
    strat_mf1 = [M.macro_f1(y.tolist(), strat_preds[r].tolist(), LABELS) for r in range(strat_preds.shape[0])]
    strat_ba = [M.balanced_accuracy(y.tolist(), strat_preds[r].tolist(), LABELS) for r in range(strat_preds.shape[0])]
    strat_pc = [M.per_class(y.tolist(), strat_preds[r].tolist(), LABELS) for r in range(strat_preds.shape[0])]
    results["methods"]["stratified_random"] = {
        "repeats": int(strat_preds.shape[0]),
        "macro_f1_mean": float(np.mean(strat_mf1)), "macro_f1_sd": float(np.std(strat_mf1)),
        "balanced_accuracy_mean": float(np.mean(strat_ba)),
        "per_class_f1_mean": {l: float(np.mean([pc[l]["f1"] for pc in strat_pc])) for l in LABELS},
    }
    # Threshold-free ranking quality of the frozen tag scores (no fitting involved).
    results["yamnet_raw_score_ranking"] = {
        lab: {"average_precision": M.average_precision(X[f"{lab}_score"], y == lab), "positives": int(np.sum(y == lab))}
        for lab in ("cat", "dog")
    }
    results["calibration_diagnostic"] = {
        lab: {
            "brier_raw_score": M.brier(X[f"{lab}_score"], y == lab),
            "brier_platt_nested": M.brier(platt_prob[lab], y == lab),
            "positives": int(np.sum(y == lab)),
            "note": "ECE/reliability curves not reported: 40 positives per class are too few for stable binning; "
                    "calibration on ESC-50 does not transfer to household audio.",
        }
        for lab in ("cat", "dog")
    }
    growl_hits = {str(c): int(n) for c, n in zip(*np.unique(cats[X["growl_score"] >= 0.3], return_counts=True))}
    results["growl_tag_score_ge_0.3_by_category"] = growl_hits

    # Deployment thresholds: selected on all 5 folds; their expected performance is the nested estimate above.
    tc_all, f1c_all = M.best_f1_threshold(X["cat_score"], y == "cat")
    td_all, f1d_all = M.best_f1_threshold(X["dog_score"], y == "dog")
    thr = {
        "format": "petperception.audio_event_thresholds/1",
        "model_id": settings.audio.yamnet.model_id,
        "selection_method": "F1-optimal threshold on raw YAMNet group scores (max over 0.975 s windows) over all 5 ESC-50 folds; generalization estimate = nested E1 result",
        "thresholds": {
            "cat_vocalization": {"threshold": tc_all, "in_sample_f1": f1c_all},
            "dog_vocalization": {"threshold": td_all, "in_sample_f1": f1d_all},
        },
        "score_kind": "yamnet_sigmoid_raw (uncalibrated)",
        "data": "ESC-50 @ 33c8ce9e (CC BY-NC 3.0 collection)",
        "use_restrictions": ["Derived from CC BY-NC 3.0 data: non-commercial research use with attribution."],
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    write_json(settings.paths.resolve("artifacts_root") / "audio_event_thresholds.json", thr)
    m_all, inf_all = linear_model.fit(X["mfcc"], y, folds.astype(str), LABELS, seed=seed)
    m_all.metadata = {"task": "ESC-50 clip-level cat/dog/other (E1 CPU baseline)", "feature_config": FeatureConfig().to_dict(),
                      "feature_names": feature_names(FeatureConfig()), "training_data": "ESC-50 @ 33c8ce9e, all 5 folds",
                      "inner_cv": inf_all, "use_restrictions": thr["use_restrictions"]}
    m_all.to_json(settings.paths.resolve("artifacts_root") / "e1_mfcc_catdogother_logreg.json")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "fold", "source_group", "category", "true"] + list(preds) + ["yamnet_cat_score", "yamnet_dog_score", "yamnet_top1"])
        for i, c in enumerate(clips):
            w.writerow([c.filename, c.fold, c.source_group, c.category, y[i]] + [preds[m][i] for m in preds] +
                       [f"{X['cat_score'][i]:.4f}", f"{X['dog_score'][i]:.4f}", X["yamnet_top1"][i]])
    summary = {
        "experiment": "E1",
        "title": "Cat/dog vs other sound events on ESC-50 (clip level)",
        "dataset": {"id": "esc50", "revision": esc50.PINNED_REVISION, "clips": len(clips), "label_counts": {l: int(np.sum(y == l)) for l in LABELS},
                    "checksum_verification": {"checked": verify["checked"], "mismatched": len(verify["mismatched_or_missing"])}},
        "protocol": {
            "outer": "5 official ESC-50 folds (source-preserving by construction)",
            "inner": "leave-one-fold-out over the 4 training folds for C; thresholds F1-optimal on training folds",
            "bootstrap": f"{n_boot} resamples of Freesound source recordings",
            "decision_rule_tags": "cat if cat_score>=t_cat, dog if dog_score>=t_dog; both -> larger score/threshold ratio; else other",
            "limitations": ["clip-level labels only; no event-timing evaluation", "ESC-50 is not household audio",
                            "4 non-pet source recordings span two folds each (verified)", "YAMNet trained on AudioSet; overlap with Freesound-derived ESC-50 unaudited"],
        },
        "folds": fold_info,
        "results": results,
        "deployment_thresholds": thr["thresholds"],
        "runtime_s": {"inputs_compute_s_if_uncached": float(X["elapsed_s"][0]), "total_s": time.perf_counter() - t_start},
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    write_json(out_dir / "metrics.json", summary)
    return summary
