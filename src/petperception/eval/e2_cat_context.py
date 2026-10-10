"""E2: CatMeows recording-context classification with animal-grouped evaluation.

Labels: the dataset's original recording contexts (brushing, waiting_for_food,
isolation). They describe the stimulus, not verified emotions or needs.

Protocol
* Outer: leave-one-cat-out (LOCO). A stricter leave-one-owner-out (LOOO)
  variant is also reported because one owner may contribute several cats.
* Inner: grouped CV by cat on the training cats only, selecting ``C``.
  Standardization is fitted inside each training partition.
* Folds whose held-out cat lacks a context are reported (per-fold macro-F1
  is computed only over classes present); pooled metrics use all predictions.
* Bootstrap 95% CIs resample cats.
* A random stratified split is run ONLY as a leakage diagnostic.
* The 8 kHz recordings are analysed at 8 kHz (fmax 4 kHz); the optional
  YAMNet-embedding comparison upsamples to 16 kHz and is flagged as
  bandwidth-limited / out-of-domain.

In this workspace the dataset is not reachable (zenodo.org denied), so
``run`` writes a ``blocked`` status with the continuation command instead of
results. ``run_on_recordings`` is exercised by synthetic unit tests only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.model_selection import StratifiedKFold

from ..audio import linear_model
from ..audio.features import FeatureConfig, extract, feature_names
from ..config import Settings
from ..data import catmeows as CM
from ..utils import utc_now, write_json
from . import metrics as M

LABELS = ["brushing", "isolation", "waiting_for_food"]
FEATURES_8K = FeatureConfig(sample_rate=8000, n_fft=256, hop_length=80, n_mels=32, n_mfcc=20, fmin=50.0, fmax=3900.0, version="mfcc-stats-8k-v1")


def _grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, list[dict]]:
    pred = np.empty(len(y), dtype=object)
    folds = []
    for g in sorted(set(groups.tolist())):
        te = groups == g
        tr = ~te
        missing_train = sorted(set(LABELS) - set(y[tr].tolist()))
        if missing_train:
            folds.append({"held_out": g, "status": "excluded", "reason": f"training data lacks {missing_train}"})
            pred[te] = "excluded"
            continue
        model, info = linear_model.fit(X[tr], y[tr], groups[tr], LABELS, seed=seed, inner_splits=5)
        pred[te] = model.predict(X[te])
        present = sorted(set(y[te].tolist()))
        folds.append({"held_out": g, "status": "ok", "n_test": int(te.sum()), "classes_present": present,
                      "missing_classes_in_test": sorted(set(LABELS) - set(present)),
                      "macro_f1_present_classes": M.macro_f1(y[te].tolist(), pred[te].tolist(), present), "selected_C": info["selected_C"]})
    return pred, folds


def _yamnet_embeddings(recs: list[CM.CatMeowsRecording], backend) -> np.ndarray:
    """Mean and max of YAMNet frame embeddings; the backend resamples 8 kHz to 16 kHz (no content above 4 kHz)."""
    from ..ingest.media import AudioTrack

    out = []
    for r in recs:
        x, sr = _load(r.path)
        ws = backend.score_track(AudioTrack("present", x, sr, 0.0), hop_s=0.48)
        out.append(np.concatenate([ws.embeddings.mean(axis=0), ws.embeddings.max(axis=0)]))
    return np.stack(out)


def run_on_recordings(recs: list[CM.CatMeowsRecording], out_dir: Path, seed: int = 0, n_boot: int = 1000, yamnet=None,
                      extra: dict | None = None) -> dict:
    X = np.stack([extract(*_load(r.path), FEATURES_8K) for r in recs])
    y = np.array([r.context for r in recs])
    cats = np.array([r.cat_id for r in recs])
    owners = np.array([r.owner_id for r in recs])
    res: dict = {"methods": {}}
    variants = [("loco_mfcc_logreg", X, cats), ("looo_mfcc_logreg", X, owners)]
    if yamnet is not None:
        # The phone-deployable variant (the app runs YAMNet); 8 kHz input upsampled, so out of YAMNet's domain.
        E = _yamnet_embeddings(recs, yamnet)
        variants += [("loco_yamnet_embedding_logreg", E, cats), ("looo_yamnet_embedding_logreg", E, owners)]
    for name, F, groups in variants:
        pred, folds = _grouped_cv(F, y, groups, seed)
        keep = pred != "excluded"
        rep = M.classification_report(y[keep].tolist(), pred[keep].tolist(), LABELS)
        rep["folds"] = folds
        rep["excluded_predictions"] = int((~keep).sum())
        rep["macro_f1_ci95_by_cat"] = M.group_bootstrap(lambda ix: M.macro_f1(y[keep][ix].tolist(), pred[keep][ix].tolist(), LABELS), cats[keep].tolist(), n_boot, seed)
        res["methods"][name] = rep
    vals, cnts = np.unique(y, return_counts=True)
    res["methods"]["majority"] = M.classification_report(y.tolist(), [vals[np.argmax(cnts)]] * len(y), LABELS)
    rng = np.random.default_rng(seed)
    strat = [M.macro_f1(y.tolist(), rng.choice(vals, size=len(y), p=cnts / cnts.sum()).tolist(), LABELS) for _ in range(200)]
    res["methods"]["stratified_random"] = {"macro_f1_mean": float(np.mean(strat)), "macro_f1_sd": float(np.std(strat))}
    # Leakage diagnostic only: windows/recordings of the same cat appear in train and test.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    leak = np.empty(len(y), dtype=object)
    for tr, te in skf.split(X, y):
        m, _ = linear_model.fit(X[tr], y[tr], cats[tr], LABELS, seed=seed, inner_splits=5)
        leak[te] = m.predict(X[te])
    res["leakage_diagnostic_random_split"] = {"macro_f1": M.macro_f1(y.tolist(), leak.tolist(), LABELS),
                                              "note": "NOT a valid estimate: the same cats appear in training and test"}
    res["dataset_observed"] = {
        "recordings": len(recs), "cats": len(set(cats.tolist())), "owners": len(set(owners.tolist())),
        "label_counts": {l: int(np.sum(y == l)) for l in LABELS},
        "sample_rates_hz": {str(k): int(v) for k, v in zip(*np.unique([r.sample_rate_hz for r in recs], return_counts=True))},
        "owner_to_cats": CM.owner_cat_relationships(recs),
    }
    res["feature_config"] = FEATURES_8K.to_dict()
    res.update(extra or {})
    write_json(out_dir / "metrics.json", {"experiment": "E2", "status": "completed", "results": res, "created_at": utc_now().isoformat(timespec="seconds")})
    model, info = linear_model.fit(X, y, cats, LABELS, seed=seed, inner_splits=5)
    model.metadata = {"task": "CatMeows recording context (original labels)", "feature_config": FEATURES_8K.to_dict(),
                      "feature_names": feature_names(FEATURES_8K), "inner_cv": info,
                      "use_restrictions": ["CatMeows: scientific research / non-commercial use; cite Ludovico et al. 2021."],
                      "label_semantics": "recording context; 'isolation' is not a fear label"}
    return {"results": res, "model": model}


def _load(path: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def run(settings: Settings, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    root = CM.dataset_dir(settings.paths.resolve("data_root"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (root / "dataset").exists():
        status = {
            "experiment": "E2", "status": "blocked",
            "reason": "CatMeows is not present: zenodo.org (and the huggingface.co mirror) are denied by this workspace's egress policy.",
            "evidence": "manifests/resources.json -> catmeows.errors",
            "next_command": "uv run petperception fetch-data catmeows && uv run petperception evaluate e2",
            "implemented": ["filename parser", "Zenodo MD5-verified fetch", "LOCO + leave-one-owner-out protocol",
                            "inner grouped C selection", "cat-level bootstrap", "random-split leakage diagnostic"],
            "tested_with": "synthetic fixtures only (tests/test_e2_protocol.py); no CatMeows metric is claimed",
            "created_at": utc_now().isoformat(timespec="seconds"),
        }
        write_json(out_dir / "status.json", status)
        return status
    recs, rejected = CM.scan(root, settings.paths.resolve("manifests_root") / "catmeows_files.csv")
    from ..audio.yamnet import YamnetBackend
    from ..registry import check_model, model_dir

    yamnet = YamnetBackend(model_dir(settings, settings.audio.yamnet.model_id), settings.runtime.num_threads, with_embeddings=True) \
        if check_model(settings, settings.audio.yamnet.model_id).ready else None
    fs = CM.dataset_dir(settings.paths.resolve("data_root")) / "fetch_status.json"
    out = run_on_recordings(recs, out_dir, yamnet=yamnet, extra={"data_source": json.loads(fs.read_text()) if fs.exists() else None})
    out["model"].to_json(settings.paths.resolve("artifacts_root") / "e2_catmeows_context_logreg.json")
    write_json(out_dir / "status.json", {"experiment": "E2", "status": "completed", "rejected_filenames": rejected,
                                         "runtime_s": time.perf_counter() - t0})
    return out["results"]
