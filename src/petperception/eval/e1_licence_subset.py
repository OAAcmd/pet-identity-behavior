"""E1-CC: audio-event thresholds refitted on CC0 / CC BY clips only.

Why: the deployed thresholds (artifacts/audio_event_thresholds.json) are
fitted on all 2,000 ESC-50 clips, a CC BY-NC 3.0 collection, so the
commercial profile cannot use them. 1,770 of the clips carry a Freesound
licence of CC0 or CC BY on their own (manifests/esc50_files.csv). This job
refits the two thresholds on that subset and measures what changes.

Protocol (same as E1, same cached YAMNet scores):
* deployment candidate: F1-optimal thresholds on the CC0 + CC BY clips of
  all 5 folds;
* generalization: for each official fold, thresholds from the CC0 + CC BY
  clips of the 4 other folds versus thresholds from all clips of those
  folds (the current method), both scored on every clip of the held-out
  fold and on its CC0 + CC BY clips only; precision / recall / F1 pooled
  over the 5 held-out folds.

What this does not settle: the class labels used for fitting are ESC-50's
own (CC BY-NC 3.0 collection). docs/ASSET_TERMS.md (A1) states that question
and the label-independent route (re-labelling the subset's cat/dog clips).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..config import Settings
from ..data import esc50
from ..utils import utc_now, write_json
from . import metrics as M
from .e1_audio_events import compute_inputs

OPEN = ("CC0", "CC-BY")


def _prf(scores: np.ndarray, pos: np.ndarray, t: float) -> dict:
    pred = scores >= t
    tp, fp, fn = int(np.sum(pred & pos)), int(np.sum(pred & ~pos)), int(np.sum(~pred & pos))
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def run(settings: Settings, out_dir: Path) -> dict:
    root = esc50.dataset_dir(settings.paths.resolve("data_root"))
    clips = sorted(esc50.load_meta(root), key=lambda c: c.filename)
    lic = {r["filename"]: r["source_license"] for r in csv.DictReader(open(settings.paths.resolve("manifests_root") / "esc50_files.csv", newline=""))}
    X = compute_inputs(settings, root, clips, settings.paths.resolve("data_root").parent / "cache")
    y = np.array([c.pet_label for c in clips])
    folds = np.array([c.fold for c in clips])
    open_ = np.array([lic[c.filename] in OPEN for c in clips])
    groups = {"cat_vocalization": ("cat_score", "cat"), "dog_vocalization": ("dog_score", "dog")}
    out: dict = {"experiment": "E1-CC", "title": "Audio-event thresholds refitted on CC0 / CC BY ESC-50 clips",
                 "protocol": __doc__.strip().splitlines(),
                 "clips": {"all": len(clips), "cc0_or_cc_by": int(open_.sum()),
                           "licence_counts": {k: int(sum(lic[c.filename] == k for c in clips)) for k in sorted(set(lic.values()))},
                           "positives_all": {g: int(np.sum(y == lab)) for g, (_, lab) in groups.items()},
                           "positives_cc0_or_cc_by": {g: int(np.sum((y == lab) & open_)) for g, (_, lab) in groups.items()}},
                 "groups": {}}
    deployed = {}
    for g, (key, lab) in groups.items():
        s, pos = X[key], y == lab
        t_all, f_all = M.best_f1_threshold(s, pos)
        t_cc, f_cc = M.best_f1_threshold(s[open_], pos[open_])
        deployed[g] = {"threshold": t_cc, "in_sample_f1": f_cc}
        nested = {"all_clip_thresholds": [], "cc_thresholds": []}
        pooled = {m: {"held_out_all": [], "held_out_cc": []} for m in nested}
        for f in sorted(set(folds)):
            tr, te = folds != f, folds == f
            ta, _ = M.best_f1_threshold(s[tr], pos[tr])
            tc, _ = M.best_f1_threshold(s[tr & open_], pos[tr & open_])
            nested["all_clip_thresholds"].append(ta)
            nested["cc_thresholds"].append(tc)
            for m, t in (("all_clip_thresholds", ta), ("cc_thresholds", tc)):
                pooled[m]["held_out_all"].append(s[te] >= t)
                pooled[m]["held_out_cc"].append(s[te & open_] >= t)
        res = {}
        for m in nested:
            res[m] = {"per_fold_threshold": nested[m]}
            for scope, sel in (("held_out_all", np.ones_like(open_)), ("held_out_cc", open_)):
                pred = np.zeros(len(clips), dtype=bool)
                for f, p in zip(sorted(set(folds)), pooled[m][scope]):
                    pred[(folds == f) & sel] = p
                res[m][scope] = _prf(pred[sel].astype(float), pos[sel], 0.5)
        out["groups"][g] = {"threshold_all_clips": t_all, "in_sample_f1_all_clips": f_all,
                            "threshold_cc0_or_cc_by": t_cc, "in_sample_f1_cc0_or_cc_by": f_cc, "nested": res}
    art = {"format": "petperception.audio_event_thresholds/1", "model_id": settings.audio.yamnet.model_id,
           "selection_method": "F1-optimal threshold on raw YAMNet group scores over the ESC-50 clips whose Freesound licence is CC0 or CC BY (all 5 folds); generalization estimate = results/e1/licence_subset.json",
           "thresholds": deployed, "score_kind": "yamnet_sigmoid_raw (uncalibrated)",
           "data": "ESC-50 @ 33c8ce9e, CC0 and CC BY clips only (manifests/esc50_files.csv lists author and URL of each)",
           "use_restrictions": ["Audio: CC0 or CC BY 3.0/4.0 per clip (attribution list: manifests/esc50_files.csv).",
                                "Labels: ESC-50 class labels (collection CC BY-NC 3.0); see docs/ASSET_TERMS.md A1."],
           "created_at": utc_now().isoformat(timespec="seconds")}
    write_json(settings.paths.resolve("artifacts_root") / "audio_event_thresholds_cc.json", art)
    out["artifact"] = "artifacts/audio_event_thresholds_cc.json"
    out["created_at"] = art["created_at"]
    write_json(out_dir / "licence_subset.json", out)
    return out
