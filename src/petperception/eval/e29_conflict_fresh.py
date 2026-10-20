"""E29: a pre-registered audio-video conflict detector, and a replication of E28, on fresh Kinetics clips.

Why: in E28 the pre-registered conflict flag never fired (the audio head
never passes its own threshold) and swapped audio went unnoticed. An
exploratory follow-up found a continuous signal (Jensen-Shannon divergence
between the audio-only and the visual-only situation distributions, ROC AUC
0.69-0.71 on E28's test clips). Adopting it needs a fresh test, fixed here
before those clips were fetched.

Data (fresh): Kinetics-700 *train*-split clips 161-210 of each of the six
classes (300 clips), fetched once for this experiment
(`kinetics_pets_train_conflict_sha256.csv`). Clips 1-160 (tuning, holdout,
and the sealed E23 set) are streamed past, never extracted. No YouTube id
may be shared with any other split (checked).

Frozen models: the E28 heads and late fusion (both variants), unchanged.

Detector: score = JS divergence between the audio head's distribution and
the late-fusion distribution with audio absent (visual only). Threshold per
variant = the 90th percentile of the genuine-pair scores on E28's 294 test
clips (development data for E29; E28's own result is already reported).
Conflict = audio and video both present and score > threshold.

Test (once): genuine fresh clips; swapped pairs built inside the fresh set
(audio of a random clip with a different label; same species and cross
species; seed 0).

Adoption into the product policy, all three required:
1. same-species swaps flagged >= 0.30 with Wilson lower bound >= 0.20;
2. genuine clips flagged <= 0.15 with Wilson upper bound <= 0.20;
3. the policy's selective accuracy on genuine fresh clips drops by at most
   0.02 (point) when flagged clips lose their interpretation.

Also reported: E28's models on the fresh clips (macro-F1 combined, visual
only, audio only; bootstrap CIs over clips) as an independent replication.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT, Settings
from ..data import kinetics_pets as KP
from ..intent import kinetics as K
from ..utils import utc_now, write_json
from .e16_breed_robustness import wilson
from .e28_av_situation import VISUAL, _metrics, _pred, swap_audio
from .e28_conflict_diagnostic import _auc, _scores

SUBDIR, SKIP, PER_CLASS = "train_conflict", 160, 50
PINS = "kinetics_pets_train_conflict_sha256.csv"
PERCENTILE = 90.0
RULE = {"same_species_flagged_min": 0.30, "same_species_wilson_low_min": 0.20, "genuine_flagged_max": 0.15, "genuine_wilson_high_max": 0.20,
        "selective_accuracy_drop_max": 0.02}


def fresh_clips(data_root: Path) -> list[K.Clip]:
    root = KP.root(data_root) / SUBDIR
    out = []
    for label, (_, species, contact) in KP.CLASSES.items():
        for p in sorted((root / label).glob("*.mp4")):
            out.append(K.Clip(f"{SUBDIR}__{label.replace(' ', '_')}__{p.stem}", p, "fresh", SUBDIR, label, species, contact, K.youtube_id(p.stem)))
    import csv

    others = {c.youtube_id for c in K.list_clips(data_root)}
    sealed = REPO_ROOT / "manifests" / KP.FINAL_PINS
    if sealed.exists():  # the sealed E23 set is checked by file name only (never opened)
        others |= {K.youtube_id(Path(r["file"]).stem) for r in csv.DictReader(open(sealed, newline=""))}
    shared = sorted({c.youtube_id for c in out} & others)
    if shared:
        raise RuntimeError(f"fresh clips share YouTube ids with other splits: {shared[:5]}")
    return out


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": k / n if n else None, "wilson95": [lo, hi]}


def _policy(m, rows: list[dict], flagged: np.ndarray, y: np.ndarray) -> dict:
    dec = m.decide(rows)
    interp = dec["status"] == "interpretation"
    out = {}
    for name, keep in (("without_detector", interp), ("with_detector", interp & ~flagged)):
        out[name] = {"coverage": float(keep.mean()), "selective_accuracy": float((dec["label"][keep] == y[keep]).mean()) if keep.any() else None}
    return out


def run(settings: Settings, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    data_root = settings.paths.resolve("data_root")
    var_root = data_root.parent
    with open(var_root / "intent" / "e28_models.pkl", "rb") as fh:
        models = pickle.load(fh)["models"]
    test28 = [r for r in K.load_pooled(K.list_clips(data_root), var_root)[0] if r["clip"].split == "test"]
    fresh, missing = K.load_pooled(fresh_clips(data_root), var_root)
    if missing or not fresh:
        raise RuntimeError(f"{len(missing)} fresh clips lack features; run the E29 fetch and feature steps")
    y = np.array([r["clip"].label for r in fresh], dtype=object)
    classes = sorted(KP.CLASSES)
    res: dict = {"fresh_clips": len(fresh), "label_counts": {c: int((y == c).sum()) for c in classes},
                 "missing_branches": {b: int(sum(r["branches"][b] is None for r in fresh)) for b in K.BRANCHES}}
    for name, m in models.items():
        out: dict = {}
        # replication of E28 on fresh clips
        out["replication"] = {cond: _metrics(y, _pred(p, classes), classes) for cond, p in (
            ("combined", m.late_proba(fresh)), ("visual_only", m.late_proba(fresh, (m.branches[0],))), ("audio_only", m.late_proba(fresh, VISUAL)))}
        # detector
        dev = _scores(m, test28)
        thr = float(np.nanpercentile(dev, PERCENTILE))
        g = _scores(m, fresh)
        ok_g = ~np.isnan(g)
        flag_g = ok_g & (g > thr)
        out["threshold"] = thr
        out["genuine"] = _rate(int(flag_g.sum()), int(ok_g.sum()))
        for same in (True, False):
            sw, src = swap_audio(fresh, same)
            s = _scores(m, sw)
            ok = ~np.isnan(s)
            flag = ok & (s > thr)
            key = f"swapped_{'same' if same else 'cross'}_species"
            out[key] = {**_rate(int(flag.sum()), int(ok.sum())), "roc_auc_vs_genuine": _auc(s[ok & ok_g], g[ok & ok_g]),
                        "policy": _policy(m, sw, flag, y)}
        out["genuine_policy"] = _policy(m, fresh, flag_g, y)
        gp = out["genuine_policy"]
        drop = (gp["without_detector"]["selective_accuracy"] or 0) - (gp["with_detector"]["selective_accuracy"] or 0)
        ss = out["swapped_same_species"]
        checks = {"1_same_species_flagged": bool(ss["rate"] >= RULE["same_species_flagged_min"] and ss["wilson95"][0] >= RULE["same_species_wilson_low_min"]),
                  "2_genuine_flagged": bool(out["genuine"]["rate"] <= RULE["genuine_flagged_max"] and out["genuine"]["wilson95"][1] <= RULE["genuine_wilson_high_max"]),
                  "3_selective_accuracy_drop": bool(drop <= RULE["selective_accuracy_drop_max"])}
        out["selective_accuracy_drop_on_genuine"] = drop
        out["adoption_checks"] = checks
        out["adopt"] = all(checks.values())
        res[name] = out
    o = {"experiment": "E29", "status": "completed", "created_at": utc_now().isoformat(timespec="seconds"), "protocol": __doc__.strip().splitlines(),
         "rule": RULE, "percentile": PERCENTILE, "results": res, "runtime_s": round(time.perf_counter() - t0, 1),
         "terms": "Kinetics annotations CC BY 4.0; clips are YouTube uploads (research only); no clip committed"}
    write_json(out_dir / "metrics.json", o)
    return o


def fetch_and_extract(settings: Settings, threads: int = 2) -> dict:
    """Fetch the fresh clips (pinned by SHA-256 on first retrieval) and extract their features (resumable)."""
    from ..intent import av_features as AF

    data_root = settings.paths.resolve("data_root")
    got = KP.fetch_train_sample(data_root, settings.paths.resolve("manifests_root"), per_class=PER_CLASS, skip=SKIP, subdir=SUBDIR, pins_name=PINS)
    clips = fresh_clips(data_root)
    comp = AF.load_components(threads)
    AF.extract_many(comp, [(c.key, c.path) for c in clips], K.feature_dir(data_root.parent))
    return {"fetch": got, "clips": len(clips)}


def apply_adoption(metrics_path: Path, art_root: Path) -> dict:
    """Write the adopted detector threshold into each adopting variant's policy.json (E28 artifacts)."""
    import json

    m = json.loads(metrics_path.read_text())
    out = {}
    for variant in ("full", "phone_now"):
        r = m["results"][variant]
        p = art_root / "kinetics_situation_v1" / variant / "policy.json"
        pol = json.loads(p.read_text())
        if r["adopt"]:
            pol["conflict_js_threshold"] = r["threshold"]
            pol["conflict_detector"] = ("E29: Jensen-Shannon divergence between the audio-only and visual-only situation distributions; "
                                        "threshold = 90th percentile of genuine pairs on E28's test clips; adopted under the pre-registered rule")
        else:
            pol.pop("conflict_js_threshold", None)
            pol["conflict_detector"] = "E29: not adopted for this variant (pre-registered rule not met)"
        write_json(p, pol)
        out[variant] = r["adopt"]
    return out
