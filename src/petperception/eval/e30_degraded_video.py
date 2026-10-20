"""E30: should the intent stage stop interpreting the situation when the video is too dark or blurred?

Why: in E28 poor video (dark, noisy, blurred copies of the test clips)
dropped the combined situation model to 0.476 macro-F1, yet the policy kept
interpreting 84 % of clips (selective accuracy 0.54). The engine already
measures per frame the mean grey level and the variance of the Laplacian
(vision/quality.py; flags at luma < 40 and variance < 25, configuration
defaults never tuned on these data).

Rule under test (fixed before any E30 number was computed): the video is
*degraded* when at least half of the sampled frames are low-light or blurry
by those default thresholds. With degraded video the situation is not
reported as an interpretation, and the result names the degraded input.

Data: development = E28's 294 test clips, clean and poor copies (reported,
nothing tuned); test = E29's 300 fresh clips, clean and a poor copy made
with the same degradation (intent/robustness.py), scored once.

Adoption (all required, per variant):
1. degraded flag on poor fresh clips >= 0.80;
2. degraded flag on clean fresh clips <= 0.05 with Wilson upper bound <= 0.10;
3. selective accuracy of the policy on clean fresh clips not lower by more
   than 0.01 with the rule.
Caveat: the degradation is synthetic; real dark or blurred home video is
not measured.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from ..config import Settings, load_settings
from ..intent import kinetics as K
from ..utils import utc_now, write_json
from .e16_breed_robustness import wilson
from .e28_av_situation import _metrics, _pred, _with_branches
from .e29_conflict_fresh import fresh_clips

FRACTION = 0.5
RULE = {"poor_flagged_min": 0.80, "clean_flagged_max": 0.05, "clean_wilson_high_max": 0.10, "clean_selective_accuracy_drop_max": 0.01}


def quality(clip: K.Clip, poor: bool) -> np.ndarray:
    """Per sampled frame (luma, Laplacian variance), decoded as the feature extractor does (2 fps, max side 480)."""
    from ..intent import av_features as AF
    from ..intent import robustness as RB

    frames = AF.decode_frames(clip.path, K.FPS)
    if poor:
        f = RB.poor_visual_fn(clip.key)
        frames = [f(x, i) for i, x in enumerate(frames)]
    return np.array([AF.frame_luma_blur(x) for x in frames], np.float32).reshape(-1, 2)


def degraded(q: np.ndarray, low_light: float, blur: float) -> bool:
    if len(q) == 0:
        return False
    bad = (q[:, 0] < low_light) | (q[:, 1] < blur)
    return bool(bad.mean() >= FRACTION)


def _policy(m, rows: list[dict], flags: np.ndarray, y: np.ndarray) -> dict:
    dec = m.decide(rows)
    interp = dec["status"] == "interpretation"
    out = {}
    for name, keep in (("without_rule", interp), ("with_rule", interp & ~flags)):
        out[name] = {"coverage": float(keep.mean()), "selective_accuracy": float((dec["label"][keep] == y[keep]).mean()) if keep.any() else None}
    return out


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": k / n if n else None, "wilson95": [lo, hi]}


def run(settings: Settings, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    data_root = settings.paths.resolve("data_root")
    var_root = data_root.parent
    qc = settings.vision.quality
    with open(var_root / "intent" / "e28_models.pkl", "rb") as fh:
        models = pickle.load(fh)["models"]
    cache = var_root / "intent" / "e30_quality.npz"
    store = dict(np.load(cache, allow_pickle=False)) if cache.exists() else {}
    sets = {"dev": [c for c in K.list_clips(data_root) if c.split == "test"], "test": fresh_clips(data_root)}
    res: dict = {"thresholds": {"low_light_luma": qc.low_light_luma, "blur_laplacian_var": qc.blur_laplacian_var, "fraction": FRACTION}}
    for split, clips in sets.items():
        rows, miss = K.load_pooled(clips, var_root)
        poor_rows, pmiss = K.load_pooled(clips, var_root, "video_poor")
        if miss or pmiss:
            raise RuntimeError(f"{split}: {len(miss)} clean / {len(pmiss)} poor clips lack features")
        y = np.array([r["clip"].label for r in rows], dtype=object)
        flags = {}
        for variant in ("clean", "poor"):
            f = []
            for c in clips:
                k = f"{variant}__{c.key}"
                if k not in store:
                    store[k] = quality(c, variant == "poor")
                f.append(degraded(store[k], qc.low_light_luma, qc.blur_laplacian_var))
            flags[variant] = np.array(f)
        np.savez(cache, **store)
        rows_poor = _with_branches(rows, {r["clip"].key: r for r in poor_rows}, ("animal", "animal_phone", "environment", "interaction"))
        out: dict = {"clips": len(rows), "flagged_clean": _rate(int(flags["clean"].sum()), len(rows)),
                     "flagged_poor": _rate(int(flags["poor"].sum()), len(rows))}
        for name, m in models.items():
            classes = m.classes
            out[name] = {"clean": {"policy": _policy(m, rows, flags["clean"], y), "macro_f1": _metrics(y, _pred(m.late_proba(rows), classes), classes,
                                                                                                        n_boot=200)["macro_f1"]},
                         "poor": {"policy": _policy(m, rows_poor, flags["poor"], y),
                                  "macro_f1": _metrics(y, _pred(m.late_proba(rows_poor), classes), classes, n_boot=200)["macro_f1"]}}
            if split == "test":
                pc = out[name]["clean"]["policy"]
                drop = (pc["without_rule"]["selective_accuracy"] or 0) - (pc["with_rule"]["selective_accuracy"] or 0)
                checks = {"1_poor_flagged": bool(out["flagged_poor"]["rate"] >= RULE["poor_flagged_min"]),
                          "2_clean_flagged": bool(out["flagged_clean"]["rate"] <= RULE["clean_flagged_max"]
                                                  and out["flagged_clean"]["wilson95"][1] <= RULE["clean_wilson_high_max"]),
                          "3_clean_selective_accuracy": bool(drop <= RULE["clean_selective_accuracy_drop_max"])}
                out[name].update({"clean_selective_accuracy_drop": drop, "adoption_checks": checks, "adopt": all(checks.values())})
        res[split] = out
    o = {"experiment": "E30", "status": "completed", "created_at": utc_now().isoformat(timespec="seconds"), "protocol": __doc__.strip().splitlines(),
         "rule": RULE, "results": res, "runtime_s": round(time.perf_counter() - t0, 1),
         "terms": "Kinetics clips: research only; degradation synthetic"}
    write_json(out_dir / "metrics.json", o)
    return o


def extract_fresh_poor(threads: int = 2) -> dict:
    """Video-only features of the poor copies of E29's fresh clips (resumable)."""
    from ..intent import av_features as AF
    from ..intent import robustness as RB

    s = load_settings()
    data_root = s.paths.resolve("data_root")
    clips = fresh_clips(data_root)
    out = K.feature_dir(data_root.parent, "video_poor")
    out.mkdir(parents=True, exist_ok=True)
    comp = AF.load_components(threads, with_audio=False)
    n = 0
    for c in clips:
        if (out / f"{c.key}.npz").exists():
            continue
        f = AF.extract_clip(comp, c.path, audio=False, frame_fn=RB.poor_visual_fn(c.key))
        np.savez(out / f"{c.key}.tmp.npz", **f)
        (out / f"{c.key}.tmp.npz").replace(out / f"{c.key}.npz")
        n += 1
    return {"extracted": n, "clips": len(clips)}


def apply_adoption(metrics_path: Path, art_root: Path) -> dict:
    import json

    m = json.loads(metrics_path.read_text())
    out = {}
    for variant in ("full", "phone_now"):
        r = m["results"]["test"][variant]
        p = art_root / "kinetics_situation_v1" / variant / "policy.json"
        pol = json.loads(p.read_text())
        if r["adopt"]:
            pol["degraded_video_rule"] = {**m["results"]["thresholds"], "source": "E30 (adopted under the pre-registered rule)"}
        else:
            pol.pop("degraded_video_rule", None)
        write_json(p, pol)
        out[variant] = r["adopt"]
    return out
