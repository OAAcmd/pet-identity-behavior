"""G1b acceptance criterion 2 (E18, real pet video) in the Python engine at the detector's per-class operating points.

Pre-registered (committed before the G1b engine variant was run on any E18 clip); the criterion itself was fixed in
docs/TRAINING_PROGRAM.md before G1 training ("G1a replaces the phone detector only if ... 2. E18 real video, paired
with M3: more clips with the right species for both species, with the lower 95 % bound of the paired difference
above 0, and no increase in only-wrong-species clips"):

* Engine: the Python reference engine as in E18 (research profile, audio and video, pose off) with
  ``vision.detector = pet_detector_oi_g1b_v1`` (the selected G1b export), ``vision.class_score_thresholds`` = its
  tuning-set points (cat 0.24, dog 0.30; ``results/g1/g1b_e7_perclass/operating_point.json``) and its own verifier
  (``artifacts/verifier/verifier_pet_detector_oi_g1b_v1.json``, fitted in acceptance 1). Everything else unchanged.
* Comparator: E18's ``phone_build`` variant (M3: TF Hub EfficientDet-Lite0 with raw decoding and its verifier),
  re-run with the same current engine code on the same 294 clips (E18's own per-clip cache no longer exists; its
  published rates were cat 59/99, dog 87/195).
* Per species: right-species clips (a track of the clip's species), only-wrong-species clips (tracks, none of the
  clip's species); paired difference G1b - M3 in right-species rate with a percentile bootstrap over clips
  (10,000 resamples, seed 0).
* Criterion 2 is met if, for both species, the right-species difference is positive with its lower 95 % bound
  above 0, and the number of only-wrong-species clips is not higher than M3's.
* Criteria 3 (E16 lowest-breed found rate), 4 (engine parity: needs per-class points in the phone engine) and 5
  (phone latency) are not run here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT, Settings
from ..data import kinetics_pets as KP
from ..utils import utc_now, write_json
from . import e18_kinetics_pet_video as E18

POINTS = json.loads((REPO_ROOT / "results" / "g1" / "g1b_e7_perclass" / "operating_point.json").read_text())["thresholds"]
VARIANT = {"vision": {"detector": "pet_detector_oi_g1b_v1", "class_score_thresholds": POINTS}}
N_BOOT, SEED = 10_000, 0


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    d = a.astype(float) - b.astype(float)
    boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(N_BOOT)]
    return {"g1b": int(a.sum()), "m3": int(b.sum()), "n": int(len(a)), "difference": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "g1b_only": int(np.sum(a & ~b)), "m3_only": int(np.sum(b & ~a))}


def run(settings: Settings, out_dir: Path) -> dict:
    data = settings.paths.resolve("data_root")
    clips = KP.clips(data)
    cache = data.parent / "cache" / "e18"
    g1b = E18.analyse(settings, clips, cache / "results_g1b.jsonl", VARIANT)
    m3 = E18.analyse(settings, clips, cache / "results_phone_build_current.jsonl", E18.VARIANTS["phone_build"])
    by_m3 = {r["clip"]: r for r in m3}
    res = {"criterion": "TRAINING_PROGRAM.md G1 acceptance 2 (E18, paired with M3)", "created_at": utc_now().isoformat(timespec="seconds"),
           "g1b_variant": VARIANT, "m3_variant": E18.VARIANTS["phone_build"], "clips": len(clips), "by_species": {},
           "g1b_summary": E18.summarize(g1b)["by_species"]}
    met = True
    for sp in ("cat", "dog"):
        other = "dog" if sp == "cat" else "cat"
        rows = [(r, by_m3[r["clip"]]) for r in g1b if r["species"] == sp and "error" not in r and "error" not in by_m3[r["clip"]]]
        right = lambda r: sp in r["track_species"]  # noqa: E731
        wrong = lambda r: bool(r["track_species"]) and sp not in r["track_species"] and other in r["track_species"]  # noqa: E731
        p = paired(np.array([right(a) for a, _ in rows]), np.array([right(b) for _, b in rows]))
        w = {"g1b": int(sum(wrong(a) for a, _ in rows)), "m3": int(sum(wrong(b) for _, b in rows))}
        ok = p["difference"] > 0 and p["ci95"][0] > 0 and w["g1b"] <= w["m3"]
        res["by_species"][sp] = {"right_species": p, "only_wrong_species": w, "met": bool(ok)}
        met = met and ok
    res["criterion_2_met"] = bool(met)
    write_json(out_dir / "acceptance2_e18.json", res)
    return res


def main() -> None:
    import argparse

    from ..config import load_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    s = load_settings()
    if a.threads:
        s.runtime.num_threads = a.threads
    r = run(s, s.paths.resolve("results_root") / "g1")
    print(json.dumps({k: v for k, v in r.items() if k != "g1b_summary"}, indent=1))


if __name__ == "__main__":
    main()
