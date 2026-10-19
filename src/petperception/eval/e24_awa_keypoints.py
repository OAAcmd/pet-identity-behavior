"""E24: cat and dog keypoints (ears, tail, limbs) on AwA-Pose, the first cat keypoint model.

Data (`petperception.data.awa_pose`): AwA-Pose annotations (MIT) on AwA2
images. 619 cat instances (persian, siamese; 257 photographers) and 1,310 dog
instances (chihuahua, collie, dalmatian, German shepherd; 533 photographers).
The points are mapped to the 24-point layout the product uses for dogs.

Protocol, fixed before training:
* Splits per species, grouped by photographer: one photographer often shot
  the same animal many times (one contributed 250 dog instances). Test: 20 %
  of photographers (seed 20261019). Development: 10 % of the rest. Fit: the
  remainder. The test split is scored once per model.
* Models (E9's architecture, hyperparameters and training code, unchanged:
  frozen EfficientNet-Lite0 features of the box crop, heatmap + offset head,
  30 epochs, the epoch chosen by development PCK@0.1, visibility gates chosen
  on development at >= 90 % precision):
  A. cat head: AwA cats only;
  B. joint head: AwA cats and dogs;
  C. reference: the E9 dog head (dog-pose data, research only) applied
     unchanged. For cats this is zero-shot; for AwA dogs it is cross-dataset.
* Metrics (E9's): PCK@{0.05, 0.1, 0.2} of the box's longer side, per
  keypoint and per region (ears, tail, legs, head). Ear-tip and tail
  elevation sign agreement with the annotation, using the product's
  feature definitions. 95 % CIs resample photographers. Also per class
  (persian / siamese, the four dog breeds).
* G2 shipping criteria (docs/TRAINING_PROGRAM.md, set before this run):
  ears only if ear PCK@0.1 >= 0.85; tail only if tail PCK@0.1 >= 0.75. The
  ear-position kappa criterion needs categorical labels this data does not
  have, so it is reported as not evaluable.

Terms: annotations MIT; images carry per-image Flickr / Wikimedia licences,
mostly "Creative Commons" without the variant stated (1,803 of 1,929
instances). Until the variants are verified, everything trained here is
research only.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ..config import REPO_ROOT, Settings
from ..data import awa_pose
from ..registry import check_model
from ..utils import sha256_file, utc_now, write_json
from ..vision import keypoints as KP
from . import e9_dog_keypoints as E9

SEED = 20261019
TEST_FRACTION, DEV_FRACTION = 0.20, 0.10
REGIONS = {"ears": ("left_ear_base", "right_ear_base", "left_ear_tip", "right_ear_tip"), "tail": ("tail_start", "tail_end"),
           "legs": tuple(k for k in KP.DOG_KEYPOINTS if any(s in k for s in ("paw", "knee", "elbow"))),
           "head": ("nose", "chin", "left_eye", "right_eye")}
G2_CRITERIA = {"ears": 0.85, "tail": 0.75}


def split_by_photographer(recs: list[dict], seed: int = SEED) -> dict[str, list[dict]]:
    ph = sorted({r["photographer"] for r in recs})
    random.Random(seed).shuffle(ph)
    n_test = int(round(len(ph) * TEST_FRACTION))
    test = set(ph[:n_test])
    rest = ph[n_test:]
    dev = set(rest[: int(round(len(rest) * DEV_FRACTION))])
    return {"fit": [r for r in recs if r["photographer"] not in test | dev], "dev": [r for r in recs if r["photographer"] in dev],
            "test": [r for r in recs if r["photographer"] in test]}


def records(data_root: Path) -> list[dict]:
    rows = awa_pose.fetch(data_root)
    out = []
    for row in rows:
        im = cv2.imread(row["image"])
        if im is None:
            continue
        rec = awa_pose.to_record(row, im.shape[:2])
        if rec["box"][2] - rec["box"][0] > 0.01 and rec["box"][3] - rec["box"][1] > 0.01:
            out.append(rec)
    return out


def _with_group(recs: list[dict]) -> list[dict]:
    """E9.evaluate resamples r['breed']; here that must be the photographer."""
    return [{**r, "breed": r["photographer"]} for r in recs]


def region_pck(per_kp: dict) -> dict:
    out = {}
    for reg, names in REGIONS.items():
        hits = n = 0
        for k in names:
            p, cnt = per_kp.get(k, (None, 0))
            if p is not None and cnt:
                hits += p * cnt
                n += cnt
        out[reg] = {"pck@0.1": hits / n if n else None, "labelled_points": n}
    return out


def score(params: dict, d: Path, recs: list[dict], gv: float, gp: float) -> dict:
    res = E9.evaluate(params, d, _with_group(recs), gv, gp)
    res["regions"] = region_pck(res["per_keypoint_pck@0.1"])
    res["g2_criteria"] = {reg: {"threshold": thr, "pck@0.1": res["regions"][reg]["pck@0.1"],
                                "met": bool(res["regions"][reg]["pck@0.1"] is not None and res["regions"][reg]["pck@0.1"] >= thr)}
                          for reg, thr in G2_CRITERIA.items()}
    res["g2_criteria"]["ear_position_kappa"] = "not evaluable: no categorical ear-position labels in AwA-Pose"
    pxy, peak, vis = E9.predict_dir(params, d)
    data = E9._load(d)
    err = np.linalg.norm(pxy - np.asarray(data["xy"]), axis=-1) * KP.CROP_SCALE
    lab = np.asarray(data["lab"])
    by_class = {}
    for cls in sorted({r["breed"] for r in recs}):
        ix = np.array([i for i, r in enumerate(recs) if r["breed"] == cls])
        by_class[cls] = {"instances": len(ix), "pck@0.1": float((err[ix][lab[ix]] <= 0.1).mean()) if lab[ix].any() else None}
    res["by_class"] = by_class
    return res


def head_metadata(fname: str, sha: str, model: str, gate: dict) -> dict:
    """Sidecar read by the engine (gates) and by people (terms)."""
    return {"file": fname, "sha256": sha, "backbone": "efficientnet_lite0_mediapipe_v1", "keypoints": KP.DOG_KEYPOINTS,
            "species": ["cat"] if model == "A_cat_head" else ["cat", "dog"], "experiment": "E24", "model": model,
            "gates": {"min_visible": gate["min_visible"], "min_peak": gate["min_peak"], "dev_precision": gate.get("dev_precision"),
                      "dev_coverage_of_labelled": gate.get("dev_coverage_of_labelled"), "rule": gate.get("rule")},
            "trained_on": f"AwA-Pose @ {awa_pose.COMMIT} on AwA2 images, photographer-grouped fit split (results/e24/metrics.json)",
            "terms": "Research only: AwA2 image licence variants unverified for most images. Annotations MIT."}


def run(settings: Settings, out_dir: Path) -> dict:
    t_start = time.perf_counter()
    data_root = settings.paths.resolve("data_root")
    rb = check_model(settings, "efficientnet_lite0_mediapipe_v1")
    if not rb.ready:
        raise RuntimeError(rb.reason)
    backbone = KP.Backbone(Path(next(iter(rb.files.values()))), settings.runtime.num_threads)
    recs = records(data_root)
    splits = {sp: split_by_photographer([r for r in recs if r["species"] == sp]) for sp in ("cat", "dog")}
    cache = data_root.parent / "cache" / "e24"
    t0 = time.perf_counter()
    for sp in ("cat", "dog"):
        for name in ("fit", "dev", "test"):
            d = cache / f"{sp}_{name}"
            if not (d / "lab.npy").exists():
                E9.extract(backbone, splits[sp][name], d, flip=(name == "fit"))
    joint = {name: splits["cat"][name] + splits["dog"][name] for name in ("fit", "dev")}
    for name in ("fit", "dev"):
        d = cache / f"joint_{name}"
        if not (d / "lab.npy").exists():
            E9.extract(backbone, joint[name], d, flip=(name == "fit"))
    extract_s = time.perf_counter() - t0
    models, logs, gates = {}, {}, {}
    for name, fit_dir, dev_dir in (("A_cat_head", cache / "cat_fit", cache / "cat_dev"), ("B_joint_head", cache / "joint_fit", cache / "joint_dev")):
        logs[name] = []
        t0 = time.perf_counter()
        models[name] = E9.train(fit_dir, dev_dir, logs[name])
        gv, gp, info = E9._choose_gates(models[name], dev_dir)
        gates[name] = {"min_visible": gv, "min_peak": gp, **info, "train_seconds": round(time.perf_counter() - t0, 1)}
    ref = REPO_ROOT / "artifacts" / "keypoints" / "dog_keypoints_head_v1.npz"
    if ref.exists():
        z = np.load(ref)
        models["C_e9_dog_head_reference"] = {k: z[k] for k in z.files if k != "version"}
        import json

        g = json.loads((ref.with_suffix(".json")).read_text())["gates"]
        gates["C_e9_dog_head_reference"] = {"min_visible": g["min_visible"], "min_peak": g["min_peak"], "note": "E9's own gates"}
    results: dict = {}
    for name, params in models.items():
        results[name] = {}
        for sp in ("cat", "dog"):
            if name == "A_cat_head" and sp == "dog":
                continue
            results[name][sp] = score(params, cache / f"{sp}_test", splits[sp]["test"], gates[name]["min_visible"], gates[name]["min_peak"])
    art = REPO_ROOT / "artifacts" / "keypoints"
    shas = {}
    for name, fname in (("A_cat_head", "cat_keypoints_head_awa_v1.npz"), ("B_joint_head", "catdog_keypoints_head_awa_v1.npz")):
        p = art / fname
        np.savez(p, **models[name], version=np.array(fname[:-4]))
        shas[fname] = sha256_file(p)
        write_json(p.with_suffix(".json"), head_metadata(fname, shas[fname], name, gates[name]))
    summary = {
        "experiment": "E24", "status": "completed", "created_at": utc_now().isoformat(timespec="seconds"),
        "protocol": __doc__.strip().splitlines(),
        "data": {"annotations": f"{awa_pose.REPO} @ {awa_pose.COMMIT} (MIT)", "images": awa_pose.ARCHIVE + " (per-image licence files)",
                 "instances": {sp: len([r for r in recs if r["species"] == sp]) for sp in ("cat", "dog")},
                 "photographers": {sp: len({r["photographer"] for r in recs if r["species"] == sp}) for sp in ("cat", "dog")},
                 "licences": dict(Counter(r["licence"] for r in recs)),
                 "splits": {sp: {k: {"instances": len(v), "photographers": len({r["photographer"] for r in v})} for k, v in s.items()} for sp, s in splits.items()},
                 "keypoint_mapping": awa_pose.TO_AWA, "seed": SEED},
        "training": {name: {"log": logs[name], "selected_epoch": max(logs[name], key=lambda r: r["dev_pck@0.1"])["epoch"], **gates[name]} for name in logs},
        "gates": gates, "results": results, "artifacts_sha256": shas, "extract_seconds": round(extract_s, 1),
        "terms": "annotations MIT; images per-image Flickr/Wikimedia licences, mostly CC without a stated variant: research only; "
                 "model C was trained on dog-pose (research only)",
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }
    write_json(out_dir / "metrics.json", summary)
    return summary
