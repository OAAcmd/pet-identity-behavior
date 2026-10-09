"""E15: pet-object relations (on / under furniture, inside a car) in still images.

Data: Open Images V7 visual relationships (data/openimages_vrd_objects.py).
Units are (pet, object) pairs of annotated boxes; for every relation the
pair's label is positive, negative (triplet in the annotation vocabulary but
not listed) or unknown (pair not used for that relation).

Pipeline (what the phone computes): the product detector runs with the
product threshold (0.35) for cats, dogs and the object classes (bed, couch,
chair, dining table, car); pet detections pass the product verifier. Each
annotated pet box is matched to an accepted detection of the same species and
each annotated object to a detection of its detector class (greedy, IoU >=
0.5). A pair is scored only when both are detected; coverage is reported and
end-to-end recall counts undetected positives as misses. Features from the
detection boxes: 12 geometry values, species, object group
(vision/object_relation.py). Model: logistic regression per relation.

Splits (by photographer, from the data builder): fit and development are
pooled for model selection by 5-fold cross-validation grouped by photographer
(regularisation by out-of-fold average precision; operating threshold by
out-of-fold F1); the model is then refitted on fit + development and the test
split (20 % of photographers, never used before) is scored once. Confidence
intervals resample photographers.

Shipping gate, fixed before the test split was scored: a relation is put in
the product artifact only if (a) at least 20 test positives are scored, (b)
the lower bound of the test AP 95 % interval is above the test positive rate
(chance AP), and (c) test precision at the chosen threshold is >= 0.70.
Baseline reported for comparison (not gated): the fraction of the pet box
inside the object box used as a score.

What this does not show: relations in video or household camera views,
whether the animal is allowed on the furniture, objects the detector has no
class for (dog beds, cat trees, crates), relations other than the three.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..config import Settings, load_settings
from ..utils import sha256_file, utc_now, write_json
from ..vision.interaction import GEOMETRY
from ..vision.object_relation import DETECTOR_CLASS, GROUPS, RELATIONS, features
from . import metrics as M
from .e13_interaction import _assign

C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
FOLDS = 5
GATE = {"min_test_positives": 20, "min_precision": 0.70}
VERSION = "pet_object_relation_v1"


def _box(r: dict, who: str) -> np.ndarray:
    return np.array([float(r[f"{who}_xmin"]), float(r[f"{who}_ymin"]), float(r[f"{who}_xmax"]), float(r[f"{who}_ymax"])])


def collect(settings: Settings, pairs: list[dict], img_dir: Path, cache: Path) -> list[dict]:
    if cache.exists():
        rows = json.loads(cache.read_text())
        for r in rows:
            r["x"] = np.array(r["x"]) if r["x"] is not None else None
        return rows
    from ..pipeline import AnalysisEngine

    s = load_settings(overrides={"audio": {"enabled": False}, "posture": {"enabled": False}, "pose": {"provider": "none"},
                                 "interaction": {"enabled": False}})
    eng = AnalysisEngine(s).load()
    by_img: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_img[p["image_id"]].append(p)
    classes = ["cat", "dog"] + sorted(set(DETECTOR_CLASS.values()))
    out = []
    for i, ps in sorted(by_img.items()):
        im = cv2.imread(str(img_dir / f"{i}.jpg"))
        if im is None:
            out.extend({**p, "covered": False, "reason": "missing_image", "x": None} for p in ps)
            continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        dets = eng.detector.detect(rgb, classes, s.vision.score_threshold, s.vision.nms_iou, 30)
        pets = [d for d in dets if d.label in ("cat", "dog") and (eng.verifier is None or (eng.verifier.verify(rgb, d) or (0.0, True))[1])]
        gq = sorted({(tuple(_box(p, "pet")), p["pet"]) for p in ps})
        go = sorted({(tuple(_box(p, "object")), p["object"]) for p in ps})
        mq: dict[int, object] = {}
        for sp in ("cat", "dog"):
            idx = [k for k, (_, s_) in enumerate(gq) if s_ == sp]
            ds = [d for d in pets if d.label == sp]
            for gi, di in _assign([np.array(gq[k][0]) for k in idx], ds).items():
                mq[idx[gi]] = ds[di]
        mo: dict[int, object] = {}
        for g in GROUPS:
            idx = [k for k, (_, g_) in enumerate(go) if g_ == g]
            ds = [d for d in dets if d.label == DETECTOR_CLASS[g]]
            for gi, di in _assign([np.array(go[k][0]) for k in idx], ds).items():
                mo[idx[gi]] = ds[di]
        for p in ps:
            qi = gq.index((tuple(_box(p, "pet")), p["pet"]))
            oi = go.index((tuple(_box(p, "object")), p["object"]))
            if qi not in mq or oi not in mo:
                out.append({**p, "covered": False, "reason": "pet_not_detected" if qi not in mq else "object_not_detected", "x": None})
                continue
            qb, ob = mq[qi].box, mo[oi].box
            out.append({**p, "covered": True, "reason": None, "x": features(ob, qb, rgb.shape[:2], p["pet"], p["object"]),
                        "det_pet": [float(v) for v in qb], "det_object": [float(v) for v in ob]})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([{**r, "x": None if r["x"] is None else r["x"].tolist()} for r in out]))
    return out


def _fit(X: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(X, y)


def _oof(X: np.ndarray, y: np.ndarray, groups: list[str], C: float) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=FOLDS).split(X, y, groups):
        p[te] = _fit(X[tr], y[tr], C).predict_proba(X[te])[:, 1]
    return p


def _report(p: np.ndarray, y: np.ndarray, groups: list[str], thr: float, rows: list[dict], positives_all: int) -> dict:
    pred = p >= thr

    def prf(ix):
        ix = np.asarray(ix)
        tp = float((pred[ix] & (y[ix] == 1)).sum())
        return tp / max(pred[ix].sum(), 1), tp / max((y[ix] == 1).sum(), 1)

    prec, rec = prf(np.arange(len(y)))
    out = {"pairs_scored": int(len(y)), "positives_scored": int(y.sum()), "average_precision": M.average_precision(p, y == 1),
           "ap_ci95": M.group_bootstrap(lambda ix: M.average_precision(p[ix], y[ix] == 1), groups, 1000, 0),
           "chance_ap_positive_rate": float(y.mean()), "threshold": thr, "precision": prec, "recall": rec,
           "precision_ci95": M.group_bootstrap(lambda ix: prf(ix)[0], groups, 1000, 0),
           "recall_ci95": M.group_bootstrap(lambda ix: prf(ix)[1], groups, 1000, 0),
           "f1": 2 * prec * rec / max(prec + rec, 1e-12),
           "end_to_end_recall_all_annotated_positives": float((pred & (y == 1)).sum()) / max(positives_all, 1)}
    for key, vals in (("species", ("cat", "dog")), ("object", GROUPS)):
        for v in vals:
            ix = [k for k, r in enumerate(rows) if r["pet" if key == "species" else "object"] == v]
            if ix and y[ix].sum():
                out.setdefault(f"by_{key}", {})[v] = {"pairs": len(ix), "positives": int(y[ix].sum()),
                                                      "average_precision": M.average_precision(p[ix], y[ix] == 1),
                                                      "precision_recall": list(prf(ix))}
    return out


def run(settings: Settings, out_dir: Path, artifact_dir: Path | None = None) -> dict:
    data = settings.paths.resolve("data_root")
    man = settings.paths.resolve("manifests_root")
    img_dir = data / "openimages_vrd" / "images"
    pinned = {r["image_id"]: r["sha256"] for r in csv.DictReader(open(man / "oi_vrd_object_sha256.csv", newline=""))}
    photographer = {r["image_id"]: (r["author_profile_url"] or f"image:{r['image_id']}")
                    for r in csv.DictReader(open(man / "oi_vrd_object_images.csv", newline=""))}
    pairs = list(csv.DictReader(open(man / "oi_vrd_object_pairs.csv", newline="")))
    bad = [i for i in pinned if (img_dir / f"{i}.jpg").exists() and sha256_file(img_dir / f"{i}.jpg") != pinned[i]]
    if bad:
        raise RuntimeError(f"{len(bad)} relation images fail their pinned SHA-256")
    missing = sorted({p["image_id"] for p in pairs} - set(pinned))
    rows = collect(settings, [p for p in pairs if p["image_id"] in pinned], img_dir, data.parent / "cache" / "e15" / "pairs.json")
    summary: dict = {"experiment": "E15", "title": "Pet-object relations (on / under furniture, inside a car) in still images, Open Images relationships",
                     "protocol": __doc__.strip().splitlines(), "gate": GATE, "images_missing": len(missing), "relations": {}}
    shipped: dict[str, dict] = {}
    for rel, groups in RELATIONS.items():
        lab = [r for r in rows if r[rel] != ""]
        dev_all = [r for r in lab if r["split"] in ("fit", "dev")]
        test_all = [r for r in lab if r["split"] == "test"]
        dev = [r for r in dev_all if r["covered"]]
        test = [r for r in test_all if r["covered"]]
        Xd, yd, gd = np.stack([r["x"] for r in dev]), np.array([int(r[rel]) for r in dev]), [photographer[r["image_id"]] for r in dev]
        cv_ap = {C: M.average_precision(_oof(Xd, yd, gd, C), yd == 1) for C in C_GRID}
        C = max(cv_ap, key=cv_ap.get)
        oof = _oof(Xd, yd, gd, C)
        thr, _ = M.best_f1_threshold(oof, yd == 1)
        m = _fit(Xd, yd, C)
        Xt, yt, gt = np.stack([r["x"] for r in test]), np.array([int(r[rel]) for r in test]), [photographer[r["image_id"]] for r in test]
        pt = m.predict_proba(Xt)[:, 1]
        pos_all = sum(1 for r in test_all if int(r[rel]) == 1)
        res = _report(pt, yt, gt, thr, test, pos_all)
        rule = np.array([r["x"][GEOMETRY.index("inter_over_pet")] for r in test])
        base = {"average_precision": M.average_precision(rule, yt == 1),
                "ap_ci95": M.group_bootstrap(lambda ix: M.average_precision(rule[ix], yt[ix] == 1), gt, 1000, 0)}
        pr = rule >= 0.5
        base["precision_recall_at_0.5"] = [float((pr & (yt == 1)).sum() / max(pr.sum(), 1)), float((pr & (yt == 1)).sum() / max(yt.sum(), 1))]
        gate = {"test_positives_scored": bool(res["positives_scored"] >= GATE["min_test_positives"]),
                "ap_lower_bound_above_chance": bool(res["ap_ci95"]["ci_low"] > res["chance_ap_positive_rate"]),
                "precision_at_threshold": bool(res["precision"] >= GATE["min_precision"])}
        reasons = {k: dict(sorted(((rs, sum(1 for r in v if r["reason"] == rs)) for rs in {r["reason"] for r in v}), key=lambda kv: str(kv[0])))
                   for k, v in (("development", dev_all), ("test", test_all))}
        summary["relations"][rel] = {
            "object_groups": list(groups),
            "pairs": {"development": len(dev_all), "test": len(test_all)}, "pairs_scored": {"development": len(dev), "test": len(test)},
            "positives": {"development": int(sum(int(r[rel]) for r in dev_all)), "test": pos_all},
            "coverage_reasons": reasons,
            "cross_validation_ap_by_C": {str(c): v for c, v in cv_ap.items()}, "selected": {"C": C, "threshold": thr},
            "development_out_of_fold": {"average_precision": cv_ap[C]},
            "test": res, "baseline_test_pet_inside_object_fraction": base,
            "gate": gate, "shipped": all(gate.values())}
        if all(gate.values()):
            scl, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
            shipped[rel] = {"groups": np.array(groups), "mean": scl.mean_, "scale": scl.scale_, "coef": lr.coef_[0],
                            "intercept": lr.intercept_, "threshold": np.array(thr)}
        fp = sorted(((float(pt[k]), test[k]) for k in range(len(test)) if pt[k] >= thr and yt[k] == 0), key=lambda v: -v[0])
        summary["relations"][rel]["test_false_positives_for_audit"] = [
            {"image_id": r["image_id"], "pet": r["pet"], "object": r["object"], "object_classes": r["object_classes"], "score": s_,
             "det_pet": r["det_pet"], "det_object": r["det_object"]} for s_, r in fp[:15]]
    summary["shipped_relations"] = sorted(shipped)
    summary["terms"] = "Open Images annotations CC BY 4.0, images CC BY 2.0 (attribution in manifests/oi_vrd_object_images.csv)."
    summary["created_at"] = utc_now().isoformat(timespec="seconds")
    if artifact_dir is not None and shipped:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        p = artifact_dir / f"{VERSION}.npz"
        arrays = {f"{r}_{k}": v for r, d in shipped.items() for k, v in d.items()}
        np.savez(p, relations=np.array(sorted(shipped)), features=np.array("geometry+species+object_group"), geometry=np.array(GEOMETRY),
                 object_groups=np.array(GROUPS), version=np.array(VERSION), **arrays)
        summary["artifact_sha256"] = sha256_file(p)
    write_json(out_dir / "metrics.json", summary)
    return summary
