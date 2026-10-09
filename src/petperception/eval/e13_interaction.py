"""E13: person-pet interaction ("holds" / "interacts with") in still images.

Data: Open Images V7 visual relationships (data/openimages_vrd.py). Units are
(person, pet) pairs of annotated boxes; the relationship annotation protocol
makes unrelated pairs reliable negatives.

Pipeline (what the phone computes): the product detector runs on the image
with the product thresholds; pet detections pass the product verifier. Each
annotated person box is matched to a COCO "person" detection and each pet box
to an accepted cat/dog detection of the same species (greedy, IoU >= 0.5).
A pair is scored only when both boxes are detected; coverage is reported and
end-to-end recall counts undetected positives as misses.
Features from the detection boxes: 12 geometry values (overlaps, offsets,
size ratio, aspect ratios, gap, containment), the species indicator, and the
pooled EfficientNet-Lite0 embedding of the square crop around the union box.
Models: logistic regression on geometry only, and on geometry + embedding.

Splits: fit on Open Images train minus a seeded 10 % image hold-out
(development: regularisation, feature set and the operating threshold, which
maximises development F1); final evaluation once on Open Images validation +
test. Confidence intervals resample images. Baselines: positive rate
(chance AP) and "boxes overlap" (IoU > 0).

What this does not show: interaction in video, the animal's response or
state, interactions with other animals or objects, household camera views.
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
from ..vision import keypoints as KP
from ..vision.interaction import GEOMETRY, _iou, geometry
from . import metrics as M

C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
MATCH_IOU = 0.5
def union_box(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])])


def _gt(r: dict, who: str) -> np.ndarray:
    return np.array([float(r[f"{who}_xmin"]), float(r[f"{who}_ymin"]), float(r[f"{who}_xmax"]), float(r[f"{who}_ymax"])])


def _assign(gts: list[np.ndarray], dets: list) -> dict[int, int]:
    """Greedy IoU matching of ground-truth boxes to detections (each used once)."""
    cand = sorted(((_iou(g, d.box), gi, di) for gi, g in enumerate(gts) for di, d in enumerate(dets)), reverse=True)
    out, used_g, used_d = {}, set(), set()
    for iou, gi, di in cand:
        if iou < MATCH_IOU:
            break
        if gi in used_g or di in used_d:
            continue
        out[gi] = di
        used_g.add(gi)
        used_d.add(di)
    return out


def collect(settings: Settings, pairs: list[dict], img_dir: Path, cache: Path) -> list[dict]:
    if cache.exists():
        rows = json.loads(cache.read_text())
        for r in rows:
            r["x_geo"], r["x_emb"] = np.array(r["x_geo"]), (np.array(r["x_emb"]) if r["x_emb"] is not None else None)
        return rows
    from ..pipeline import AnalysisEngine

    s = load_settings(overrides={"audio": {"enabled": False}, "posture": {"enabled": False}, "pose": {"provider": "none"}})
    eng = AnalysisEngine(s).load()
    from ..registry import check_model

    bb = KP.Backbone(Path(next(iter(check_model(s, s.posture.backbone).files.values()))), s.runtime.num_threads)
    by_img: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_img[p["image_id"]].append(p)
    out = []
    for i, ps in sorted(by_img.items()):
        im = cv2.imread(str(img_dir / f"{i}.jpg"))
        if im is None:
            out.extend({**p, "covered": False, "reason": "missing_image", "x_geo": None, "x_emb": None} for p in ps)
            continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        dets = eng.detector.detect(rgb, ["cat", "dog", "person"], s.vision.score_threshold, s.vision.nms_iou, 20)
        persons = [d for d in dets if d.label == "person"]
        pets = [d for d in dets if d.label in ("cat", "dog") and (eng.verifier is None or (eng.verifier.verify(rgb, d) or (0.0, True))[1])]
        gp = sorted({tuple(_gt(p, "person")) for p in ps})
        gq = sorted({(tuple(_gt(p, "pet")), p["pet"]) for p in ps})
        mp = _assign([np.array(g) for g in gp], persons)
        mq = {}
        for sp in ("cat", "dog"):
            idx = [k for k, (_, s_) in enumerate(gq) if s_ == sp]
            ds = [d for d in pets if d.label == sp]
            for gi, di in _assign([np.array(gq[k][0]) for k in idx], ds).items():
                mq[idx[gi]] = ds[di]
        for p in ps:
            pi = gp.index(tuple(_gt(p, "person")))
            qi = gq.index((tuple(_gt(p, "pet")), p["pet"]))
            if pi not in mp or qi not in mq:
                out.append({**p, "covered": False, "reason": "person_not_detected" if pi not in mp else "pet_not_detected", "x_geo": None, "x_emb": None})
                continue
            pb, qb = persons[mp[pi]].box, mq[qi].box
            emb = bb.features(KP.square_crop(rgb, union_box(pb, qb))[0])["pooled"]
            out.append({**p, "covered": True, "reason": None, "x_geo": geometry(pb, qb, rgb.shape[:2]), "x_emb": emb,
                        "det_person": [float(v) for v in pb], "det_pet": [float(v) for v in qb]})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([{**r, "x_geo": None if r["x_geo"] is None else r["x_geo"].tolist(),
                                  "x_emb": None if r["x_emb"] is None else r["x_emb"].tolist()} for r in out]))
    return out


def _X(rows: list[dict], which: str) -> np.ndarray:
    sp = lambda r: [1.0, 0.0] if r["pet"] == "cat" else [0.0, 1.0]  # noqa: E731
    if which == "geometry":
        return np.stack([np.concatenate([r["x_geo"], sp(r)]) for r in rows])
    return np.stack([np.concatenate([r["x_geo"], sp(r), r["x_emb"]]) for r in rows])


def _fit(X: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(X, y)


def _best_f1_threshold(p: np.ndarray, y: np.ndarray) -> float:
    best, thr = -1.0, 0.5
    for t in np.unique(np.round(p, 3)):
        pred = p >= t
        tp = float((pred & (y == 1)).sum())
        f1 = 2 * tp / max(pred.sum() + (y == 1).sum(), 1)
        if f1 > best:
            best, thr = f1, float(t)
    return thr


def _report(rows: list[dict], p: np.ndarray, thr: float, all_pairs: list[dict]) -> dict:
    y = np.array([r["interaction"] for r in rows])
    pred = p >= thr
    groups = [r["image_id"] for r in rows]

    def prf(ix):
        ix = np.asarray(ix)
        tp = float((pred[ix] & (y[ix] == 1)).sum())
        return tp / max(pred[ix].sum(), 1), tp / max((y[ix] == 1).sum(), 1)

    prec, rec = prf(np.arange(len(y)))
    out = {"pairs_scored": int(len(y)), "positives_scored": int(y.sum()), "average_precision": M.average_precision(p, y == 1),
           "ap_ci95": M.group_bootstrap(lambda ix: M.average_precision(p[ix], y[ix] == 1), groups, 1000, 0),
           "chance_ap_positive_rate": float(y.mean()), "threshold": thr, "precision": prec, "recall": rec,
           "f1": 2 * prec * rec / max(prec + rec, 1e-12),
           "f1_ci95": M.group_bootstrap(lambda ix: (lambda a, b: 2 * a * b / max(a + b, 1e-12))(*prf(ix)), groups, 1000, 0)}
    for sp in ("cat", "dog"):
        ix = [k for k, r in enumerate(rows) if r["pet"] == sp]
        if ix:
            out[sp] = {"pairs": len(ix), "positives": int(y[ix].sum()), "average_precision": M.average_precision(p[ix], y[ix] == 1),
                       "precision_recall": list(prf(ix))}
    pos_all = sum(1 for r in all_pairs if r["interaction"])
    out["end_to_end_recall_all_annotated_positives"] = float((pred & (y == 1)).sum()) / max(pos_all, 1)
    return out


def run(settings: Settings, out_dir: Path, artifact_dir: Path | None = None) -> dict:
    data = settings.paths.resolve("data_root")
    man = settings.paths.resolve("manifests_root")
    img_dir = data / "openimages_vrd" / "images"
    pinned = {r["image_id"]: r["sha256"] for r in csv.DictReader(open(man / "oi_vrd_interaction_sha256.csv", newline=""))}
    pairs = list(csv.DictReader(open(man / "oi_vrd_interaction_pairs.csv", newline="")))
    for p in pairs:
        p["interaction"] = int(p["interaction"])
    missing = sorted({p["image_id"] for p in pairs} - set(pinned))
    bad = [i for i in pinned if (img_dir / f"{i}.jpg").exists() and sha256_file(img_dir / f"{i}.jpg") != pinned[i]]
    if bad:
        raise RuntimeError(f"{len(bad)} interaction images fail their pinned SHA-256")
    rows = collect(settings, [p for p in pairs if p["image_id"] in pinned], img_dir, data.parent / "cache" / "e13" / "pairs.json")
    split = {"fit": ("train",), "dev": ("train_dev",), "final": ("validation", "test")}
    cov = {k: [r for r in rows if r["split"] in v] for k, v in split.items()}
    sc = {k: [r for r in v if r["covered"]] for k, v in cov.items()}
    y = {k: np.array([r["interaction"] for r in v]) for k, v in sc.items()}
    dev_ap: dict = {}
    for which in ("geometry", "geometry+embedding"):
        for C in C_GRID:
            m = _fit(_X(sc["fit"], which), y["fit"], C)
            dev_ap[(which, C)] = M.average_precision(m.predict_proba(_X(sc["dev"], which))[:, 1], y["dev"] == 1)
    which, C = max(dev_ap, key=dev_ap.get)
    results = {}
    chosen = None
    for feats in ("geometry", "geometry+embedding"):
        Cf = max((c for (w, c) in dev_ap if w == feats), key=lambda c: dev_ap[(feats, c)])
        m = _fit(_X(sc["fit"], feats), y["fit"], Cf)
        thr = _best_f1_threshold(m.predict_proba(_X(sc["dev"], feats))[:, 1], y["dev"])
        results[feats] = {"C": Cf, "dev": _report(sc["dev"], m.predict_proba(_X(sc["dev"], feats))[:, 1], thr, cov["dev"]),
                          "final": _report(sc["final"], m.predict_proba(_X(sc["final"], feats))[:, 1], thr, cov["final"])}
        if feats == which:
            chosen = (m, thr)
    overlap = np.array([r["x_geo"][0] > 0 for r in sc["final"]], dtype=float)
    reasons = {k: dict(sorted({rs: sum(1 for r in v if r["reason"] == rs) for rs in {r["reason"] for r in v}}.items(), key=lambda kv: str(kv[0])))
               for k, v in cov.items()}
    summary = {
        "experiment": "E13", "title": "Person-pet interaction (holds / interacts with) in still images, Open Images relationships",
        "protocol": __doc__.strip().splitlines(),
        "pairs": {k: len(v) for k, v in cov.items()}, "pairs_scored": {k: len(v) for k, v in sc.items()},
        "positives": {k: int(sum(r["interaction"] for r in v)) for k, v in cov.items()},
        "coverage_reasons": reasons, "images_missing": len(missing),
        "development_ap_by_features_and_C": {f"{w}@{c}": v for (w, c), v in dev_ap.items()},
        "selected_on_development": {"features": which, "C": C},
        "results": results,
        "baseline_final_boxes_overlap_rule": _report(sc["final"], overlap, 0.5, cov["final"]),
        "terms": "Open Images annotations CC BY 4.0, images CC BY 2.0 (attribution in manifests/oi_vrd_interaction_images.csv).",
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    if artifact_dir is not None and chosen is not None:
        m, thr = chosen
        scl, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        p = artifact_dir / "person_pet_interaction_v1.npz"
        np.savez(p, mean=scl.mean_, scale=scl.scale_, coef=lr.coef_[0], intercept=lr.intercept_, threshold=np.array(thr),
                 features=np.array(which), geometry=np.array(GEOMETRY), version=np.array("person_pet_interaction_v1"))
        summary["artifact_sha256"] = sha256_file(p)
    write_json(out_dir / "metrics.json", summary)
    return summary
