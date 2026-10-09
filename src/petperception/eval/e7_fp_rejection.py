"""E7: false-positive rejection for cat/dog detection, with its cost in recall.

Data separation (enforced in code):
* **Tuning** (fit + every selection decision): Open Images V7 *test* split
  subset ``openimages_v7_test_tuning`` (manifests/oi_test_tuning_*.csv).
* **Final evaluation**: the unchanged E4 Open Images V7 *validation* subset.
  It is touched once per configuration, after all choices are frozen.
The builder asserts the two subsets share no image IDs; this module checks it
again.

Stage 1 candidates: detections with score >= 0.20 from each detector.
Stage 2 verifier feature sets (per detection):
* ``crop``: EfficientNet-Lite0 ImageNet group masses on the box crop;
* ``redetect``: the same detector re-run on the zoomed crop (same-class and
  other-class maximum scores) - no extra model;
* ``anchor``: the detector's own 90-class score vector at the anchor that
  produced the box (raw-anchor builds only);
plus detector score and box geometry in every set. A per-class logistic
regression is fitted per feature set.

Tuning labels: on images where the class is verified, greedy IoU>=0.5
matching to non-depiction, non-group boxes gives 1/0 (matches to
depiction/group boxes are skipped); on unverified images, a detection
overlapping (IoU>=0.5) a box of a confusable class or of the other species,
and no box of its own class, is a negative; anything else is skipped.

Selection rule (fixed before looking at E4): operating rule = detector score
>= 0.35 (Phase 1 threshold) AND verifier score >= tau. tau is the largest
value that keeps at least 98% of the detector-only true-positive detections at
the operating threshold, measured with out-of-fold verifier scores
(5-fold grouped by Flickr author) on the tuning set. The baseline
alternative - simply raising the detector threshold - gets the same 98%
budget. The feature set used by the product is the one with the best
out-of-fold tuning specificity on verified-negative images; all are
reported on E4.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..config import Settings
from ..data import openimages as OI
from ..registry import check_model
from ..utils import sha256_file, utc_now, write_json
from ..vision import verifier as V
from ..vision.detector import Detection, load_detector
from . import e4_visual as E4
from . import metrics as M

CLASSES = ["cat", "dog"]
FLOOR = 0.20
OP = 0.35
RETAIN = 0.98
DETECTORS = ["efficientdet_lite0_mediapipe_v1", "efficientdet_lite2_mediapipe_v1", "efficientdet_lite0_tfhub_v1"]
CLASSIFIER = "efficientnet_lite0_mediapipe_v1"
FEATURE_SETS = V.FEATURE_SETS


def _tuning_subset(man: Path) -> tuple[list[dict], list[OI.GtBox], list[dict]]:
    imgs = list(csv.DictReader(open(man / "oi_test_tuning_images.csv", newline="", encoding="utf-8")))
    boxes, conf = [], []
    for r in csv.DictReader(open(man / "oi_test_tuning_boxes.csv", newline="", encoding="utf-8")):
        b = OI.GtBox(r["image_id"], r["label"], float(r["xmin"]), float(r["xmax"]), float(r["ymin"]), float(r["ymax"]),
                     int(r["is_occluded"]), int(r["is_truncated"]), int(r["is_group_of"]), int(r["is_depiction"]), 0)
        (boxes if r["label"] in CLASSES else conf).append(b)
    return imgs, boxes, [{"image_id": b.image_id, "label": b.label, "box": E4._xyxy(b)} for b in conf]


def _verify_images(img_dir: Path, sums: Path) -> int:
    pinned = {r["image_id"]: r["sha256"] for r in csv.DictReader(open(sums, newline=""))}
    bad = [i for i, h in pinned.items() if not (img_dir / f"{i}.jpg").exists() or sha256_file(img_dir / f"{i}.jpg") != h]
    if bad:
        raise RuntimeError(f"{len(bad)} images missing or failing their pinned SHA-256 in {img_dir}")
    return len(pinned)


def collect(det, clf: V.CropClassifier, images: list[dict], img_dir: Path) -> list[dict]:
    rows = []
    for r in images:
        rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{r['image_id']}.jpg")), cv2.COLOR_BGR2RGB)
        for d in det.detect(rgb, CLASSES, FLOOR, 0.5, 20):
            rows.append({"image_id": r["image_id"], **V.describe(det, clf, rgb, d, CLASSES)})
    return rows


def features(row: dict, fset: str) -> np.ndarray:
    return V.features(row, fset)


def label_tuning(rows: list[dict], images: list[dict], boxes: list[OI.GtBox], conf: list[dict]) -> list[int | None]:
    status = {r["image_id"]: {"cat": r["cat_status"], "dog": r["dog_status"]} for r in images}
    by_img: dict[str, list[OI.GtBox]] = defaultdict(list)
    for b in boxes:
        by_img[b.image_id].append(b)
    conf_by: dict[str, list[np.ndarray]] = defaultdict(list)
    for c in conf:
        conf_by[c["image_id"]].append(c["box"])
    labels: list[int | None] = [None] * len(rows)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for k, r in enumerate(rows):
        groups[(r["image_id"], r["label"])].append(k)
    for (img, c), idx in groups.items():
        st = status[img][c]
        own = [b for b in by_img[img] if b.label == c]
        if st == "pos":
            gts = [E4._xyxy(b) for b in own if not b.is_depiction and not b.is_group_of]
            ign = [E4._xyxy(b) for b in own if b.is_depiction or b.is_group_of]
            res = E4._match([(rows[k]["score"], np.array(rows[k]["box"])) for k in idx], gts, ign)
            order = sorted(idx, key=lambda k: -rows[k]["score"])
            for k, (_, kind, _) in zip(order, res):
                labels[k] = None if kind == "ignore" else int(kind == "tp")
        elif st == "neg":
            for k in idx:
                labels[k] = 0
        else:
            other = [E4._xyxy(b) for b in by_img[img] if b.label != c]
            rivals = conf_by.get(img, []) + other
            for k in idx:
                bx = np.array(rows[k]["box"])[None]
                own_iou = M.iou_xyxy(bx, np.array([E4._xyxy(b) for b in own]))[0].max() if own else 0.0
                riv_iou = M.iou_xyxy(bx, np.array(rivals))[0].max() if rivals else 0.0
                if riv_iou >= 0.5 and own_iou < 0.3:
                    labels[k] = 0
    return labels


def _fit(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)


def _oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 5) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        out[te] = _fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


def _tau_for_retention(v_tp: np.ndarray, retain: float) -> float:
    """Largest tau keeping >= retain of the given true-positive verifier scores."""
    if v_tp.size == 0:
        return 0.0
    s = np.sort(v_tp)
    drop = int(np.floor((1 - retain) * s.size))
    return float(s[drop]) if drop < s.size else float(s[-1])


def _image_spec(rows: list[dict], keep: np.ndarray, images: list[dict], c: str) -> dict:
    neg = {r["image_id"] for r in images if r[f"{c}_status"] == "neg"}
    hit = {rows[k]["image_id"] for k in range(len(rows)) if keep[k] and rows[k]["label"] == c and rows[k]["image_id"] in neg}
    return {"verified_negative_images": len(neg), "with_detection": len(hit), "specificity": 1 - len(hit) / max(len(neg), 1)}


def _preds(rows: list[dict], keep: np.ndarray, image_ids: list[str]) -> dict[str, list[Detection]]:
    out: dict[str, list[Detection]] = {i: [] for i in image_ids}
    for k, r in enumerate(rows):
        if keep[k]:
            out[r["image_id"]].append(Detection(r["label"], r["score"], np.array(r["box"])))
    return out


def _cache_rows(path: Path, fn) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text())
    rows = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def run(settings: Settings, out_dir: Path, detectors: list[str] | None = None, params_out: Path | None = None) -> dict:
    man = settings.paths.resolve("manifests_root")
    data = settings.paths.resolve("data_root")
    cache = data.parent / "cache" / "e7"
    tune_imgs, tune_boxes, tune_conf = _tuning_subset(man)
    e4_imgs, e4_boxes = OI.read_subset(man)
    overlap = {r["image_id"] for r in tune_imgs} & {r["image_id"] for r in e4_imgs}
    if overlap:
        raise RuntimeError(f"tuning and evaluation subsets overlap: {sorted(overlap)[:5]}")
    tune_dir = data / "openimages_test" / "images"
    e4_dir = OI.dataset_dir(data) / "images"
    n_tune = _verify_images(tune_dir, man / "oi_test_tuning_sha256.csv")
    n_e4 = _verify_images(e4_dir, man / "openimages_catdog_sha256.csv")
    rc = check_model(settings, CLASSIFIER)
    if not rc.ready:
        raise RuntimeError(f"{CLASSIFIER}: {rc.reason}")
    clf_path = Path(next(iter(rc.files.values())))
    clf = V.CropClassifier(clf_path, settings.runtime.num_threads)
    author_t = {r["image_id"]: r["author_profile_url"] or r["author"] for r in tune_imgs}
    results: dict = {}
    frozen: dict = {}
    for mid in detectors or DETECTORS:
        r = check_model(settings, mid)
        if not r.ready:
            results[mid] = {"status": "unavailable", "reason": r.reason}
            continue
        det = load_detector(Path(next(iter(r.files.values()))), settings.runtime.num_threads)
        t0 = time.perf_counter()
        trows = _cache_rows(cache / f"{mid}_tuning.json", lambda: collect(det, clf, tune_imgs, tune_dir))
        erows = _cache_rows(cache / f"{mid}_e4.json", lambda: collect(det, clf, e4_imgs, e4_dir))
        collect_s = time.perf_counter() - t0
        tlab = label_tuning(trows, tune_imgs, tune_boxes, tune_conf)
        fsets = [f for f in FEATURE_SETS if trows and (trows[0]["anchor"] is not None or "anchor" not in f and f != "all")]
        res_m: dict = {"tuning_detections": len(trows), "e4_detections": len(erows), "collect_seconds_incl_cache": round(collect_s, 1),
                       "tuning": {}, "e4": {}}
        base_keep = np.array([x["score"] >= OP for x in erows])
        res_m["e4"]["detector_only@0.35"] = E4.score_predictions(_preds(erows, base_keep, [i["image_id"] for i in e4_imgs]), e4_imgs, e4_boxes, OP,
                                                                 extra_ci=True)
        choice: dict[str, dict] = {}
        for c in CLASSES:
            idx = [k for k, x in enumerate(trows) if x["label"] == c and tlab[k] is not None]
            y = np.array([tlab[k] for k in idx])
            g = np.array([author_t[trows[k]["image_id"]] for k in idx])
            sc = np.array([trows[k]["score"] for k in idx])
            at_op = sc >= OP
            tp_scores = sc[(y == 1) & at_op]
            # Baseline: raise the detector threshold under the same recall budget.
            thr_raise = max(OP, _tau_for_retention(tp_scores, RETAIN))
            spec_raise = _image_spec(trows, np.array([x["score"] >= thr_raise for x in trows]), tune_imgs, c)
            tune_c: dict = {"labelled_detections": len(idx), "positives": int(y.sum()), "negatives": int((y == 0).sum()),
                            "negatives_on_verified_negative_images": int(sum(1 for k in idx if tlab[k] == 0 and
                                                                             next(r for r in tune_imgs if r["image_id"] == trows[k]["image_id"])[f"{c}_status"] == "neg")),
                            "detector_only@0.35": _image_spec(trows, np.array([x["score"] >= OP for x in trows]), tune_imgs, c),
                            "raised_threshold": {"threshold": thr_raise, **spec_raise}, "verifier": {}}
            for fs in fsets:
                X = np.stack([features(trows[k], fs) for k in idx])
                oof = _oof(X, y, g)
                tau = _tau_for_retention(oof[(y == 1) & at_op], RETAIN)
                v_all = np.full(len(trows), np.inf)
                v_all[idx] = oof
                # Unlabelled detections on verified-negative images do not exist (all are labelled 0), so
                # specificity uses the out-of-fold scores directly.
                keep_t = np.array([x["score"] >= OP and x["label"] == c for x in trows]) & (v_all >= tau)
                tune_c["verifier"][fs] = {"tau": tau, "oof_ap_detection": M.average_precision(oof, y == 1), **_image_spec(trows, keep_t, tune_imgs, c)}
            best = max(tune_c["verifier"], key=lambda f: (tune_c["verifier"][f]["specificity"], -len(f)))
            tune_c["selected_feature_set"] = best
            res_m["tuning"][c] = tune_c
            choice[c] = {"feature_set": best, "tau": tune_c["verifier"][best]["tau"], "raise_thr": thr_raise}
        # Freeze: refit selected models on all tuning data, then score E4 once per configuration.
        models: dict[str, dict] = {}
        for fs in fsets:
            models[fs] = {}
            for c in CLASSES:
                idx = [k for k, x in enumerate(trows) if x["label"] == c and tlab[k] is not None]
                X = np.stack([features(trows[k], fs) for k in idx])
                models[fs][c] = (_fit(X, np.array([tlab[k] for k in idx])), res_m["tuning"][c]["verifier"][fs]["tau"])
        ids = [i["image_id"] for i in e4_imgs]
        keep_raise = np.array([x["score"] >= choice[x["label"]]["raise_thr"] for x in erows])
        res_m["e4"]["raised_threshold"] = {"thresholds": {c: choice[c]["raise_thr"] for c in CLASSES},
                                           **E4.score_predictions(_preds(erows, keep_raise, ids), e4_imgs, e4_boxes, OP, extra_ci=True)}
        for fs in fsets:
            v = np.array([models[fs][x["label"]][0].predict_proba(features(x, fs)[None])[0, 1] for x in erows]) if erows else np.zeros(0)
            keep = base_keep & np.array([v[k] >= models[fs][x["label"]][1] for k, x in enumerate(erows)])
            res_m["e4"][f"verifier[{fs}]"] = {"tau": {c: models[fs][c][1] for c in CLASSES},
                                              **E4.score_predictions(_preds(erows, keep, ids), e4_imgs, e4_boxes, OP, extra_ci=True)}
        res_m["selected"] = {c: choice[c]["feature_set"] for c in CLASSES}
        for k in list(res_m["e4"]):
            for c in CLASSES:
                res_m["e4"][k]["classes"][c].pop("in_sample_best_box_f1_threshold_diagnostic", None)
        results[mid] = res_m
        frozen[mid] = {c: {"feature_set": choice[c]["feature_set"], "model": models[choice[c]["feature_set"]][c][0],
                           "tau": models[choice[c]["feature_set"]][c][1]} for c in CLASSES}
    summary = {
        "experiment": "E7",
        "title": "False-positive rejection for cat/dog detection: tuning on Open Images test split, final evaluation on the E4 validation subset",
        "protocol": __doc__.strip().splitlines(),
        "tested_distributions": {
            "tuning": {"split": "Open Images V7 test", "images": n_tune,
                       "roles": _count(tune_imgs, "role"),
                       "confusable_boxes_by_class": _count(tune_conf, "label")},
            "final_evaluation": {"split": "Open Images V7 validation (E4 subset, unchanged)", "images": n_e4,
                                 "cat_status": _count(e4_imgs, "cat_status"), "dog_status": _count(e4_imgs, "dog_status")},
            "not_tested": ["household camera footage", "night/IR video", "motion blur from moving pets", "fisheye or ceiling cameras",
                           "other species common in homes (rabbits, ferrets, birds) as primary subjects"],
        },
        "overlap_between_tuning_and_evaluation": 0,
        "results": results,
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    write_json(out_dir / "metrics.json", summary)
    if params_out is not None:
        _export_params(frozen, results, params_out)
    return summary


def _e4_digest(e4: dict, fset: str, c: str) -> dict:
    base, ver = e4["detector_only@0.35"]["classes"][c], e4[f"verifier[{fset}]"]["classes"][c]
    keys = ("precision_at_op", "recall_at_op")
    return {"detector_only": {k: base[k] for k in keys} | {"image_specificity": base["image_level"]["specificity_on_verified_negatives"],
                                                           "image_recall": base["image_level"]["recall"]},
            "with_verifier": {k: ver[k] for k in keys} | {"image_specificity": ver["image_level"]["specificity_on_verified_negatives"],
                                                          "image_recall": ver["image_level"]["recall"]}}


def _count(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        out[str(r[key])] += 1
    return dict(sorted(out.items()))


def _export_params(frozen: dict, results: dict, out: Path) -> None:
    """Write the frozen per-class verifier parameters for the runtime DetectionVerifier."""
    for mid, per in frozen.items():
        classes = {}
        for c, f in per.items():
            sc, lr = f["model"].named_steps["standardscaler"], f["model"].named_steps["logisticregression"]
            classes[c] = {"feature_set": f["feature_set"], "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                          "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "tau": f["tau"],
                          "e4_final": _e4_digest(results[mid]["e4"], f["feature_set"], c)}
        write_json(out / f"verifier_{mid}.json", {
            "schema": "petperception.verifier/1", "detector_id": mid, "classifier_id": CLASSIFIER,
            "operating_threshold": OP, "retain_target": RETAIN, "classes": classes,
            "fitted_on": "openimages_v7_test_tuning (manifests/oi_test_tuning_images.csv)",
            "note": "Verification scores are rejection scores fitted on the tuning distribution, not calibrated probabilities.",
            "created_at": utc_now().isoformat(timespec="seconds")})
