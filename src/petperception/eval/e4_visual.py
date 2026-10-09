"""E4: cat/dog detection on the Open Images V7 validation subset.

Protocol (per class C in {cat, dog}):
* evaluation images = subset images where C is human-verified (boxed
  positive or verified negative); detections of C on unverified images are
  not scored (Open Images convention);
* ground truth = C boxes; ``IsDepiction`` and ``IsGroupOf`` boxes are ignore
  regions (a detection matching them is neither TP nor FP);
* AP@IoU0.5 with the all-point interpolated precision envelope; precision /
  recall / F1 at the configured operating threshold (set a priori in
  configs/cpu.yaml, not tuned here);
* image-level presence metrics on verified images;
* 95% CIs by bootstrap over Flickr authors (images by one author are
  correlated), 1000 resamples.

Diagnostics: cross-species confusion on images where the other species is not
verified absent (labelled as unverified), recall by box size / occlusion /
truncation, and verified-negative images that still trigger detections.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..config import Settings
from ..data import openimages as OI
from ..registry import check_model
from ..utils import sha256_file, utc_now, write_json
from ..vision.detector import EfficientDetDetector
from . import metrics as M

CLASSES = ["cat", "dog"]


def _xyxy(b: OI.GtBox) -> np.ndarray:
    return np.array([b.xmin, b.ymin, b.xmax, b.ymax])


def _ap(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(float)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _match(dets: list[tuple[float, np.ndarray]], gts: list[np.ndarray], ign: list[np.ndarray], thr: float = 0.5):
    """Greedy matching for one image; returns per-detection status and matched GT indices."""
    status = []
    used = set()
    G = np.array(gts) if gts else np.zeros((0, 4))
    I = np.array(ign) if ign else np.zeros((0, 4))
    for score, box in sorted(dets, key=lambda d: -d[0]):
        ious = M.iou_xyxy(box[None], G)[0] if len(G) else np.zeros(0)
        best, bi = 0.0, -1
        for j, v in enumerate(ious):
            if j not in used and v > best:
                best, bi = v, j
        if best >= thr:
            used.add(bi)
            status.append((score, "tp", bi))
        elif len(I) and M.iou_xyxy(box[None], I)[0].max() >= thr:
            status.append((score, "ignore", -1))
        else:
            status.append((score, "fp", -1))
    return status


def evaluate_detector(det: EfficientDetDetector, images: list[dict], boxes: list[OI.GtBox], img_dir: Path, op_thr: float) -> dict:
    preds: dict[str, list] = {}
    t0 = time.perf_counter()
    for r in images:
        rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{r['image_id']}.jpg")), cv2.COLOR_BGR2RGB)
        preds[r["image_id"]] = det.detect(rgb, CLASSES, 0.05, 0.5, 50)
    infer_s = time.perf_counter() - t0
    out = score_predictions(preds, images, boxes, op_thr)
    out = {"inference_ms_per_image_including_jpeg_decode": 1000 * infer_s / len(images), **out}
    out["per_image_predictions"] = {i: [(d.label, round(d.score, 4), [round(float(v), 4) for v in d.box]) for d in preds[i] if d.score >= 0.2] for i in preds}
    return out


def score_predictions(preds: dict[str, list], images: list[dict], boxes: list[OI.GtBox], op_thr: float, n_boot: int = 1000,
                      extra_ci: bool = False) -> dict:
    """Score per-image detections with the E4 protocol (shared with E7)."""
    by_img: dict[str, list[OI.GtBox]] = defaultdict(list)
    for b in boxes:
        by_img[b.image_id].append(b)
    status = {r["image_id"]: {"cat": r["cat_status"], "dog": r["dog_status"]} for r in images}
    author = {r["image_id"]: r["author_profile_url"] or r["author"] for r in images}
    out: dict = {"classes": {}}
    for c in CLASSES:
        ev_imgs = [i for i in status if status[i][c] in ("pos", "neg")]
        rows = []  # (image, score, status)
        n_gt_img: dict[str, int] = {}
        gt_meta: dict[str, list[OI.GtBox]] = {}
        for i in ev_imgs:
            gts = [b for b in by_img[i] if b.label == c and not b.is_depiction and not b.is_group_of]
            ign = [b for b in by_img[i] if b.label == c and (b.is_depiction or b.is_group_of)]
            dets = [(d.score, d.box) for d in preds[i] if d.label == c]
            st = _match(dets, [_xyxy(b) for b in gts], [_xyxy(b) for b in ign])
            n_gt_img[i] = len(gts)
            gt_meta[i] = gts
            rows += [(i, s, k, gi) for s, k, gi in st]
        scored = [(i, s, k, gi) for i, s, k, gi in rows if k != "ignore"]

        rows_by_img: dict[str, list[tuple[float, bool]]] = defaultdict(list)
        for i, s, k, _ in scored:
            rows_by_img[i].append((s, k == "tp"))

        def ap_for(img_list: list[str]) -> float:
            # Bootstrap resamples contain repeated images; each repetition counts
            # its detections and its ground truth again.
            pairs = [x for i in img_list for x in rows_by_img.get(i, [])]
            sc = np.array([p[0] for p in pairs])
            tp = np.array([p[1] for p in pairs], dtype=bool)
            return _ap(tp, sc, sum(n_gt_img[i] for i in img_list))

        ap = ap_for(ev_imgs)
        n_gt = sum(n_gt_img.values())
        tp_op = sum(1 for i, s, k, _ in scored if k == "tp" and s >= op_thr)
        fp_op = sum(1 for i, s, k, _ in scored if k == "fp" and s >= op_thr)
        prec = tp_op / max(tp_op + fp_op, 1)
        rec = tp_op / max(n_gt, 1)
        # Recall breakdown at operating threshold by GT attributes.
        matched = {(i, gi) for i, s, k, gi in scored if k == "tp" and s >= op_thr}
        breakdown: dict[str, dict[str, float]] = {}
        for name, fn in {
            "area_lt_1pct": lambda b: (b.xmax - b.xmin) * (b.ymax - b.ymin) < 0.01,
            "area_1_to_10pct": lambda b: 0.01 <= (b.xmax - b.xmin) * (b.ymax - b.ymin) < 0.10,
            "area_ge_10pct": lambda b: (b.xmax - b.xmin) * (b.ymax - b.ymin) >= 0.10,
            "occluded": lambda b: b.is_occluded == 1,
            "truncated": lambda b: b.is_truncated == 1,
        }.items():
            sel = [(i, gi) for i in ev_imgs for gi, b in enumerate(gt_meta[i]) if fn(b)]
            breakdown[name] = {"gt_boxes": len(sel), "recall": (sum((i, gi) in matched for i, gi in sel) / len(sel)) if sel else None}
        # Image-level presence on verified images.
        y_true = np.array([status[i][c] == "pos" for i in ev_imgs])
        y_pred = np.array([any(d.label == c and d.score >= op_thr for d in preds[i]) for i in ev_imgs])
        tp_i, fp_i = int(np.sum(y_true & y_pred)), int(np.sum(~y_true & y_pred))
        fn_i, tn_i = int(np.sum(y_true & ~y_pred)), int(np.sum(~y_true & ~y_pred))
        groups = [author[i] for i in ev_imgs]
        idx_imgs = np.array(ev_imgs)

        def f1_img(ix: np.ndarray) -> float:
            t, p = y_true[ix], y_pred[ix]
            a, b, cc = np.sum(t & p), np.sum(~t & p), np.sum(t & ~p)
            return float(2 * a / max(2 * a + b + cc, 1))

        def spec_img(ix: np.ndarray) -> float:
            t, p = y_true[ix], y_pred[ix]
            return float(np.sum(~t & ~p) / max(np.sum(~t), 1))

        def rec_img(ix: np.ndarray) -> float:
            t, p = y_true[ix], y_pred[ix]
            return float(np.sum(t & p) / max(np.sum(t), 1))

        ap_ci = M.group_bootstrap(lambda ix: ap_for(list(idx_imgs[ix])), groups, n_boot, 0)
        f1_ci = M.group_bootstrap(f1_img, groups, n_boot, 0)
        # In-sample diagnostic only: threshold maximizing box-level F1 on this set.
        cand = sorted({round(s, 3) for _, s, _, _ in scored})
        best = max(((2 * sum(1 for _, s, k, _ in scored if k == "tp" and s >= t) /
                     max(sum(1 for _, s, k, _ in scored if s >= t) + n_gt, 1)), t) for t in cand) if cand else (0.0, None)
        neg_hits = sorted({i for i in ev_imgs if status[i][c] == "neg" and y_pred[ev_imgs.index(i)]})
        other = "dog" if c == "cat" else "cat"
        cross_imgs = [i for i in status if status[i][other] == "pos" and status[i][c] != "pos"]
        cross = sum(any(d.label == c and d.score >= op_thr for d in preds[i]) for i in cross_imgs)
        out["classes"][c] = {
            "eval_images": len(ev_imgs), "verified_positive_images": int(y_true.sum()), "verified_negative_images": int((~y_true).sum()),
            "gt_boxes_scored": n_gt, "ignored_gt_boxes_depiction_or_groupof": sum(1 for i in ev_imgs for b in by_img[i] if b.label == c and (b.is_depiction or b.is_group_of)),
            "ap50": ap, "ap50_ci95": ap_ci,
            "operating_threshold": op_thr, "precision_at_op": prec, "recall_at_op": rec,
            "f1_at_op": 2 * prec * rec / max(prec + rec, 1e-12), "tp_at_op": tp_op, "fp_at_op": fp_op,
            "image_level": {"tp": tp_i, "fp": fp_i, "fn": fn_i, "tn": tn_i,
                            "precision": tp_i / max(tp_i + fp_i, 1), "recall": tp_i / max(tp_i + fn_i, 1),
                            "specificity_on_verified_negatives": tn_i / max(tn_i + fp_i, 1), "f1": f1_img(np.arange(len(ev_imgs))), "f1_ci95": f1_ci,
                            **({"specificity_ci95": M.group_bootstrap(spec_img, groups, n_boot, 0),
                                "recall_ci95": M.group_bootstrap(rec_img, groups, n_boot, 0)} if extra_ci else {})},
            "recall_breakdown_at_op": breakdown,
            "in_sample_best_box_f1_threshold_diagnostic": {"f1": best[0], "threshold": best[1], "note": "chosen on the test set itself; NOT used for deployment"},
            "verified_negative_images_with_detection_at_op": neg_hits,
            "cross_species_diagnostic": {"images_where_other_species_positive_and_this_not_positive": len(cross_imgs),
                                         "with_this_species_detection_at_op": int(cross),
                                         "note": "this species is often unverified on these images; not a verified false-positive rate"},
        }
    return out


def run(settings: Settings, out_dir: Path, detectors: list[str] | None = None) -> dict:
    man = settings.paths.resolve("manifests_root")
    data = settings.paths.resolve("data_root")
    images, boxes = OI.read_subset(man)
    img_dir = OI.dataset_dir(data) / "images"
    pinned = {r["image_id"]: r["sha256"] for r in csv.DictReader(open(man / "openimages_catdog_sha256.csv", newline=""))}
    missing = [r["image_id"] for r in images if not (img_dir / f"{r['image_id']}.jpg").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} subset images missing; run: uv run petperception fetch-data openimages-catdog")
    bad = [i for i in pinned if sha256_file(img_dir / f"{i}.jpg") != pinned[i]]
    if bad:
        raise RuntimeError(f"checksum mismatch for {len(bad)} images")
    results = {}
    for mid in detectors or [settings.vision.detector]:
        r = check_model(settings, mid)
        if not r.ready:
            results[mid] = {"status": "unavailable", "reason": r.reason}
            continue
        det = EfficientDetDetector(Path(next(iter(r.files.values()))), settings.runtime.num_threads)
        res = evaluate_detector(det, images, boxes, img_dir, settings.vision.score_threshold)
        preds = res.pop("per_image_predictions")
        write_json(out_dir / f"predictions_{mid}.json", preds)
        results[mid] = res
    summary = {
        "experiment": "E4",
        "title": "Cat/dog detection on Open Images V7 validation subset (human-verified labels)",
        "dataset": {"id": "openimages_v7_val_catdog", "images": len(images), "boxes": len(boxes),
                    "selection": "all rotation-safe cat-positive images (339), 350 sampled dog-positive (seed 20261009), all 134 verified negatives",
                    "checksums_verified": len(pinned)},
        "protocol": __doc__.strip().splitlines()[2:20],
        "contamination": "Detectors were trained on COCO 2017 (Flickr); Open Images is also Flickr-sourced. Photo-level overlap is unaudited.",
        "results": results,
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    write_json(out_dir / "metrics.json", summary)
    return summary
