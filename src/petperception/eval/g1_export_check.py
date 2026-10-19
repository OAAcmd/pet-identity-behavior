"""G1 export check: a trained detector export scored with the E4 protocol, next to the shipped detectors.

Scores a Model Maker export (float32, float16 or int8 TFLite) on the E4 Open
Images validation subset with `e4_visual.evaluate_detector`, the code that
produced E4. The same images are scored with M3 (the phone build's detector)
and M2 (the reference detector), on the same images; each AP50 carries E4's bootstrap interval over Flickr authors. G1's training
data excludes every photographer of this subset (`training/g1_data.py`).

This is detector-only AP50 (no verifier): verifiers are fitted per detector,
and a new detector needs its own before the full E14 protocol applies
(docs/TRAINING_PROGRAM.md, G1 acceptance 1). Only cat and dog are scored;
the subset has no verified labels for the other G1 classes.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..data import openimages as OI
from ..registry import check_model, resource
from ..utils import utc_now, write_json
from ..vision.detector import EfficientDetDetector, load_detector
from . import e4_visual as E4

BASELINES = ["efficientdet_lite0_tfhub_rawdecode_v1", "efficientdet_lite0_mediapipe_v1"]


def _digest(r: dict) -> dict:
    out = {}
    for c, v in r["classes"].items():
        il = v["image_level"]
        out[c] = {"ap50": v["ap50"], "ap50_ci95": [v["ap50_ci95"]["ci_low"], v["ap50_ci95"]["ci_high"]], "image_recall_at_op": il["recall"],
                  "image_specificity_at_op": il["specificity_on_verified_negatives"], "box_recall_at_op": v["recall_at_op"],
                  "box_precision_at_op": v["precision_at_op"], "small_box_recall_at_op": v["recall_breakdown_at_op"]["area_lt_1pct"]["recall"]}
    out["inference_ms_per_image_including_jpeg_decode"] = r["inference_ms_per_image_including_jpeg_decode"]
    return out


def run(settings: Settings, out_path: Path, models: dict[str, Path]) -> dict:
    man, data = settings.paths.resolve("manifests_root"), settings.paths.resolve("data_root")
    images, boxes = OI.read_subset(man)
    img_dir = OI.dataset_dir(data) / "images"
    res = {name: E4.evaluate_detector(EfficientDetDetector(p, settings.runtime.num_threads), images, boxes, img_dir, settings.vision.score_threshold)
           for name, p in models.items()}
    for mid in BASELINES:
        r = check_model(settings, mid)
        if r.ready:
            det = load_detector(Path(next(iter(r.files.values()))), settings.runtime.num_threads, resource(mid).get("decode"))
            res[mid] = E4.evaluate_detector(det, images, boxes, img_dir, settings.vision.score_threshold)
    for v in res.values():
        v.pop("per_image_predictions")
    diffs = {f"{name}-{mid}/{c}": res[name]["classes"][c]["ap50"] - res[mid]["classes"][c]["ap50"]
             for name in models for mid in BASELINES if mid in res for c in E4.CLASSES}
    out = {"check": "G1 export on the E4 subset (detector only, E4 protocol)", "created_at": utc_now().isoformat(timespec="seconds"),
           "images": len(images), "models": {k: str(v) for k, v in models.items()}, "operating_threshold": settings.vision.score_threshold,
           "results": {k: _digest(v) for k, v in res.items()}, "ap50_difference_point": diffs,
           "protocol": __doc__.strip().splitlines()}
    write_json(out_path, out)
    return out
