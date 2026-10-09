"""Second-stage crop verifier for cat/dog detections (false-positive rejection).

Stage 1 (the detector) proposes boxes. Stage 2 crops each box with context,
classifies the crop with an ImageNet classifier (EfficientNet-Lite0, TFLite,
Apache-2.0 per its embedded metadata) and summarizes the 1000-way softmax
into coarse groups (domestic cat, dog, wild cat, wild canid, bear, other
animal, teddy bear, non-animal). A per-class logistic regression on those
group masses plus the detector score and box geometry produces a
**verification score**.

The verification score is a ranking/rejection score fitted on the Open
Images *test-split* tuning set. It is not a calibrated probability and must
not be displayed as one. Thresholds are chosen on the tuning set only (see
``eval/e7_fp_rejection.py``); final numbers come from the untouched E4
validation subset.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..eval.metrics import iou_xyxy
from .detector import Detection
from .tflite_io import dequantize, encode_image
from .tflite_meta import associated_files, input_normalization

# Group index ranges in ImageNet-1k order (labels without background). The
# endpoints are checked by name when the model loads.
_RANGES = {
    "dog": (151, 268, "Chihuahua", "Mexican hairless"),
    "wild_canid": (269, 280, "timber wolf", "grey fox"),
    "cat": (281, 285, "tabby", "Egyptian cat"),
    "wild_cat": (286, 293, "cougar", "cheetah"),
    "bear": (294, 297, "brown bear", "sloth bear"),
}
_ANIMAL_END = 397  # indices 0..397 are animals in ImageNet-1k
_TEDDY = (850, "teddy")
GROUPS = ["cat", "dog", "wild_cat", "wild_canid", "bear", "other_animal", "teddy", "non_animal"]
EPS = 1e-4


@dataclass
class CropScores:
    groups: dict[str, float]
    top1: float


class CropClassifier:
    """EfficientNet-Lite0 ImageNet classifier applied to detection crops."""

    name = "efficientnet_lite0_imagenet"

    def __init__(self, model_path: Path, num_threads: int = 2, margin: float = 0.15):
        from ai_edge_litert.interpreter import Interpreter

        data = model_path.read_bytes()
        norm = input_normalization(data)
        if norm is None:
            raise RuntimeError("input NormalizationOptions missing from classifier metadata")
        self.mean, self.std = float(norm[0][0]), float(norm[1][0])
        files = associated_files(data)
        name = next((k for k in files if k.startswith("labels")), None)
        if name is None:
            raise RuntimeError("label file missing from classifier metadata")
        self.labels = [line.strip() for line in files[name].decode().splitlines()]
        if len(self.labels) != 1000:
            raise RuntimeError(f"expected 1000 ImageNet labels, found {len(self.labels)}")
        for g, (a, b, la, lb) in _RANGES.items():
            if self.labels[a] != la or self.labels[b] != lb:
                raise RuntimeError(f"ImageNet label order differs from expectation for group {g}")
        if self.labels[_TEDDY[0]] != _TEDDY[1]:
            raise RuntimeError("ImageNet label order differs from expectation (teddy)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        self.in_detail = inp
        self.in_idx, self.in_h, self.in_w = inp["index"], int(inp["shape"][1]), int(inp["shape"][2])
        out = self.interp.get_output_details()[0]
        self.out_idx, self.out_q = out["index"], out["quantization"]
        self.margin = margin
        masks = {g: np.zeros(1000, bool) for g in GROUPS}
        for g, (a, b, _, _) in _RANGES.items():
            masks[g][a:b + 1] = True
        masks["teddy"][_TEDDY[0]] = True
        animal = np.zeros(1000, bool)
        animal[: _ANIMAL_END + 1] = True
        named = np.zeros(1000, bool)
        for g in ("cat", "dog", "wild_cat", "wild_canid", "bear"):
            named |= masks[g]
        masks["other_animal"] = animal & ~named
        masks["non_animal"] = ~animal & ~masks["teddy"]
        self.masks = masks

    def crop(self, rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box)
        bw, bh = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
        x1, x2 = max(0.0, x1 - self.margin * bw), min(1.0, x2 + self.margin * bw)
        y1, y2 = max(0.0, y1 - self.margin * bh), min(1.0, y2 + self.margin * bh)
        c = rgb[int(y1 * h): max(int(y2 * h), int(y1 * h) + 1), int(x1 * w): max(int(x2 * w), int(x1 * w) + 1)]
        return cv2.resize(c, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA if min(c.shape[:2]) > self.in_h else cv2.INTER_LINEAR)

    def scores(self, rgb: np.ndarray, box: np.ndarray) -> CropScores:
        self.interp.set_tensor(self.in_idx, encode_image(self.crop(rgb, box), self.in_detail, self.mean, self.std))
        self.interp.invoke()
        p = dequantize(self.interp.get_tensor(self.out_idx)[0], self.out_q).astype(np.float64)
        p = p / max(p.sum(), 1e-12)
        return CropScores({g: float(p[m].sum()) for g, m in self.masks.items()}, float(p.max()))


def feature_vector(det_score: float, box: np.ndarray, cs: CropScores, image_hw: tuple[int, int]) -> np.ndarray:
    s = min(max(det_score, EPS), 1 - EPS)
    h, w = image_hw
    bw, bh = max(float(box[2] - box[0]), 1e-4), max(float(box[3] - box[1]), 1e-4)
    return np.array([np.log(s / (1 - s)), *[np.log(cs.groups[g] + EPS) for g in GROUPS],
                     np.log(bw * bh), np.log((bw * w) / (bh * h)), np.log(cs.top1 + EPS)], dtype=np.float64)


# COCO label-map indices used by the anchor feature set (raw-anchor detector builds).
ANCHOR_CLASSES = {"person": 0, "bird": 15, "cat": 16, "dog": 17, "horse": 18, "sheep": 19, "cow": 20, "bear": 22, "teddy bear": 87}
FEATURE_SETS = ["crop", "redetect", "anchor", "crop+redetect", "all"]
_PARTS = {"crop": ["crop"], "redetect": ["redetect"], "anchor": ["anchor"], "crop+redetect": ["crop", "redetect"], "all": ["crop", "redetect", "anchor"]}


def parts_for(fset: str) -> list[str]:
    return _PARTS[fset]


def redetect(detector, rgb: np.ndarray, d: Detection, classes: list[str], margin: float = 0.25) -> tuple[float, float]:
    """Re-run the detector on the zoomed box: (same-class max score, other-class max score)."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in d.box)
    bw, bh = x2 - x1, y2 - y1
    cx1, cy1 = max(0.0, x1 - margin * bw), max(0.0, y1 - margin * bh)
    cx2, cy2 = min(1.0, x2 + margin * bw), min(1.0, y2 + margin * bh)
    crop = rgb[int(cy1 * h): max(int(cy2 * h), int(cy1 * h) + 2), int(cx1 * w): max(int(cx2 * w), int(cx1 * w) + 2)]
    inner = np.array([[(x1 - cx1) / max(cx2 - cx1, 1e-6), (y1 - cy1) / max(cy2 - cy1, 1e-6),
                       (x2 - cx1) / max(cx2 - cx1, 1e-6), (y2 - cy1) / max(cy2 - cy1, 1e-6)]])
    same, other = 0.0, 0.0
    for r in detector.detect(crop, classes, 0.05, 0.5, 5):
        if r.label == d.label and iou_xyxy(r.box[None], inner)[0, 0] >= 0.3:
            same = max(same, r.score)
        elif r.label != d.label:
            other = max(other, r.score)
    return same, other


def describe(detector, classifier: CropClassifier | None, rgb: np.ndarray, d: Detection, classes: list[str],
             need: set[str] | None = None) -> dict:
    """Raw verifier inputs for one detection (only the parts in ``need`` are computed)."""
    need = need or {"crop", "redetect", "anchor"}
    row: dict = {"label": d.label, "score": d.score, "box": [float(v) for v in d.box], "hw": list(rgb.shape[:2]),
                 "groups": None, "top1": None, "redetect": None, "anchor": None}
    if "crop" in need and classifier is not None:
        cs = classifier.scores(rgb, d.box)
        row["groups"], row["top1"] = cs.groups, cs.top1
    if "redetect" in need:
        row["redetect"] = list(redetect(detector, rgb, d, classes))
    if "anchor" in need and d.class_scores is not None:
        row["anchor"] = {k: float(d.class_scores[i]) for k, i in ANCHOR_CLASSES.items()}
    return row


def features(row: dict, fset: str) -> np.ndarray:
    """Feature vector for a feature set: detector logit + box geometry + the set's parts."""
    s = min(max(row["score"], EPS), 1 - EPS)
    h, w = row["hw"]
    b = row["box"]
    bw, bh = max(b[2] - b[0], 1e-4), max(b[3] - b[1], 1e-4)
    out = [np.log(s / (1 - s)), np.log(bw * bh), np.log((bw * w) / (bh * h))]
    for part in _PARTS[fset]:
        if part == "crop":
            out += [np.log(row["groups"][g] + EPS) for g in GROUPS] + [np.log(row["top1"] + EPS)]
        elif part == "redetect":
            out += [np.log(row["redetect"][0] + EPS), np.log(row["redetect"][1] + EPS)]
        else:
            a = row["anchor"]
            if a is None:
                raise ValueError("anchor features need a raw-anchor detector build")
            own, oth = (a["cat"], a["dog"]) if row["label"] == "cat" else (a["dog"], a["cat"])
            rivals = [a[k] for k in ("bird", "horse", "sheep", "cow", "bear", "teddy bear")]
            out += [np.log(oth + EPS), np.log(max(rivals) + EPS), np.log(a["person"] + EPS), np.log(own + EPS) - np.log(max([*rivals, oth]) + EPS)]
    return np.array(out, dtype=np.float64)


@dataclass
class ClassVerifier:
    feature_set: str
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    tau: float

    def score(self, x: np.ndarray) -> float:
        z = float(((x - self.mean) / self.scale) @ self.coef + self.intercept)
        return float(1.0 / (1.0 + np.exp(-z)))


class DetectionVerifier:
    """Applies the frozen per-class verifier (parameters exported by E7) to detections."""

    def __init__(self, detector, classifier: CropClassifier | None, params: dict, classes: list[str]):
        if params.get("schema") != "petperception.verifier/1":
            raise RuntimeError("unsupported verifier parameter file")
        self.detector, self.classifier, self.params, self.classes = detector, classifier, params, classes
        self.detector_id = params["detector_id"]
        self.per_class = {
            c: ClassVerifier(v["feature_set"], np.array(v["mean"]), np.array(v["scale"]), np.array(v["coef"]), float(v["intercept"]), float(v["tau"]))
            for c, v in params["classes"].items()
        }
        for c, cv in self.per_class.items():
            if "crop" in _PARTS[cv.feature_set] and classifier is None:
                raise RuntimeError(f"verifier for {c} needs the crop classifier")

    def verify(self, rgb: np.ndarray, det: Detection) -> tuple[float, bool] | None:
        """Return (verification score, accepted), or None when the class is not covered."""
        cv = self.per_class.get(det.label)
        if cv is None:
            return None
        row = describe(self.detector, self.classifier, rgb, det, self.classes, set(_PARTS[cv.feature_set]))
        v = cv.score(features(row, cv.feature_set))
        return v, v >= cv.tau
