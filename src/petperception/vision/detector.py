"""Cat/dog detection with the MediaPipe EfficientDet-Lite TFLite build.

Decoding reproduces MediaPipe's TensorsToDetections (YXHW box format, fixed
anchors and scales read from the embedded DETECTOR_METADATA, exponential box
sizes). Scores already pass an in-graph LOGISTIC op. Non-maximum suppression
is per class. Boxes are normalized image coordinates (x1, y1, x2, y2) in
[0, 1] of the frame as given (the frame is stretched to the square model
input, as MediaPipe does, so normalized coordinates map back directly).

Class map: COCO category slots from the embedded labels.txt (90 slots with
'???' placeholders). Only explicitly configured classes are returned.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..eval.metrics import iou_xyxy
from .tflite_io import dequantize, encode_image
from .tflite_meta import DetectorDecoding, associated_files, input_normalization, mediapipe_detector_options


@dataclass
class Detection:
    label: str
    score: float
    box: np.ndarray  # normalized x1, y1, x2, y2
    # Raw-anchor builds only: all class scores of the anchor that produced the box
    # (used by the false-positive verifier; empty for post-processed builds).
    class_scores: np.ndarray | None = field(default=None, repr=False)

    @property
    def area(self) -> float:
        b = self.box
        return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_out: int) -> list[int]:
    order = list(np.argsort(-scores))
    keep: list[int] = []
    while order and len(keep) < max_out:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        ious = iou_xyxy(boxes[i:i + 1], boxes[order])[0]
        order = [j for j, v in zip(order, ious) if v <= iou_thr]
    return keep


# G1 detectors (training/g1, Open Images class names) name two classes differently from the
# COCO label maps of the other detectors; the engines, verifier and relation rules use the COCO names.
LABEL_ALIASES = {"table": "dining table", "teddy_bear": "teddy bear"}


def label_list(text: bytes) -> list[str]:
    """Detector label map from the model's embedded label file, with LABEL_ALIASES applied."""
    return [LABEL_ALIASES.get(x.strip(), x.strip()) for x in text.decode().splitlines()]


class EfficientDetDetector:
    name = "efficientdet_lite_mediapipe"

    def __init__(self, model_path: Path, num_threads: int = 2):
        data = model_path.read_bytes()
        norm = input_normalization(data)
        if norm is None:
            raise RuntimeError("input NormalizationOptions missing from model metadata")
        files = associated_files(data)
        if "labels.txt" not in files:
            raise RuntimeError("labels.txt missing from model metadata")
        self._setup(model_path, data, mediapipe_detector_options(data), label_list(files["labels.txt"]), norm, num_threads)

    @classmethod
    def from_postprocessed(cls, model_path: Path, num_threads: int = 2) -> EfficientDetDetector:
        """A detector with in-graph ``TFLite_Detection_PostProcess`` (TF Hub build), with that op
        removed and its raw outputs decoded here - the same path the phone uses (LiteRT.js
        cannot run the op). NMS is then per class like the MediaPipe builds, not the op's
        class-agnostic top-25, so results differ from the in-graph variant (evaluated in E14)."""
        from ..ondevice.tflite_patch import strip_detection_postprocess

        src = model_path.read_bytes()
        data, anchors, opt = strip_detection_postprocess(src)
        norm = input_normalization(src)
        if norm is None:
            raise RuntimeError("input NormalizationOptions missing from model metadata")
        files = associated_files(src)
        lab = files.get("labelmap.txt") or files.get("labels.txt")
        if lab is None:
            raise RuntimeError("label map missing from model metadata")
        dec = DetectorDecoding(anchors=anchors[:, [1, 0, 3, 2]].astype(np.float32),  # (y, x, h, w) -> (x, y, w, h)
                               num_classes=int(opt["num_classes"]), num_boxes=int(anchors.shape[0]), num_coords=4,
                               x_scale=float(opt["x_scale"]), y_scale=float(opt["y_scale"]), w_scale=float(opt["w_scale"]),
                               h_scale=float(opt["h_scale"]), apply_exponential_on_box_size=True, sigmoid_score=False)
        det = cls.__new__(cls)
        det._setup(model_path, data, dec, label_list(lab), norm, num_threads)
        det.model_bytes = data
        return det

    def _setup(self, model_path: Path, data: bytes, decoding, labels: list[str], norm, num_threads: int) -> None:
        from ai_edge_litert.interpreter import Interpreter

        self.model_path = model_path
        self.decoding = decoding
        self.norm_mean, self.norm_std = float(norm[0][0]), float(norm[1][0])
        self.labels = labels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.interp = Interpreter(model_content=data, num_threads=num_threads)
        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        self.in_detail = inp
        self.in_idx = inp["index"]
        self.in_h, self.in_w = int(inp["shape"][1]), int(inp["shape"][2])
        outs = self.interp.get_output_details()
        box_o = next(o for o in outs if o["shape"][-1] == self.decoding.num_coords)
        score_o = next(o for o in outs if o["shape"][-1] == self.decoding.num_classes)
        self.box_idx, self.score_idx = box_o["index"], score_o["index"]
        self.box_q, self.score_q = box_o["quantization"], score_o["quantization"]
        if len(self.decoding.anchors) != self.decoding.num_boxes:
            raise RuntimeError("anchor count does not match num_boxes")

    def class_map(self) -> dict[int, str]:
        return {i: l for i, l in enumerate(self.labels) if l != "???"}

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        d = self.decoding
        a = d.anchors
        yc = raw[:, 0] / d.y_scale * a[:, 3] + a[:, 1]
        xc = raw[:, 1] / d.x_scale * a[:, 2] + a[:, 0]
        if d.apply_exponential_on_box_size:
            h = np.exp(raw[:, 2] / d.h_scale) * a[:, 3]
            w = np.exp(raw[:, 3] / d.w_scale) * a[:, 2]
        else:
            h = raw[:, 2] / d.h_scale * a[:, 3]
            w = raw[:, 3] / d.w_scale * a[:, 2]
        return np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)

    def detect(self, rgb: np.ndarray, classes: list[str], score_threshold: float, iou_thr: float = 0.5,
               max_per_class: int = 10) -> list[Detection]:
        img = cv2.resize(rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        self.interp.set_tensor(self.in_idx, encode_image(img, self.in_detail, self.norm_mean, self.norm_std))
        self.interp.invoke()
        raw_boxes = dequantize(self.interp.get_tensor(self.box_idx)[0], self.box_q)
        scores = dequantize(self.interp.get_tensor(self.score_idx)[0], self.score_q)
        if self.decoding.sigmoid_score:
            scores = 1.0 / (1.0 + np.exp(-scores))
        boxes = np.clip(self._decode(raw_boxes), 0.0, 1.0)
        out: list[Detection] = []
        for cls in classes:
            if cls not in self.labels:
                raise ValueError(f"class '{cls}' not in detector label map")
            ci = self.labels.index(cls)
            s = scores[:, ci]
            cand = np.where(s >= score_threshold)[0]
            if cand.size == 0:
                continue
            keep = nms(boxes[cand], s[cand], iou_thr, max_per_class)
            out.extend(Detection(cls, float(s[cand[k]]), boxes[cand[k]].copy(), scores[cand[k]].copy()) for k in keep)
        out.sort(key=lambda d: -d.score)
        return out


class PostprocessedDetector:
    """TFLite Task Library detector with in-graph ``TFLite_Detection_PostProcess``.

    Used for the TF Hub ``lite-model_efficientdet_lite0_detection_metadata_1``
    build, whose embedded model metadata states the Apache-2.0 license.
    Interface (verified on the file): uint8 RGB input (quantization scale
    1/128, zero point 127, i.e. raw pixel values), outputs boxes
    ``[1, N, 4]`` as (ymin, xmin, ymax, xmax), class indices, scores and a
    detection count. The in-graph NMS is class-agnostic with one class per
    box and N = 25, so low-scoring alternatives for the same region are not
    available (lower AP resolution than the raw-anchor builds).
    """

    name = "efficientdet_lite0_tfhub_postprocessed"

    def __init__(self, model_path: Path, num_threads: int = 2):
        from ai_edge_litert.interpreter import Interpreter

        data = model_path.read_bytes()
        self.model_path = model_path
        files = associated_files(data)
        lab = files.get("labelmap.txt") or files.get("labels.txt")
        if lab is None:
            raise RuntimeError("label map missing from model metadata")
        self.labels = [l.strip() for l in lab.decode().splitlines()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        if inp["dtype"] != np.uint8:
            raise RuntimeError(f"expected uint8 input, got {inp['dtype']}")
        self.in_idx = inp["index"]
        self.in_h, self.in_w = int(inp["shape"][1]), int(inp["shape"][2])
        outs = self.interp.get_output_details()
        self.box_idx = next(o["index"] for o in outs if len(o["shape"]) == 3 and o["shape"][-1] == 4)
        flat = [o for o in outs if len(o["shape"]) == 2]
        count = [o for o in outs if len(o["shape"]) == 1]
        if len(flat) != 2 or len(count) != 1:
            raise RuntimeError("unexpected output layout for a post-processed detector")
        self.count_idx = count[0]["index"]
        # Class and score tensors share a shape; the Task Library metadata order is
        # (location, category, score, count), which maps to the names :3, :2, :1, :0.
        flat.sort(key=lambda o: o["name"], reverse=True)
        self.class_idx, self.score_idx = flat[0]["index"], flat[1]["index"]

    def class_map(self) -> dict[int, str]:
        return {i: l for i, l in enumerate(self.labels) if l != "???"}

    def detect(self, rgb: np.ndarray, classes: list[str], score_threshold: float, iou_thr: float = 0.5,
               max_per_class: int = 10) -> list[Detection]:
        img = cv2.resize(rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        self.interp.set_tensor(self.in_idx, np.ascontiguousarray(img[None], dtype=np.uint8))
        self.interp.invoke()
        n = int(self.interp.get_tensor(self.count_idx)[0])
        boxes = self.interp.get_tensor(self.box_idx)[0][:n]
        cls = self.interp.get_tensor(self.class_idx)[0][:n].astype(int)
        scores = self.interp.get_tensor(self.score_idx)[0][:n]
        for c in classes:
            if c not in self.labels:
                raise ValueError(f"class '{c}' not in detector label map")
        want = {self.labels.index(c): c for c in classes}
        out: list[Detection] = []
        per: dict[str, int] = {}
        for b, k, s in zip(boxes, cls, scores):
            if k in want and s >= score_threshold and per.get(want[k], 0) < max_per_class:
                ymin, xmin, ymax, xmax = (float(v) for v in b)
                out.append(Detection(want[k], float(s), np.clip(np.array([xmin, ymin, xmax, ymax]), 0.0, 1.0)))
                per[want[k]] = per.get(want[k], 0) + 1
        out.sort(key=lambda d: -d.score)
        return out


def detect_at_operating_point(detector, rgb: np.ndarray, classes: list[str], vision, max_detections: int | None = None) -> list:
    """Detections at the configured operating point: ``vision.score_threshold`` for every class, or the per-class
    ``vision.class_score_thresholds`` where given (the detector runs at the lowest of them, then each class is cut at
    its own point)."""
    per = vision.class_score_thresholds or {}
    lo = min([vision.score_threshold, *per.values()]) if per else vision.score_threshold
    dets = detector.detect(rgb, classes, lo, vision.nms_iou, max_detections or vision.max_detections_per_frame)
    return [d for d in dets if d.score >= per.get(d.label, vision.score_threshold)] if per else dets


def load_detector(model_path: Path, num_threads: int = 2, decode: str | None = None) -> EfficientDetDetector | PostprocessedDetector:
    """Pick the decoder that matches the model's output layout. ``decode="strip_postprocess"``
    (set per model in manifests/resources.json) removes in-graph post-processing and decodes
    the raw tensors, as on the phone."""
    if decode == "strip_postprocess":
        return EfficientDetDetector.from_postprocessed(model_path, num_threads)
    from ai_edge_litert.interpreter import Interpreter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe = Interpreter(model_path=str(model_path))
    n_out = len(probe.get_output_details())
    del probe
    return PostprocessedDetector(model_path, num_threads) if n_out == 4 else EfficientDetDetector(model_path, num_threads)
