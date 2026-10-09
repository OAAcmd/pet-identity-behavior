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
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..eval.metrics import iou_xyxy
from .tflite_meta import associated_files, input_normalization, mediapipe_detector_options


@dataclass
class Detection:
    label: str
    score: float
    box: np.ndarray  # normalized x1, y1, x2, y2

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


class EfficientDetDetector:
    name = "efficientdet_lite_mediapipe"

    def __init__(self, model_path: Path, num_threads: int = 2):
        from ai_edge_litert.interpreter import Interpreter

        data = model_path.read_bytes()
        self.model_path = model_path
        self.decoding = mediapipe_detector_options(data)
        norm = input_normalization(data)
        if norm is None:
            raise RuntimeError("input NormalizationOptions missing from model metadata")
        self.norm_mean, self.norm_std = float(norm[0][0]), float(norm[1][0])
        files = associated_files(data)
        if "labels.txt" not in files:
            raise RuntimeError("labels.txt missing from model metadata")
        self.labels = [l.strip() for l in files["labels.txt"].decode().splitlines()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        self.in_idx = inp["index"]
        self.in_h, self.in_w = int(inp["shape"][1]), int(inp["shape"][2])
        outs = self.interp.get_output_details()
        self.box_idx = next(o["index"] for o in outs if o["shape"][-1] == self.decoding.num_coords)
        self.score_idx = next(o["index"] for o in outs if o["shape"][-1] == self.decoding.num_classes)
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
        x = ((img.astype(np.float32) - self.norm_mean) / self.norm_std)[None]
        self.interp.set_tensor(self.in_idx, x)
        self.interp.invoke()
        raw_boxes = self.interp.get_tensor(self.box_idx)[0]
        scores = self.interp.get_tensor(self.score_idx)[0]
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
            out.extend(Detection(cls, float(s[cand[k]]), boxes[cand[k]].copy()) for k in keep)
        out.sort(key=lambda d: -d.score)
        return out
