"""Per-frame and per-detection quality limits reported as observations."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import VisionQualityConfig
from ..eval.metrics import iou_xyxy
from .tracking import AnimalDetection


@dataclass
class FrameQuality:
    t_s: float
    mean_luma: float
    laplacian_var: float
    low_light: bool
    blurry: bool


def frame_quality(t_s: float, rgb: np.ndarray, cfg: VisionQualityConfig) -> tuple[FrameQuality, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luma = float(gray.mean())
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return FrameQuality(t_s, luma, lap, luma < cfg.low_light_luma, lap < cfg.blur_laplacian_var), gray


def detection_flags(dets: list[AnimalDetection], cfg: VisionQualityConfig, border: float = 0.01) -> None:
    """Annotate detections in place with tiny/truncated/overlap flags."""
    boxes = np.array([d.box for d in dets]) if dets else np.zeros((0, 4))
    ious = iou_xyxy(boxes, boxes) if len(dets) > 1 else np.zeros((len(dets), len(dets)))
    for i, d in enumerate(dets):
        if d.area < cfg.tiny_area_fraction:
            d.flags.append("tiny_subject")
        b = d.box
        if b[0] <= border or b[1] <= border or b[2] >= 1 - border or b[3] >= 1 - border:
            d.flags.append("truncated_at_frame_edge")
        if len(dets) > 1 and np.any(np.delete(ious[i], i) > 0.05):
            d.flags.append("overlaps_other_animal_possible_occlusion")
