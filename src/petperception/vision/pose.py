"""Pose-provider interface and visibility-aware pose features.

No verified cat/dog pose checkpoint is available in Phase 1 (see
manifests/resources.json: the Ultralytics dog-pose resources were not
reachable, and a generic human-pose model is not an animal pose model). The
default provider therefore reports itself unavailable, and every pose-derived
feature is returned as ``available: false`` with a reason. Features are only
computed from keypoints that a provider actually declares and that are
visible above a threshold; they are never estimated from a bounding box.

Coordinate convention: normalized image coordinates, x to the right, y down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class KeypointSet:
    t_s: float
    names: list[str]
    xy: np.ndarray  # [K, 2] normalized image coords
    visibility: np.ndarray  # [K] in [0, 1]

    def get(self, name: str, min_vis: float) -> np.ndarray | None:
        if name not in self.names:
            return None
        i = self.names.index(name)
        return self.xy[i] if self.visibility[i] >= min_vis else None


class PoseProvider(Protocol):
    name: str
    species: tuple[str, ...]
    keypoint_vocabulary: tuple[str, ...]
    available: bool
    unavailable_reason: str | None

    def estimate(self, rgb: np.ndarray, box: np.ndarray, t_s: float) -> KeypointSet | None: ...


@dataclass
class UnavailablePoseProvider:
    name: str = "none"
    species: tuple[str, ...] = ()
    keypoint_vocabulary: tuple[str, ...] = ()
    available: bool = False
    unavailable_reason: str | None = "no verified cat/dog pose checkpoint is installed (Phase 1)"

    def estimate(self, rgb: np.ndarray, box: np.ndarray, t_s: float) -> KeypointSet | None:
        return None


@dataclass
class FeatureValue:
    name: str
    available: bool
    value: float | bool | None = None
    units: str | None = None
    reason: str | None = None
    keypoints_used: list[str] = field(default_factory=list)


FEATURE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    # Angle of the ear (base->tip) relative to image vertical; needs ear keypoints.
    "ear_angle_deg": ("left_ear_base", "left_ear_tip"),
    # Tail-tip height relative to tail base, normalized by withers-to-tail-base length.
    "tail_height": ("tail_base", "tail_tip", "withers"),
    # Front lowered relative to hindquarters (dog play-bow *candidate* only).
    "play_bow_candidate": ("left_front_paw", "right_front_paw", "left_hip", "right_hip", "withers"),
    # Left/right oscillation of the tail tip over time relative to the tail base.
    "tail_lateral_oscillation": ("tail_base", "tail_tip"),
}


def _unavailable(name: str, provider: PoseProvider | None, missing: list[str] | None = None, reason: str | None = None) -> FeatureValue:
    if provider is None or not provider.available:
        reason = reason or (provider.unavailable_reason if provider else "no pose provider")
    elif missing:
        reason = f"keypoints not provided by '{provider.name}': {', '.join(missing)}"
    return FeatureValue(name, False, reason=reason)


def compute_features(provider: PoseProvider | None, series: list[KeypointSet], min_vis: float = 0.5) -> list[FeatureValue]:
    out: list[FeatureValue] = []
    vocab = set(provider.keypoint_vocabulary) if provider is not None else set()
    for name, req in FEATURE_REQUIREMENTS.items():
        missing = [k for k in req if k not in vocab]
        if provider is None or not provider.available or missing:
            out.append(_unavailable(name, provider, missing))
            continue
        if not series:
            out.append(FeatureValue(name, False, reason="no keypoint estimates for this track"))
            continue
        if name == "ear_angle_deg":
            vals = []
            for ks in series:
                b, t = ks.get("left_ear_base", min_vis), ks.get("left_ear_tip", min_vis)
                if b is not None and t is not None:
                    v = t - b
                    vals.append(float(np.degrees(np.arctan2(v[0], -v[1]))))
            out.append(FeatureValue(name, bool(vals), float(np.median(vals)) if vals else None, "degrees_from_image_vertical",
                                    None if vals else "keypoints below visibility threshold", list(req)))
        elif name == "tail_height":
            vals = []
            for ks in series:
                b, t, w = (ks.get(k, min_vis) for k in req)
                if b is not None and t is not None and w is not None:
                    L = float(np.linalg.norm(w - b)) or 1e-6
                    vals.append(float((b[1] - t[1]) / L))  # positive = tip above base (y down)
            out.append(FeatureValue(name, bool(vals), float(np.median(vals)) if vals else None, "back_lengths_above_tail_base",
                                    None if vals else "keypoints below visibility threshold", list(req)))
        elif name == "play_bow_candidate":
            flags = []
            for ks in series:
                pts = [ks.get(k, min_vis) for k in req]
                if all(p is not None for p in pts):
                    lf, rf, lh, rh, w = pts
                    front = (lf[1] + rf[1]) / 2
                    hips = (lh[1] + rh[1]) / 2
                    flags.append(bool(w[1] > hips + 0.05 and front > hips))
            out.append(FeatureValue(name, bool(flags), any(flags) if flags else None, "boolean_candidate",
                                    None if flags else "keypoints below visibility threshold", list(req)))
        elif name == "tail_lateral_oscillation":
            xs = []
            for ks in series:
                b, t = ks.get("tail_base", min_vis), ks.get("tail_tip", min_vis)
                if b is not None and t is not None:
                    xs.append(float(t[0] - b[0]))
            ok = len(xs) >= 4
            sign_changes = int(np.sum(np.diff(np.sign(np.array(xs))) != 0)) if ok else None
            out.append(FeatureValue(name, ok, sign_changes, "sign_changes_of_tip_offset",
                                    None if ok else "needs >= 4 visible tail estimates (and sufficient frame rate)", list(req)))
    return out
