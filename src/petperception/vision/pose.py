"""Pose-provider interface and visibility-aware pose features.

Dogs: ``vision/keypoints.py`` provides 20 annotated keypoints (research
profile only; see E9). Cats: no cat keypoint data or model is available, so
the cat provider reports itself unavailable and every cat pose feature is
returned as ``available: false`` with a reason. Features are only
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
    unavailable_reason: str | None = "no verified pose model is installed for this species"

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
    # Ear-tip height above the ear base, in units of the distance between the two ear
    # bases (positive = tip above base). Dominated by breed ear type (erect vs floppy):
    # only changes within one animal over time are informative.
    "ear_tip_elevation": ("left_ear_base", "left_ear_tip", "right_ear_base", "right_ear_tip"),
    # Tail-end height above the tail start, in units of nose-to-tail-start length.
    "tail_elevation": ("tail_start", "tail_end", "nose"),
    # Forelegs folded (front elbows near the front paws) while hind legs stay extended:
    # a geometric play-bow *candidate* (dogs only; never validated on behavior labels).
    "play_bow_candidate": ("front_left_paw", "front_right_paw", "front_left_elbow", "front_right_elbow",
                           "rear_left_paw", "rear_right_paw", "rear_left_elbow", "rear_right_elbow"),
    # Left/right swings of the tail end relative to the tail start over time.
    "tail_lateral_oscillation": ("tail_start", "tail_end"),
}


def _pair(ks: KeypointSet, a: str, b: str, min_vis: float) -> tuple[np.ndarray, np.ndarray] | None:
    pa, pb = ks.get(a, min_vis), ks.get(b, min_vis)
    return (pa, pb) if pa is not None and pb is not None else None


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
        if name == "ear_tip_elevation":
            vals = []
            for ks in series:
                bases = _pair(ks, "left_ear_base", "right_ear_base", min_vis)
                width = float(np.linalg.norm(bases[0] - bases[1])) if bases else 0.0
                per = []
                for side in ("left", "right"):
                    pr = _pair(ks, f"{side}_ear_base", f"{side}_ear_tip", min_vis)
                    if pr is not None and width > 1e-4:
                        per.append(float(pr[0][1] - pr[1][1]) / width)
                if per:
                    vals.append(float(np.mean(per)))
            out.append(FeatureValue(name, bool(vals), float(np.median(vals)) if vals else None, "ear_base_widths_above_base",
                                    None if vals else "ear keypoints below visibility threshold", list(req)))
        elif name == "tail_elevation":
            vals = []
            for ks in series:
                tail, body = _pair(ks, "tail_start", "tail_end", min_vis), _pair(ks, "nose", "tail_start", min_vis)
                if tail is not None and body is not None:
                    L = float(np.linalg.norm(body[0] - body[1])) or 1e-6
                    vals.append(float(tail[0][1] - tail[1][1]) / L)  # positive = end above start (y down)
            out.append(FeatureValue(name, bool(vals), float(np.median(vals)) if vals else None, "body_lengths_above_tail_start",
                                    None if vals else "tail or nose keypoints below visibility threshold", list(req)))
        elif name == "play_bow_candidate":
            flags = []
            for ks in series:
                pts = {k: ks.get(k, min_vis) for k in req}
                if all(v is not None for v in pts.values()):
                    front = np.mean([pts[f"front_{s}_paw"][1] - pts[f"front_{s}_elbow"][1] for s in ("left", "right")])
                    rear = np.mean([pts[f"rear_{s}_paw"][1] - pts[f"rear_{s}_elbow"][1] for s in ("left", "right")])
                    flags.append(bool(rear > 0 and front < 0.35 * rear))
            out.append(FeatureValue(name, bool(flags), any(flags) if flags else None, "boolean_candidate",
                                    None if flags else "leg keypoints below visibility threshold", list(req)))
        elif name == "tail_lateral_oscillation":
            xs = []
            for ks in series:
                pr = _pair(ks, "tail_start", "tail_end", min_vis)
                if pr is not None:
                    xs.append(float(pr[1][0] - pr[0][0]))
            ok = len(xs) >= 4
            sign_changes = int(np.sum(np.diff(np.sign(np.array(xs))) != 0)) if ok else None
            out.append(FeatureValue(name, ok, sign_changes, "sign_changes_of_tail_end_offset",
                                    None if ok else "needs >= 4 visible tail estimates (and sufficient frame rate)", list(req)))
    return out
