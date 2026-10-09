"""Species merging, IoU and Kalman (SORT-style) trackers, camera-motion estimation and movement features.

Units: boxes and centroids are normalized image coordinates in [0, 1].
Speeds are reported as image diagonals per second and as box-size units per
second (displacement / sqrt(box area)); neither is a physical speed.
A track ID is a per-analysis label for a temporally linked set of boxes; it is
NOT a verified pet identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..eval.metrics import iou_xyxy
from .detector import Detection


@dataclass
class AnimalDetection:
    t_s: float
    frame_index: int
    box: np.ndarray  # normalized x1, y1, x2, y2
    species_scores: dict[str, float]  # e.g. {"cat": 0.39, "dog": 0.52}
    flags: list[str] = field(default_factory=list)
    source: Detection | None = field(default=None, repr=False)  # highest-scoring raw detection
    verify_score: float | None = None  # second-stage verifier score (uncalibrated)
    keypoints: object | None = field(default=None, repr=False)  # pose.KeypointSet when estimated
    posture: tuple[str, float] | None = None  # (label or "uncertain", uncalibrated top score)

    @property
    def species(self) -> str:
        return max(self.species_scores, key=self.species_scores.get)

    @property
    def score(self) -> float:
        return max(self.species_scores.values())

    @property
    def area(self) -> float:
        b = self.box
        return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])


def merge_species(dets: list[Detection], t_s: float, frame_index: int, iou_thr: float = 0.7) -> list[AnimalDetection]:
    """Merge heavily overlapping cat/dog boxes into one detection with per-species scores."""
    out: list[AnimalDetection] = []
    for d in sorted(dets, key=lambda x: -x.score):
        merged = False
        for a in out:
            if iou_xyxy(a.box[None], d.box[None])[0, 0] >= iou_thr:
                a.species_scores[d.label] = max(a.species_scores.get(d.label, 0.0), d.score)
                merged = True
                break
        if not merged:
            out.append(AnimalDetection(t_s, frame_index, d.box.copy(), {d.label: d.score}, source=d))
    for a in out:
        s = sorted(a.species_scores.values(), reverse=True)
        if len(s) > 1 and s[1] >= 0.6 * s[0]:
            a.flags.append("species_ambiguous")
    return out


@dataclass
class CameraMotion:
    t_from: float
    t_to: float
    dx: float  # normalized translation of background (fraction of width)
    dy: float
    scale: float
    inlier_ratio: float
    reliable: bool


def estimate_camera_motion(prev_gray: np.ndarray, gray: np.ndarray, masks: list[np.ndarray],
                           t_from: float, t_to: float) -> CameraMotion | None:
    """Global similarity transform from ORB matches on the background.

    Animal boxes (``masks`` in normalized coords) are excluded so the animal's
    own motion does not count as camera motion. Returns None when there are
    too few features (e.g. featureless or dark scenes): motion is then unknown.
    """
    h, w = gray.shape
    mask = np.full((h, w), 255, np.uint8)
    for b in masks:
        x1, y1, x2, y2 = (b * np.array([w, h, w, h])).astype(int)
        mask[max(0, y1):y2, max(0, x1):x2] = 0
    orb = cv2.ORB_create(nfeatures=500)
    k1, d1 = orb.detectAndCompute(prev_gray, mask)
    k2, d2 = orb.detectAndCompute(gray, mask)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 12:
        return None
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inl is None:
        return None
    ratio = float(inl.mean())
    scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    return CameraMotion(t_from, t_to, float(M[0, 2] / w), float(M[1, 2] / h), scale, ratio, reliable=ratio >= 0.5 and int(inl.sum()) >= 10)


@dataclass
class Track:
    track_id: str
    detections: list[AnimalDetection] = field(default_factory=list)
    missed_frames: int = 0
    verification: tuple[int, float, float, bool] | None = None  # (checked, median score, threshold, accepted)

    @property
    def first_t(self) -> float:
        return self.detections[0].t_s

    @property
    def last_t(self) -> float:
        return self.detections[-1].t_s

    def species_summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for sp in {s for d in self.detections for s in d.species_scores}:
            vals = np.array([d.species_scores.get(sp, 0.0) for d in self.detections])
            out[sp] = {"mean": float(vals.mean()), "max": float(vals.max()), "frames_top": int(sum(d.species == sp for d in self.detections))}
        return out

    @property
    def species(self) -> str:
        s = self.species_summary()
        return max(s, key=lambda k: s[k]["mean"])


class IouTracker:
    """Greedy-optimal (Hungarian) IoU association with a time-based gap limit."""

    def __init__(self, iou_threshold: float = 0.3, max_gap_s: float = 1.5):
        self.iou_threshold = iou_threshold
        self.max_gap_s = max_gap_s
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next = 1

    def update(self, t_s: float, dets: list[AnimalDetection], shift: tuple[float, float] = (0.0, 0.0)) -> None:
        still = []
        for tr in self.active:
            (still if t_s - tr.last_t <= self.max_gap_s else self.finished).append(tr)
        self.active = still
        if self.active and dets:
            prev = np.array([tr.detections[-1].box for tr in self.active])
            prev = prev + np.array([shift[0], shift[1], shift[0], shift[1]])  # camera-motion prediction
            cost = 1.0 - iou_xyxy(prev, np.array([d.box for d in dets]))
            rows, cols = linear_sum_assignment(cost)
            used_t, used_d = set(), set()
            for r, c in zip(rows, cols):
                if 1.0 - cost[r, c] >= self.iou_threshold:
                    self.active[r].detections.append(dets[c])
                    used_t.add(r)
                    used_d.add(c)
            for r, tr in enumerate(self.active):
                if r not in used_t:
                    tr.missed_frames += 1
            new = [d for i, d in enumerate(dets) if i not in used_d]
        else:
            for tr in self.active:
                tr.missed_frames += 1
            new = dets
        for d in new:
            self.active.append(Track(f"track-{self._next}", [d]))
            self._next += 1

    def tracks(self, min_hits: int) -> list[Track]:
        all_tracks = self.finished + self.active
        return sorted((t for t in all_tracks if len(t.detections) >= min_hits), key=lambda t: t.first_t)


class KalmanTracker:
    """SORT-style tracker: constant-velocity Kalman filter on (cx, cy, w, h), time-based.

    At the low sampling rates used here (default 2 fps) an animal can move more
    than its own width between samples, so plain IoU association breaks tracks.
    The filter predicts each track forward by the actual time step; association
    uses IoU with the predicted box, and also accepts a pair whose centers are
    within ``center_gate`` predicted-box sizes when IoU is too low (fast motion).
    Same interface and Track objects as IouTracker.
    """

    def __init__(self, iou_threshold: float = 0.3, max_gap_s: float = 1.5, center_gate: float = 1.0,
                 accel_std: float = 0.5, meas_std: float = 0.02):
        self.iou_threshold = iou_threshold
        self.max_gap_s = max_gap_s
        self.center_gate = center_gate
        self.accel_std = accel_std  # normalized units / s^2
        self.meas_std = meas_std  # normalized units
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._kf: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        self._next = 1
        self.H = np.hstack([np.eye(4), np.zeros((4, 2))])

    @staticmethod
    def _z(box: np.ndarray) -> np.ndarray:
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])

    @staticmethod
    def _box(x: np.ndarray) -> np.ndarray:
        w, h = max(x[2], 1e-4), max(x[3], 1e-4)
        return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])

    def _predict(self, tid: str, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        x, P, t0 = self._kf[tid]
        dt = max(t_s - t0, 0.0)
        F = np.eye(6)
        F[0, 4] = F[1, 5] = dt
        q = self.accel_std ** 2
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [0, 0], [0, 0], [dt, 0], [0, dt]])
        Q = G @ G.T * q + np.diag([0, 0, (0.05 * dt) ** 2, (0.05 * dt) ** 2, 0, 0])
        return F @ x, F @ P @ F.T + Q

    def _init(self, tid: str, d: AnimalDetection) -> None:
        x = np.concatenate([self._z(d.box), [0.0, 0.0]])
        P = np.diag([self.meas_std ** 2] * 4 + [0.5 ** 2, 0.5 ** 2])
        self._kf[tid] = (x, P, d.t_s)

    def _correct(self, tid: str, x: np.ndarray, P: np.ndarray, d: AnimalDetection) -> None:
        R = np.eye(4) * self.meas_std ** 2
        y = self._z(d.box) - self.H @ x
        S = self.H @ P @ self.H.T + R
        Kg = P @ self.H.T @ np.linalg.inv(S)
        self._kf[tid] = (x + Kg @ y, (np.eye(6) - Kg @ self.H) @ P, d.t_s)

    def update(self, t_s: float, dets: list[AnimalDetection], shift: tuple[float, float] = (0.0, 0.0)) -> None:
        still = []
        for tr in self.active:
            (still if t_s - tr.last_t <= self.max_gap_s else self.finished).append(tr)
        self.active = still
        preds = {}
        for tr in self.active:
            x, P = self._predict(tr.track_id, t_s)
            x = x.copy()
            x[0] += shift[0]
            x[1] += shift[1]
            preds[tr.track_id] = (x, P)
        used_d: set[int] = set()
        if self.active and dets:
            pb = np.array([self._box(preds[tr.track_id][0]) for tr in self.active])
            db = np.array([d.box for d in dets])
            iou = iou_xyxy(pb, db)
            cost = 1.0 - iou
            size = np.sqrt(np.maximum((pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1]), 1e-6))
            pc = (pb[:, :2] + pb[:, 2:]) / 2
            dc = (db[:, :2] + db[:, 2:]) / 2
            dist = np.linalg.norm(pc[:, None, :] - dc[None, :, :], axis=-1) / size[:, None]
            weak = (iou < self.iou_threshold) & (dist <= self.center_gate)
            cost = np.where(weak, 1.0 - self.iou_threshold + 0.01 + 0.01 * dist, cost)
            ok = (iou >= self.iou_threshold) | weak
            cost = np.where(ok, cost, 10.0)
            rows, cols = linear_sum_assignment(cost)
            used_t = set()
            for r, c in zip(rows, cols):
                if ok[r, c]:
                    tr = self.active[r]
                    tr.detections.append(dets[c])
                    self._correct(tr.track_id, *preds[tr.track_id], dets[c])
                    used_t.add(r)
                    used_d.add(c)
            for r, tr in enumerate(self.active):
                if r not in used_t:
                    tr.missed_frames += 1
        else:
            for tr in self.active:
                tr.missed_frames += 1
        for i, d in enumerate(dets):
            if i in used_d:
                continue
            tid = f"track-{self._next}"
            self._next += 1
            self.active.append(Track(tid, [d]))
            self._init(tid, d)

    def tracks(self, min_hits: int) -> list[Track]:
        all_tracks = self.finished + self.active
        return sorted((t for t in all_tracks if len(t.detections) >= min_hits), key=lambda t: t.first_t)


def make_tracker(method: str, iou_threshold: float, max_gap_s: float) -> IouTracker | KalmanTracker:
    if method == "kalman":
        return KalmanTracker(iou_threshold, max_gap_s)
    if method == "iou":
        return IouTracker(iou_threshold, max_gap_s)
    raise ValueError(f"unknown tracker method '{method}'")


def movement_summary(track: Track, frame_w: int, frame_h: int, camera: dict[float, tuple[float, float]] | None,
                     moving_threshold: float = 0.05) -> dict:
    """Movement statistics with explicit units; camera translation subtracted if known."""
    dets = track.detections
    diag = float(np.hypot(frame_w, frame_h))
    if len(dets) < 2:
        return {"available": False, "reason": "fewer than two detections", "units": "image_diagonals_per_second"}
    speeds, rel_speeds, path = [], [], 0.0
    compensated = 0
    for a, b in zip(dets[:-1], dets[1:]):
        dt = b.t_s - a.t_s
        if dt <= 0:
            continue
        d = b.center - a.center
        if camera is not None and b.t_s in camera:
            cdx, cdy = camera[b.t_s]
            d = d - np.array([cdx, cdy])
            compensated += 1
        dist = float(np.hypot(d[0] * frame_w, d[1] * frame_h)) / diag
        path += dist
        speeds.append(dist / dt)
        size_px = np.sqrt(max(a.area, 1e-9) * frame_w * frame_h)  # geometric-mean box side in pixels
        rel_speeds.append(float(np.hypot(d[0] * frame_w, d[1] * frame_h)) / size_px / dt)
    if not speeds:
        return {"available": False, "reason": "non-increasing timestamps", "units": "image_diagonals_per_second"}
    s = np.array(speeds)
    areas = np.array([d.area for d in dets])
    return {
        "available": True,
        "units": "image_diagonals_per_second",
        "median_speed": float(np.median(s)),
        "p90_speed": float(np.percentile(s, 90)),
        "path_length_diagonals": path,
        "moving_fraction": float(np.mean(s > moving_threshold)),
        "moving_threshold": moving_threshold,
        "median_speed_box_units_per_second": float(np.median(rel_speeds)),
        "box_area_change_ratio": float(areas[-1] / max(areas[0], 1e-9)),
        "camera_compensated_steps": compensated,
        "steps": len(speeds),
    }
