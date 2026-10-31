"""Turning-in-place (circling / tail chasing) recognizer from the animal's track over time (E32).

Frame-level vision-language scores miss most tail chasing (E31: 3 of 8 clips): a
spinning animal looks like an ordinary animal in any single frame, the detector
loses it in motion blur, and a curled body resembles resting or kneading. What
does separate circling is its time course: the animal stays in place while its
box shape oscillates (side view <-> end view twice per turn), the horizontal
motion inside its box flips direction, the detector drops frames, and its
appearance repeats with a period of about 0.5-2 s. Window features (2 s,
hop 0.5 s) describe that course; a small logistic regression, fitted on the
E31 collection (development data only), turns them into a per-window
probability. Clip score = mean of the top quarter of window probabilities; bouts =
merged windows above the threshold. Stored as JSON (no pickles).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import MOTION_FPS

WIN = 12  # samples (2 s at 6 fps)
HOP = 3  # samples (0.5 s)
FEATURES = ("detected", "filled", "displacement", "path", "speed", "omega_abs", "omega_net", "dx_flips", "aspect_cv", "area_cv", "periodicity",
            "zs_circling_max", "zs_circling_mean", "zs_neutral_mean")
MODEL_VERSION = "circling-v1"


def _periodicity(app: np.ndarray) -> float:
    good = ~np.isnan(app[:, 0])
    if good.sum() < 8:
        return 0.0
    A = app[good].astype(np.float32)
    A = (A - A.mean(0)) / (A.std(0) + 1e-6)
    curve = [float(np.mean(np.sum(A[:-lag] * A[lag:], axis=1)) / A.shape[1]) for lag in range(1, min(9, len(A) - 2))]
    if len(curve) < 3:
        return 0.0
    c = np.array(curve)
    k = int(np.argmax(c[1:])) + 1  # best repeat after the first lag
    return float(max(0.0, c[k] - c[: k + 1].min()))


def window_features(z, zs_frames: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """-> (window start times [W], features [W, len(FEATURES)]); zs_frames = (times, circling prob, neutral prob) at 2 fps."""
    if "box" not in z.files or len(z["box"]) < WIN:
        return np.zeros(0, np.float32), np.zeros((0, len(FEATURES)), np.float32)
    box, det, flow, app = z["box"], z["det"], z["flow"], z["app"].astype(np.float32)
    T = len(box)
    t = np.arange(T) / MOTION_FPS
    rows, starts = [], []
    for s in range(0, T - WIN + 1, HOP):
        e = s + WIN
        b = box[s:e]
        ok = ~np.isnan(b[:, 0])
        detected = float(np.mean(det[s:e, 0] > 0))
        filled = float(ok.mean())
        if ok.sum() >= 3:
            w = b[ok, 2] - b[ok, 0]
            h = b[ok, 3] - b[ok, 1]
            size = float(np.sqrt(np.mean(w * h))) + 1e-3
            ctr = np.stack([(b[ok, 0] + b[ok, 2]) / 2, (b[ok, 1] + b[ok, 3]) / 2], 1)
            disp = float(np.linalg.norm(ctr[-1] - ctr[0])) / size
            path = float(np.sum(np.linalg.norm(np.diff(ctr, axis=0), axis=1))) / size
            asp = w / np.maximum(h, 1e-3)
            aspect_cv = float(np.std(asp) / (np.mean(asp) + 1e-6))
            area = w * h
            area_cv = float(np.std(area) / (np.mean(area) + 1e-6))
        else:
            disp = path = aspect_cv = area_cv = 0.0
        f = flow[s:e]
        f = f[~np.isnan(f[:, 0])]
        if len(f) >= 2:
            speed = float(np.mean(f[:, 0]))
            omega_abs = float(np.mean(np.abs(f[:, 1])))
            omega_net = float(abs(np.mean(f[:, 1])))
            dx = f[:, 2]
            sig = dx[np.abs(dx) > 0.02]
            dx_flips = float(np.sum(np.diff(np.sign(sig)) != 0)) / WIN if len(sig) > 1 else 0.0
        else:
            speed = omega_abs = omega_net = dx_flips = 0.0
        per = _periodicity(app[s:e])
        zmax = zmean = nmean = 0.0
        if zs_frames is not None and len(zs_frames[0]):
            zt, zc, zn = zs_frames
            m = (zt >= t[s]) & (zt < t[e - 1] + 1e-6)
            if m.any():
                zmax, zmean, nmean = float(zc[m].max()), float(zc[m].mean()), float(zn[m].mean())
        rows.append([detected, filled, disp, path, speed, omega_abs, omega_net, dx_flips, aspect_cv, area_cv, per, zmax, zmean, nmean])
        starts.append(t[s])
    return np.array(starts, np.float32), np.array(rows, np.float32)


@dataclass
class CirclingModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    threshold: float
    meta: dict = field(default_factory=dict)

    def window_probs(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        zlin = ((X - self.mean) / self.scale) @ self.coef + self.intercept
        return 1 / (1 + np.exp(-zlin))

    @staticmethod
    def clip_score(p: np.ndarray) -> float:
        if len(p) == 0:
            return 0.0
        top = np.sort(p)[-max(2, len(p) // 4):]  # top quarter: one noisy window in a long clip must not decide (development finding)
        return float(top.mean())

    def bouts(self, starts: np.ndarray, p: np.ndarray) -> list[tuple[float, float, float]]:
        out: list[list[float]] = []
        for s, v in zip(starts, p):
            if v < self.threshold:
                continue
            e = float(s) + WIN / MOTION_FPS
            if out and s <= out[-1][1]:
                out[-1][1] = e
                out[-1][2] = max(out[-1][2], float(v))
            else:
                out.append([float(s), e, float(v)])
        return [tuple(b) for b in out]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": MODEL_VERSION, "features": list(FEATURES), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                                    "coef": self.coef.tolist(), "intercept": self.intercept, "threshold": self.threshold, **self.meta}, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> CirclingModel:
        m = json.loads(Path(path).read_text())
        if tuple(m["features"]) != FEATURES:
            raise ValueError("circling model was fitted on a different feature list")
        meta = {k: v for k, v in m.items() if k not in ("features", "mean", "scale", "coef", "intercept", "threshold")}
        return cls(np.array(m["mean"]), np.array(m["scale"]), np.array(m["coef"]), float(m["intercept"]), float(m["threshold"]), meta)


def fit(X: np.ndarray, y: np.ndarray, C: float, sample_weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    mean, scale = X.mean(0), X.std(0)
    scale[scale < 1e-8] = 1.0
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    clf.fit((X - mean) / scale, y, sample_weight=sample_weight)
    return mean, scale, clf.coef_[0], float(clf.intercept_[0])
