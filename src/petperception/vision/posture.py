"""Coarse posture (lying / sitting / standing-or-walking) for cats and dogs.

Model: multinomial logistic regression on the pooled EfficientNet-Lite0
embedding of the square animal crop plus box shape, fitted by E10 on WEAK
labels parsed from Localized Narratives captions (Open Images train split).
Scores are uncalibrated; a label is reported only when the top score passes
a gate chosen on the development split, otherwise the frame is "uncertain".
Agreement figures are agreement with caption-derived labels, whose own noise
was not measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

POSTURES = ("lying", "sitting", "standing")
SPECIES = ("cat", "dog")


def posture_features(pooled: np.ndarray, box: np.ndarray, image_hw: tuple[int, int], species: str) -> np.ndarray:
    h, w = image_hw
    bw, bh = max(float(box[2] - box[0]), 1e-4), max(float(box[3] - box[1]), 1e-4)
    shape = [np.log((bw * w) / (bh * h)), np.log(bw * bh)]
    sp = [1.0 if species == s else 0.0 for s in SPECIES]
    return np.concatenate([pooled.astype(np.float64), shape, sp])


@dataclass
class PostureModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # [3, D]
    intercept: np.ndarray  # [3]
    gate: float
    version: str

    @classmethod
    def load(cls, path: Path) -> PostureModel:
        with np.load(path, allow_pickle=False) as z:
            return cls(z["mean"], z["scale"], z["coef"], z["intercept"], float(z["gate"]), str(z["version"]))

    def scores(self, x: np.ndarray) -> np.ndarray:
        z = ((x - self.mean) / self.scale) @ self.coef.T + self.intercept
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def predict(self, pooled: np.ndarray, box: np.ndarray, image_hw: tuple[int, int], species: str) -> tuple[str, float]:
        """(label or 'uncertain', top score)."""
        p = self.scores(posture_features(pooled, box, image_hw, species))
        k = int(np.argmax(p))
        return (POSTURES[k] if p[k] >= self.gate else "uncertain"), float(p[k])
