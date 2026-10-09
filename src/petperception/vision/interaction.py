"""Person-pet interaction from detection-box geometry (E13).

A logistic regression over 12 geometry values of a (person box, pet box) pair
plus a species indicator, fitted on Open Images visual relationships
("holds" / "interacts with"; unrelated pairs are reliable negatives). It is a
still-image model: in video it scores each sampled frame, and runs of frames
above the threshold become ``person_interaction_candidate`` segments. The
temporal grouping is not validated, cats were not evaluated (too few scorable
pairs), and nothing is inferred about the animal's response.
Same definitions in web/src/engine/interaction.ts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

GEOMETRY = ["iou", "inter_over_pet", "inter_over_person", "dx", "dy", "log_area_ratio", "person_aspect", "pet_aspect",
            "gap", "pet_center_inside_person", "person_area", "pet_area"]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def geometry(person: np.ndarray, pet: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Boxes as normalized (x1, y1, x2, y2)."""
    h, w = image_hw
    pw, ph = max(person[2] - person[0], 1e-4), max(person[3] - person[1], 1e-4)
    qw, qh = max(pet[2] - pet[0], 1e-4), max(pet[3] - pet[1], 1e-4)
    ix = max(0.0, min(person[2], pet[2]) - max(person[0], pet[0]))
    iy = max(0.0, min(person[3], pet[3]) - max(person[1], pet[1]))
    inter = ix * iy
    pcx, pcy = (person[0] + person[2]) / 2, (person[1] + person[3]) / 2
    qcx, qcy = (pet[0] + pet[2]) / 2, (pet[1] + pet[3]) / 2
    gx = max(0.0, max(person[0], pet[0]) - min(person[2], pet[2]))
    gy = max(0.0, max(person[1], pet[1]) - min(person[3], pet[3]))
    return np.array([
        _iou(person, pet), inter / (qw * qh), inter / (pw * ph), (qcx - pcx) / pw, (qcy - pcy) / ph, np.log((qw * qh) / (pw * ph)),
        np.log((pw * w) / (ph * h)), np.log((qw * w) / (qh * h)), np.hypot(gx, gy) / np.sqrt(pw * ph),
        float(person[0] <= qcx <= person[2] and person[1] <= qcy <= person[3]), pw * ph, qw * qh], dtype=np.float64)


@dataclass
class InteractionModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    threshold: float
    version: str

    @classmethod
    def load(cls, path: Path) -> InteractionModel:
        with np.load(path, allow_pickle=False) as z:
            if str(z["features"]) != "geometry":
                raise RuntimeError(f"{path.name}: only the geometry feature set is supported at runtime")
            return cls(z["mean"], z["scale"], z["coef"], float(np.ravel(z["intercept"])[0]), float(z["threshold"]), str(z["version"]))

    def score(self, person: np.ndarray, pet: np.ndarray, image_hw: tuple[int, int], species: str) -> float:
        x = np.concatenate([geometry(person, pet, image_hw), [1.0, 0.0] if species == "cat" else [0.0, 1.0]])
        z = float(((x - self.mean) / self.scale) @ self.coef + self.intercept)
        return float(1.0 / (1.0 + np.exp(-z)))
