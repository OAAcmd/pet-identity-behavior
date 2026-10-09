"""Pet-object relations from detection-box geometry (E15).

One logistic regression per relation over the 12 E13 geometry values of an
(object box, pet box) pair (the object takes the reference role), a species
indicator and an object-group indicator, fitted on Open Images visual
relationships (data/openimages_vrd_objects.py). Relations: the pet is **on** a
bed, couch, chair or table; **under** a table or chair; **inside** a car.
Objects come from the product detector's COCO classes. Only relations that
passed the E15 gate are in the shipped artifact.

It is a still-image model: in video each sampled frame is scored and runs of
frames above the threshold become ``object_relation_candidate`` segments; the
temporal grouping is not validated, and nothing is inferred about whether the
animal is allowed there or why it is there.
Same definitions in web/src/engine/objectrelation.ts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .interaction import geometry

GROUPS = ("bed", "couch", "chair", "table", "car")
DETECTOR_CLASS = {"bed": "bed", "couch": "couch", "chair": "chair", "table": "dining table", "car": "car"}
GROUP_OF_CLASS = {v: k for k, v in DETECTOR_CLASS.items()}
RELATIONS = {"on": ("bed", "couch", "chair", "table"), "under": ("table", "chair"), "inside_of": ("car",)}
PHRASE = {"on": "on", "under": "under", "inside_of": "inside"}


def features(obj: np.ndarray, pet: np.ndarray, image_hw: tuple[int, int], species: str, group: str) -> np.ndarray:
    """Boxes as normalized (x1, y1, x2, y2): 12 geometry values, species (cat, dog), object group one-hot."""
    return np.concatenate([geometry(obj, pet, image_hw), [1.0, 0.0] if species == "cat" else [0.0, 1.0],
                           [1.0 if group == g else 0.0 for g in GROUPS]])


@dataclass
class RelationHead:
    relation: str
    groups: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    threshold: float

    def score(self, x: np.ndarray) -> float:
        z = float(((x - self.mean) / self.scale) @ self.coef + self.intercept)
        return float(1.0 / (1.0 + np.exp(-z)))


@dataclass
class ObjectRelationModel:
    heads: dict[str, RelationHead]
    version: str

    @classmethod
    def load(cls, path: Path) -> ObjectRelationModel:
        with np.load(path, allow_pickle=False) as z:
            heads = {}
            for r in [str(v) for v in z["relations"]]:
                if r not in RELATIONS:
                    raise RuntimeError(f"{path.name}: unknown relation {r}")
                heads[r] = RelationHead(r, tuple(str(g) for g in z[f"{r}_groups"]), z[f"{r}_mean"], z[f"{r}_scale"], z[f"{r}_coef"],
                                        float(np.ravel(z[f"{r}_intercept"])[0]), float(z[f"{r}_threshold"]))
            return cls(heads, str(z["version"]))

    @property
    def detector_classes(self) -> list[str]:
        return sorted({DETECTOR_CLASS[g] for h in self.heads.values() for g in h.groups})

    def best(self, objects: list[tuple[str, np.ndarray]], pet: np.ndarray, image_hw: tuple[int, int],
             species: str) -> dict[str, tuple[float, str]]:
        """Highest score per relation over the detected objects: {relation: (score, object group)}."""
        out: dict[str, tuple[float, str]] = {}
        for label, box in objects:
            g = GROUP_OF_CLASS.get(label)
            if g is None:
                continue
            x = features(box, pet, image_hw, species, g)
            for r, h in self.heads.items():
                if g in h.groups:
                    s = h.score(x)
                    if r not in out or s > out[r][0]:
                        out[r] = (s, g)
        return out
