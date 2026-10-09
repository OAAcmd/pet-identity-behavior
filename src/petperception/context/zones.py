"""User-configured scene zones (e.g. where the food bowl or the door is).

Zones describe the *current scene*. They are context evidence, never
supervised intent labels: an animal standing in a ``food_area`` zone is an
observation, not a food request.

Zone file format (JSON)::

    {
      "schema": "petperception.zones/1",
      "coordinate_space": "normalized_image",   # x right, y down, both in [0, 1]
      "anchor": "bottom_center",                 # point of the box tested for membership
      "zones": [
        {"zone_id": "bowl", "label": "Food bowl corner", "semantic_tag": "food_area",
         "polygon": [[0.0, 0.6], [0.3, 0.6], [0.3, 1.0], [0.0, 1.0]]}
      ]
    }
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

SemanticTag = Literal["food_area", "water_area", "exit_door", "rest_area", "play_area", "litter_area", "window", "other"]


class Zone(BaseModel):
    zone_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,64}$")
    label: str = Field(max_length=120)
    semantic_tag: SemanticTag
    polygon: list[tuple[float, float]]

    @field_validator("polygon")
    @classmethod
    def _valid_polygon(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(v) < 3:
            raise ValueError("polygon needs at least 3 points")
        for x, y in v:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("polygon coordinates must be normalized to [0, 1]")
        return v


class ZoneConfig(BaseModel):
    schema_: Literal["petperception.zones/1"] = Field("petperception.zones/1", alias="schema")
    coordinate_space: Literal["normalized_image"] = "normalized_image"
    anchor: Literal["bottom_center", "center"] = "bottom_center"
    zones: list[Zone] = Field(default_factory=list, max_length=32)

    model_config = {"populate_by_name": True}


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def anchor_point(box: np.ndarray, anchor: str) -> tuple[float, float]:
    x = float((box[0] + box[2]) / 2)
    y = float(box[3]) if anchor == "bottom_center" else float((box[1] + box[3]) / 2)
    return x, y


def occupancy(times: list[float], boxes: list[np.ndarray], cfg: ZoneConfig, max_gap_s: float) -> list[dict]:
    """Per-zone dwell intervals for one track (sampled detections only)."""
    out = []
    for z in cfg.zones:
        inside = [point_in_polygon(*anchor_point(b, cfg.anchor), z.polygon) for b in boxes]
        intervals: list[list[float]] = []
        for t, ins in zip(times, inside):
            if not ins:
                continue
            if intervals and t - intervals[-1][1] <= max_gap_s:
                intervals[-1][1] = t
            else:
                intervals.append([t, t])
        n_in = int(sum(inside))
        if n_in:
            out.append({
                "zone_id": z.zone_id, "semantic_tag": z.semantic_tag, "label": z.label,
                "intervals_s": [(a, b) for a, b in intervals],
                "dwell_s": float(sum(b - a for a, b in intervals)),
                "fraction_of_track_detections": n_in / len(boxes),
                "anchor": cfg.anchor,
            })
    return out
