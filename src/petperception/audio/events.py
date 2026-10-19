"""Turn frame-level YAMNet tags into timestamped sound events.

Species groups are derived from the AudioSet ontology (not hand-picked
display names): ``cat_vocalization`` = Cat and its descendants,
``dog_vocalization`` = Dog and its descendants. "Growling" is a child of both
Cat and Dog in the ontology, so it is excluded from species group scores and
reported as the species-ambiguous ``growl`` subtype.

Scores stay raw sigmoid scores (``score_kind = yamnet_sigmoid_raw``). The
decision thresholds were selected on ESC-50 development folds (E1) and do not
make the scores calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..utils import read_json
from .yamnet import WindowScores, YamnetBackend

CAT_MID = "/m/01yrx"
DOG_MID = "/m/0bt9lr"
GROWLING_MID = "/m/0ghcn6"

# Subtypes the fusion layer may reference. Keys are stable identifiers.
SUBTYPE_MIDS = {
    "meow": "/m/07qrkrw",
    "purr": "/m/02yds9",
    "hiss": "/m/07rjwbb",
    "caterwaul": "/m/0ch8v",
    "bark": "/m/05tny_",
    "yip": "/m/07r_k2n",
    "howl": "/m/07qf0zm",
    "bow_wow": "/m/07rc7d9",
    "whimper_dog": "/m/07srf8z",
    "growl": GROWLING_MID,
}

# Environmental context tags (observations only, never intent labels).
CONTEXT_TAG_MIDS = {
    "speech": "/m/09x0r",
    "door": "/m/02dgv",
    "doorbell": "/m/03wwcy",
    "knock": "/m/07r4wb8",
    "television": "/m/07c52",
    "music": "/m/04rlf",
    "vacuum_cleaner": "/m/0d31p",
    "dishes_pots_pans": "/m/04brg2",
    "cutlery": "/m/023pjk",
    "thunder": "/m/0ngt1",
    "fireworks": "/m/0g6b5",
    "alarm": "/m/07pp_mv",
    "siren": "/m/03kmc9",
    "baby_cry": "/t/dd00002",
}


@dataclass
class GroupSpec:
    name: str
    indices: list[int]
    labels: list[str]


@dataclass
class SoundEventRaw:
    label: str  # cat_vocalization | dog_vocalization | context:<tag> | growl
    start_s: float
    end_s: float
    peak_score: float
    mean_score: float
    n_windows: int
    threshold: float
    subtypes: dict[str, float] = field(default_factory=dict)  # subtype -> peak raw score in event
    top_audioset_labels: list[tuple[str, float]] = field(default_factory=list)


# A subtype becomes a fusion tag when its peak score inside a species (or growl) event reaches this value.
SUBTYPE_MIN = 0.3
# Subtypes that make a vocalization request-like (food, outside, attention candidates).
REQUEST_LIKE = {"meow", "caterwaul", "bark", "yip", "bow_wow", "howl", "whimper_dog"}
TAG_SUBTYPES = {"hiss", "growl", "purr", "bark", "howl"}


SubtypeStatus = str  # passed | failed | not_evaluable | not_evaluated


def load_subtype_validation(path: Path | None) -> dict[str, SubtypeStatus]:
    """Validation status per vocalization subtype (E21 gate, artifacts/audio_subtype_validation.json).

    Without the file no subtype counts as validated, so none is shown or used
    as an interpretation trigger.
    """
    if not path or not path.exists():
        return {}
    return {k: v["status"] for k, v in read_json(path)["subtypes"].items()}


def shown_subtypes(validation: dict[str, SubtypeStatus]) -> set[str]:
    """Subtypes that passed the pre-set gate (F1 >= 0.7, lower 95 % bound >= 0.6) and may be shown."""
    return {k for k, v in validation.items() if v == "passed"}


def vocal_tags(label: str, subtypes: dict[str, float], shown: set[str] | None = None) -> set[str]:
    """Fusion tags of one species or growl event (the rule both engines apply).

    A species-group event without any confident subtype, or with a request-like
    one, is a request-like "vocalization"; hiss, growl and purr alone are not.
    Only subtypes in `shown` (validated, see `shown_subtypes`) become tags;
    `shown=None` means every subtype counts (detection-only analyses such as E21).
    Unvalidated confident subtypes still keep an event from counting as a request.
    """
    strong = {k for k, v in subtypes.items() if v >= SUBTYPE_MIN}
    eligible = TAG_SUBTYPES if shown is None else TAG_SUBTYPES & shown
    tags = strong & eligible
    if label != "growl" and (strong & REQUEST_LIKE or not strong):
        tags.add("vocalization")
    if label == "growl" and (shown is None or "growl" in shown):
        tags.add("growl")
    return tags


def vocal_conflict_tags(label: str, subtypes: dict[str, float], shown: set[str]) -> set[str]:
    """Confident subtypes that are not validated: they may block a hypothesis (a possible growl
    rules out "relaxed") but never open or support one."""
    strong = {k for k, v in subtypes.items() if v >= SUBTYPE_MIN}
    out = (strong & TAG_SUBTYPES) - shown
    if label == "growl" and "growl" not in shown:
        out.add("growl")
    return out


def build_groups(backend: YamnetBackend) -> dict[str, GroupSpec]:
    groups: dict[str, GroupSpec] = {}
    for name, root in (("cat_vocalization", CAT_MID), ("dog_vocalization", DOG_MID)):
        mids = [m for m in backend.descendants(root) if m != GROWLING_MID and m in backend.mid_to_index]
        idx = [backend.mid_to_index[m] for m in mids]
        groups[name] = GroupSpec(name, idx, [backend.labels[i] for i in idx])
    return groups


def group_scores(ws: WindowScores, groups: dict[str, GroupSpec]) -> dict[str, np.ndarray]:
    return {g: (ws.scores[:, spec.indices].max(axis=1) if ws.scores.size else np.zeros(0)) for g, spec in groups.items()}


def load_thresholds(path: Path | None, defaults: dict[str, float]) -> tuple[dict[str, float], dict[str, str]]:
    """Per-label thresholds and their source (data-selected file vs. config default)."""
    th = dict(defaults)
    src = {k: "config default (not selected on data)" for k in defaults}
    if path and path.exists():
        data = read_json(path)
        for k, v in data["thresholds"].items():
            th[k] = float(v["threshold"])
            src[k] = f"{path.name}: {data.get('selection_method', 'unspecified')}"
    return th, src


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(mask) - 1))
    return segs


def extract_events(ws: WindowScores, backend: YamnetBackend, groups: dict[str, GroupSpec],
                   thresholds: dict[str, float], context_threshold: float, min_gap_s: float) -> list[SoundEventRaw]:
    """Merge above-threshold windows into intervals per group/tag."""
    if ws.scores.shape[0] == 0:
        return []
    events: list[SoundEventRaw] = []
    gs = group_scores(ws, groups)
    sub_idx = {k: backend.mid_to_index[m] for k, m in SUBTYPE_MIDS.items() if m in backend.mid_to_index}
    series: dict[str, tuple[np.ndarray, float]] = {g: (gs[g], thresholds[g]) for g in groups}
    series["growl"] = (ws.scores[:, sub_idx["growl"]], thresholds.get("growl", 0.5))
    for tag, mid in CONTEXT_TAG_MIDS.items():
        if mid in backend.mid_to_index:
            series[f"context:{tag}"] = (ws.scores[:, backend.mid_to_index[mid]], context_threshold)
    for label, (s, th) in series.items():
        segs = _segments(s >= th)
        merged: list[tuple[int, int]] = []
        for a, b in segs:
            if merged and ws.t_start[a] - ws.t_end[merged[-1][1]] <= min_gap_s:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        for a, b in merged:
            sl = slice(a, b + 1)
            ev = SoundEventRaw(
                label=label, start_s=float(ws.t_start[a]), end_s=float(ws.t_end[b]),
                peak_score=float(s[sl].max()), mean_score=float(s[sl].mean()), n_windows=b - a + 1, threshold=float(th),
            )
            if label in groups or label == "growl":
                ev.subtypes = {k: float(ws.scores[sl, i].max()) for k, i in sub_idx.items() if ws.scores[sl, i].max() >= 0.1}
                top = np.argsort(ws.scores[sl].max(axis=0))[::-1][:5]
                ev.top_audioset_labels = [(backend.labels[i], float(ws.scores[sl, i].max())) for i in top]
            events.append(ev)
    events.sort(key=lambda e: (e.start_s, e.label))
    return events
