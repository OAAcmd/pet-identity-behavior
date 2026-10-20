"""Transparent rule baselines, written before the E27/E28 test data were scored and never tuned.

Each rule reads interpretable cues only (YAMNet class scores, COCO objects
seen by the detector, species of the detected animal, person-animal
interaction) and applies them in a fixed order. The thresholds are a-priori
round values: a YAMNet class counts as heard when its score reaches 0.2 in
any 0.96 s window (the order of the product's sound-event thresholds); a COCO
object counts as seen when detected (score >= the product's 0.35) in at least
2 sampled frames. The SHA-256 of this file is recorded with every result.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

HEARD = 0.2
SEEN_FRAMES = 2

WATER = ("Water", "Water tap, faucet", "Sink (filling or washing)", "Bathtub (filling or washing)", "Splash, splatter", "Pour", "Drip",
         "Fill (with liquid)", "Liquid")
GROOMING_TOOLS_SOUND = ("Hair dryer", "Electric shaver, electric razor", "Scissors")
GROOMING_TOOLS_SEEN = ("scissors", "hair drier", "toothbrush")
BATHROOM_SEEN = ("sink", "toilet")
STREET_SOUND = ("Vehicle", "Car", "Motor vehicle (road)", "Traffic noise, roadway noise", "Car passing by", "Outside, urban or manmade",
                "Outside, rural or natural", "Wind", "Wind noise (microphone)", "Bird vocalization, bird call, bird song")
STREET_SEEN = ("car", "bicycle", "motorcycle", "bus", "truck", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench")
DOG_SOUND = ("Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)")
CAT_SOUND = ("Cat", "Purr", "Meow", "Hiss", "Caterwaul")
DOOR_SOUND = ("Doorbell", "Ding-dong", "Knock", "Door", "Sliding door")
WHIMPER = ("Whimper (dog)", "Whimper", "Crying, sobbing")
REST_SOUND = ("Snoring", "Breathing")
TOY_SOUND = ("Squeak",)
FOOD_SOUND = ("Dishes, pots, and pans", "Cutlery, silverware", "Chewing, mastication", "Chopping (food)", "Frying (food)", "Microwave oven")
PEOPLE_PLAY = ("Laughter", "Children playing", "Children shouting", "Speech")


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _heard(score_max: dict[str, float], names: tuple[str, ...]) -> bool:
    return any(score_max.get(n, 0.0) >= HEARD for n in names)


def _seen(object_frames: dict[str, int], names: tuple[str, ...]) -> bool:
    return any(object_frames.get(n, 0) >= SEEN_FRAMES for n in names)


def kinetics_situation(score_max: dict[str, float] | None, object_frames: dict[str, int] | None, dog_fraction: float | None,
                       pet_frames: int) -> tuple[str, list[str]]:
    """E28 rule baseline -> (one of the six Kinetics labels, reasons). Missing channels are passed as None."""
    sm = score_max or {}
    of = object_frames or {}
    why: list[str] = []
    if pet_frames > 0 and dog_fraction is not None:
        dog = dog_fraction >= 0.5
        why.append(f"species from detector ({'dog' if dog else 'cat'} in {dog_fraction:.0%} of animal frames)")
    elif sm:
        dog = max(sm.get(n, 0.0) for n in DOG_SOUND) >= max(sm.get(n, 0.0) for n in CAT_SOUND)
        why.append(f"species from sound ({'dog' if dog else 'cat'})")
    else:
        dog = True
        why.append("species unknown; dog assumed (4 of 6 classes)")
    if not dog:
        if _heard(sm, GROOMING_TOOLS_SOUND) or _seen(of, GROOMING_TOOLS_SEEN):
            return "grooming cat", why + ["grooming tool heard or seen"]
        return "petting cat", why + ["default for cats"]
    if _heard(sm, WATER) or _seen(of, BATHROOM_SEEN):
        return "bathing dog", why + ["water heard or sink/toilet seen"]
    if _heard(sm, GROOMING_TOOLS_SOUND) or _seen(of, GROOMING_TOOLS_SEEN):
        return "grooming dog", why + ["grooming tool heard or seen"]
    if _heard(sm, STREET_SOUND) or _seen(of, STREET_SEEN):
        return "walking the dog", why + ["street or outdoor cue heard or seen"]
    return "training dog", why + ["default for dogs"]


def barkopedia_activity(score_max: dict[str, float]) -> tuple[str, list[str]]:
    """E27 rule baseline for the dog audio channel (eight Barkopedia activities)."""
    sm = score_max
    if _heard(sm, WATER):
        return "taking shower", ["water heard"]
    if _heard(sm, DOOR_SOUND):
        return "alerting to sounds", ["door sound heard"]
    if _heard(sm, FOOD_SOUND):
        return "begging for food", ["kitchen or chewing sound heard"]
    if _heard(sm, TOY_SOUND):
        return "playing with toy", ["squeak heard"]
    if _heard(sm, WHIMPER):
        return "seeking attention", ["whimper heard"]
    if _heard(sm, REST_SOUND) or max(sm.get(n, 0.0) for n in DOG_SOUND) < HEARD:
        return "rest", ["snoring/breathing heard or no clear dog vocalization"]
    if _heard(sm, PEOPLE_PLAY):
        return "playing with human", ["people heard with the dog"]
    if sm.get("Growling", 0.0) >= HEARD:
        return "playing with other animals", ["growling (play growls with other dogs are common)"]
    return "alerting to sounds", ["barking without other cues"]


def score_max_by_name(scores: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Max over windows per YAMNet class name. scores: [W, 521] or [1042] (mean|max layout)."""
    s = np.asarray(scores, np.float32)
    m = s.max(axis=0) if s.ndim == 2 else s[len(labels):]
    return {n: float(v) for n, v in zip(labels, m)}
