"""Context observed in the recording, and what happens before and after a behaviour (E32).

Kept apart from what the user writes: every item names the instrument that
produced it (detector, sound model, scene prompts) and its score, and is
labelled "from the recording". Recording cues can reorder candidate
explanations, with half the weight of the user's own words, and are listed
separately in the answer. Thresholds are fixed, not fitted: detector >= 0.5,
person-animal overlap >= 0.2, sound classes >= 0.3, place >= 0.5 probability.
"""

from __future__ import annotations

import numpy as np

from .features import CONTEXT_CLASSES, MOTION_FPS

DET_MIN, NEAR_MIN, SOUND_MIN, PLACE_MIN = 0.5, 0.2, 0.3, 0.5
OBJECTS = {"bowl": ("a food or water bowl", "bowl food"), "sports ball": ("a ball", "toy ball"), "frisbee": ("a frisbee or toy", "toy"),
           "teddy bear": ("a soft toy", "toy"), "bed": ("a bed", "bed"), "couch": ("a sofa", "sofa"), "toilet": ("a toilet", "bathroom"),
           "sink": ("a sink", "bathroom"), "car": ("a car", "car")}
SOUNDS = {"laughter": (("Laughter", "Giggle", "Chuckle, chortle", "Belly laugh", "Snicker", "Baby laughter"), "people laughing", "laugh"),
          "speech": (("Speech", "Child speech, kid speaking", "Conversation"), "people talking", "talking"),
          "doorbell": (("Doorbell", "Ding-dong", "Knock"), "a doorbell or knock", "doorbell door"),
          "tv_music": (("Television", "Music"), "TV or music", "")}
PLACES = {"living room": "a photo of a living room", "kitchen": "a photo of a kitchen", "bathroom": "a photo of a bathroom",
          "bedroom": "a photo of a bedroom", "garden or yard": "a photo of a garden or a yard", "park or street": "a photo of a park or a street",
          "car": "a photo of the inside of a car", "veterinary clinic": "a photo of a veterinary clinic"}
PLACE_CUES = {"kitchen": "kitchen food", "garden or yard": "outdoor outside", "park or street": "outdoor outside", "car": "car"}
VOCAL_TYPES = {"cat": ("Meow", "Purr", "Hiss", "Caterwaul", "Growling"), "dog": ("Bark", "Howl", "Whimper (dog)", "Growling", "Yip")}


class PlaceModel:
    def __init__(self, encoder):
        self.encoder = encoder
        self.names = list(PLACES)
        self.T = encoder.texts([PLACES[n] for n in self.names])

    def probs(self, scenes: np.ndarray) -> dict[str, float]:
        logits = self.encoder.scale * (np.asarray(scenes, np.float32) @ self.T.T) + self.encoder.bias
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return dict(zip(self.names, p.mean(axis=0).tolist()))


def _times(mask: np.ndarray, t: np.ndarray) -> list[list[float]]:
    out: list[list[float]] = []
    for i in np.where(mask)[0]:
        if out and t[i] - out[-1][1] <= 1.0:
            out[-1][1] = float(t[i]) + 1 / MOTION_FPS
        else:
            out.append([float(t[i]), float(t[i]) + 1 / MOTION_FPS])
    return [[round(a, 1), round(b, 1)] for a, b in out]


def recording_context(z, yam_labels: list[str] | None, places: PlaceModel | None) -> tuple[list[dict], str]:
    """-> (items, cue text). Items: {"what", "from", "score", "times_s"}."""
    items: list[dict] = []
    cues: list[str] = []
    if "ctx" in z.files and len(z["ctx"]):
        ctx = z["ctx"]
        t = np.arange(len(ctx)) / MOTION_FPS
        col = {c: j for j, c in enumerate(CONTEXT_CLASSES)}
        near = z["ctx_person_near"] if "ctx_person_near" in z.files else np.zeros(len(ctx))
        if (near >= NEAR_MIN).any():
            items.append({"what": "a person close to the animal", "from": "detector (person box overlapping the animal)", "score": round(float(near.max()), 2),
                          "times_s": _times(near >= NEAR_MIN, t)})
            cues.append("person near")
        elif (ctx[:, col["person"]] >= DET_MIN).any():
            items.append({"what": "a person in view", "from": "detector", "score": round(float(ctx[:, col["person"]].max()), 2),
                          "times_s": _times(ctx[:, col["person"]] >= DET_MIN, t)})
        for k, (desc, cue) in OBJECTS.items():
            v = ctx[:, col[k]]
            if (v >= DET_MIN).mean() >= 0.1:
                items.append({"what": desc, "from": "detector", "score": round(float(v.max()), 2), "times_s": _times(v >= DET_MIN, t)})
                cues.append(cue)
    if yam_labels is not None and "a_scores" in z.files:
        s = z["a_scores"].astype(np.float32)
        ta = z["a_t"]
        for _, (classes, desc, cue) in SOUNDS.items():
            cols = [yam_labels.index(c) for c in classes if c in yam_labels]
            if not cols:
                continue
            v = s[:, cols].max(axis=1)
            if (v >= SOUND_MIN).any():
                items.append({"what": desc, "from": "sound model (YAMNet)", "score": round(float(v.max()), 2),
                              "times_s": [[round(float(ta[i]), 1), round(float(ta[i]) + 0.96, 1)] for i in np.where(v >= SOUND_MIN)[0][:6]]})
                if cue:
                    cues.append(cue)
    if places is not None:
        sc = z["scenes"] if "scenes" in z.files else (z["scene"][None] if "scene" in z.files else None)
        if sc is not None:
            p = places.probs(sc)
            best = max(p, key=p.get)
            if p[best] >= PLACE_MIN:
                items.append({"what": f"place: {best}", "from": "scene prompts (SigLIP)", "score": round(p[best], 2), "times_s": []})
                cues.append(PLACE_CUES.get(best, ""))
    return items, " ".join(c for c in cues if c)


def vocal_types(z, species: str, yam_labels: list[str] | None) -> list[dict]:
    if yam_labels is None or "a_scores" not in z.files:
        return []
    s = z["a_scores"].astype(np.float32)
    ta = z["a_t"]
    out = []
    for name in VOCAL_TYPES[species]:
        if name not in yam_labels:
            continue
        v = s[:, yam_labels.index(name)]
        if (v >= SOUND_MIN).any():
            out.append({"type": name, "score": round(float(v.max()), 2), "first_s": round(float(ta[int(np.argmax(v >= SOUND_MIN))]), 1)})
    return sorted(out, key=lambda x: -x["score"])


def events_around(seen: list[dict], context: list[dict], window_s: float = 5.0, limit: int = 3) -> list[dict]:
    """For the first bouts of each recognized behaviour: other behaviours and recording context within ``window_s`` before and after."""
    out = []
    for b in seen[:limit]:
        for t0, t1 in b["bouts_s"][:2]:
            before, after = [], []
            for o in seen:
                if o["behavior"] == b["behavior"]:
                    continue
                for a0, a1 in o["bouts_s"]:
                    if t0 - window_s <= a1 <= t0 + 0.5:
                        before.append(f"{o['behavior']} ({a0:.1f}-{a1:.1f} s)")
                    elif t1 - 0.5 <= a0 <= t1 + window_s:
                        after.append(f"{o['behavior']} ({a0:.1f}-{a1:.1f} s)")
            for c in context:
                for a0, a1 in c.get("times_s", []):
                    if t0 - window_s <= a1 <= t0 + 0.5:
                        before.append(f"{c['what']} ({a0:.1f}-{a1:.1f} s)")
                    elif t1 - 0.5 <= a0 <= t1 + window_s:
                        after.append(f"{c['what']} ({a0:.1f}-{a1:.1f} s)")
            out.append({"behavior": b["behavior"], "bout_s": [t0, t1], "before": sorted(set(before))[:6], "after": sorted(set(after))[:6]})
    return out
