"""Behaviour recognition from per-clip evidence: zero-shot vision-language scores over time, motion pattern, sound.

Outputs, per behaviour: a clip score in [0, 1], the bouts (start, end, peak)
in which it is the most likely reading of the animal crop, and the evidence
that produced it. Temporal pattern descriptors (how much the animal moves,
whether it stays in place, how much it turns, whether the motion repeats) are
reported separately because several explanations depend on them (bout
length, recurrence, turning in place).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..features import EMB_FPS, MOTION_FPS
from . import vocab as V

DEFAULT_SEEN = 0.30  # zero-shot clip score above which a behaviour with a bout counts as seen, fixed before E31
VOCAL_CLASSES = {"cat": ("Cat", "Meow", "Purr", "Hiss", "Caterwaul"), "dog": ("Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)")}
VOCAL_HEARD = 0.3
BOUT_MIN_P = 0.35
BOUT_GAP_S = 1.0


@dataclass
class Bout:
    t0: float
    t1: float
    peak: float


@dataclass
class BehaviorReading:
    behavior: str
    score: float
    bouts: list[Bout]
    evidence: dict = field(default_factory=dict)


@dataclass
class RecognizerConfig:
    """Which scorer each behaviour uses and the score above which it counts as seen (artifacts/understand/recognizer.json)."""

    version: str = "zero-shot-v1"
    method: str = "zero_shot"  # zero_shot | head | mean (of zero-shot and head)
    head_behaviors: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    default_threshold: float = DEFAULT_SEEN
    vocal_threshold: float = VOCAL_HEARD
    provenance: dict = field(default_factory=dict)

    def threshold(self, behavior: str) -> float:
        if behavior == "vocalizing":
            return self.thresholds.get(behavior, self.vocal_threshold)
        return self.thresholds.get(behavior, self.default_threshold)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> RecognizerConfig:
        return cls(**json.loads(Path(path).read_text()))


class ZeroShot:
    """SigLIP text embeddings of the behaviour prompts per species (computed once)."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.cache: dict[str, tuple[np.ndarray, list[str], list[str]]] = {}

    def prompts(self, species: str) -> tuple[np.ndarray, list[str], list[str]]:
        if species not in self.cache:
            texts, owners = V.prompts_for(species)
            self.cache[species] = (self.encoder.texts(texts), texts, owners)
        return self.cache[species]

    def frame_probs(self, emb: np.ndarray, species: str) -> tuple[np.ndarray, list[str]]:
        """Per frame, softmax over behaviours (+ neutral) of the best prompt logit per behaviour -> [N, B], names."""
        T, _, owners = self.prompts(species)
        names = list(dict.fromkeys(owners))
        logits = self.encoder.scale * (np.asarray(emb, np.float32) @ T.T) + self.encoder.bias
        per = np.stack([logits[:, [i for i, o in enumerate(owners) if o == n]].max(axis=1) for n in names], axis=1)
        z = per - per.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return p, names


def bouts_from(t: np.ndarray, p: np.ndarray, is_top: np.ndarray, step: float) -> list[Bout]:
    on = (p >= BOUT_MIN_P) & is_top
    out: list[Bout] = []
    for i in np.where(on)[0]:
        if out and t[i] - out[-1].t1 <= BOUT_GAP_S + step:
            out[-1].t1 = float(t[i] + step)
            out[-1].peak = max(out[-1].peak, float(p[i]))
        else:
            out.append(Bout(float(t[i]), float(t[i] + step), float(p[i])))
    return out


def temporal_pattern(z) -> dict:
    """Motion of the located animal: activity, staying in place, turning, repetition."""
    box, flow = z["box"], z["flow"]
    ok = ~np.isnan(box[:, 0])
    T = len(box)
    out: dict = {"frames": int(T), "seconds": float(T / MOTION_FPS), "animal_found_fraction": float(ok.mean()) if T else 0.0}
    if ok.sum() < 3:
        out["available"] = False
        return out
    ctr = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], 1)
    size = np.sqrt(np.nanmean((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])))
    idx = np.where(ok)[0]
    steps = np.linalg.norm(np.diff(ctr[idx], axis=0), axis=1)
    path = float(np.nansum(steps)) / max(size, 1e-3)
    net = float(np.linalg.norm(ctr[idx[-1]] - ctr[idx[0]])) / max(size, 1e-3)
    f = flow[~np.isnan(flow[:, 0])]
    speed = float(np.mean(f[:, 0])) if len(f) else 0.0
    omega = f[:, 1] if len(f) else np.zeros(0)
    turned = float(np.sum(omega) / MOTION_FPS) if len(f) else 0.0
    abs_turn = float(np.sum(np.abs(omega)) / MOTION_FPS) if len(f) else 0.0
    # repetition: autocorrelation of the region's appearance, lags 0.33-3 s
    app = z["app"].astype(np.float32)
    good = ~np.isnan(app[:, 0])
    best_r, best_lag = 0.0, None
    if good.sum() > 12:
        A = app[good]
        A = (A - A.mean(0)) / (A.std(0) + 1e-6)
        base = []
        for lag in range(2, min(int(3 * MOTION_FPS), len(A) // 2) + 1):
            base.append(float(np.mean(np.sum(A[:-lag] * A[lag:], axis=1)) / A.shape[1]))
        if base:
            curve = np.array(base)
            # a peak after the curve first dips: repetition rather than slow drift
            dips = np.where(np.diff(curve) > 0)[0]
            if len(dips):
                k = dips[0] + 1 + int(np.argmax(curve[dips[0] + 1:]))
                best_r = float(curve[k] - curve[: k + 1].min())
                best_lag = (k + 2) / MOTION_FPS
    out.update({"available": True, "mean_speed_body_per_s": round(speed, 4), "path_body_lengths": round(path, 3), "net_displacement_body_lengths": round(net, 3),
                "stays_in_place": bool(net < 1.0 and speed > 0.08), "turned_rad": round(turned, 3), "abs_turned_rad": round(abs_turn, 3),
                "turning_consistency": round(abs(turned) / max(abs_turn, 1e-6), 3), "repetition_strength": round(best_r, 3),
                "repetition_period_s": round(best_lag, 2) if best_lag else None,
                "detector_dog_vote": round(float(np.mean(z["det"][ok, 1])), 3)})
    return out


def vocal_bouts(z, species: str, labels: list[str]) -> tuple[float, list[Bout]]:
    if "a_scores" not in z.files:
        return 0.0, []
    s = z["a_scores"].astype(np.float32)
    cols = [labels.index(n) for n in VOCAL_CLASSES[species] if n in labels]
    v = s[:, cols].max(axis=1)
    t = z["a_t"]
    top = np.ones(len(v), bool)
    b = [x for x in bouts_from(t, v, top, 0.96) if x.peak >= VOCAL_HEARD]
    return float(v.max()) if len(v) else 0.0, b


def zero_shot_scores(z, species: str, zs: ZeroShot) -> dict[str, float]:
    """Clip-level zero-shot score per visual behaviour: mean of the top quarter of per-frame probabilities."""
    emb = z["emb"].astype(np.float32) if "emb" in z.files else np.zeros((0, 768), np.float32)
    if not len(emb):
        return {}
    p, names = zs.frame_probs(emb, species)
    return {n: float(np.mean(np.sort(p[:, j])[-max(1, len(p) // 4):])) for j, n in enumerate(names) if n != "neutral"}


def recognize(z, species: str, zs: ZeroShot, yamnet_labels: list[str] | None = None, head=None,
              config: RecognizerConfig | None = None) -> tuple[list[BehaviorReading], dict]:
    cfg = config or RecognizerConfig()
    readings: list[BehaviorReading] = []
    pattern = temporal_pattern(z) if "box" in z.files else {"available": False}
    emb = z["emb"].astype(np.float32) if "emb" in z.files else np.zeros((0, 768), np.float32)
    if len(emb):
        p, names = zs.frame_probs(emb, species)
        t = z["emb_t"]
        top = p.argmax(axis=1)
        trained = head.predict(z, species) if head is not None and cfg.method != "zero_shot" else {}
        for j, n in enumerate(names):
            if n == "neutral":
                continue
            frac_top = float(np.mean(top == j))
            zscore = float(np.mean(np.sort(p[:, j])[-max(1, len(p) // 4):]))
            ev = {"zero_shot_top_quartile_mean": round(zscore, 4), "share_of_frames_most_likely": round(frac_top, 3)}
            score = zscore
            if n in trained and n in cfg.head_behaviors:
                score = trained[n] if cfg.method == "head" else 0.5 * zscore + 0.5 * trained[n]
                ev["trained_head"] = round(trained[n], 4)
                ev["clip_level"] = True
            readings.append(BehaviorReading(n, round(score, 4), bouts_from(t, p[:, j], top == j, 1.0 / EMB_FPS), ev))
    if yamnet_labels is not None:
        vmax, vb = vocal_bouts(z, species, yamnet_labels)
        readings.append(BehaviorReading("vocalizing", round(vmax, 4), vb, {"yamnet_max_vocal_score": round(vmax, 4)}))
    readings.sort(key=lambda r: -r.score)
    return readings, pattern
