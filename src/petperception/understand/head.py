"""Trained behaviour head: one-vs-rest logistic regression on clip-level SigLIP evidence.

Input per clip (``clip_vector``): the mean of the animal-crop embeddings
(L2-normalized), the scene embedding of the middle frame and a species flag;
1,537 numbers. The vision-language encoder stays frozen; only the linear
heads are trained, which is what the CPU and the few hundred labelled clips
support. ``C`` per behaviour is chosen by grouped cross-validation inside the
training clips (average precision), never on evaluation clips.

The head is stored as plain arrays (``save``/``load``: JSON metadata + NPZ
weights, about 100 kB), so it can be versioned in Git and mirrored to a
phone engine without pickles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

C_GRID = (0.01, 0.1, 1.0)
HEAD_VERSION = "behavior-head-v1"


def clip_vector(z, species: str) -> np.ndarray | None:
    if "emb" not in z.files or len(z["emb"]) < 2:
        return None
    e = z["emb"].astype(np.float32).mean(axis=0)
    e /= np.linalg.norm(e) + 1e-8
    s = z["scene"].astype(np.float32)
    s /= np.linalg.norm(s) + 1e-8
    return np.concatenate([e, s, [1.0 if species == "dog" else 0.0]]).astype(np.float32)


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def _fit_one(X: np.ndarray, y: np.ndarray, C: float, seed: int = 0):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
    clf.fit(X, y)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def choose_c(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 3) -> float:
    from sklearn.model_selection import GroupKFold

    if len(set(groups[y == 1])) < 2:
        return C_GRID[1]
    k = min(folds, len(set(groups)))
    best, best_c = -1.0, C_GRID[1]
    for C in C_GRID:
        scores = np.zeros(len(y))
        for tr, te in GroupKFold(k).split(X, y, groups):
            if y[tr].sum() == 0 or y[tr].sum() == len(tr):
                scores[te] = 0.0
                continue
            w, b = _fit_one(X[tr], y[tr], C)
            scores[te] = X[te] @ w + b
        ap = _ap(y, scores)
        if ap == ap and ap > best:
            best, best_c = ap, C
    return best_c


@dataclass
class BehaviorHead:
    behaviors: list[str]
    mean: np.ndarray
    scale: np.ndarray
    W: np.ndarray  # [B, D]
    b: np.ndarray  # [B]
    C: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, X: np.ndarray, Y: dict[str, np.ndarray], groups: np.ndarray, meta: dict | None = None) -> BehaviorHead:
        X = np.asarray(X, np.float64)
        mean, scale = X.mean(0), X.std(0)
        scale[scale < 1e-8] = 1.0
        Xs = (X - mean) / scale
        names, W, b, Cs = [], [], [], {}
        for n, y in Y.items():
            y = np.asarray(y, int)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            C = choose_c(Xs, y, groups)
            w, c = _fit_one(Xs, y, C)
            names.append(n)
            W.append(w)
            b.append(c)
            Cs[n] = C
        return cls(names, mean, scale, np.array(W), np.array(b), Cs, meta or {})

    def scores(self, x: np.ndarray) -> dict[str, float]:
        """Sigmoid of the logit per behaviour (class-balanced training: relative, not calibrated)."""
        if not self.behaviors:
            return {}
        logit = self.W @ ((np.asarray(x, np.float64) - self.mean) / self.scale) + self.b
        return {n: float(1 / (1 + np.exp(-v))) for n, v in zip(self.behaviors, logit)}

    def predict(self, z, species: str) -> dict[str, float]:
        x = clip_vector(z, species)
        return {} if x is None else self.scores(x)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".npz"), mean=self.mean.astype(np.float32), scale=self.scale.astype(np.float32), W=self.W.astype(np.float32),
                 b=self.b.astype(np.float32))
        path.with_suffix(".json").write_text(json.dumps({"version": HEAD_VERSION, "behaviors": self.behaviors, "C": self.C, **self.meta}, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> BehaviorHead:
        m = json.loads(path.with_suffix(".json").read_text())
        a = np.load(path.with_suffix(".npz"))
        return cls(m["behaviors"], a["mean"].astype(np.float64), a["scale"].astype(np.float64), a["W"].astype(np.float64), a["b"].astype(np.float64),
                   m.get("C", {}), {k: v for k, v in m.items() if k not in ("behaviors", "C")})
