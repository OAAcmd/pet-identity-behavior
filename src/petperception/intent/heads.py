"""Lightweight trained heads (multinomial logistic regression) with calibration, abstention and fusion.

* ``fit_head``: standardized multinomial logistic regression (class-balanced);
  ``C`` chosen by grouped K-fold cross-validation on the training data only
  (pooled out-of-fold macro-F1); the out-of-fold (OOF) logits at the selected
  ``C`` give a temperature (NLL minimisation), an abstention threshold and the
  inputs of the late-fusion stacker. Selecting C on the same folds makes the
  OOF scores slightly optimistic; they never touch test data.
* ``LateFusion``: a multinomial logistic regression over the branch
  log-probabilities, each multiplied by its availability flag, plus the
  flags. Training rows are duplicated with branches removed (modality
  dropout), so the same stacker serves every combination of present inputs;
  an absent branch contributes nothing, never a guess.
* ``EarlyFusion``: concatenated per-branch standardized features with
  availability flags, trained with the same modality dropout.

Scores are softmax outputs after temperature scaling: calibrated on the
training distribution only, not probabilities of truth on new data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from ..audio import linear_model
from ..utils import read_json, utc_now, write_json

C_GRID = (0.001, 0.01, 0.1, 1.0)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_fixed(X: np.ndarray, y: np.ndarray, classes: list[str], C: float, seed: int = 0, standardize: bool = True) -> linear_model.LinearModel:
    X = np.asarray(X, np.float64)
    mean = X.mean(axis=0) if standardize else np.zeros(X.shape[1])
    scale = X.std(axis=0) if standardize else np.ones(X.shape[1])
    scale[scale < 1e-8] = 1.0
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
    clf.fit((X - mean) / scale, y)
    present = list(clf.classes_)
    coef = np.full((len(classes), X.shape[1]), 0.0)
    intercept = np.full(len(classes), -30.0)  # a class absent from the training fold is never predicted
    if len(present) == 2 and clf.coef_.shape[0] == 1:
        for sign, c in ((-0.5, present[0]), (0.5, present[1])):
            coef[classes.index(c)] = sign * clf.coef_[0]
            intercept[classes.index(c)] = sign * clf.intercept_[0]
    else:
        for i, c in enumerate(present):
            coef[classes.index(c)] = clf.coef_[i]
            intercept[classes.index(c)] = clf.intercept_[i]
    return linear_model.LinearModel(list(classes), mean, scale, coef, intercept, C)


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    def nll(t: float) -> float:
        p = softmax(logits / t)
        return float(-np.mean(np.log(np.maximum(p[np.arange(len(y_idx)), y_idx], 1e-12))))

    lo, hi = np.log(0.05), np.log(20.0)
    g = (np.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(60):
        if nll(np.exp(a)) < nll(np.exp(b)):
            hi = b
        else:
            lo = a
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    return float(np.exp((lo + hi) / 2))


def selective_threshold(proba: np.ndarray, y_idx: np.ndarray, target_accuracy: float) -> float:
    """Smallest max-score threshold whose covered OOF predictions reach the target accuracy (1.01 if none)."""
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_idx
    order = np.argsort(-conf)
    cum = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.where(cum >= target_accuracy)[0]
    if len(ok) == 0:
        return 1.01
    k = ok.max()
    return float(conf[order][k])


def ece(proba: np.ndarray, y_idx: np.ndarray, bins: int = 10) -> float:
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_idx).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


@dataclass
class Head:
    name: str
    model: linear_model.LinearModel
    temperature: float = 1.0
    abstain_below: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def classes(self) -> list[str]:
        return self.model.classes

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision(np.atleast_2d(X)) / self.temperature

    def proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.logits(X))

    def to_json(self, path: Path) -> None:
        m = self.model
        write_json(path, {"format": "petperception.intent_head/1", "name": self.name, "classes": m.classes, "standardize_mean": m.mean.tolist(),
                          "standardize_scale": m.scale.tolist(), "coef": m.coef.tolist(), "intercept": m.intercept.tolist(), "C": m.C,
                          "temperature": self.temperature, "abstain_below": self.abstain_below,
                          "score_semantics": "softmax(logits / temperature); temperature fitted on out-of-fold training scores; not a probability of truth",
                          "metadata": self.metadata, "saved_at": utc_now().isoformat(timespec="seconds")})

    @classmethod
    def from_json(cls, path: Path) -> Head:
        d = read_json(path)
        if d.get("format") != "petperception.intent_head/1":
            raise ValueError(f"{path}: unsupported head format")
        m = linear_model.LinearModel(d["classes"], np.array(d["standardize_mean"]), np.array(d["standardize_scale"]), np.array(d["coef"]),
                                     np.array(d["intercept"]), float(d["C"]))
        return cls(d["name"], m, float(d["temperature"]), float(d.get("abstain_below", 0.0)), d.get("metadata", {}))


def cv_select(X: np.ndarray, y: np.ndarray, groups: np.ndarray, classes: list[str], C_grid=C_GRID, splits: int = 5, seed: int = 0,
              standardize: bool = True) -> tuple[float, np.ndarray, dict[float, float]]:
    """One grouped K-fold pass per C: returns the C with the best pooled out-of-fold macro-F1 (ties: smaller C),
    its out-of-fold logits, and the macro-F1 of every C."""
    yi = np.array([classes.index(v) for v in y])
    folds = list(GroupKFold(n_splits=min(splits, len(set(groups.tolist())))).split(X, y, groups))
    best, scores = None, {}
    for C in sorted(C_grid):
        oof = np.zeros((len(y), len(classes)))
        for tr, va in folds:
            oof[va] = fit_fixed(X[tr], y[tr], classes, C, seed, standardize).decision(X[va])
        f = _macro_f1(yi, oof.argmax(1), len(classes))
        scores[C] = f
        if best is None or f > best[0]:
            best = (f, C, oof)
    return best[1], best[2], scores


def fit_head(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, classes: list[str], C_grid=C_GRID, splits: int = 5,
             target_accuracy: float = 0.8, seed: int = 0, metadata: dict | None = None) -> tuple[Head, dict, np.ndarray]:
    X = np.asarray(X, np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    C, oof, scores = cv_select(X, y, groups, classes, C_grid, splits, seed)
    model = fit_fixed(X, y, classes, C, seed)
    yi = np.array([classes.index(v) for v in y])
    T = fit_temperature(oof, yi)
    p = softmax(oof / T)
    tau = selective_threshold(p, yi, target_accuracy)
    info = {"cv": f"GroupKFold({splits}) pooled out-of-fold macro-F1", "C_grid": list(C_grid), "oof_macro_f1_by_C": scores, "selected_C": C,
            "temperature": T, "oof_macro_f1": _macro_f1(yi, p.argmax(1), len(classes)), "oof_ece_raw": ece(softmax(oof), yi),
            "oof_ece_calibrated": ece(p, yi), "abstain_below": tau, "target_selective_accuracy": target_accuracy,
            "oof_coverage_at_threshold": float((p.max(1) >= tau).mean())}
    return Head(name, model, T, tau, dict(metadata or {})), info, p


def _macro_f1(y: np.ndarray, p: np.ndarray, k: int) -> float:
    f = []
    for c in range(k):
        tp, fp, fn = np.sum((y == c) & (p == c)), np.sum((y != c) & (p == c)), np.sum((y == c) & (p != c))
        if tp + fn > 0:
            f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f)) if f else 0.0


# ---------------------------------------------------------------- fusion
def dropout_patterns(n_branches: int) -> list[tuple[int, ...]]:
    """All non-empty availability patterns (1 = present)."""
    pats = []
    for k in range(1, n_branches + 1):
        for keep in combinations(range(n_branches), k):
            pats.append(tuple(1 if i in keep else 0 for i in range(n_branches)))
    return pats


@dataclass
class LateFusion:
    branches: list[str]
    classes: list[str]
    model: linear_model.LinearModel
    temperature: float = 1.0
    abstain_below: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def design(logp: dict[str, np.ndarray | None], branches: list[str], k: int, n: int) -> np.ndarray:
        """Rows of a branch that are NaN (or a branch that is None) are absent: zero block, flag 0."""
        cols, flags = [], []
        for b in branches:
            v = logp.get(b)
            if v is None:
                v = np.full((n, k), np.nan)
            ok = ~np.isnan(v).any(axis=1)
            c = np.zeros((n, k))
            c[ok] = v[ok] - v[ok].mean(axis=1, keepdims=True)
            cols.append(c)
            flags.append(ok[:, None].astype(float))
        return np.hstack(cols + flags)

    def proba(self, proba_by_branch: dict[str, np.ndarray | None]) -> np.ndarray:
        n = next(len(v) for v in proba_by_branch.values() if v is not None)
        logp = {b: (np.log(np.maximum(v, 1e-6)) if v is not None else None) for b, v in proba_by_branch.items()}
        X = self.design(logp, self.branches, len(self.classes), n)
        return softmax(self.model.decision(X) / self.temperature)

    @classmethod
    def fit(cls, proba_by_branch: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray, classes: list[str], C_grid=C_GRID,
            target_accuracy: float = 0.8, seed: int = 0) -> tuple[LateFusion, dict]:
        branches = list(proba_by_branch)
        n, k = len(y), len(classes)
        Xs, ys, gs = [], [], []
        for pat in dropout_patterns(len(branches)):
            logp = {b: (np.log(np.maximum(proba_by_branch[b], 1e-6)) if a else None) for b, a in zip(branches, pat)}  # NaN rows stay NaN
            Xs.append(cls.design(logp, branches, k, n))
            ys.append(y)
            gs.append(groups)
        X, Y, G = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
        C, oof, scores = cv_select(X, Y, G, classes, C_grid, 5, seed)
        model = fit_fixed(X, Y, classes, C, seed)
        info = {"C_grid": list(C_grid), "oof_macro_f1_by_C": scores, "selected_C": C}
        # temperature and abstention on the all-present rows, out of fold
        yi = np.array([classes.index(v) for v in Y])
        T = fit_temperature(oof, yi)
        full = slice((len(Xs) - 1) * n, len(Xs) * n)  # last pattern = all present
        p_full = softmax(oof[full] / T)
        tau = selective_threshold(p_full, yi[full], target_accuracy)
        info.update({"temperature": T, "abstain_below": tau, "patterns": len(Xs), "oof_macro_f1_all_present": _macro_f1(yi[full], p_full.argmax(1), k)})
        return cls(branches, list(classes), model, T, tau, {"dropout_patterns": len(Xs)}), info

    def to_json(self, path: Path) -> None:
        m = self.model
        write_json(path, {"format": "petperception.intent_late_fusion/1", "branches": self.branches, "classes": self.classes,
                          "standardize_mean": m.mean.tolist(), "standardize_scale": m.scale.tolist(), "coef": m.coef.tolist(),
                          "intercept": m.intercept.tolist(), "C": m.C, "temperature": self.temperature, "abstain_below": self.abstain_below,
                          "design": "per branch: centred log-probabilities x availability, then one availability flag per branch",
                          "metadata": self.metadata, "saved_at": utc_now().isoformat(timespec="seconds")})

    @classmethod
    def from_json(cls, path: Path) -> LateFusion:
        d = read_json(path)
        if d.get("format") != "petperception.intent_late_fusion/1":
            raise ValueError(f"{path}: unsupported fusion format")
        m = linear_model.LinearModel(d["classes"], np.array(d["standardize_mean"]), np.array(d["standardize_scale"]), np.array(d["coef"]),
                                     np.array(d["intercept"]), float(d["C"]))
        return cls(d["branches"], d["classes"], m, float(d["temperature"]), float(d["abstain_below"]), d.get("metadata", {}))


@dataclass
class EarlyFusion:
    branches: list[str]
    dims: list[int]
    means: list[np.ndarray]
    scales: list[np.ndarray]
    model: linear_model.LinearModel
    temperature: float = 1.0

    def design(self, feats: dict[str, np.ndarray | None], n: int) -> np.ndarray:
        """NaN rows of a branch (or a None branch) are absent: zero block, flag 0."""
        cols, flags = [], []
        for b, d, mu, sd in zip(self.branches, self.dims, self.means, self.scales):
            v = feats.get(b)
            v = np.full((n, d), np.nan) if v is None else np.asarray(v, np.float64)
            ok = ~np.isnan(v).any(axis=1)
            cols.append(np.where(ok[:, None], np.nan_to_num((v - mu) / sd), 0.0))
            flags.append(ok[:, None].astype(float))
        return np.hstack(cols + flags)

    def proba(self, feats: dict[str, np.ndarray | None]) -> np.ndarray:
        n = next(len(v) for v in feats.values() if v is not None)
        return softmax(self.model.decision(self.design(feats, n)) / self.temperature)

    @classmethod
    def fit(cls, feats: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray, classes: list[str], C_grid=C_GRID, seed: int = 0,
            patterns: list[tuple[int, ...]] | None = None) -> tuple[EarlyFusion, dict]:
        branches = list(feats)
        means, scales, dims = [], [], []
        for b in branches:
            v = np.asarray(feats[b], np.float64)
            v = v[~np.isnan(v).any(axis=1)]
            sd = v.std(axis=0)
            sd[sd < 1e-8] = 1.0
            means.append(v.mean(axis=0))
            scales.append(sd)
            dims.append(v.shape[1])
        proto = cls(branches, dims, means, scales, None)  # type: ignore[arg-type]
        n = len(y)
        Xs, ys, gs = [], [], []
        for pat in patterns or dropout_patterns(len(branches)):
            Xs.append(proto.design({b: (feats[b] if a else None) for b, a in zip(branches, pat)}, n))
            ys.append(y)
            gs.append(groups)
        X, Y, G = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
        # columns are already standardized: fit without re-standardizing so absent blocks stay exactly zero
        C, oof, scores = cv_select(X, Y, G, classes, C_grid, 5, seed, standardize=False)
        info = {"C_grid": list(C_grid), "oof_macro_f1_by_C": scores}
        model = fit_fixed(X, Y, classes, C, seed, standardize=False)
        T = fit_temperature(oof, np.array([classes.index(v) for v in Y]))
        info.update({"selected_C": C, "temperature": T, "patterns": len(Xs)})
        return cls(branches, dims, means, scales, model, T), info
