"""Standardized multinomial logistic regression with safe JSON persistence.

Used for the CPU MFCC baselines (E1 cat/dog/other, E2 CatMeows context) and
the optional frozen-embedding comparison. Scaling and the regularization
strength are fitted only on training partitions; ``C`` is chosen with an inner
grouped cross-validation. Outputs are softmax scores of a linear model and are
explicitly *uncalibrated*.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from ..utils import read_json, utc_now, write_json


@dataclass
class LinearModel:
    classes: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # [n_classes, n_features]
    intercept: np.ndarray  # [n_classes]
    C: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def decision(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, np.float64) - self.mean) / self.scale
        logits = Z @ self.coef.T + self.intercept
        if logits.shape[1] == 1:  # binary sklearn layout
            logits = np.hstack([-logits, logits]) / 2.0
        return logits

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        z = self.decision(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> list[str]:
        return [self.classes[i] for i in np.argmax(self.decision(X), axis=1)]

    def to_json(self, path: Path) -> None:
        write_json(path, {
            "format": "petperception.linear_model/1",
            "classes": self.classes,
            "standardize_mean": self.mean.tolist(),
            "standardize_scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "C": self.C,
            "score_semantics": "softmax of linear logits; uncalibrated, not a probability of truth",
            "metadata": self.metadata,
            "saved_at": utc_now().isoformat(timespec="seconds"),
        })

    @classmethod
    def from_json(cls, path: Path) -> LinearModel:
        d = read_json(path)
        if d.get("format") != "petperception.linear_model/1":
            raise ValueError(f"{path}: unsupported model format")
        return cls(d["classes"], np.array(d["standardize_mean"]), np.array(d["standardize_scale"]),
                   np.array(d["coef"]), np.array(d["intercept"]), float(d["C"]), d.get("metadata", {}))


def _fit_once(X: np.ndarray, y: np.ndarray, C: float, class_weight: str | None, seed: int) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=5000, random_state=seed)
    clf.fit((X - mean) / scale, y)
    return clf, mean, scale


def fit(X: np.ndarray, y: Sequence[str], groups: Sequence[str], classes: list[str],
        C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0), class_weight: str | None = "balanced",
        inner_splits: int | None = None, seed: int = 0) -> tuple[LinearModel, dict[str, Any]]:
    """Fit with ``C`` selected by inner grouped CV (macro-F1) on the given data only."""
    X = np.asarray(X, np.float64)
    y_arr = np.asarray(y)
    g_arr = np.asarray(groups)
    n_groups = len(set(g_arr.tolist()))
    splitter = LeaveOneGroupOut() if inner_splits is None and n_groups <= 12 else GroupKFold(n_splits=min(inner_splits or 5, n_groups))
    cv_scores: dict[float, list[float]] = {}
    skipped = 0
    for C in C_grid:
        scores = []
        for tr, va in splitter.split(X, y_arr, g_arr):
            if len(set(y_arr[tr].tolist())) < len(classes):
                skipped += 1  # a class is missing from this inner training fold
                continue
            clf, m, s = _fit_once(X[tr], y_arr[tr], C, class_weight, seed)
            pred = clf.predict((X[va] - m) / s)
            scores.append(f1_score(y_arr[va], pred, labels=classes, average="macro", zero_division=0))
        cv_scores[C] = scores
    valid = {C: float(np.mean(v)) for C, v in cv_scores.items() if v}
    best_C = max(valid, key=lambda c: (valid[c], -c)) if valid else 1.0
    clf, mean, scale = _fit_once(X, y_arr, best_C, class_weight, seed)
    if clf.coef_.shape[0] == len(classes):
        order = [list(clf.classes_).index(c) for c in classes]
        coef, intercept = clf.coef_[order], clf.intercept_[order]
    else:
        # Binary layout: one logit for clf.classes_[1]; align it with classes[1].
        sign = 1.0 if list(clf.classes_) == list(classes) else -1.0
        coef, intercept = sign * clf.coef_, sign * clf.intercept_
    model = LinearModel(classes, mean, scale, coef, intercept, best_C)
    info = {"inner_cv": type(splitter).__name__, "C_grid": list(C_grid), "inner_macro_f1": valid,
            "selected_C": best_C, "inner_folds_skipped_missing_class": skipped}
    return model, info
