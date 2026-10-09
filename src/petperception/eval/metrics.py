"""Classification, detection and resampling metrics."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def confusion(y_true: Sequence[str], y_pred: Sequence[str], labels: list[str]) -> list[list[int]]:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m.tolist()


def per_class(y_true: Sequence[str], y_pred: Sequence[str], labels: list[str]) -> dict[str, dict[str, float]]:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    out = {}
    for l in labels:
        tp = int(np.sum((yt == l) & (yp == l)))
        fp = int(np.sum((yt != l) & (yp == l)))
        fn = int(np.sum((yt == l) & (yp != l)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[l] = {"precision": prec, "recall": rec, "f1": f1, "support": int(np.sum(yt == l)), "tp": tp, "fp": fp, "fn": fn}
    return out


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: list[str]) -> float:
    pc = per_class(y_true, y_pred, labels)
    return float(np.mean([pc[l]["f1"] for l in labels]))


def balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str], labels: list[str]) -> float:
    pc = per_class(y_true, y_pred, labels)
    recs = [pc[l]["recall"] for l in labels if pc[l]["support"] > 0]
    return float(np.mean(recs)) if recs else 0.0


def classification_report(y_true: Sequence[str], y_pred: Sequence[str], labels: list[str]) -> dict:
    return {
        "labels": labels,
        "n": len(y_true),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, labels),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if len(y_true) else 0.0,
        "per_class": per_class(y_true, y_pred, labels),
        "confusion_matrix": {"rows_true_cols_pred": labels, "matrix": confusion(y_true, y_pred, labels)},
    }


def group_bootstrap(stat: Callable[[np.ndarray], float], groups: Sequence[str], n_boot: int = 1000,
                    seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Percentile CI resampling whole groups (e.g. source recordings, animals)."""
    g = np.asarray(groups)
    uniq = np.unique(g)
    members = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[u] for u in pick])
        vals.append(stat(idx))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot, "n_groups": int(len(uniq)), "unit": "group"}


def average_precision(scores: np.ndarray, is_pos: np.ndarray) -> float:
    """Area under the precision-recall step curve (all-point, ranking only)."""
    order = np.argsort(-scores, kind="stable")
    pos = is_pos[order].astype(float)
    if pos.sum() == 0:
        return float("nan")
    tp = np.cumsum(pos)
    prec = tp / np.arange(1, len(pos) + 1)
    return float(np.sum(prec * pos) / pos.sum())


def best_f1_threshold(scores: np.ndarray, is_pos: np.ndarray) -> tuple[float, float]:
    """Threshold maximizing F1 on the given (training/development) data.

    Ties are broken toward the higher threshold (fewer false positives).
    """
    cands = np.unique(scores)
    best = (0.0, float(cands.max()) if cands.size else 0.5)
    P = is_pos.sum()
    for t in cands:
        pred = scores >= t
        tp = np.sum(pred & is_pos)
        fp = np.sum(pred & ~is_pos)
        f1 = 2 * tp / (2 * tp + fp + (P - tp)) if P else 0.0
        if f1 > best[0] + 1e-12 or (abs(f1 - best[0]) <= 1e-12 and t > best[1]):
            best = (float(f1), float(t))
    return best[1], best[0]


def brier(prob: np.ndarray, is_pos: np.ndarray) -> float:
    return float(np.mean((prob - is_pos.astype(float)) ** 2))


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between [N,4] and [M,4] boxes in (x1, y1, x2, y2)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-12)
