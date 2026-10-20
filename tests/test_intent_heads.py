from __future__ import annotations

import numpy as np
import pytest

from petperception.intent import heads as H


def _data(n=240, d=6, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array(["a", "b", "c", "d"])[rng.integers(0, 4, n)]
    g = np.array([f"g{i // 2}" for i in range(n)])
    # branch A separates {a,b} from {c,d} and a from b; branch B separates c from d
    A = rng.normal(0, 1, (n, d))
    A[:, 0] += np.where(np.isin(y, ["a", "b"]), 2.5, -2.5)
    A[:, 1] += np.where(y == "a", 2.0, np.where(y == "b", -2.0, 0.0))
    B = rng.normal(0, 1, (n, d))
    B[:, 0] += np.where(y == "c", 2.5, np.where(y == "d", -2.5, 0.0))
    return A, B, y, g


def test_fit_head_round_trip(tmp_path):
    A, _, y, g = _data()
    head, info, oof = H.fit_head("a", A, y, g, ["a", "b", "c", "d"], C_grid=(0.1, 1.0))
    assert oof.shape == (len(y), 4) and np.allclose(oof.sum(1), 1)
    assert info["temperature"] > 0 and 0 <= info["abstain_below"] <= 1.01
    head.to_json(tmp_path / "h.json")
    h2 = H.Head.from_json(tmp_path / "h.json")
    assert np.allclose(head.proba(A[:10]), h2.proba(A[:10]))


def test_late_fusion_combines_and_handles_missing(tmp_path):
    A, B, y, g = _data()
    classes = ["a", "b", "c", "d"]
    _, _, pa = H.fit_head("A", A, y, g, classes, C_grid=(1.0,))
    _, _, pb = H.fit_head("B", B, y, g, classes, C_grid=(1.0,))
    pb_nan = pb.copy()
    pb_nan[:20] = np.nan  # naturally missing rows
    fus, info = H.LateFusion.fit({"A": pa, "B": pb_nan}, y, g, classes, C_grid=(1.0,))
    acc = lambda p: float((np.array(classes)[p.argmax(1)] == y).mean())  # noqa: E731
    both = fus.proba({"A": pa, "B": pb})
    only_a = fus.proba({"A": pa, "B": None})
    assert acc(both) > acc(only_a) + 0.1  # B resolves c vs d
    assert np.allclose(fus.proba({"A": pa, "B": np.full_like(pb, np.nan)}), only_a)  # NaN rows == absent branch
    fus.to_json(tmp_path / "f.json")
    assert np.allclose(H.LateFusion.from_json(tmp_path / "f.json").proba({"A": pa, "B": pb}), both)


def test_early_fusion_absent_block_is_zero():
    A, B, y, g = _data()
    classes = ["a", "b", "c", "d"]
    ef, _ = H.EarlyFusion.fit({"A": A, "B": B}, y, g, classes, C_grid=(1.0,))
    X = ef.design({"A": A[:5], "B": None}, 5)
    assert np.all(X[:, A.shape[1]:A.shape[1] + B.shape[1]] == 0) and np.all(X[:, -1] == 0) and np.all(X[:, -2] == 1)
    assert np.allclose(ef.proba({"A": A[:5], "B": np.full((5, B.shape[1]), np.nan)}), ef.proba({"A": A[:5], "B": None}))


def test_selective_threshold_and_temperature():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 3, (500, 3))
    y = z.argmax(1)
    flip = rng.random(500) < 0.3
    y[flip] = rng.integers(0, 3, flip.sum())
    p = H.softmax(z)
    tau = H.selective_threshold(p, y, 0.9)
    cov = p.max(1) >= tau
    assert cov.any() and (p.argmax(1)[cov] == y[cov]).mean() >= 0.9
    assert H.selective_threshold(p, y, 1.01) == pytest.approx(1.01)
    # labels drawn from softmax(z); the model reports 4z (over-confident): temperature ~4 restores calibration
    z2 = rng.normal(0, 1.5, (3000, 3))
    pt = H.softmax(z2)
    y2 = np.array([rng.choice(3, p=row) for row in pt])
    T = H.fit_temperature(4 * z2, y2)
    assert 3.0 < T < 5.5
    assert H.ece(H.softmax(4 * z2 / T), y2) < H.ece(H.softmax(4 * z2), y2)


def test_dropout_patterns():
    assert H.dropout_patterns(2) == [(1, 0), (0, 1), (1, 1)]
    assert len(H.dropout_patterns(4)) == 15
