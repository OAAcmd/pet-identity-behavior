"""Split-leakage, grouping and metric tests."""

import csv
from collections import defaultdict

import numpy as np
import pytest

from petperception.audio import linear_model
from petperception.config import REPO_ROOT
from petperception.eval import metrics as M

MAN = REPO_ROOT / "manifests"


def test_esc50_pet_sources_never_cross_folds():
    folds = defaultdict(set)
    labels = {}
    with open(MAN / "esc50_files.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            folds[r["source_group"]].add(r["fold"])
            labels[r["source_group"]] = r["pet_label"]
    crossing = {g for g, f in folds.items() if len(f) > 1}
    assert not any(labels[g] in ("cat", "dog") for g in crossing)
    # The verified exceptions are recorded in the register; any change must be reviewed.
    assert crossing == {"freesound:131943", "freesound:134049", "freesound:209698", "freesound:234879"}


def test_openimages_subset_is_consistent():
    with open(MAN / "openimages_catdog_images.csv", newline="") as fh:
        imgs = list(csv.DictReader(fh))
    ids = [r["image_id"] for r in imgs]
    assert len(ids) == len(set(ids))
    with open(MAN / "openimages_catdog_boxes.csv", newline="") as fh:
        assert {r["image_id"] for r in csv.DictReader(fh)} <= set(ids)
    with open(MAN / "openimages_catdog_sha256.csv", newline="") as fh:
        assert {r["image_id"] for r in csv.DictReader(fh)} == set(ids)
    assert all(r["license"] and r["author_profile_url"] for r in imgs)


@pytest.mark.synthetic
def test_standardization_fitted_on_training_partition_only():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(5.0, 2.0, size=(60, 3))
    y = np.array(["a", "b", "c"] * 20)
    g = np.array([str(i % 6) for i in range(60)])
    m, info = linear_model.fit(Xtr, y, g, ["a", "b", "c"])
    np.testing.assert_allclose(m.mean, Xtr.mean(axis=0))
    assert info["inner_cv"] == "LeaveOneGroupOut"


@pytest.mark.synthetic
def test_linear_model_json_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 4))
    y = np.array(["x", "y"] * 20)
    X[y == "y"] += 2
    m, _ = linear_model.fit(X, y, [str(i % 4) for i in range(40)], ["x", "y"])
    m.to_json(tmp_path / "m.json")
    m2 = linear_model.LinearModel.from_json(tmp_path / "m.json")
    np.testing.assert_allclose(m.predict_scores(X), m2.predict_scores(X))
    assert m.predict(X) == m2.predict(X)
    assert (tmp_path / "m.json").read_text().startswith("{")  # numeric JSON, no pickle


@pytest.mark.synthetic
def test_group_bootstrap_resamples_groups():
    groups = ["g1"] * 50 + ["g2"] * 50
    vals = np.array([1.0] * 50 + [0.0] * 50)
    ci = M.group_bootstrap(lambda ix: float(vals[ix].mean()), groups, n_boot=200, seed=0)
    assert ci["n_groups"] == 2 and ci["ci_low"] == 0.0 and ci["ci_high"] == 1.0


@pytest.mark.synthetic
def test_metrics_basic():
    yt = ["cat", "dog", "other", "other"]
    yp = ["cat", "other", "other", "dog"]
    pc = M.per_class(yt, yp, ["cat", "dog", "other"])
    assert pc["cat"]["f1"] == 1.0 and pc["dog"]["f1"] == 0.0
    assert M.balanced_accuracy(yt, yp, ["cat", "dog", "other"]) == pytest.approx((1 + 0 + 0.5) / 3)
    t, f1 = M.best_f1_threshold(np.array([0.1, 0.4, 0.8, 0.9]), np.array([False, False, True, True]))
    assert f1 == 1.0 and t == 0.8


@pytest.mark.synthetic
def test_iou():
    a = np.array([[0, 0, 1, 1.0]])
    b = np.array([[0.5, 0, 1.5, 1.0], [2, 2, 3, 3.0]])
    np.testing.assert_allclose(M.iou_xyxy(a, b), [[1 / 3, 0.0]])
