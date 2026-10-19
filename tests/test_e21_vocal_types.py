"""E21 scoring logic and the shared vocal tag rule (synthetic inputs; not evidence of accuracy)."""

from __future__ import annotations

import numpy as np
import pytest

from petperception.audio.events import SUBTYPE_MIN, vocal_tags
from petperception.eval import e21_vocal_types as E

pytestmark = pytest.mark.synthetic


def test_vocal_tags_rule():
    assert vocal_tags("cat_vocalization", {}) == {"vocalization"}  # no confident subtype: request-like
    assert vocal_tags("cat_vocalization", {"meow": 0.9}) == {"vocalization"}
    assert vocal_tags("cat_vocalization", {"hiss": 0.5}) == {"hiss"}  # hiss alone is not a request
    assert vocal_tags("cat_vocalization", {"hiss": SUBTYPE_MIN - 0.01}) == {"vocalization"}  # below the gate: ignored
    assert vocal_tags("dog_vocalization", {"bark": 0.4, "growl": 0.35}) == {"bark", "growl", "vocalization"}
    assert vocal_tags("growl", {}) == {"growl"}  # a growl event never counts as a request


def test_targets_are_species_conjunctive_where_fsd50k_labels_are_not():
    steam = frozenset({"Hiss", "Vehicle"})
    cat_hiss = frozenset({"Hiss", "Cat", "Animal"})
    wolf = frozenset({"Growling", "Wild_animals", "Animal"})
    assert not E.TYPE_TARGETS["hiss_cat"](steam) and E.TYPE_TARGETS["hiss_cat"](cat_hiss)
    assert E.TYPE_TARGETS["growl_any"](wolf) and not E.TYPE_TARGETS["growl_dog"](wolf)


def test_report_counts_ci_and_gate():
    y = np.array([True] * 10 + [False] * 30)
    pred = np.array([True] * 9 + [False] + [True] * 1 + [False] * 29)
    groups = np.array([f"u{i % 8}" for i in range(40)])
    r = E._report(y, pred, groups, score=pred.astype(float), n_boot=200, seed=0)
    assert (r["tp"], r["fp"], r["fn"]) == (9, 1, 1)
    assert r["precision"] == pytest.approx(0.9) and r["recall"] == pytest.approx(0.9) and r["f1"] == pytest.approx(0.9)
    assert r["f1_ci95_by_uploader"]["n_groups"] == 8
    assert r["passes_gate"] is (r["f1_ci95_by_uploader"]["ci_low"] >= 0.6)
    none = E._report(np.zeros(5, bool), np.zeros(5, bool), np.array(list("abcde")), None, 50, 0)
    assert none["positives"] == 0 and none["passes_gate"] is False and none["f1_ci95_by_uploader"] is None
