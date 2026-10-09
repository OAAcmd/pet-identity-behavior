"""E2 grouped-protocol mechanics on SYNTHETIC CatMeows-style files.

These recordings are generated tones with CatMeows-like names. They only test
that the leave-one-cat-out machinery, filename parsing and reporting work;
they say nothing about CatMeows performance.
"""

import numpy as np
import pytest
import soundfile as sf

from petperception.data import catmeows as CM
from petperception.eval import e2_cat_context as E2

pytestmark = pytest.mark.synthetic


@pytest.fixture
def fake_catmeows(tmp_path):
    d = tmp_path / "catmeows" / "dataset"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    freqs = {"B": 500, "F": 900, "I": 1400}
    owners = {"CAT01": "OWN01", "CAT02": "OWN01", "CAT03": "OWN02", "CAT04": "OWN03", "CAT05": "OWN03"}
    for cat, owner in owners.items():
        for ctx, f in freqs.items():
            if cat == "CAT05" and ctx == "I":
                continue  # a cat missing one context, as can happen in real data
            for k in range(3):
                t = np.arange(8000) / 8000
                x = 0.3 * np.sin(2 * np.pi * (f + rng.normal(0, 40)) * t) + 0.01 * rng.normal(size=t.size)
                sf.write(d / f"{ctx}_{cat}_MC_FN_{owner}_1{k:02d}.wav", x.astype(np.float32), 8000)
    (d / "README_unparseable.wav").write_bytes(b"")
    return tmp_path / "catmeows"


def test_scan_parses_and_rejects(fake_catmeows):
    recs, rejected = CM.scan(fake_catmeows)
    assert len(recs) == 5 * 3 * 3 - 3 and rejected == ["dataset/README_unparseable.wav"]
    assert {r.sample_rate_hz for r in recs} == {8000}
    assert CM.owner_cat_relationships(recs)["OWN01"] == ["CAT01", "CAT02"]


def test_loco_holds_out_whole_cats_and_reports_missing_classes(fake_catmeows, tmp_path):
    recs, _ = CM.scan(fake_catmeows)
    out = E2.run_on_recordings(recs, tmp_path / "out", n_boot=50)
    loco = out["results"]["methods"]["loco_mfcc_logreg"]
    held = [f["held_out"] for f in loco["folds"]]
    assert sorted(held) == ["CAT01", "CAT02", "CAT03", "CAT04", "CAT05"]
    cat5 = next(f for f in loco["folds"] if f["held_out"] == "CAT05")
    assert cat5["missing_classes_in_test"] == ["isolation"]
    looo = out["results"]["methods"]["looo_mfcc_logreg"]
    assert sorted(f["held_out"] for f in looo["folds"]) == ["OWN01", "OWN02", "OWN03"]
    assert "NOT a valid estimate" in out["results"]["leakage_diagnostic_random_split"]["note"]
    assert out["model"].metadata["label_semantics"].startswith("recording context")


def test_e2_run_reports_blocked_without_data(settings, tmp_path):
    s = settings.model_copy(deep=True)
    s.paths.data_root = str(tmp_path / "empty")
    st = E2.run(s, tmp_path / "res")
    assert st["status"] == "blocked" and "fetch-data catmeows" in st["next_command"]
