"""E28 follow-up (exploratory, after the pre-registered result): is there any conflict signal at all?

The pre-registered conflict flag (both the audio head and the visual-only
model pass their own abstention thresholds and disagree) never fired in E28,
because the audio head never reaches its threshold (it cannot attain 0.8
out-of-fold selective accuracy, so its threshold is unattainable). This
follow-up asks whether a continuous disagreement score separates genuine
audio-video pairs from swapped ones (audio of a different-situation clip),
without choosing any threshold on the test clips:

* score = Jensen-Shannon divergence between the audio-only and the
  visual-only situation distributions (both from the trained E28 models);
* reported: ROC AUC genuine vs swapped (same species, cross species), and
  the rate at which the swapped score exceeds the 95th percentile of the
  genuine *training* out-of-fold scores (a threshold from training data).

Exploratory: it does not change E28's result or the shipped policy.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..config import Settings
from ..intent import kinetics as K
from ..utils import utc_now, write_json
from .e28_av_situation import VISUAL, swap_audio


def _js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = (p + q) / 2
    kl = lambda a, b: np.sum(a * (np.log(np.maximum(a, 1e-9)) - np.log(np.maximum(b, 1e-9))), axis=1)  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _scores(m, rows: list[dict]) -> np.ndarray:
    bp = m.branch_proba(rows)
    ab = m.branches[0]
    pa = bp[ab]
    pv = m.late.proba({**dict.fromkeys(m.branches), **{b: bp[b] for b in m.branches if b in VISUAL}})
    ok = ~np.isnan(pa).any(axis=1)
    out = np.full(len(rows), np.nan)
    out[ok] = _js(pa[ok], pv[ok])
    return out


def run(settings: Settings, out_dir: Path) -> dict:
    data_root = settings.paths.resolve("data_root")
    with open(data_root.parent / "intent" / "e28_models.pkl", "rb") as fh:
        models = pickle.load(fh)["models"]
    clips = K.list_clips(data_root)
    rows, _ = K.load_pooled(clips, data_root.parent)
    train = [r for r in rows if r["clip"].split == "train"]
    test = [r for r in rows if r["clip"].split == "test"]
    res = {}
    for name, m in models.items():
        g_tr = _scores(m, train)  # in-sample for the heads (no OOF heads stored): a lenient reference
        thr = float(np.nanpercentile(g_tr, 95))
        g = _scores(m, test)
        out = {"threshold_95th_percentile_genuine_train": thr, "genuine_test_above_threshold": float(np.nanmean(g > thr))}
        for same in (True, False):
            sw, _ = swap_audio(test, same)
            s = _scores(m, sw)
            ok = ~np.isnan(s) & ~np.isnan(g)
            out[f"swapped_{'same' if same else 'cross'}_species"] = {"roc_auc_vs_genuine": _auc(s[ok], g[ok]),
                                                                     "above_threshold": float(np.nanmean(s > thr))}
        res[name] = out
    o = {"experiment": "E28-conflict-diagnostic", "status": "completed (exploratory)", "created_at": utc_now().isoformat(timespec="seconds"),
         "protocol": __doc__.strip().splitlines(), "results": res,
         "caveat": "training-set reference scores use heads fitted on those clips (in-sample), so the threshold is lenient"}
    write_json(out_dir / "conflict_diagnostic.json", o)
    return o
