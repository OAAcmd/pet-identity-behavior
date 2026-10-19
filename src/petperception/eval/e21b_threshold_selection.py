"""E21b: audio decision thresholds selected on FSD50K dev, tested once on FSD50K eval.

Why. E21 tested the deployed operating point on FSD50K eval. The species
thresholds were selected on ESC-50 (40 clips per species, E1). The subtype
gate (0.3), the growl threshold (0.5) and the context-tag threshold (0.3)
were never selected on data. FSD50K dev has 36,796 + 4,170 labelled clips
from uploaders disjoint from eval (verified in E21). Its CC0 and CC BY clips
also allow thresholds whose data permits commercial use (the open question
Q-A2 for the ESC-50 thresholds).

Protocol, fixed before the dev audio was scored:
* Selection data: FSD50K dev (train + val), variant `all` (every clip) and
  variant `cc0_ccby` (CC0 and CC BY clips only). Eval is not touched
  during selection.
* What is selected, in this order, each as the F1-optimal value on a grid
  of 0.05 to 0.95 in steps of 0.005:
  1. species thresholds (cat_vocalization, dog_vocalization) on the clip's
     maximum group score; targets "Cat" and "Dog";
  2. with those fixed: a per-subtype minimum for meow, purr, bark and growl,
     applied to the subtype's peak inside species events, the rule both
     engines use with one shared value today (0.3). The growl event
     threshold (today 0.5) is selected jointly with the growl subtype
     minimum. Hiss is not selected: dev has 13 cat-hiss clips and eval 2;
  3. a per-tag threshold for every evaluable context tag (today 0.3 for
     all), on the clip's maximum tag score.
* Targets are the E21 definitions (species-conjunctive for growl).
* Honest dev estimate: 5-fold cross-validation grouped by uploader on dev
  (select on 4 folds, score the 5th, pool).
* Test: each variant's thresholds are scored once on eval, with the exact
  engine rule (`extract_events` + `vocal_tags` logic with per-type minimums).
  The comparison with the deployed operating point is paired: a bootstrap
  over uploaders gives the 95 % CI of the F1 difference.
* Adoption rule (fixed now): a selected threshold replaces the deployed one
  only if the paired F1 difference on eval has a lower 95 % bound above 0.
  For the `cc0_ccby` variant, which would remove a licence question, the
  bound is above -0.02 (non-inferiority). The E21 gate (F1 >= 0.7, lower
  bound >= 0.6) still decides whether a decision may be shown at all.
* Out-of-domain check: the selected species thresholds are also scored on
  ESC-50 (all 2,000 clips, the E1 cache), the data the deployed thresholds
  came from.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ..audio.events import CONTEXT_TAG_MIDS, build_groups, extract_events
from ..config import REPO_ROOT, Settings
from ..data import fsd50k
from ..utils import utc_now, write_json
from . import e21_vocal_types as E21
from . import metrics as M

log = logging.getLogger("petperception.e21b")
GRID = np.round(np.arange(0.05, 0.9501, 0.005), 3)
SUBTYPES = ["meow", "purr", "bark", "growl"]
SUB_MIDS = {"meow": "/m/07qrkrw", "purr": "/m/02yds9", "hiss": "/m/07rjwbb", "bark": "/m/05tny_", "growl": "/m/0ghcn6"}
SUB_TARGET = {"meow": "meow", "purr": "purr", "bark": "bark", "growl": "growl_any"}
ADOPT_LOWER = {"all": 0.0, "cc0_ccby": -0.02}


def _f1(y: np.ndarray, p: np.ndarray) -> float:
    tp, fp, fn = np.sum(y & p), np.sum(~y & p), np.sum(y & ~p)
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def best_threshold(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """F1-optimal grid value (ties: the lowest threshold)."""
    best = (float(GRID[0]), -1.0)
    for t in GRID:
        f = _f1(y, score >= t)
        if f > best[1] + 1e-12:
            best = (float(t), f)
    return best


def best_growl(gev: np.ndarray, gsub: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Joint (growl event threshold, growl subtype minimum) on a coarse-then-fine grid."""
    best = (0.5, 0.3, -1.0)
    coarse = GRID[::10]
    for te in coarse:
        for ts in coarse:
            f = _f1(y, (gev >= te) | (gsub >= ts))
            if f > best[2] + 1e-12:
                best = (float(te), float(ts), f)
    fine = lambda c: GRID[(GRID >= c - 0.05) & (GRID <= c + 0.05)]  # noqa: E731
    te0, ts0 = best[0], best[1]
    for te in fine(te0):
        for ts in fine(ts0):
            f = _f1(y, (gev >= te) | (gsub >= ts))
            if f > best[2] + 1e-12:
                best = (float(te), float(ts), f)
    return best


class ClipScores:
    """Per-clip quantities from cached window scores, exact for the engine rule."""

    def __init__(self, sc: dict, backend, groups, y_cfg):
        self.sc, self.backend, self.groups, self.y_cfg = sc, backend, groups, y_cfg
        n = len(sc["fname"])
        self.n = n
        self.sub_idx = {k: backend.mid_to_index[m] for k, m in SUB_MIDS.items()}
        self.ctx_idx = {t: backend.mid_to_index[m] for t, m in CONTEXT_TAG_MIDS.items() if m in backend.mid_to_index}
        self.gmax = {g: np.zeros(n, np.float32) for g in groups}
        self.ctx = {t: np.zeros(n, np.float32) for t in self.ctx_idx}
        self.growl_max = np.zeros(n, np.float32)
        for i in range(n):
            S = self.window(i)
            for g, spec in groups.items():
                self.gmax[g][i] = S[:, spec.indices].max()
            for t, j in self.ctx_idx.items():
                self.ctx[t][i] = S[:, j].max()
            self.growl_max[i] = S[:, self.sub_idx["growl"]].max()

    def window(self, i: int) -> np.ndarray:
        return self.sc["scores"][self.sc["offset"][i]:self.sc["offset"][i + 1]]

    def ws(self, i: int):
        from ..audio.yamnet import WindowScores

        S = self.window(i)
        t = np.arange(S.shape[0]) * self.y_cfg.hop_s
        return WindowScores(t, t + 0.975, S)

    def subtype_peaks(self, species_th: dict[str, float], growl_th: float) -> dict[str, np.ndarray]:
        """Each subtype's highest peak inside species or growl events (0 if none), as `vocal_tags` sees them."""
        out = {k: np.zeros(self.n, np.float32) for k in self.sub_idx}
        th = {**species_th, "growl": growl_th}
        cand = np.where((self.gmax["cat_vocalization"] >= species_th["cat_vocalization"]) | (self.gmax["dog_vocalization"] >= species_th["dog_vocalization"])
                        | (self.growl_max >= growl_th))[0]
        for i in cand:
            for ev in extract_events(self.ws(i), self.backend, self.groups, th, 2.0, self.y_cfg.min_event_gap_s):
                if ev.label in self.groups or ev.label == "growl":
                    for k, v in ev.subtypes.items():
                        if k in out:
                            out[k][i] = max(out[k][i], v)
        return out

    def decisions(self, p: dict) -> dict[str, np.ndarray]:
        """Clip decisions under parameters p (species, growl_event, subtype_min per type, context per tag): the engine rule."""
        peaks = self.subtype_peaks(p["species"], p["growl_event"])
        d = {g: self.gmax[g] >= p["species"][g] for g in self.groups}
        for k in SUBTYPES:
            d[k] = peaks[k] >= p["subtype_min"][k]
        d["growl"] = d["growl"] | (self.growl_max >= p["growl_event"])
        d["hiss"] = peaks["hiss"] >= p["subtype_min"].get("hiss", 0.3)
        for t, th in p["context"].items():
            d[f"context:{t}"] = self.ctx[t] >= th
        return d


def targets(clips: list[fsd50k.Clip], ctx_labels: dict[str, str]) -> dict[str, np.ndarray]:
    L = [c.labels for c in clips]
    y = {k: np.array([f(l) for l in L]) for k, f in {**E21.SPECIES_TARGETS, **E21.TYPE_TARGETS}.items()}
    for t, lab in ctx_labels.items():
        y[f"context:{t}"] = np.array([lab in l for l in L])
    return y


def select(cs: ClipScores, y: dict, mask: np.ndarray, growl_start: float) -> dict:
    """Species thresholds, then subtype minimums with growl events at `growl_start`, then the growl pair, then context tags."""
    sp = {g: best_threshold(cs.gmax[g][mask], y[g][mask])[0] for g in ("cat_vocalization", "dog_vocalization")}
    peaks = cs.subtype_peaks(sp, growl_start)
    sub = {k: best_threshold(peaks[k][mask], y[SUB_TARGET[k]][mask])[0] for k in ("meow", "purr", "bark")}
    ge, gs, _ = best_growl(cs.growl_max[mask], peaks["growl"][mask], y["growl_any"][mask])
    sub["growl"] = gs
    ctx = {t: best_threshold(cs.ctx[t][mask], y[f"context:{t}"][mask])[0] for t in cs.ctx if f"context:{t}" in y}
    return {"species": sp, "growl_event": ge, "subtype_min": sub, "context": ctx}


DECISION_TARGET = {"cat_vocalization": "cat_vocalization", "dog_vocalization": "dog_vocalization", "meow": "meow", "purr": "purr",
                   "bark": "bark", "growl": "growl_any"}


def deployed_params(settings: Settings, ctx_tags: list[str]) -> dict:
    from ..audio.events import SUBTYPE_MIN, load_thresholds

    y_cfg = settings.audio.yamnet
    th, _ = load_thresholds(REPO_ROOT / y_cfg.thresholds_file, y_cfg.default_thresholds)
    return {"species": {g: th[g] for g in ("cat_vocalization", "dog_vocalization")}, "growl_event": th.get("growl", 0.5),
            "subtype_min": {k: SUBTYPE_MIN for k in [*SUBTYPES, "hiss"]}, "context": {t: y_cfg.context_tag_threshold for t in ctx_tags}}


def run(settings: Settings, out_dir: Path, workers: int = 3, n_boot: int = 1000, seed: int = 0) -> dict:
    from ..audio.yamnet import YamnetBackend
    from ..registry import model_dir

    t0 = time.perf_counter()
    data_root = settings.paths.resolve("data_root")
    dev = fsd50k.load_clips(data_root, "dev")
    ev_clips = [c for c in fsd50k.load_clips(data_root, "eval") if c.fname not in E21.esc50_freesound_ids(settings)]
    sc_dev = E21.score_clips(settings, "dev", [c.fname for c in dev], data_root / "fsd50k" / f"{E21.CACHE_VERSION}_dev.npz", workers)
    sc_ev = E21.score_clips(settings, "eval", [c.fname for c in ev_clips], data_root / "fsd50k" / f"{E21.CACHE_VERSION}_eval.npz", workers)
    backend = YamnetBackend(model_dir(settings, settings.audio.yamnet.model_id), 1)
    groups = build_groups(backend)
    y_cfg = settings.audio.yamnet
    cs_dev, cs_ev = ClipScores(sc_dev, backend, groups, y_cfg), ClipScores(sc_ev, backend, groups, y_cfg)
    vocab = {}
    with open(fsd50k.root(data_root) / "meta" / "FSD50K.ground_truth" / "vocabulary.csv", encoding="utf-8") as fh:
        for line in fh:
            _, lab, mid = line.strip().split(",")
            vocab[mid] = lab
    ctx_labels = {t: vocab[m] for t, m in CONTEXT_TAG_MIDS.items() if m in vocab and t in cs_dev.ctx}
    y_dev, y_ev = targets(dev, ctx_labels), targets(ev_clips, ctx_labels)
    lic_dev = np.array([c.licence for c in dev])
    up_dev, up_ev = np.array([c.uploader for c in dev]), np.array([c.uploader for c in ev_clips])
    masks = {"all": np.ones(len(dev), bool), "cc0_ccby": np.isin(lic_dev, ["CC0", "CC BY"])}
    deployed = deployed_params(settings, list(ctx_labels))
    d_dep = cs_ev.decisions(deployed)
    dec_targets = {**DECISION_TARGET, **{f"context:{t}": f"context:{t}" for t in ctx_labels}}

    out: dict = {"variants": {}}
    rng = np.random.default_rng(seed)
    uniq_dev = np.unique(up_dev)
    fold_of = dict(zip(rng.permutation(uniq_dev), np.arange(len(uniq_dev)) % 5))
    folds = np.array([fold_of[u] for u in up_dev])
    for vname, vm in masks.items():
        p = select(cs_dev, y_dev, vm, deployed["growl_event"])
        # honest dev estimate: uploader-grouped 5-fold CV on the same variant
        cv_pred = {k: np.zeros(len(dev), bool) for k in dec_targets}
        for f in range(5):
            pf = select(cs_dev, y_dev, vm & (folds != f), deployed["growl_event"])
            df = cs_dev.decisions(pf)
            te = vm & (folds == f)
            for k in dec_targets:
                cv_pred[k][te] = df[k][te]
        d_new = cs_ev.decisions(p)
        rows = {}
        for k, tk in dec_targets.items():
            y = y_ev[tk]
            yd = y_dev[tk][vm]
            diff = M.group_bootstrap(lambda ix, y=y, k=k: _f1(y[ix], d_new[k][ix]) - _f1(y[ix], d_dep[k][ix]), up_ev.tolist(), n_boot, seed)
            new_rep = E21._report(y, d_new[k], up_ev, None, n_boot, seed)
            lower = ADOPT_LOWER[vname]
            rows[k] = {"eval_positives": int(y.sum()), "dev_positives": int(yd.sum()),
                       "dev_cv_f1": _f1(yd, cv_pred[k][vm]),
                       "eval_f1_deployed": _f1(y, d_dep[k]), "eval_f1_selected": new_rep["f1"],
                       "eval_precision_selected": new_rep["precision"], "eval_recall_selected": new_rep["recall"],
                       "eval_f1_selected_ci95": new_rep["f1_ci95_by_uploader"], "paired_f1_difference_ci95": diff,
                       "f1_difference": new_rep["f1"] - _f1(y, d_dep[k]),
                       "adopt": bool(diff["ci_low"] > lower), "passes_gate_selected": new_rep["passes_gate"]}
        out["variants"][vname] = {"selection_clips": int(vm.sum()), "selected": p, "results": rows,
                                  "adoption_rule": f"paired F1 difference lower 95% bound > {ADOPT_LOWER[vname]}"}
    out["esc50_out_of_domain"] = _esc50_check(settings, out, deployed)
    out["consistency_with_e21"] = _e21_consistency({k: _f1(y_ev[tk], d_dep[k]) for k, tk in dec_targets.items()})
    res = {"experiment": "E21b", "status": "completed", "created_at": utc_now().isoformat(timespec="seconds"),
           "deployed": deployed, "grid": [float(GRID[0]), float(GRID[-1]), 0.005],
           "data": {"dev_clips": len(dev), "eval_clips": len(ev_clips), "dev_licences": {k: int(v) for k, v in zip(*np.unique(lic_dev, return_counts=True))},
                    "dev_audio_hours": float(np.sum(sc_dev["duration_s"]) / 3600), "dev_scoring_seconds": float(sc_dev["elapsed_s"][0])},
           **out, "runtime_s": round(time.perf_counter() - t0, 1)}
    write_json(out_dir / "metrics.json", res)
    write_json(REPO_ROOT / CONTEXT_ARTIFACT, context_artifact(res))
    return res


CONTEXT_ARTIFACT = "artifacts/audio_context_thresholds.json"


def context_artifact(res: dict, variant: str = "cc0_ccby") -> dict:
    """Context-tag thresholds adopted under the pre-set rule, from the licence-clean variant (usable in every profile)."""
    v = res["variants"][variant]
    tags = {}
    for k, r in v["results"].items():
        if k.startswith("context:") and r["adopt"]:
            t = k.split(":", 1)[1]
            tags[t] = {"threshold": v["selected"]["context"][t], "eval_f1_deployed": round(r["eval_f1_deployed"], 4),
                       "eval_f1_selected": round(r["eval_f1_selected"], 4),
                       "paired_f1_difference_ci95": [round(r["paired_f1_difference_ci95"]["ci_low"], 4), round(r["paired_f1_difference_ci95"]["ci_high"], 4)],
                       "passes_gate": r["passes_gate_selected"]}
    return {"format": "petperception.audio_context_thresholds/1", "source": f"E21b, variant {variant} (FSD50K dev CC0 / CC BY clips; FSD50K is CC BY 4.0)",
            "rule": v["adoption_rule"], "other_tags": "context_tag_threshold from the configuration", "created_at": res["created_at"], "tags": tags}


def _e21_consistency(f1_deployed: dict[str, float]) -> dict:
    """The deployed-point F1s recomputed here must equal E21's (same cached scores, same rule)."""
    import json

    p = REPO_ROOT / "results" / "e21" / "metrics.json"
    if not p.exists():
        return {"checked": False}
    r = json.loads(p.read_text())["results"]
    ref = {"cat_vocalization": r["species_presence"]["cat_vocalization"], "dog_vocalization": r["species_presence"]["dog_vocalization"],
           **{k: r["vocalization_type"][t] for k, t in DECISION_TARGET.items() if t in r["vocalization_type"]},
           **{f"context:{t}": v for t, v in r["context_tags"].items()}}
    diffs = {k: abs(f1_deployed[k] - ref[k]["all"]["f1"]) for k in f1_deployed if k in ref}
    return {"checked": True, "max_abs_f1_difference": max(diffs.values()), "identical": max(diffs.values()) < 1e-12, "per_decision": diffs}


def _esc50_check(settings: Settings, out: dict, deployed: dict) -> dict:
    """Species F1 on all 2,000 ESC-50 clips (E1's cached clip scores) at the deployed and the selected thresholds."""
    from ..data import esc50
    from .e1_audio_events import compute_inputs

    data_root = settings.paths.resolve("data_root")
    root = esc50.dataset_dir(data_root)
    clips = sorted(esc50.load_meta(root), key=lambda c: c.filename)
    X = compute_inputs(settings, root, clips, data_root.parent / "cache")
    y_cat = np.array([c.pet_label == "cat" for c in clips])
    y_dog = np.array([c.pet_label == "dog" for c in clips])
    res: dict = {"clips": len(clips), "note": "each species scored on its own group score (E1's reported rule is exclusive cat / dog / other); "
                                               "the deployed thresholds were fitted on these clips, so 'deployed' is in-sample"}
    for name, p in [("deployed", deployed)] + [(v, out["variants"][v]["selected"]) for v in out["variants"]]:
        res[name] = {"cat_f1": _f1(y_cat, X["cat_score"] >= p["species"]["cat_vocalization"]),
                     "dog_f1": _f1(y_dog, X["dog_score"] >= p["species"]["dog_vocalization"])}
    return res
