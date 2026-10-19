"""E21: species vocalization, vocalization type and context sounds on FSD50K eval, at the deployed operating point.

Question. Do the audio decisions the product makes today hold on a large,
independent, human-labelled set? E1 measured species presence on ESC-50
(40 cat and 40 dog clips). Vocalization types and context sounds have never
been evaluated: no type-labelled set was reachable.

Data. FSD50K v1.0 eval split (10,231 Freesound clips, human-verified AudioSet
labels, per-clip CC licences; `petperception.data.fsd50k`). Each archive file is
checked against the MD5 published on the Zenodo record.

Protocol, fixed before any FSD50K clip was scored:
* Operating point: exactly what both engines ship. Species thresholds come
  from artifacts/audio_event_thresholds.json (E1). The growl threshold (0.5),
  the context-tag threshold (0.3) and the event gap come from configs/cpu.yaml.
  Events come from `extract_events`. Tags come from `vocal_tags`, which
  applies the subtype gate SUBTYPE_MIN = 0.3 inside species or growl events.
  Nothing is tuned on FSD50K here; E21 only scores. Selecting new thresholds
  needs the dev split and is a separate experiment.
* Clip-level decisions:
  - species present: at least one event of that species group;
  - subtype k (meow, purr, hiss, bark, growl): k is a confident subtype
    (>= SUBTYPE_MIN) of a species or growl event, or, for growl, a growl
    event exists (the `vocal_tags` rule);
  - context tag: at least one context event of that tag.
* Targets, from FSD50K labels (ancestors included by the publishers):
  - cat_vocalization: Cat. dog_vocalization: Dog.
  - meow: Meow. purr: Purr. bark: Bark.
  - hiss_cat: Hiss and Cat. Most FSD50K "Hiss" clips are steam, fire or
    traffic; those are hard negatives for the product's hiss tag.
  - growl_any: Growling (the product's growl is species-ambiguous).
    growl_dog: Growling and Dog.
  - context tags: the FSD50K label with the same AudioSet MID as the
    product's tag; tags without one are listed as not evaluable.
* Leakage control: ESC-50, whose clips selected the E1 thresholds, is cut
  from Freesound recordings. Eval clips whose Freesound id is an ESC-50
  source are removed. YAMNet was trained on AudioSet (YouTube audio), not on
  Freesound.
* Metrics per target: precision, recall and F1 at the deployed decision. 95 %
  CIs come from a bootstrap that resamples Freesound uploaders, since one
  uploader often contributes many similar clips. Also reported: average
  precision of the ungated clip score (maximum window score), a
  threshold-free view; false positives by their most frequent FSD50K labels;
  and the CC0 + CC BY subset.
* Acceptance (docs/TRAINING_PROGRAM.md G4, set before this run): a
  vocalization type may be shown on the phone only if F1 >= 0.7 and the
  lower 95 % bound >= 0.6. Context tags and species presence use the same
  gate here.

Terms: FSD50K is CC BY 4.0 as a collection, with per-clip CC licences; it is
used for evaluation and nothing derived from it ships.
"""

from __future__ import annotations

import csv
import io
import logging
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio.events import CONTEXT_TAG_MIDS, SUBTYPE_MIN, build_groups, extract_events, load_thresholds, vocal_tags
from ..config import REPO_ROOT, Settings
from ..data import fsd50k
from ..ingest.media import AudioTrack
from ..utils import utc_now, write_json
from . import metrics as M

log = logging.getLogger("petperception.e21")
CACHE_VERSION = "yamnet_scores_v1"
GENERIC = {"Animal", "Domestic_animals_and_pets"}
SPECIES_TARGETS = {
    "cat_vocalization": lambda L: "Cat" in L,
    "dog_vocalization": lambda L: "Dog" in L,
}
TYPE_TARGETS = {
    "meow": lambda L: "Meow" in L,
    "purr": lambda L: "Purr" in L,
    "hiss_cat": lambda L: "Hiss" in L and "Cat" in L,
    "bark": lambda L: "Bark" in L,
    "growl_any": lambda L: "Growling" in L,
    "growl_dog": lambda L: "Growling" in L and "Dog" in L,
}
TYPE_DECISION = {"meow": "meow", "purr": "purr", "hiss_cat": "hiss", "bark": "bark", "growl_any": "growl", "growl_dog": "growl"}
NOT_EVALUABLE_TYPES = {k: "no FSD50K label" for k in ("caterwaul", "yip", "howl", "bow_wow", "whimper_dog")}
GATE = {"f1": 0.7, "ci_low": 0.6}
MIN_POSITIVES = 20  # fewer eval positives than this: "not_evaluable" (the E15 gate used the same minimum)
# Product subtype key -> E21 target (the product's growl is species-ambiguous: growl_any).
SUBTYPE_TARGET = {"meow": "meow", "purr": "purr", "hiss": "hiss_cat", "bark": "bark", "growl": "growl_any"}
ARTIFACT = "artifacts/audio_subtype_validation.json"


def _score_shard(args: tuple) -> dict:
    """Worker: YAMNet window scores for a list of clips (one interpreter per process)."""
    data_root, split, fnames, model_dir, hop_s = args
    from ..audio.yamnet import YamnetBackend

    backend = YamnetBackend(Path(model_dir), 1)
    z = fsd50k.audio_archive(Path(data_root), split)
    out = {"fname": [], "n": [], "scores": [], "sr": [], "dur": []}
    for f in fnames:
        x, sr = sf.read(io.BytesIO(z.read(fsd50k.member_name(split, f))), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        ws = backend.score_track(AudioTrack("present", x, sr, 0.0), hop_s=hop_s)
        out["fname"].append(f)
        out["n"].append(ws.scores.shape[0])
        out["scores"].append(ws.scores.astype(np.float32))
        out["sr"].append(sr)
        out["dur"].append(len(x) / sr)
    return out


def score_clips(settings: Settings, split: str, fnames: list[str], cache: Path, workers: int) -> dict:
    """Window scores of every clip, cached as one array with per-clip offsets."""
    from ..registry import model_dir

    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        if list(d["fname"]) == fnames:
            return {k: d[k] for k in d.files}
        log.warning("cache %s lists other clips; rescoring", cache)
    mdir = str(model_dir(settings, settings.audio.yamnet.model_id))
    shards = [fnames[i::workers * 8] for i in range(workers * 8)]
    t0 = time.perf_counter()
    parts = []
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_score_shard, [(str(settings.paths.resolve("data_root")), split, s, mdir, settings.audio.yamnet.hop_s) for s in shards])):
            parts.append(r)
            log.info("E21 scored shard %d/%d (%.0f s)", k + 1, len(shards), time.perf_counter() - t0)
    order = {f: i for i, f in enumerate(fnames)}
    rows = sorted(((f, n, s, sr, du) for p in parts for f, n, s, sr, du in zip(p["fname"], p["n"], p["scores"], p["sr"], p["dur"])), key=lambda r: order[r[0]])
    n = np.array([r[1] for r in rows])
    out = {"fname": np.array([r[0] for r in rows]), "offset": np.concatenate([[0], np.cumsum(n)]), "scores": np.concatenate([r[2] for r in rows]),
           "sample_rate": np.array([r[3] for r in rows]), "duration_s": np.array([r[4] for r in rows]),
           "elapsed_s": np.array([time.perf_counter() - t0]), "workers": np.array([workers])}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def esc50_freesound_ids(settings: Settings) -> set[str]:
    with open(settings.paths.resolve("manifests_root") / "esc50_files.csv", newline="", encoding="utf-8") as fh:
        return {r["source_group"].split(":", 1)[1] for r in csv.DictReader(fh) if r["source_group"].startswith("freesound:")}


def _report(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, score: np.ndarray | None, n_boot: int, seed: int) -> dict:
    tp, fp, fn = int(np.sum(y & pred)), int(np.sum(~y & pred)), int(np.sum(y & ~pred))

    def f1(ix: np.ndarray) -> float:
        a, b, c = np.sum(y[ix] & pred[ix]), np.sum(~y[ix] & pred[ix]), np.sum(y[ix] & ~pred[ix])
        return float(2 * a / max(1, 2 * a + b + c))

    rep = {"positives": int(y.sum()), "negatives": int((~y).sum()), "tp": tp, "fp": fp, "fn": fn,
           "precision": tp / max(1, tp + fp) if tp + fp else None, "recall": tp / max(1, tp + fn) if tp + fn else None,
           "f1": f1(np.arange(len(y))), "false_positive_rate": fp / max(1, int((~y).sum())),
           "positive_uploaders": int(len(set(groups[y].tolist())))}
    rep["f1_ci95_by_uploader"] = M.group_bootstrap(f1, groups.tolist(), n_boot, seed) if y.sum() else None
    if score is not None and y.sum():
        rep["average_precision_ungated_score"] = float(M.average_precision(score, y))
    lo = (rep["f1_ci95_by_uploader"] or {}).get("ci_low")
    rep["passes_gate"] = bool(y.sum() and rep["f1"] >= GATE["f1"] and lo is not None and lo >= GATE["ci_low"])
    return rep


def run(settings: Settings, out_dir: Path, workers: int = 3, n_boot: int = 1000, seed: int = 0) -> dict:
    from ..audio.yamnet import YamnetBackend
    from ..registry import model_dir

    t0 = time.perf_counter()
    data_root = settings.paths.resolve("data_root")
    clips = fsd50k.load_clips(data_root, "eval")
    dev = fsd50k.load_clips(data_root, "dev")
    esc = esc50_freesound_ids(settings)
    excluded = sorted(c.fname for c in clips if c.fname in esc)
    clips = [c for c in clips if c.fname not in esc]
    fnames = [c.fname for c in clips]
    cache = data_root / "fsd50k" / f"{CACHE_VERSION}_eval.npz"
    sc = score_clips(settings, "eval", fnames, cache, workers)

    backend = YamnetBackend(model_dir(settings, settings.audio.yamnet.model_id), 1)
    groups = build_groups(backend)
    y_cfg = settings.audio.yamnet
    th_path = REPO_ROOT / y_cfg.thresholds_file
    thresholds, th_src = load_thresholds(th_path, y_cfg.default_thresholds)
    from ..audio.yamnet import WindowScores

    sub_idx = {k: backend.mid_to_index[m] for k, m in {"meow": "/m/07qrkrw", "purr": "/m/02yds9", "hiss": "/m/07rjwbb", "bark": "/m/05tny_",
                                                       "growl": "/m/0ghcn6"}.items()}
    ctx_idx = {t: backend.mid_to_index[m] for t, m in CONTEXT_TAG_MIDS.items() if m in backend.mid_to_index}
    n = len(clips)
    dec_species = {k: np.zeros(n, bool) for k in SPECIES_TARGETS}
    dec_sub = {k: np.zeros(n, bool) for k in sub_idx}
    dec_ctx = {t: np.zeros(n, bool) for t in ctx_idx}
    s_species = {k: np.zeros(n, np.float32) for k in SPECIES_TARGETS}
    s_sub = {k: np.zeros(n, np.float32) for k in sub_idx}
    s_ctx = {t: np.zeros(n, np.float32) for t in ctx_idx}
    for i in range(n):
        S = sc["scores"][sc["offset"][i]:sc["offset"][i + 1]]
        t = np.arange(S.shape[0]) * y_cfg.hop_s
        ws = WindowScores(t, t + 0.975, S)
        for k in SPECIES_TARGETS:
            s_species[k][i] = S[:, groups[k].indices].max()
        for k, j in sub_idx.items():
            s_sub[k][i] = S[:, j].max()
        for tg, j in ctx_idx.items():
            s_ctx[tg][i] = S[:, j].max()
        for ev in extract_events(ws, backend, groups, thresholds, y_cfg.context_tag_threshold, y_cfg.min_event_gap_s):
            if ev.label in dec_species:
                dec_species[ev.label][i] = True
            if ev.label.startswith("context:"):
                dec_ctx[ev.label.split(":", 1)[1]][i] = True
                continue
            strong = {k for k, v in ev.subtypes.items() if v >= SUBTYPE_MIN}
            for k in (strong | vocal_tags(ev.label, ev.subtypes)) & set(dec_sub):
                dec_sub[k][i] = True

    labels = [c.labels for c in clips]
    uploaders = np.array([c.uploader for c in clips])
    lic = np.array([c.licence for c in clips])
    mid_of_label: dict[str, str] = {}
    meta = fsd50k.root(data_root) / "meta" / "FSD50K.ground_truth" / "vocabulary.csv"
    with open(meta, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            mid_of_label[row[1]] = row[2]  # index, label, AudioSet id
    subsets = {"all": np.ones(n, bool), "cc0_ccby": np.isin(lic, ["CC0", "CC BY"])}

    def fp_labels(y: np.ndarray, pred: np.ndarray) -> list:
        return Counter(l for i in np.where(~y & pred)[0] for l in labels[i] if l not in GENERIC).most_common(8)

    results: dict = {"species_presence": {}, "vocalization_type": {}, "context_tags": {}}
    for name, fn_, pred, score, key in ([(k, SPECIES_TARGETS[k], dec_species[k], s_species[k], "species_presence") for k in SPECIES_TARGETS]
                                        + [(k, TYPE_TARGETS[k], dec_sub[TYPE_DECISION[k]], s_sub[TYPE_DECISION[k]], "vocalization_type") for k in TYPE_TARGETS]):
        y = np.array([fn_(L) for L in labels])
        results[key][name] = {sub: _report(y[m], pred[m], uploaders[m], score[m], n_boot, seed) for sub, m in subsets.items()}
        results[key][name]["false_positives_by_label"] = fp_labels(y, pred)
        results[key][name]["decision"] = (f"event of group {name}" if key == "species_presence" else
                                          f"'{TYPE_DECISION[name]}' confident subtype (>= {SUBTYPE_MIN}) of a species or growl event, or a growl event for growl")
    not_eval_ctx = {}
    for tg, mid in CONTEXT_TAG_MIDS.items():
        lab = next((k for k, v in mid_of_label.items() if v == mid), None)
        if lab is None or tg not in ctx_idx:
            not_eval_ctx[tg] = f"no FSD50K label with AudioSet id {mid}"
            continue
        y = np.array([lab in L for L in labels])
        results["context_tags"][tg] = {sub: _report(y[m], dec_ctx[tg][m], uploaders[m], s_ctx[tg][m], n_boot, seed) for sub, m in subsets.items()}
        results["context_tags"][tg]["fsd50k_label"] = lab
        results["context_tags"][tg]["false_positives_by_label"] = fp_labels(y, dec_ctx[tg])

    eval_up, dev_up = set(uploaders.tolist()), {c.uploader for c in dev}
    out = {
        "experiment": "E21", "status": "completed", "created_at": utc_now().isoformat(timespec="seconds"),
        "data": {"dataset": "FSD50K v1.0 eval split (Zenodo 4060432; archive MD5s verified by the importer)", "clips_scored": n,
                 "excluded_esc50_sources": {"count": len(excluded), "fnames": excluded},
                 "uploaders": len(eval_up), "uploaders_shared_with_dev": len(eval_up & dev_up),
                 "licences": dict(Counter(lic.tolist())), "audio_hours": float(np.sum(sc["duration_s"]) / 3600),
                 "sample_rates_hz": {str(k): int(v) for k, v in zip(*np.unique(sc["sample_rate"], return_counts=True))}},
        "operating_point": {"thresholds": thresholds, "threshold_source": th_src, "subtype_min": SUBTYPE_MIN,
                            "context_tag_threshold": y_cfg.context_tag_threshold, "min_event_gap_s": y_cfg.min_event_gap_s, "hop_s": y_cfg.hop_s,
                            "selected_on_fsd50k": False},
        "gate": GATE, "results": results,
        "not_evaluable": {"vocalization_type": NOT_EVALUABLE_TYPES, "context_tags": not_eval_ctx},
        "scoring_seconds": float(sc["elapsed_s"][0]), "scoring_workers": int(sc["workers"][0]),
        "runtime_s": round(time.perf_counter() - t0, 1),
        "terms": "FSD50K: CC BY 4.0 collection, per-clip CC licences; evaluation only, nothing derived ships",
    }
    write_json(out_dir / "metrics.json", out)
    write_json(out_dir / "summary.json", _summary(out))
    write_json(REPO_ROOT / ARTIFACT, validation_artifact(out))
    return out


def validation_artifact(out: dict) -> dict:
    """Per-subtype status under the pre-set gate, read by both engines (shown / trigger only if 'passed')."""
    subs: dict = {}
    for k, t in SUBTYPE_TARGET.items():
        a = out["results"]["vocalization_type"][t]["all"]
        ci = a["f1_ci95_by_uploader"] or {}
        if a["positives"] < MIN_POSITIVES:
            subs[k] = {"status": "not_evaluable", "reason": f"{a['positives']} positives in FSD50K eval (minimum {MIN_POSITIVES})", "target": t}
        else:
            subs[k] = {"status": "passed" if a["passes_gate"] else "failed", "target": t, "positives": a["positives"],
                       "f1": round(a["f1"], 4), "f1_ci95": [round(ci["ci_low"], 4), round(ci["ci_high"], 4)],
                       "precision": round(a["precision"], 4) if a["precision"] is not None else None,
                       "recall": round(a["recall"], 4) if a["recall"] is not None else None}
    for k, why in NOT_EVALUABLE_TYPES.items():
        subs[k] = {"status": "not_evaluable", "reason": why}
    return {"format": "petperception.audio_subtype_validation/1", "source": "E21, FSD50K v1.0 eval, deployed operating point (results/e21/metrics.json)",
            "gate": {**GATE, "min_positives": MIN_POSITIVES}, "rule": "a vocalization subtype is shown on the phone or used as an interpretation trigger only if its status is 'passed' (docs/TRAINING_PROGRAM.md G4)",
            "created_at": out["created_at"], "subtypes": subs}


def _summary(out: dict) -> dict:
    rows = {}
    for group, items in out["results"].items():
        for name, r in items.items():
            a = r["all"]
            ci = a.get("f1_ci95_by_uploader") or {}
            rows[f"{group}/{name}"] = {"positives": a["positives"], "precision": a["precision"], "recall": a["recall"], "f1": a["f1"],
                                      "f1_ci95": [ci.get("ci_low"), ci.get("ci_high")], "ap_ungated": a.get("average_precision_ungated_score"),
                                      "fpr": a["false_positive_rate"], "passes_gate": a["passes_gate"],
                                      "f1_cc0_ccby": r["cc0_ccby"]["f1"], "positives_cc0_ccby": r["cc0_ccby"]["positives"]}
    return {"experiment": "E21", "gate": out["gate"], "rows": rows, "clips": out["data"]["clips_scored"]}

