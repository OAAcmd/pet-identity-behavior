"""E19: choosing the video operating point on real pet video (no weight training).

Why: the detector threshold (0.35) and the verifier thresholds were chosen
on web photos (E4, E7). On real pet video the pet is found in only about
half of the clips (E18), and removing the verifier recovers some clips at
the cost of species confusion. This experiment selects the operating point
on real video and checks it on separate real video once.

Data:
* tuning: Kinetics-700-2020 *train* clips, the first 50 of each of the six
  pet-action classes (300 clips; data/kinetics_pets.fetch_train_sample);
* evaluation: the 294 validation clips of E18, scored once;
* photo guard: per species, the E4 images verified negative for it.

Method: each clip is decoded once at the engine's sampling (2 fps). Every
cat/dog detection down to score 0.15 is cached with its verifier score (the
top six per frame), together with the camera-motion estimate. The engine's
own merge, tracker and track-level verifier rule are then replayed for
every setting of the pre-specified grid:
* detector threshold: 0.20, 0.25, 0.30, 0.35 (current);
* verifier threshold scale: 1 (current), 0.5, 0.25, 0 (verifier off).
The current setting replayed must reproduce the engine (fidelity is
reported against E18's engine runs).

Selection rule, fixed before any run: on the tuning clips, take the
setting with the highest mean over species of "right species tracked",
among settings whose rate of "a wrong-species track appears" does not
exceed the current setting's rate by more than 0.03 for either species.
Ties go to the higher detector threshold, then the higher verifier scale.
The selected setting is compared once with the current one on the
evaluation clips (paired exact McNemar per species) and on the photo guard
(per species, image false-positive rate on images verified negative for
that species).
"""

from __future__ import annotations

import gzip
import json
from math import comb
from pathlib import Path

import numpy as np

from ..config import Settings, load_settings
from ..utils import utc_now, write_json
from .e16_breed_robustness import wilson

LOW = 0.15
TOP_VERIFY = 6
THRESHOLDS = (0.20, 0.25, 0.30, 0.35)
TAU_SCALES = (1.0, 0.5, 0.25, 0.0)
CURRENT = (0.35, 1.0)
MAX_WRONG_INCREASE = 0.03
DETECTORS = ("efficientdet_lite0_tfhub_rawdecode_v1", "efficientdet_lite0_mediapipe_v1")


def _engine(detector: str):
    from ..pipeline import AnalysisEngine

    s = load_settings(overrides={"release_profile": "research", "pose": {"provider": "none"}, "vision": {"detector": detector}})
    return s, AnalysisEngine(s).load()


def extract(s: Settings, eng, path: str) -> list[dict]:
    """Per sampled frame: detections >= LOW with verifier scores, camera motion."""
    from ..ingest import media as M
    from ..vision.detector import Detection
    from ..vision.quality import frame_quality
    from ..vision.tracking import estimate_camera_motion

    mp = M.probe(path, s.limits)
    frames, prev_gray, prev_t, prev_boxes = [], None, None, []
    for fr in M.iter_frames(mp, s.vision.sample_fps, s.vision.max_side, s.limits):
        _, gray = frame_quality(fr.t_s, fr.rgb, s.vision.quality)
        dets = eng.detector.detect(fr.rgb, ["cat", "dog"], LOW, s.vision.nms_iou, s.vision.max_detections_per_frame)
        dets = sorted(dets, key=lambda d: -d.score)
        rows = []
        for k, d in enumerate(dets):
            v = None
            if eng.verifier is not None and k < TOP_VERIFY:
                r = eng.verifier.verify(fr.rgb, Detection(d.label, d.score, d.box, d.class_scores))
                v = None if r is None else float(r[0])
            rows.append({"label": d.label, "score": float(d.score), "box": [float(x) for x in d.box], "v": v,
                         "cs": None if d.class_scores is None else [float(x) for x in d.class_scores]})
        cm = None
        cur_boxes = [np.array(r["box"]) for r in rows if r["score"] >= CURRENT[0]]
        if prev_gray is not None and not mp.video.is_still_image:
            c = estimate_camera_motion(prev_gray, gray, prev_boxes + cur_boxes, prev_t, fr.t_s)
            cm = None if c is None else {"dx": c.dx, "dy": c.dy, "scale": c.scale, "reliable": bool(c.reliable)}
        frames.append({"t": fr.t_s, "i": fr.index, "dets": rows, "cm": cm})
        prev_gray, prev_t, prev_boxes = gray, fr.t_s, cur_boxes
    return frames


def replay(s: Settings, taus: dict[str, float], frames: list[dict], thr: float | dict[str, float], tau_scale: float) -> set[str]:
    """Species with at least one reported track, using the engine's merge, tracker and verifier rule.

    `thr` is one detector threshold, or one per species (E19b)."""
    from ..vision.detector import Detection
    from ..vision.tracking import make_tracker, merge_species

    th = thr if isinstance(thr, dict) else {"cat": thr, "dog": thr}
    tracker = make_tracker(s.vision.tracker.method, s.vision.tracker.iou_threshold, s.vision.tracker.max_gap_s)
    for f in frames:
        raw = [(Detection(r["label"], r["score"], np.array(r["box"]), None if r["cs"] is None else np.array(r["cs"])), r["v"])
               for r in f["dets"] if r["score"] >= th[r["label"]]]
        vmap = {id(d): v for d, v in raw}
        adets = merge_species([d for d, _ in raw], f["t"], f["i"])
        for a in sorted(adets, key=lambda x: -x.score)[: s.vision.verifier.max_checks_per_frame]:
            a.verify_score = vmap.get(id(a.source))
        shift = (0.0, 0.0)
        cm = f["cm"]
        if cm is not None and cm["reliable"]:
            if (np.hypot(cm["dx"], cm["dy"]) > s.vision.quality.camera_motion_threshold or abs(cm["scale"] - 1) > 0.02) and s.vision.camera_motion_compensation:
                shift = (cm["dx"], cm["dy"])
        tracker.update(f["t"], adets, shift)
    n = len(frames)
    min_hits = s.vision.tracker.min_hits if n >= 2 * s.vision.tracker.min_hits else 1
    out = set()
    for tr in tracker.tracks(min_hits):
        vs = [d.verify_score for d in tr.detections if d.verify_score is not None]
        tau = taus.get(tr.species)
        if tau is None or not vs or float(np.median(vs)) >= tau * tau_scale:
            out.add(tr.species)
    return out


def _cached(s: Settings, eng, rows: list[dict], cache: Path) -> dict[str, list[dict]]:
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for r in rows:
        p = cache / f"{r['clip']}.json.gz"
        if p.exists():
            out[r["clip"]] = json.loads(gzip.decompress(p.read_bytes()))
            continue
        try:
            fr = extract(s, eng, r["path"])
        except Exception as exc:  # noqa: BLE001 - a clip that cannot be decoded is recorded, not fatal
            fr = {"error": type(exc).__name__}
        p.write_bytes(gzip.compress(json.dumps(fr).encode()))
        out[r["clip"]] = fr
    return out


def _metrics(rows: list[dict], species_sets: dict[str, set[str]]) -> dict:
    out = {}
    for sp in ("cat", "dog"):
        rs = [r for r in rows if r["species"] == sp and r["clip"] in species_sets]
        other = "dog" if sp == "cat" else "cat"
        k_right = sum(sp in species_sets[r["clip"]] for r in rs)
        k_wrong = sum(other in species_sets[r["clip"]] for r in rs)
        out[sp] = {"n": len(rs), "right_species": k_right / len(rs) if rs else None, "right_ci95": wilson(k_right, len(rs)),
                   "wrong_species_track": k_wrong / len(rs) if rs else None, "k_right": k_right, "k_wrong": k_wrong}
    return out


def _mcnemar(rows: list[dict], a: dict[str, set[str]], b: dict[str, set[str]], sp: str) -> dict:
    rs = [r for r in rows if r["species"] == sp and r["clip"] in a and r["clip"] in b]
    only_a = sum(sp in a[r["clip"]] and sp not in b[r["clip"]] for r in rs)
    only_b = sum(sp in b[r["clip"]] and sp not in a[r["clip"]] for r in rs)
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"only_current": only_a, "only_selected": only_b, "exact_p": p}


def photo_guard(s: Settings, eng, taus: dict[str, float], setting: tuple[float, float], manifest_dir: Path, image_dir: Path) -> dict:
    """Per species, on E4 images verified negative for it: share with a detection of that species passing detector and verifier."""
    import csv

    import cv2

    from ..vision.detector import Detection

    rows = list(csv.DictReader(open(manifest_dir / "openimages_catdog_images.csv", newline="")))
    out = {}
    for sp in ("cat", "dog"):
        fp, n = 0, 0
        for r in rows:
            if r[f"{sp}_status"] != "neg":
                continue
            p = image_dir / f"{r['image_id']}.jpg"
            bgr = cv2.imread(str(p)) if p.exists() else None
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            n += 1
            dets = sorted(eng.detector.detect(rgb, ["cat", "dog"], setting[0], s.vision.nms_iou, s.vision.max_detections_per_frame), key=lambda d: -d.score)
            for d in dets[:3]:
                if d.label != sp:
                    continue
                v = eng.verifier.verify(rgb, Detection(d.label, d.score, d.box, d.class_scores)) if eng.verifier is not None else None
                if v is None or v[0] >= taus.get(sp, 0.0) * setting[1]:
                    fp += 1
                    break
        out[sp] = {"verified_negative_images": n, "false_positive_images": fp, "rate": fp / n if n else None, "ci95": wilson(fp, n)}
    return out


def run(settings: Settings, out_dir: Path) -> dict:
    from ..data import kinetics_pets as KP

    data = settings.paths.resolve("data_root")
    tune_rows, eval_rows = KP.train_sample(data), KP.clips(data)
    summary: dict = {"experiment": "E19", "title": "Video operating point chosen on real pet video (Kinetics train), checked once (Kinetics val)",
                     "protocol": __doc__.strip().splitlines(), "grid": {"thresholds": THRESHOLDS, "tau_scales": TAU_SCALES, "current": CURRENT},
                     "clips": {"tuning": len(tune_rows), "evaluation": len(eval_rows)}, "detectors": {}}
    for det in DETECTORS:
        s, eng = _engine(det)
        taus = {c: cv.tau for c, cv in eng.verifier.per_class.items()} if eng.verifier is not None else {}
        cache = data.parent / "cache" / "e19" / det
        tune = _cached(s, eng, tune_rows, cache / "tune")
        ev = _cached(s, eng, eval_rows, cache / "eval")
        tune_ok = [r for r in tune_rows if isinstance(tune[r["clip"]], list)]
        eval_ok = [r for r in eval_rows if isinstance(ev[r["clip"]], list)]
        grid = {}
        for thr in THRESHOLDS:
            for ts in TAU_SCALES:
                sets = {r["clip"]: replay(s, taus, tune[r["clip"]], thr, ts) for r in tune_ok}
                grid[(thr, ts)] = _metrics(tune_ok, sets)
        cur = grid[CURRENT]
        allowed = [k for k, m in grid.items()
                   if all(m[sp]["wrong_species_track"] <= cur[sp]["wrong_species_track"] + MAX_WRONG_INCREASE for sp in ("cat", "dog"))]
        sel = max(allowed, key=lambda k: (np.mean([grid[k][sp]["right_species"] for sp in ("cat", "dog")]), k[0], k[1]))
        cur_sets = {r["clip"]: replay(s, taus, ev[r["clip"]], *CURRENT) for r in eval_ok}
        sel_sets = {r["clip"]: replay(s, taus, ev[r["clip"]], *sel) for r in eval_ok}
        e18_cache = data.parent / "cache" / "e18" / ("results_phone_build.jsonl" if det == DETECTORS[0] else "results.jsonl")
        fidelity = None
        if e18_cache.exists():
            e18 = {json.loads(line)["clip"]: set(json.loads(line).get("track_species", [])) for line in e18_cache.read_text().splitlines()}
            same = [r["clip"] for r in eval_ok if r["clip"] in e18]
            fidelity = {"clips": len(same), "species_sets_identical": sum(cur_sets[c] == e18[c] for c in same)}
        block = {"verifier_taus": taus, "tuning_grid": {f"thr={k[0]},tau_scale={k[1]}": v for k, v in grid.items()},
                 "allowed_settings": [f"thr={k[0]},tau_scale={k[1]}" for k in allowed], "selected": {"threshold": sel[0], "tau_scale": sel[1]},
                 "evaluation": {"current": _metrics(eval_ok, cur_sets), "selected": _metrics(eval_ok, sel_sets),
                                "paired": {sp: _mcnemar(eval_ok, cur_sets, sel_sets, sp) for sp in ("cat", "dog")}},
                 "replay_fidelity_vs_engine_e18": fidelity,
                 "decode_failures": {"tuning": len(tune_rows) - len(tune_ok), "evaluation": len(eval_rows) - len(eval_ok)}}
        if sel != CURRENT:
            img_dir = data / "openimages" / "images"
            block["photo_guard"] = {"current": photo_guard(s, eng, taus, CURRENT, settings.paths.resolve("manifests_root"), img_dir),
                                    "selected": photo_guard(s, eng, taus, sel, settings.paths.resolve("manifests_root"), img_dir)}
        summary["detectors"][det] = block
    summary["terms"] = "Kinetics-700-2020 (CVDF mirror): research evaluation only; clips not committed."
    summary["created_at"] = utc_now().isoformat(timespec="seconds")
    write_json(out_dir / "metrics.json", summary)
    return summary


E19B_DOC = """E19b: per-species detector thresholds, phone-build detector, fresh holdout.

E19 found that a lower detector threshold helps cats and costs dogs with
the phone build's detector, on both its tuning and its evaluation clips.
Choosing per-species thresholds from the E19 evaluation clips would reuse
them, so:
* grid: cat threshold x dog threshold in {0.20, 0.25, 0.30, 0.35}^2,
  verifier unchanged (E19 showed lowering it adds species confusion);
* selection on the E19 tuning clips (train clips 1-50 per class) with
  E19's rule: highest mean right-species rate, any wrong-species track
  rate at most 0.03 above the current setting for both species;
* evaluation once on a fresh holdout: train clips 51-100 per class
  (300 clips; no clip shared with tuning; validation untouched).
Adoption rule, fixed before the run: adopt if on the holdout neither
species loses right-species clips (selected >= current), at least one
species gains with paired exact McNemar p < 0.05, and neither species'
wrong-species track rate rises by more than 0.03.
"""


def run_per_species(settings: Settings, out_dir: Path) -> dict:
    from ..data import kinetics_pets as KP

    det = DETECTORS[0]
    data = settings.paths.resolve("data_root")
    s, eng = _engine(det)
    taus = {c: cv.tau for c, cv in eng.verifier.per_class.items()}
    cache = data.parent / "cache" / "e19" / det
    tune_rows, hold_rows = KP.train_sample(data), KP.train_sample(data, subdir="train_holdout")
    tune = _cached(s, eng, tune_rows, cache / "tune")
    hold = _cached(s, eng, hold_rows, cache / "holdout")
    tune_ok = [r for r in tune_rows if isinstance(tune[r["clip"]], list)]
    hold_ok = [r for r in hold_rows if isinstance(hold[r["clip"]], list)]
    grid = {}
    for tc in THRESHOLDS:
        for td in THRESHOLDS:
            sets = {r["clip"]: replay(s, taus, tune[r["clip"]], {"cat": tc, "dog": td}, 1.0) for r in tune_ok}
            grid[(tc, td)] = _metrics(tune_ok, sets)
    cur_key = (CURRENT[0], CURRENT[0])
    cur = grid[cur_key]
    allowed = [k for k, m in grid.items() if all(m[sp]["wrong_species_track"] <= cur[sp]["wrong_species_track"] + MAX_WRONG_INCREASE for sp in ("cat", "dog"))]
    sel = max(allowed, key=lambda k: (np.mean([grid[k][sp]["right_species"] for sp in ("cat", "dog")]), k[1], k[0]))
    cur_sets = {r["clip"]: replay(s, taus, hold[r["clip"]], {"cat": cur_key[0], "dog": cur_key[1]}, 1.0) for r in hold_ok}
    sel_sets = {r["clip"]: replay(s, taus, hold[r["clip"]], {"cat": sel[0], "dog": sel[1]}, 1.0) for r in hold_ok}
    mc, ms = _metrics(hold_ok, cur_sets), _metrics(hold_ok, sel_sets)
    paired = {sp: _mcnemar(hold_ok, cur_sets, sel_sets, sp) for sp in ("cat", "dog")}
    adopt = (all(ms[sp]["k_right"] >= mc[sp]["k_right"] for sp in ("cat", "dog"))
             and any(paired[sp]["exact_p"] < 0.05 and ms[sp]["k_right"] > mc[sp]["k_right"] for sp in ("cat", "dog"))
             and all(ms[sp]["wrong_species_track"] <= mc[sp]["wrong_species_track"] + MAX_WRONG_INCREASE for sp in ("cat", "dog")))
    out = {"experiment": "E19b", "detector": det, "protocol": E19B_DOC.strip().splitlines(), "verifier_taus": taus,
           "clips": {"tuning": len(tune_ok), "holdout": len(hold_ok)},
           "tuning_grid": {f"cat={k[0]},dog={k[1]}": v for k, v in grid.items()},
           "allowed": [f"cat={k[0]},dog={k[1]}" for k in allowed], "selected": {"cat": sel[0], "dog": sel[1]},
           "holdout": {"current": mc, "selected": ms, "paired": paired}, "adopted": bool(adopt),
           "created_at": utc_now().isoformat(timespec="seconds")}
    if sel != cur_key:
        img_dir = data / "openimages" / "images"
        guard = {}
        for name, k in (("current", cur_key), ("selected", sel)):
            g = {}
            for sp, th in (("cat", k[0]), ("dog", k[1])):
                g[sp] = photo_guard(s, eng, taus, (th, 1.0), settings.paths.resolve("manifests_root"), img_dir)[sp]
            guard[name] = g
        out["photo_guard"] = guard
    write_json(out_dir / "per_species.json", out)
    return out
