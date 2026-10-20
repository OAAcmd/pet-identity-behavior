"""E20: pet detection on home smartphone video (CoP3D), per species and per detector.

Question. E18 measured detection on YouTube clips (person-centred actions).
How often do the product's detectors find the pet in people's own phone
videos of their cat or dog at home, and how often do they report the other
species?

Data. CoP3D v1 (`petperception.data.cop3d`): every sequence of `cat_001.zip`
and `dog_001.zip`, 10 frames per sequence, evenly spaced. The sequence's
category is the species label. Each frame has a model-generated foreground
mask; its bounding box serves as the reference box. Terms: CC BY-NC 4.0,
research evaluation only.

Protocol, fixed before scoring:
* Detector configurations:
  - the phone build (TF Hub EfficientDet-Lite0, raw decoding, M3) with its
    verifier;
  - the reference detector (MediaPipe EfficientDet-Lite0, M2) with its
    verifier;
  - both detector-only;
  - optionally G1 exports, detector-only (`petperception g1-eval --e20`).
  At the product operating point: score >= `vision.score_threshold`,
  per-class NMS, classes cat and dog.
* Verifier: applied per detection (E7 protocol). The engine decides per
  track by the median verifier score, so this is a per-frame
  approximation of the engine.
* Frame outcome for species S:
  - found: an accepted S detection;
  - found (box): an accepted S detection with IoU >= 0.5 against the mask
    box, scored only on frames whose mask covers 0.5 % to 95 % of the image;
  - wrong species only: no accepted S detection but an accepted detection
    of the other species;
  - missed: neither.
* Sequence outcome: the share of its frames with the pet found; a sequence
  counts as found when at least half its frames are.
* 95 % CIs: bootstrap over contributors. A CoP3D sequence name starts with
  a number shared by many sequences, treated as the contributor (probably one
  household and pet): 14 contributors for the 209 cat sequences, 9 for the
  147 dog sequences. Comparisons with the phone build are paired by sequence
  and resampled by contributor. (Until 2026-10-20 the bootstrap resampled
  sequences, which gave intervals that were too narrow.)

Limitations stated up front: the reference boxes come from model-generated
masks; CoP3D videos circle a mostly static, centred pet (unlike home-camera
footage); the contributor grouping is inferred from sequence names, not
documented by the dataset.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..config import Settings
from ..data import cop3d
from ..utils import utc_now, write_json
from ..vision.detector import EfficientDetDetector
from . import metrics as M

SPECIES = ["cat", "dog"]
MASK_MIN, MASK_MAX = 0.005, 0.95


def contributor(sequence: str) -> str:
    """CoP3D contributor: the leading number of the sequence name (shared by many sequences)."""
    return sequence.split("_")[0]


def _iou(a: tuple, b: tuple) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score_frames(det, verifier, frames: list[cop3d.Frame], settings: Settings) -> list[dict]:
    """Per-frame outcome records for one detector configuration."""
    v = settings.vision
    out = []
    for f in frames:
        rgb = cv2.cvtColor(cv2.imread(f.path), cv2.COLOR_BGR2RGB)
        dets = det.detect(rgb, SPECIES, v.score_threshold, v.nms_iou, v.max_detections_per_frame)
        acc = []
        for d in dets:
            if verifier is not None:
                r = verifier.verify(rgb, d)
                if r is not None and not r[1]:
                    continue
            acc.append(d)
        own = [d for d in acc if d.label == f.species]
        other = [d for d in acc if d.label != f.species]
        box_ok = f.mask_box is not None and MASK_MIN <= f.mask_area <= MASK_MAX
        out.append({"species": f.species, "sequence": f.sequence, "frame": f.frame, "found": bool(own), "wrong_only": bool(other) and not own,
                    "box_scored": box_ok, "found_box": box_ok and any(_iou(tuple(d.box), f.mask_box) >= 0.5 for d in own),
                    "best_iou": max((_iou(tuple(d.box), f.mask_box) for d in own), default=0.0) if box_ok else None})
    return out


def summarize(rows: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    res = {}
    for sp in SPECIES:
        r = [x for x in rows if x["species"] == sp]
        if not r:
            continue
        seqs = sorted({x["sequence"] for x in r})
        by = {s: [x for x in r if x["sequence"] == s] for s in seqs}
        frac = np.array([np.mean([x["found"] for x in by[s]]) for s in seqs])
        wrong = np.array([np.mean([x["wrong_only"] for x in by[s]]) for s in seqs])
        boxed = [x for x in r if x["box_scored"]]
        groups = [contributor(x["sequence"]) for x in r]
        found = np.array([x["found"] for x in r])
        wonly = np.array([x["wrong_only"] for x in r])
        res[sp] = {
            "sequences": len(seqs), "frames": len(r), "contributors": len(set(groups)),
            "frame_found": float(found.mean()), "frame_found_ci95": M.group_bootstrap(lambda ix: float(found[ix].mean()), groups, n_boot, seed),
            "frame_wrong_species_only": float(wonly.mean()),
            "frame_wrong_species_only_ci95": M.group_bootstrap(lambda ix: float(wonly[ix].mean()), groups, n_boot, seed),
            "frame_missed": float(np.mean([not x["found"] and not x["wrong_only"] for x in r])),
            "frame_found_box_iou50": float(np.mean([x["found_box"] for x in boxed])) if boxed else None, "box_scored_frames": len(boxed),
            "median_best_iou_when_found": float(np.median([x["best_iou"] for x in boxed if x["found"]])) if any(x["found"] for x in boxed) else None,
            "sequence_found_half_frames": float(np.mean(frac >= 0.5)), "sequence_any_frame": float(np.mean(frac > 0)),
            "sequence_mean_found_fraction": float(frac.mean()), "sequence_mean_wrong_only_fraction": float(wrong.mean()),
        }
    return res


def paired(rows_a: list[dict], rows_b: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """Per-sequence found-fraction difference (a minus b), bootstrap over sequences."""
    out = {}
    for sp in SPECIES:
        fa = {}
        for x in rows_a:
            if x["species"] == sp:
                fa.setdefault(x["sequence"], []).append(x["found"])
        fb = {}
        for x in rows_b:
            if x["species"] == sp:
                fb.setdefault(x["sequence"], []).append(x["found"])
        seqs = sorted(set(fa) & set(fb))
        if not seqs:
            continue
        d = np.array([np.mean(fa[s]) - np.mean(fb[s]) for s in seqs])
        by_c: dict[str, list[int]] = {}
        for i, sq in enumerate(seqs):
            by_c.setdefault(contributor(sq), []).append(i)
        cs = sorted(by_c)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            ix = np.concatenate([by_c[cs[j]] for j in rng.integers(0, len(cs), len(cs))])
            boots.append(float(d[ix].mean()))
        out[sp] = {"mean_difference": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                   "sequences_better": int((d > 0).sum()), "sequences_worse": int((d < 0).sum()), "sequences": len(seqs)}
    return out


def shipped_configs(settings: Settings) -> dict:
    """(detector, verifier) of the phone build and the reference detector, loaded as the engine loads them."""
    from ..pipeline import AnalysisEngine

    out = {}
    for name, mid in (("phone_build_m3", "efficientdet_lite0_tfhub_rawdecode_v1"), ("reference_m2", "efficientdet_lite0_mediapipe_v1")):
        s = settings.model_copy(deep=True)
        s.vision.detector = mid
        s.audio.enabled = False
        eng = AnalysisEngine(s).load()
        if eng.detector is None:
            continue
        out[f"{name}+verifier"] = (eng.detector, eng.verifier)
        out[f"{name}_detector_only"] = (eng.detector, None)
    return out


def frames_all(settings: Settings, k: int = 10) -> list[cop3d.Frame]:
    data_root = settings.paths.resolve("data_root")
    return [f for sp in SPECIES for f in cop3d.sample(data_root, sp, k)]


def run(settings: Settings, out_dir: Path, extra_models: dict[str, Path] | None = None, shipped: bool = True) -> dict:
    """shipped=False (G1 exports): only the extra models and the phone build's detector alone, the like-for-like baseline."""
    t0 = time.perf_counter()
    frames = frames_all(settings)
    configs = shipped_configs(settings)
    if not shipped:
        configs = {k: v for k, v in configs.items() if k == "phone_build_m3_detector_only"}
    for name, p in (extra_models or {}).items():
        configs[f"{name}_detector_only"] = (EfficientDetDetector(p, settings.runtime.num_threads), None)
    rows = {name: score_frames(det, ver, frames, settings) for name, (det, ver) in configs.items()}
    base = "phone_build_m3+verifier" if shipped else "phone_build_m3_detector_only"
    res = {
        "experiment": "E20", "status": "completed", "created_at": utc_now().isoformat(timespec="seconds"),
        "data": {"dataset": "CoP3D v1 (dl.fbaipublicfiles.com/cop3d), frame archives " + ", ".join(a for v in cop3d.FRAME_ARCHIVES.values() for a in v),
                 "terms": "CC BY-NC 4.0 (LICENSE in the archive); research evaluation only",
                 "frames_per_sequence": 10, "frames": len(frames),
                 "sequences": {sp: len({f.sequence for f in frames if f.species == sp}) for sp in SPECIES},
                 "frame_sha256_digest": __import__("hashlib").sha256("".join(sorted(f.sha256 for f in frames)).encode()).hexdigest(),
                 "frames_without_usable_mask": sum(1 for f in frames if not (f.mask_box and MASK_MIN <= f.mask_area <= MASK_MAX))},
        "operating_point": {"score_threshold": settings.vision.score_threshold, "nms_iou": settings.vision.nms_iou,
                            "verifier": "per detection (E7 protocol); the engine decides per track"},
        "results": {name: summarize(r) for name, r in rows.items()},
        "paired_baseline": base,
        "paired_vs_phone_build": {name: paired(r, rows[base]) for name, r in rows.items() if name != base and base in rows},
        "protocol": __doc__.strip().splitlines(),
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    write_json(out_dir / "metrics.json", res)
    return res


def _clip(frames: list[cop3d.Frame], dst: Path, fps: float = 2.0) -> Path:
    """The sequence's sampled frames as a short H.264 clip, one frame per engine sample (2 fps)."""
    import subprocess

    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(f.path).resolve()}'\nduration {1 / fps}\n" for f in frames) + f"file '{Path(frames[-1].path).resolve()}'\n")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=10",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", str(dst)], check=True)
    return dst


def engine_level(settings: Settings, out_dir: Path, detector_id: str = "efficientdet_lite0_tfhub_rawdecode_v1", n_boot: int = 1000) -> dict:
    """E20b: the whole engine (detector, verifier at track level, tracker, species merge) on each sequence's frames
    shown as a 5 s clip; the outcome is the species of the reported tracks."""
    from ..pipeline import AnalysisEngine

    t0 = time.perf_counter()
    s = settings.model_copy(deep=True)
    s.vision.detector = detector_id
    s.audio.enabled = False
    eng = AnalysisEngine(s).load()
    frames = frames_all(settings)
    by_seq: dict[tuple[str, str], list[cop3d.Frame]] = {}
    for f in frames:
        by_seq.setdefault((f.species, f.sequence), []).append(f)
    clip_dir = settings.paths.resolve("data_root").parent / "cache" / "e20_clips"
    rows = []
    for (sp, seq), fr in sorted(by_seq.items()):
        r = eng.analyze(_clip(sorted(fr, key=lambda x: x.frame), clip_dir / sp / f"{seq}.mp4"), modalities=("video",))
        species = [t.species for t in r.observations.animal_tracks]
        right = sp in species
        rows.append({"species": sp, "sequence": seq, "right_species_track": right, "only_other_species": bool(species) and not right,
                     "no_track": not species, "tracks": len(species),
                     "right_species_duration_s": sum(t.duration_s for t in r.observations.animal_tracks if t.species == sp)})
    res = {}
    for sp in SPECIES:
        rr = [x for x in rows if x["species"] == sp]
        y = np.array([x["right_species_track"] for x in rr])
        w = np.array([x["only_other_species"] for x in rr])
        rng = np.random.default_rng(0)
        by_c: dict[str, list[int]] = {}
        for i, x in enumerate(rr):
            by_c.setdefault(contributor(x["sequence"]), []).append(i)
        cs = sorted(by_c)
        boot = [float(y[np.concatenate([by_c[cs[j]] for j in rng.integers(0, len(cs), len(cs))])].mean()) for _ in range(n_boot)]
        res[sp] = {"sequences": len(rr), "right_species_track": float(y.mean()), "right_species_track_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                   "only_other_species": float(w.mean()), "no_track": float(np.mean([x["no_track"] for x in rr])),
                   "mean_tracks_per_sequence": float(np.mean([x["tracks"] for x in rr]))}
    out = {"experiment": "E20b", "status": "completed", "created_at": utc_now().isoformat(timespec="seconds"), "detector": detector_id,
           "clips": "the 10 sampled frames of each sequence at 2 fps (the engine's sampling rate), H.264, video only",
           "results": res, "runtime_s": round(time.perf_counter() - t0, 1),
           "note": "engine-level species decisions (track verifier median, species merge); frames 1/20 of the sequence apart, so camera motion between samples is large"}
    write_json(out_dir / "engine_level.json", out)
    return out
