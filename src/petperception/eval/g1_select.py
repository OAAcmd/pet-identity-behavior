"""G1b candidate selection on development data (rule in docs/TRAINING_PROGRAM.md, fixed before the runs finished).

For each candidate export: per-class operating points recall-matched on the Open Images tuning set to M3
at 0.35; the trainer's final Open Images development AP50 per class; and frame-level results on the CoP3D
development frames (contributors disjoint from training and from E20), detector only, at those points.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..config import Settings
from ..data import cop3d
from ..utils import utc_now, write_json
from ..vision.detector import EfficientDetDetector
from . import e7_fp_rejection as E7
from . import e20_cop3d_home_video as E20

GRID = [round(x, 2) for x in np.arange(0.05, 0.501, 0.01)]


def per_class_points(det, reference, settings: Settings) -> dict:
    ref = E7.tuning_recall_curve(reference, settings, [E7.OP])[E7.OP]
    cur = E7.tuning_recall_curve(det, settings, GRID)
    pts = {c: min(GRID, key=lambda t, c=c: (abs(cur[t][c] - ref[c]), -t)) for c in E7.CLASSES}
    return {"thresholds": pts, "reference_tuning_recall": ref, "candidate_tuning_recall": {c: cur[pts[c]][c] for c in E7.CLASSES}}


def dev_frames(store: Path, csv_path: Path) -> list[cop3d.Frame]:
    out = []
    for r in csv.DictReader(open(csv_path, newline="", encoding="utf-8")):
        if r["split"] != "development":
            continue
        box = tuple(float(r[c]) for c in ("xmin", "ymin", "xmax", "ymax"))
        out.append(cop3d.Frame(r["species"], r["sequence"], r["file_name"], str(store / r["file_name"]), r["source_sha256"], 0, 0, box, float(r["mask_area"])))
    return out


class PerClass:
    def __init__(self, det, pts: dict[str, float]):
        self.det, self.pts, self.labels = det, pts, det.labels

    def detect(self, rgb, classes, thr, iou=0.5, k=10):
        return [d for d in self.det.detect(rgb, classes, min(self.pts.values()), iou, k) if d.score >= self.pts.get(d.label, thr)]


def run(settings: Settings, out_path: Path, candidates: dict[str, tuple[Path, Path]]) -> dict:
    """candidates: name -> (export .tflite, run directory with final.json)."""
    root = Path(settings.paths.resolve("data_root")).parent / "training" / "g1"
    frames = dev_frames(root / "images", root / "cop3d_frames.csv")
    m3 = E20.shipped_configs(settings)["phone_build_m3_detector_only"][0]
    rows = {}
    for name, (export, run_dir) in candidates.items():
        det = EfficientDetDetector(export, settings.runtime.num_threads)
        pts = per_class_points(det, m3, settings)
        hist = json.loads((run_dir / "final.json").read_text())["history"][-1]
        res = E20.summarize(E20.score_frames(PerClass(det, pts["thresholds"]), None, frames, settings))
        rows[name] = {"points": pts, "oi_dev_ap50": {c: hist["dev_ap50"][c]["ap50"] for c in ("cat", "dog")}, "oi_dev_map50": hist["dev_map50"],
                      "cop3d_dev": {sp: {k: v for k, v in r.items() if k in ("frames", "sequences", "contributors", "frame_found", "frame_wrong_species_only")}
                                    for sp, r in res.items()}}
    base = rows["g1a_full"]
    for r in rows.values():
        r["constraint_dog_dev_ap50"] = r["oi_dev_ap50"]["dog"] >= base["oi_dev_ap50"]["dog"] - 0.01
        r["constraint_dog_only_cat"] = r["cop3d_dev"]["dog"]["frame_wrong_species_only"] <= base["cop3d_dev"]["dog"]["frame_wrong_species_only"] + 0.02
    ok = [n for n, r in rows.items() if r["constraint_dog_dev_ap50"] and r["constraint_dog_only_cat"]]
    winner = max(ok, key=lambda n: rows[n]["oi_dev_ap50"]["cat"]) if ok else "g1a_full"
    out = {"selection": "G1b (docs/TRAINING_PROGRAM.md)", "created_at": utc_now().isoformat(timespec="seconds"), "candidates": rows, "winner": winner,
           "cop3d_dev_frames": len(frames)}
    write_json(out_path, out)
    return out
