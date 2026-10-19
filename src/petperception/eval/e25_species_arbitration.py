"""E25: species arbitration with the verifier's crop classifier (cat vs dog relabelling).

Why. On home phone video (E20) 14 % of dog frames show only a "cat", and on
the engine level 16 % of dog videos are reported as showing only a cat. The
verifier already runs an ImageNet classifier on every detection crop. Its
group probabilities (domestic cat breeds vs about 120 dog breeds) are
species evidence that the detector's label ignores.

Rule. A detection labelled S is relabelled to the other species S' when
log(p_crop(S') / p_crop(S)) > tau.

Protocol, fixed before any E20 frame was scored with it:
* Selection: the E7 tuning set (Open Images test split). The phone build's
  detections at the operating point are matched to cat or dog ground-truth
  boxes at IoU >= 0.5; the matched box gives the true species. tau is the
  grid value (0 to 6, step 0.25) with the highest species accuracy on these
  detections. Ties go to the larger tau, i.e. fewer relabels.
* Test, once:
  - E20 frames (home video), phone build with its verifier, scored as in
    E20, with and without arbitration, paired by sequence;
  - guard: the E4 subset, image-level species accuracy of matched
    detections.
* Adoption rule:
  1. the rate of dog frames with only a cat falls, with the paired 95 %
     CI's upper bound below 0;
  2. on cat sequences, frames found does not fall by more than 0.02
     (paired lower bound above -0.02);
  3. dog frames found does not fall (paired lower bound above -0.02);
  4. on E4, matched-detection species accuracy does not fall.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import Settings
from ..data import openimages as OI
from ..utils import utc_now, write_json
from . import e7_fp_rejection as E7
from . import e20_cop3d_home_video as E20

GRID = np.arange(0.0, 6.0001, 0.25)
EPS = 1e-6
OTHER = {"cat": "dog", "dog": "cat"}


def margin(groups: dict, label: str) -> float:
    return float(np.log((groups[OTHER[label]] + EPS) / (groups[label] + EPS)))


def _matched(dets: list[tuple], gts: list[tuple]) -> list[tuple[str, float]]:
    """(true species, margin) for detections overlapping a cat/dog ground-truth box at IoU >= 0.5."""
    out = []
    for label, box, groups in dets:
        best, sp = 0.0, None
        for g_label, g_box in gts:
            iou = E20._iou(tuple(box), tuple(g_box))
            if iou > best:
                best, sp = iou, g_label
        if best >= 0.5:
            out.append((label, sp, margin(groups, label)))
    return out


def collect(det, clf, images: list[dict], boxes: list, img_dir: Path, threshold: float) -> list[tuple[str, str, float]]:
    by_img: dict[str, list] = {}
    for b in boxes:
        if b.label in ("cat", "dog") and not b.is_depiction:
            by_img.setdefault(b.image_id, []).append((b.label, E7.E4._xyxy(b)))
    rows = []
    for r in images:
        p = img_dir / f"{r['image_id']}.jpg"
        if not p.exists() or r["image_id"] not in by_img:
            continue
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        dets = [(d.label, d.box, clf.scores(rgb, d.box).groups) for d in det.detect(rgb, ["cat", "dog"], threshold, 0.5, 10)]
        rows += _matched(dets, by_img[r["image_id"]])
    return rows


def accuracy(rows: list[tuple[str, str, float]], tau: float | None) -> float:
    ok = [(OTHER[lab] if tau is not None and m > tau else lab) == true for lab, true, m in rows]
    return float(np.mean(ok)) if ok else float("nan")


def select_tau(rows: list[tuple[str, str, float]]) -> tuple[float, list]:
    curve = [(float(t), accuracy(rows, float(t))) for t in GRID]
    best = max(curve, key=lambda x: (round(x[1], 9), x[0]))
    return best[0], curve


class Arbitrated:
    """A detector wrapper that relabels cat/dog detections by the crop classifier's species margin."""

    def __init__(self, det, clf, tau: float):
        self.det, self.clf, self.tau = det, clf, tau
        self.labels = det.labels

    def detect(self, rgb, classes, score_threshold, iou_thr=0.5, max_per_class=10):
        out = []
        for d in self.det.detect(rgb, classes, score_threshold, iou_thr, max_per_class):
            if d.label in OTHER and OTHER[d.label] in classes and margin(self.clf.scores(rgb, d.box).groups, d.label) > self.tau:
                d = dataclasses.replace(d, label=OTHER[d.label])
            out.append(d)
        return out


def run(settings: Settings, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    man, data = settings.paths.resolve("manifests_root"), settings.paths.resolve("data_root")
    cfg = E20.shipped_configs(settings)
    det, ver = cfg["phone_build_m3+verifier"]
    clf = ver.classifier
    op = settings.vision.score_threshold
    tune_imgs, tune_boxes, _ = E7._tuning_subset(man)
    tune = collect(det, clf, tune_imgs, tune_boxes, data / "openimages_test" / "images", op)
    tau, curve = select_tau(tune)
    e4_imgs, e4_boxes = OI.read_subset(man)
    guard_rows = collect(det, clf, e4_imgs, e4_boxes, OI.dataset_dir(data) / "images", op)
    frames = E20.frames_all(settings)
    base = E20.score_frames(det, ver, frames, settings)
    arb = E20.score_frames(Arbitrated(det, clf, tau), ver, frames, settings)
    pf = E20.paired(arb, base)

    def paired_wrong(sp: str) -> dict:
        fa, fb = {}, {}
        for x in arb:
            if x["species"] == sp:
                fa.setdefault(x["sequence"], []).append(x["wrong_only"])
        for x in base:
            if x["species"] == sp:
                fb.setdefault(x["sequence"], []).append(x["wrong_only"])
        seqs = sorted(fa)
        d = np.array([np.mean(fa[s]) - np.mean(fb[s]) for s in seqs])
        rng = np.random.default_rng(0)
        boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(1000)]
        return {"mean_difference": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}

    pw = {sp: paired_wrong(sp) for sp in E20.SPECIES}
    guard = {"accuracy_without": accuracy(guard_rows, None), "accuracy_with": accuracy(guard_rows, tau), "matched_detections": len(guard_rows)}
    rule = {
        "1_dog_only_cat_frames_fall": pw["dog"]["ci95"][1] < 0,
        "2_cat_found_not_worse": pf["cat"]["ci95"][0] > -0.02,
        "3_dog_found_not_worse": pf["dog"]["ci95"][0] > -0.02,
        "4_e4_species_accuracy_not_worse": guard["accuracy_with"] >= guard["accuracy_without"],
    }
    res = {"experiment": "E25", "status": "completed", "created_at": utc_now().isoformat(timespec="seconds"), "protocol": __doc__.strip().splitlines(),
           "selection": {"matched_tuning_detections": len(tune), "species_accuracy_without": accuracy(tune, None), "tau": tau,
                         "species_accuracy_with": accuracy(tune, tau), "curve": curve,
                         "relabels_on_tuning": int(sum(m > tau for _, _, m in tune))},
           "e20_frames": {"without": E20.summarize(base), "with": E20.summarize(arb), "paired_found_difference": pf, "paired_wrong_only_difference": pw},
           "e4_guard": guard, "adoption_rule": rule, "adopt": all(rule.values()),
           "runtime_s": round(time.perf_counter() - t0, 1)}
    write_json(out_dir / "metrics.json", res)
    return res
