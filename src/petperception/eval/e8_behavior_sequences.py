"""E8: tracking and behavior segments on SEMI-SYNTHETIC sequences.

What is real: the animal pixels (Open Images test-split cat/dog box crops) and
backgrounds (test-split images with no cat, dog or confusable-animal boxes),
the detector, verifier, tracker and behavior code (the full ``analyze`` path on
encoded MP4 files). What is synthetic: the motion. Each animal crop is moved
along a scripted path, so the ground-truth boxes, identities and behavior
segments are exact. This measures whether the software recovers known motion
patterns under real detection noise. It says nothing about how real pets
move, how often they pace, or whether these segments match human behavior
annotations; no such labelled household data was available.

Scripts: still->walk->still, run burst, pacing, zone visit, exit and return,
two animals crossing, occlusion, camera pan over a still animal.

Seeds: development 0-15 (2 per script; used to check the code and to choose
the default tracker), test 1000-1047 (6 per script; reported). Behavior
thresholds are a-priori configuration values and were not tuned on either.

Matching: interval segments by temporal IoU >= 0.3, instant events within
1.0 s; tracking by per-sample Hungarian IoU >= 0.5 (MOTA, IDF1, ID switches).
"""

from __future__ import annotations

import csv
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import Settings, load_settings
from ..context.zones import ZoneConfig
from ..utils import utc_now, write_json
from . import metrics as M

W, H, FPS = 640, 480, 10
DIAG = float(np.hypot(W, H))
SCRIPTS = ["still_walk_still", "run_burst", "pacing", "zone_visit", "exit_return", "two_cross", "occlusion", "camera_pan"]
DEV_SEEDS = list(range(16))
TEST_SEEDS = list(range(1000, 1048))
INSTANT = {"enter_zone", "leave_zone", "entered_view", "left_view"}


@dataclass
class Actor:
    patch: np.ndarray  # RGB crop of the animal box
    species: str
    path: list[tuple[float, float]]  # center per frame, normalized scene coords (None-free)
    visible: list[bool]
    height: float  # patch height as a fraction of frame height


@dataclass
class Sequence:
    seq_id: str
    script: str
    seed: int
    background: np.ndarray
    actors: list[Actor]
    n_frames: int
    gt_segments: list[dict] = field(default_factory=list)
    zones: ZoneConfig | None = None
    pan_px_per_frame: float = 0.0
    occluder: tuple[int, int, int, int] | None = None  # x1,y1,x2,y2 in px, drawn on visible frames
    eval_kinds: frozenset[str] = frozenset()  # segment kinds scored for this script (others ignored)


def _rows(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


INDOOR = {"/m/02crq1": "Couch", "/m/03ssj5": "Bed", "/m/01mzpv": "Chair", "/m/0h8n5zk": "Kitchen & dining room table", "/m/0b3fp9": "Countertop",
          "/m/0642b4": "Cupboard", "/m/03tw93": "Fireplace", "/m/0gjbg72": "Shelf", "/m/01lynh": "Stairs", "/m/02vkqh8": "Wardrobe",
          "/m/026qbn5": "Studio couch", "/m/03m3pdh": "Sofa bed"}
ANIMAL_MIDS = {"/m/0jbk", "/m/04rky", "/m/01lrl", "/m/015p6", "/m/01yrx", "/m/0bt9lr", "/m/0kmg4", "/m/0138tl", "/m/0167gd", "/m/06msq"}
BG_SEED = 20261012


def build_background_manifest(data_root: Path, man: Path, n: int = 60) -> dict:
    """Indoor test-split images (furniture boxes) with no animal, toy, doll or sculpture boxes or positive labels."""
    import random

    d = data_root / "openimages_test"
    boxes = defaultdict(set)
    for r in _rows(d / "test-annotations-bbox.csv"):
        boxes[r["ImageID"]].add(r["LabelName"])
    animal_pos = {r["ImageID"] for r in _rows(d / "oidv7-test-annotations-human-imagelabels.csv")
                  if r["LabelName"] in ANIMAL_MIDS and float(r["Confidence"]) >= 0.5}
    cand = sorted(i for i, labs in boxes.items() if len(labs & INDOOR.keys()) >= 2 and not (labs & ANIMAL_MIDS) and i not in animal_pos)
    tuning = {r["image_id"] for r in _rows(man / "oi_test_tuning_images.csv")}
    cand = [i for i in cand if i not in tuning]
    sel = random.Random(BG_SEED).sample(cand, min(n, len(cand)))
    with open(man / "e8_backgrounds.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "indoor_classes", "download_url"])
        for i in sorted(sel):
            w.writerow([i, ";".join(sorted(INDOOR[m] for m in boxes[i] & INDOOR.keys())),
                        f"https://open-images-dataset.s3.amazonaws.com/test/{i}.jpg"])
    return {"candidates": len(cand), "selected": len(sel), "seed": BG_SEED}


def load_assets(data_root: Path, man: Path, detector=None, verifier=None) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray], dict]:
    """Animal crops and animal-free indoor backgrounds.

    Screening (documented in results): a crop is kept only if, pasted on three
    random backgrounds at a typical size, the product detector finds it
    (score >= 0.35, IoU >= 0.5) and the verifier accepts it at least twice.
    E8 therefore measures tracking/behavior given detectable animals;
    detection itself is measured by E4/E7.
    """
    imgs = {r["image_id"]: r for r in _rows(man / "oi_test_tuning_images.csv")}
    boxes = defaultdict(list)
    for r in _rows(man / "oi_test_tuning_boxes.csv"):
        boxes[r["image_id"]].append(r)
    img_dir = data_root / "openimages_test" / "images"
    backgrounds = []
    for r in _rows(man / "e8_backgrounds.csv"):
        im = cv2.imread(str(img_dir / f"{r['image_id']}.jpg"))
        if im is not None:
            backgrounds.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    raw: dict[str, list[np.ndarray]] = {"cat": [], "dog": []}
    for i, r in sorted(imgs.items()):
        if r["role"] not in ("cat_positive", "dog_positive"):
            continue
        sp = r["role"].split("_")[0]
        bx = boxes.get(i, [])
        own = [b for b in bx if b["label"] == sp]
        if len(own) != 1 or len(bx) != 1:
            continue
        b = own[0]
        if any(int(b[k]) for k in ("is_occluded", "is_group_of", "is_depiction")):
            continue
        area = (float(b["xmax"]) - float(b["xmin"])) * (float(b["ymax"]) - float(b["ymin"]))
        if not 0.03 <= area <= 0.6:
            continue
        im = cv2.imread(str(img_dir / f"{i}.jpg"))
        if im is None:
            continue
        h, w = im.shape[:2]
        crop = im[int(float(b["ymin"]) * h): int(float(b["ymax"]) * h), int(float(b["xmin"]) * w): int(float(b["xmax"]) * w)]
        if min(crop.shape[:2]) >= 100 and 0.4 <= crop.shape[1] / crop.shape[0] <= 2.5:
            raw[sp].append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    screening = {"candidates": {k: len(v) for k, v in raw.items()}}
    if detector is None:
        return raw, backgrounds, screening
    from ..vision.detector import Detection  # noqa: F401  (verifier input type)

    rng = np.random.default_rng(BG_SEED)
    kept: dict[str, list[np.ndarray]] = {"cat": [], "dog": []}
    for sp, lst in raw.items():
        for patch in lst:
            ok = 0
            for _ in range(3):
                bg = backgrounds[int(rng.integers(len(backgrounds)))]
                scene = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
                x1, y1, x2, y2, _ = _paste_px(scene, patch, float(rng.uniform(0.3, 0.7)) * W, float(rng.uniform(0.45, 0.6)), 0.42)
                gtb = np.array([[x1 / W, y1 / H, x2 / W, y2 / H]])
                dets = [d for d in detector.detect(scene, [sp], 0.35, 0.5, 5) if M.iou_xyxy(d.box[None], gtb)[0, 0] >= 0.5]
                if dets and (verifier is None or (verifier.verify(scene, dets[0]) or (0, True))[1]):
                    ok += 1
            if ok >= 2:
                kept[sp].append(patch)
    screening["kept"] = {k: len(v) for k, v in kept.items()}
    return kept, backgrounds, screening


def _hold(p: tuple[float, float], n: int) -> list[tuple[float, float]]:
    return [p] * n


def _move(a: tuple[float, float], b: tuple[float, float], speed_diag: float) -> list[tuple[float, float]]:
    dist = np.hypot((b[0] - a[0]) * W, (b[1] - a[1]) * H) / DIAG
    n = max(2, int(round(dist / speed_diag * FPS)))
    return [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]


def build_sequence(script: str, seed: int, patches: dict[str, list[np.ndarray]], backgrounds: list[np.ndarray]) -> Sequence:
    rng = np.random.default_rng(seed)
    sp = "cat" if rng.random() < 0.5 else "dog"
    patch = patches[sp][int(rng.integers(len(patches[sp])))]
    bg = backgrounds[int(rng.integers(len(backgrounds)))]
    hgt = float(rng.uniform(0.38, 0.48))
    y = float(rng.uniform(0.45, 0.6))
    segs: list[dict] = []
    t = lambda frames: frames / FPS  # noqa: E731
    zones = None
    occ = None
    pan = 0.0
    actors: list[Actor]
    if script == "still_walk_still":
        a, b = (0.25, y), (0.72, y)
        n1, n3 = int(rng.integers(50, 70)), int(rng.integers(40, 60))
        walk = _move(a, b, float(rng.uniform(0.08, 0.15)))
        path = _hold(a, n1) + walk + _hold(b, n3)
        segs = [{"kind": "stationary", "t0": 0.0, "t1": t(n1)}, {"kind": "moving", "t0": t(n1), "t1": t(n1 + len(walk))},
                {"kind": "stationary", "t0": t(n1 + len(walk)), "t1": t(len(path) - 1)}]
        actors = [Actor(patch, sp, path, [True] * len(path), hgt)]
    elif script == "run_burst":
        a, b = (0.2, y), (0.78, y)
        n1, n3 = 40, 40
        run = _move(a, b, float(rng.uniform(0.35, 0.5)))
        path = _hold(a, n1) + run + _hold(b, n3)
        segs = [{"kind": "stationary", "t0": 0.0, "t1": t(n1)}, {"kind": "moving", "t0": t(n1), "t1": t(n1 + len(run))},
                {"kind": "active_bout", "t0": t(n1), "t1": t(n1 + len(run))}, {"kind": "stationary", "t0": t(n1 + len(run)), "t1": t(len(path) - 1)}]
        actors = [Actor(patch, sp, path, [True] * len(path), hgt)]
    elif script == "pacing":
        amp = float(rng.uniform(0.25, 0.4))
        a, b = (0.5 - amp / 2, y), (0.5 + amp / 2, y)
        legs = int(rng.integers(4, 7))
        speed = float(rng.uniform(0.08, 0.14))
        path = _hold(a, 10)
        for k in range(legs):
            path += _move(a, b, speed) if k % 2 == 0 else _move(b, a, speed)
        path += _hold(path[-1], 10)
        segs = [{"kind": "pacing_candidate", "t0": t(10), "t1": t(len(path) - 10)}, {"kind": "moving", "t0": t(10), "t1": t(len(path) - 10)}]
        actors = [Actor(patch, sp, path, [True] * len(path), hgt)]
    elif script == "zone_visit":
        zx = float(rng.uniform(0.62, 0.7))
        zones = ZoneConfig.model_validate({"schema": "petperception.zones/1", "anchor": "center", "zones": [
            {"zone_id": "z1", "semantic_tag": "food_area", "label": "Bowl", "polygon": [[zx, 0.2], [0.98, 0.2], [0.98, 0.9], [zx, 0.9]]}]})
        a, inside, out = (0.18, y), (min(0.9, zx + 0.12), y), (0.2, y)
        n0 = 20
        approach = _move(a, inside, float(rng.uniform(0.08, 0.14)))
        enter_k = next(k for k, p in enumerate(approach) if p[0] >= zx)
        dwell = int(rng.integers(50, 70))
        leave = _move(inside, out, float(rng.uniform(0.1, 0.15)))
        path = _hold(a, n0) + approach + _hold(inside, dwell) + leave + _hold(out, 10)
        t_enter = t(n0 + enter_k)
        exit_k = next(k for k, p in enumerate(leave) if p[0] < zx)
        t_exit = t(n0 + len(approach) + dwell + exit_k)
        segs = [{"kind": "approach_zone", "t0": t(n0), "t1": t_enter}, {"kind": "enter_zone", "t0": t_enter, "t1": t_enter},
                {"kind": "dwell_zone", "t0": t_enter, "t1": t_exit}, {"kind": "leave_zone", "t0": t_exit, "t1": t_exit}]
        actors = [Actor(patch, sp, path, [True] * len(path), hgt)]
    elif script == "exit_return":
        a, off = (0.5, y), (-0.3, y)
        out_walk = _move(a, off, 0.15)
        absent = int(rng.integers(30, 45))
        back = _move(off, a, 0.15)
        path = _hold(a, 20) + out_walk + _hold(off, absent) + back + _hold(a, 20)
        vis = [(-0.15 < p[0] < 1.15) for p in path]
        # Ground truth for view events: last/first frame with >= half of the patch in view.
        half = [abs(p[0] - 0.5) < 0.5 for p in path]
        last_in = max(k for k in range(20 + len(out_walk)) if half[k])
        first_back = next(k for k in range(20 + len(out_walk) + absent, len(path)) if half[k])
        segs = [{"kind": "left_view", "t0": t(last_in), "t1": t(last_in)}, {"kind": "entered_view", "t0": t(first_back), "t1": t(first_back)}]
        actors = [Actor(patch, sp, path, vis, hgt)]
    elif script == "two_cross":
        other = "dog" if sp == "cat" else "cat"
        p2 = patches[other][int(rng.integers(len(patches[other])))]
        y1, y2 = y - 0.06, y + 0.06
        n = int(rng.integers(70, 90))
        path1 = [(0.15 + 0.7 * k / n, y1) for k in range(n)]
        path2 = [(0.85 - 0.7 * k / n, y2) for k in range(n)]
        actors = [Actor(patch, sp, path1, [True] * n, hgt), Actor(p2, other, path2, [True] * n, hgt)]
    elif script == "occlusion":
        a, b = (0.1, y), (0.9, y)
        walk = _move(a, b, float(rng.uniform(0.1, 0.13)))
        path = _hold(a, 10) + walk + _hold(b, 10)
        mid = 10 + len(walk) // 2
        gap = int(rng.integers(8, 14))  # 0.8-1.3 s hidden (tracker max gap 1.5 s)
        vis = [not (mid - gap // 2 <= k < mid + gap // 2) for k in range(len(path))]
        actors = [Actor(patch, sp, path, vis, hgt)]
    elif script == "camera_pan":
        n = int(rng.integers(70, 90))
        pan = float(rng.uniform(2.0, 4.0))  # px per frame (0.03-0.06 frame widths per second at 10 fps)
        scene_x = float(rng.uniform(0.45, 0.6))
        path = [(scene_x, y)] * n
        segs = [{"kind": "stationary", "t0": 0.0, "t1": t(n - 1)}]
        actors = [Actor(patch, sp, path, [True] * n, hgt)]
    else:
        raise ValueError(script)
    n_frames = max(len(a.path) for a in actors)
    motion = {"stationary", "moving", "active_bout", "pacing_candidate"}
    kinds = {"still_walk_still": motion, "run_burst": motion, "pacing": {"pacing_candidate", "moving", "active_bout"},
             "zone_visit": {"approach_zone", "enter_zone", "dwell_zone", "leave_zone"}, "exit_return": {"left_view", "entered_view"},
             "camera_pan": motion, "two_cross": set(), "occlusion": set()}[script]
    return Sequence(f"{script}-{seed}", script, seed, bg, actors, n_frames, segs, zones, pan, occ, frozenset(kinds))


def _paste_px(scene: np.ndarray, patch: np.ndarray, cx_px: float, cy: float, hgt: float) -> tuple[float, float, float, float, float]:
    """Paste a resized patch centred at (cx_px, cy*H) on the scene; returns the unclipped box in scene px and its area."""
    ph = int(hgt * H)
    pw = max(8, int(patch.shape[1] * ph / patch.shape[0]))
    p = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)
    x1, y1 = int(round(cx_px - pw / 2)), int(round(cy * H - ph / 2))
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(scene.shape[1], x1 + pw), min(scene.shape[0], y1 + ph)
    if sx2 > sx1 and sy2 > sy1:
        scene[sy1:sy2, sx1:sx2] = p[sy1 - y1: sy2 - y1, sx1 - x1: sx2 - x1]
    return float(x1), float(y1), float(x1 + pw), float(y1 + ph), float(pw * ph)


def render(seq: Sequence, out: Path) -> dict[int, dict[int, tuple[float, float, float, float]]]:
    """Encode the sequence to MP4; return GT boxes (view-normalized) per frame per actor with >= 50% of it in view.

    The scene is a background wider than the view by the total pan; the view window
    moves right by ``pan_px_per_frame`` per frame (camera pan). Actor paths are in
    scene coordinates normalized by the view size.
    """
    sw = W + int(np.ceil(seq.pan_px_per_frame * seq.n_frames)) + 1
    bg = cv2.resize(seq.background, (sw, H), interpolation=cv2.INTER_AREA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    gt: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    for k in range(seq.n_frames):
        off = int(round(seq.pan_px_per_frame * k))
        scene = bg.copy()
        boxes: dict[int, tuple[float, float, float, float]] = {}
        for ai, a in enumerate(seq.actors):
            if k >= len(a.path) or not a.visible[k]:
                continue
            x1, y1, x2, y2, area = _paste_px(scene, a.patch, a.path[k][0] * W, a.path[k][1], a.height)
            vx1, vx2 = max(0.0, x1 - off), min(float(W), x2 - off)
            vy1, vy2 = max(0.0, y1), min(float(H), y2)
            if vx2 > vx1 and vy2 > vy1 and (vx2 - vx1) * (vy2 - vy1) / area >= 0.5:
                boxes[ai] = (vx1 / W, vy1 / H, vx2 / W, vy2 / H)
        gt[k] = boxes
        proc.stdin.write(np.ascontiguousarray(scene[:, off: off + W]).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return gt


def tracking_metrics(gt: dict[int, dict[int, tuple]], tracks: list[dict], sample_times: list[float]) -> dict:
    """CLEAR-MOT style counts on the sampled frames."""
    pred_at: dict[float, dict[str, np.ndarray]] = defaultdict(dict)
    for tr in tracks:
        for d in tr["detections"]:
            pred_at[round(d["t_s"], 3)][tr["track_id"]] = np.array(d["box_xyxy_norm"])
    tp = fp = fn = idsw = 0
    last: dict[int, str] = {}
    pair_counts: dict[tuple[int, str], int] = defaultdict(int)
    n_gt = n_pred = 0
    for ts in sample_times:
        k = int(round(ts * FPS))
        g = gt.get(k, {})
        p = pred_at.get(round(ts, 3), {})
        n_gt += len(g)
        n_pred += len(p)
        gi, pi = list(g), list(p)
        matched = set()
        if gi and pi:
            iou = M.iou_xyxy(np.array([g[i] for i in gi]), np.array([p[j] for j in pi]))
            r, c = linear_sum_assignment(-iou)
            for a, b in zip(r, c):
                if iou[a, b] >= 0.5:
                    matched.add((gi[a], pi[b]))
        tp += len(matched)
        fn += len(gi) - len(matched)
        fp += len(pi) - len(matched)
        for gid, tid in matched:
            pair_counts[(gid, tid)] += 1
            if gid in last and last[gid] != tid:
                idsw += 1
            last[gid] = tid
    gids = sorted({a for a, _ in pair_counts})
    tids = sorted({b for _, b in pair_counts})
    idtp = 0
    if gids and tids:
        m = np.array([[pair_counts.get((a, b), 0) for b in tids] for a in gids])
        r, c = linear_sum_assignment(-m)
        idtp = int(m[r, c].sum())
    return {"gt": n_gt, "pred": n_pred, "tp": tp, "fp": fp, "fn": fn, "idsw": idsw, "idtp": idtp,
            "n_tracks": len(tracks)}


def _tiou(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["t1"], b["t1"]) - max(a["t0"], b["t0"]))
    union = max(a["t1"], b["t1"]) - min(a["t0"], b["t0"])
    return inter / union if union > 0 else (1.0 if abs(a["t0"] - b["t0"]) < 1e-6 else 0.0)


def segment_matches(gt: list[dict], pred: list[dict], kinds: frozenset[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k in kinds:
        G = [g for g in gt if g["kind"] == k]
        P = [p for p in pred if p["kind"] == k]
        used = set()
        hits = 0
        errs = []
        for g in G:
            best, bj = -1.0, None
            for j, p in enumerate(P):
                if j in used:
                    continue
                if k in INSTANT:
                    sc = 1.0 - abs(p["t0"] - g["t0"]) if abs(p["t0"] - g["t0"]) <= 1.0 else -1.0
                else:
                    sc = _tiou(g, p) if _tiou(g, p) >= 0.3 else -1.0
                if sc > best:
                    best, bj = sc, j
            if bj is not None and best >= 0:
                used.add(bj)
                hits += 1
                errs.append(max(abs(P[bj]["t0"] - g["t0"]), abs(P[bj]["t1"] - g["t1"])))
        out[k] = {"gt": len(G), "pred": len(P), "matched": hits, "max_boundary_errors_s": errs}
    return out


def run_split(settings_by_tracker: dict, seeds: list[int], patches, backgrounds, work: Path) -> dict:
    from ..pipeline import AnalysisEngine

    engines = {m: AnalysisEngine(s).load() for m, s in settings_by_tracker.items()}
    per_seq = []
    for script in SCRIPTS:
        for seed in seeds[SCRIPTS.index(script)::len(SCRIPTS)]:
            seq = build_sequence(script, seed, patches, backgrounds)
            mp4 = work / f"{seq.seq_id}.mp4"
            gt = render(seq, mp4)
            row = {"seq_id": seq.seq_id, "script": script, "species": seq.actors[0].species, "n_frames": seq.n_frames, "by_tracker": {}}
            for m, eng in engines.items():
                res = eng.analyze(mp4, zones=seq.zones, modalities=("video",)).model_dump(mode="json")
                tracks = res["observations"]["animal_tracks"]
                sample_times = sorted({round(d["t_s"], 3) for tr in tracks for d in tr["detections"]} |
                                      {round(k / settings_by_tracker[m].vision.sample_fps, 3) for k in range(int(seq.n_frames / FPS * settings_by_tracker[m].vision.sample_fps) + 1)})
                tm = tracking_metrics(gt, tracks, sample_times)
                pred = [{"kind": b["kind"], "t0": b["t_start_s"], "t1": b["t_end_s"]} for tr in tracks for b in tr["behavior_segments"]]
                sm = segment_matches(seq.gt_segments, pred, seq.eval_kinds)
                rejected = [q for q in res["quality_flags"] if q["code"] == "detections_rejected_by_verifier"]
                row["by_tracker"][m] = {"tracking": tm, "segments": sm, "verifier_rejections": len(rejected),
                                        "predicted_kinds": sorted({p["kind"] for p in pred})}
            per_seq.append(row)
    return {"sequences": per_seq, "summary": summarize(per_seq, list(engines))}


def summarize(rows: list[dict], trackers: list[str]) -> dict:
    out = {}
    for m in trackers:
        t = defaultdict(int)
        seg: dict[str, dict[str, float]] = defaultdict(lambda: {"gt": 0, "pred": 0, "matched": 0, "errs": []})
        by_script: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in rows:
            tr = r["by_tracker"][m]["tracking"]
            for k in ("gt", "pred", "tp", "fp", "fn", "idsw", "idtp"):
                t[k] += tr[k]
                by_script[r["script"]][k] += tr[k]
            by_script[r["script"]]["tracks"] += tr["n_tracks"]
            by_script[r["script"]]["sequences"] += 1
            for k, v in r["by_tracker"][m]["segments"].items():
                seg[k]["gt"] += v["gt"]
                seg[k]["pred"] += v["pred"]
                seg[k]["matched"] += v["matched"]
                seg[k]["errs"] += v["max_boundary_errors_s"]
        mota = 1 - (t["fn"] + t["fp"] + t["idsw"]) / max(t["gt"], 1)
        idf1 = 2 * t["idtp"] / max(t["gt"] + t["pred"], 1)
        out[m] = {
            "tracking": {**dict(t), "mota": mota, "idf1": idf1, "detection_recall": t["tp"] / max(t["gt"], 1),
                         "detection_precision": t["tp"] / max(t["pred"], 1)},
            "tracking_by_script": {s: {**dict(v), "idf1": 2 * v["idtp"] / max(v["gt"] + v["pred"], 1)} for s, v in by_script.items()},
            "segments": {k: {"gt": v["gt"], "pred": v["pred"], "matched": v["matched"],
                             "recall": v["matched"] / v["gt"] if v["gt"] else None,
                             "precision": v["matched"] / v["pred"] if v["pred"] else None,
                             "median_max_boundary_error_s": float(np.median(v["errs"])) if v["errs"] else None}
                         for k, v in sorted(seg.items())},
        }
    return out


def run(settings: Settings, out_dir: Path) -> dict:
    data = settings.paths.resolve("data_root")
    man = settings.paths.resolve("manifests_root")
    from ..pipeline import AnalysisEngine

    if not (man / "e8_backgrounds.csv").exists():
        build_background_manifest(data, man)
    from ..data import openimages as OI

    fetched = OI.fetch_images(data, man, image_dir=data / "openimages_test" / "images", images_csv="e8_backgrounds.csv", sums_csv="e8_backgrounds_sha256.csv")
    screen_engine = AnalysisEngine(load_settings(overrides={"audio": {"enabled": False}})).load()
    patches, backgrounds, screening = load_assets(data, man, screen_engine.detector, screen_engine.verifier)
    if min(len(patches["cat"]), len(patches["dog"])) < 5 or len(backgrounds) < 10:
        raise RuntimeError(f"not enough crops/backgrounds after screening: {screening}; run: petperception fetch-data openimages-tuning")
    work = data.parent / "e8"
    by = {}
    for m in ("iou", "kalman"):
        s = load_settings(overrides={"vision": {"tracker": {"method": m}}, "audio": {"enabled": False}})
        s.paths = settings.paths
        by[m] = s
    dev = run_split(by, DEV_SEEDS, patches, backgrounds, work / "dev")
    best = max(("iou", "kalman"), key=lambda m: (dev["summary"][m]["tracking"]["idf1"], dev["summary"][m]["tracking"]["mota"]))
    test = run_split(by, TEST_SEEDS, patches, backgrounds, work / "test")
    summary = {
        "experiment": "E8",
        "title": "Tracking and behavior segments on semi-synthetic sequences (real animal pixels, scripted motion)",
        "protocol": __doc__.strip().splitlines(),
        "assets": {"cat_crops": len(patches["cat"]), "dog_crops": len(patches["dog"]), "backgrounds": len(backgrounds),
                   "crop_screening": screening, "background_fetch_errors": len(fetched.get("errors", {})),
                   "source": "Open Images V7 test split: crops from openimages_v7_test_tuning, indoor backgrounds from manifests/e8_backgrounds.csv"},
        "render": {"width": W, "height": H, "fps": FPS, "codec": "libx264 crf 20", "analysis_sample_fps": settings.vision.sample_fps},
        "dev": dev["summary"],
        "tracker_selected_on_dev": best,
        "test": test["summary"],
        "test_sequences": test["sequences"],
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    write_json(out_dir / "metrics.json", summary)
    return summary
