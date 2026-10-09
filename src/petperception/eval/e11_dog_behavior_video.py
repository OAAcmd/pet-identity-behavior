"""E11: dog behavior recognition from real video (MammalNet, dog breeds only).

Data: MammalNet detection annotations (temporal behavior segments in YouTube
videos; genus Canis, domestic-dog breed keywords only - wolves and wolfdogs
excluded). MammalNet code is MIT; no dataset licence is published and the
videos are third-party YouTube uploads, so data and the trained head are
research-profile only. There are no domestic cats in MammalNet (only wild
Felis species), so this experiment covers dogs only.

Classes (household-relevant, enough examples): sleeps, eats_food,
drinks_water, fights (against other animals), eliminates (pees or poops).
Other MammalNet behaviors are excluded from training and evaluation.

Pipeline (identical to what the phone computes): frames at 2 fps inside each
annotated segment (streamed with FFmpeg HTTP range reads, never stored),
product detector -> largest dog box -> EfficientNet-Lite0 pooled embedding of
the square crop, plus the pooled embedding of the whole frame; segment
features = mean/max of crop embeddings, mean of frame embeddings, box motion.
Model: multinomial logistic regression.

Splits: MammalNet's own training / validation / testing subsets (video
level). Regularisation is chosen on validation; testing is used once.
Confidence intervals resample videos. Breed-stratified results use the
MammalNet keyword (bulldog, terrier, malamute, hound, ...).
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..config import Settings, load_settings
from ..utils import sha256_file, utc_now, write_json
from . import metrics as M

BASE = "https://mammalnet.s3.amazonaws.com/full_videos/{vid}.mp4"
ANNOTATION_SHA256 = "055af7e2ae5184af1a292bd1605d7dd09d4937d25d2e81c1b333c654b97caee0"
DOG = {"bulldog", "terrier", "malamute", "hound", "husky dog", "shepherd dog", "eskimo dog", "foxhound"}
CLASSES = {"sleeps": "sleeps", "eats food": "eats_food", "drinks water": "drinks_water", "fights against other animals": "fights",
           "pees": "eliminates", "poops": "eliminates"}
LABELS = ["sleeps", "eats_food", "drinks_water", "fights", "eliminates"]
FPS = 2.0
MAX_SEG_S = 30.0
MAX_SEGS_PER_VIDEO = 3
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)


def segments(ann_path: Path) -> list[dict]:
    d = json.loads(ann_path.read_text())
    out = []
    for vid, v in sorted(d.items()):
        tax = v.get("taxnomy") or []
        if {t.get("genus") for t in tax} != {"canis"}:
            continue
        kws = {t.get("keyword") for t in tax} & DOG
        if not kws:
            continue
        n = 0
        for a in v["annotations"]:
            lab = CLASSES.get(a["label"])
            if lab is None or n >= MAX_SEGS_PER_VIDEO:
                continue
            s0, s1 = float(a["segment"][0]), float(a["segment"][1])
            if s1 - s0 < 1.0:
                continue
            out.append({"video": vid, "subset": v["subset"], "breed": sorted(kws)[0], "label": lab, "t0": s0, "t1": min(s1, s0 + MAX_SEG_S)})
            n += 1
    return out


def stream_frames(vid: str, t0: float, t1: float, size: tuple[int, int] = (480, 360)) -> list[np.ndarray]:
    """Decode frames at FPS inside [t0, t1] directly from the remote file (HTTP range reads)."""
    w, h = size
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", BASE.format(vid=vid),
           "-vf", f"fps={FPS},scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        return []
    n = len(r.stdout) // (w * h * 3)
    return [np.frombuffer(r.stdout[i * w * h * 3:(i + 1) * w * h * 3], np.uint8).reshape(h, w, 3) for i in range(n)]


def featurize(eng, backbone, frames: list[np.ndarray]) -> dict | None:
    import cv2

    from ..vision.keypoints import square_crop

    crops, wholes, boxes = [], [], []
    for rgb in frames:
        dets = [d for d in eng.detector.detect(rgb, ["dog"], 0.35, 0.5, 5)]
        whole = backbone.features(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA))["pooled"]
        wholes.append(whole)
        if not dets:
            continue
        d = max(dets, key=lambda x: x.area)
        crops.append(backbone.features(square_crop(rgb, d.box)[0])["pooled"])
        boxes.append(d.box)
    if not wholes:
        return None
    if crops:
        C = np.stack(crops)
        B = np.stack(boxes)
        ctr = (B[:, :2] + B[:, 2:]) / 2
        sp = np.linalg.norm(np.diff(ctr, axis=0), axis=1) * FPS if len(B) > 1 else np.zeros(1)
        area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
        motion = [float(np.median(sp)), float(np.percentile(sp, 90)), float(area.mean()), float(np.std(area)), len(crops) / len(frames)]
        crop_mean, crop_max = C.mean(0), C.max(0)
    else:
        motion = [0.0, 0.0, 0.0, 0.0, 0.0]
        crop_mean = crop_max = np.zeros(1280, np.float32)
    return {"crop_mean": crop_mean, "crop_max": crop_max, "frame_mean": np.stack(wholes).mean(0), "motion": np.array(motion, np.float32),
            "frames": len(frames), "frames_with_dog": len(crops)}


def build_features(settings: Settings, segs: list[dict], cache: Path, workers: int = 3) -> list[dict]:
    from ..pipeline import AnalysisEngine
    from ..registry import check_model
    from ..vision.keypoints import Backbone

    cache.mkdir(parents=True, exist_ok=True)
    eng = AnalysisEngine(load_settings(overrides={"audio": {"enabled": False}, "posture": {"enabled": False}, "pose": {"provider": "none"}})).load()
    rb = check_model(settings, "efficientnet_lite0_mediapipe_v1")
    backbone = Backbone(Path(next(iter(rb.files.values()))), settings.runtime.num_threads)
    todo = [s for s in segs if not (cache / f"{s['video']}_{s['t0']:.2f}.npz").exists()]

    def fetch(s):
        return s, stream_frames(s["video"], s["t0"], s["t1"])

    def store(s: dict, frames: list[np.ndarray]) -> None:
        p = cache / f"{s['video']}_{s['t0']:.2f}.npz"
        f = featurize(eng, backbone, frames) if frames else None
        if f is None:
            np.savez(p, missing=np.array(1))
        else:
            np.savez(p, **f)

    # Bounded prefetch: at most 2 x workers decoded segments in memory (Executor.map would
    # submit every download at once and buffer all decoded frames).
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending: deque = deque()
        for s in todo:
            pending.append(ex.submit(fetch, s))
            if len(pending) >= 2 * workers:
                store(*pending.popleft().result())
        while pending:
            store(*pending.popleft().result())
    out = []
    for s in segs:
        with np.load(cache / f"{s['video']}_{s['t0']:.2f}.npz") as z:
            if "missing" in z.files:
                continue
            out.append({**s, **{k: z[k] for k in z.files}})
    return out


def _X(rows: list[dict], which: str) -> np.ndarray:
    if which == "motion_only":
        return np.stack([r["motion"] for r in rows])
    return np.stack([np.concatenate([r["crop_mean"], r["crop_max"], r["frame_mean"], r["motion"]]) for r in rows])


def _fit(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(X, y)


def _report(rows: list[dict], pred: list[str]) -> dict:
    y = [r["label"] for r in rows]
    groups = [r["video"] for r in rows]
    out = {"n": len(rows), "macro_f1": M.macro_f1(y, pred, LABELS), "accuracy": float(np.mean([a == b for a, b in zip(y, pred)])),
           "per_class": M.per_class(y, pred, LABELS), "confusion_rows_true_cols_pred": M.confusion(y, pred, LABELS),
           "macro_f1_ci95": M.group_bootstrap(lambda ix: M.macro_f1([y[i] for i in ix], [pred[i] for i in ix], LABELS), groups, 1000, 0)}
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["breed"]].append(i)
    out["by_breed"] = {b: {"n": len(ix), "accuracy": float(np.mean([y[i] == pred[i] for i in ix]))} for b, ix in sorted(by.items())}
    return out


def run(settings: Settings, out_dir: Path, artifact_dir: Path | None = None) -> dict:
    root = settings.paths.resolve("data_root") / "mammalnet"
    ann = root / "_download" / "x" / "annotation" / "detection_annotations.json"
    tar = root / "_download" / "annotation.tar"
    if not ann.exists() or sha256_file(tar) != ANNOTATION_SHA256:
        raise FileNotFoundError("MammalNet annotations missing or changed; see docs/DATA_AND_MODEL_REGISTER.md")
    segs = segments(ann)
    rows = build_features(settings, segs, settings.paths.resolve("data_root").parent / "cache" / "e11")
    by = {s: [r for r in rows if r["subset"] == s] for s in ("training", "validation", "testing")}
    ytr = [r["label"] for r in by["training"]]
    results = {}
    chosen = {}
    for which in ("full", "motion_only"):
        Xtr, Xva, Xte = _X(by["training"], which), _X(by["validation"], which), _X(by["testing"], which)
        dev = {C: M.macro_f1([r["label"] for r in by["validation"]], list(_fit(Xtr, ytr, C).predict(Xva)), LABELS) for C in C_GRID}
        C = max(dev, key=dev.get)
        m = _fit(Xtr, ytr, C)
        chosen[which] = (m, C)
        results[which] = {"C": C, "validation_macro_f1_by_C": {str(k): v for k, v in dev.items()},
                          "validation": _report(by["validation"], list(m.predict(Xva))), "test": _report(by["testing"], list(m.predict(Xte)))}
    maj = Counter(ytr).most_common(1)[0][0]
    summary = {
        "experiment": "E11", "title": "Dog behavior recognition from real video segments (MammalNet, dog breeds)",
        "protocol": __doc__.strip().splitlines(),
        "segments": {s: len(v) for s, v in by.items()}, "segments_requested": len(segs), "segments_unavailable": len(segs) - len(rows),
        "label_counts": {s: dict(Counter(r["label"] for r in v)) for s, v in by.items()},
        "breeds": dict(Counter(r["breed"] for r in rows)),
        "frames_with_dog_fraction": float(np.mean([r["frames_with_dog"] / max(r["frames"], 1) for r in rows])) if rows else None,
        "results": results,
        "baseline_majority_test_macro_f1": M.macro_f1([r["label"] for r in by["testing"]], [maj] * len(by["testing"]), LABELS),
        "terms": "MammalNet videos are third-party YouTube uploads without a published dataset licence: research profile only.",
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    if artifact_dir is not None:
        m, C = chosen["full"]
        sc, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        p = artifact_dir / "dog_behavior_video_v1.npz"
        np.savez(p, mean=sc.mean_, scale=sc.scale_, coef=lr.coef_, intercept=lr.intercept_, classes=np.array(list(lr.classes_)), version=np.array("dog_behavior_video_v1"))
        summary["artifact_sha256"] = sha256_file(p)
    write_json(out_dir / "metrics.json", summary)
    return summary
