"""Per-frame and per-window features of synchronized audio-visual clips, computed with the product's components.

Everything here is what the phone pipeline computes (or can compute once the
bundle exposes the same tensors): the COCO detector, the crop verifier, the
EfficientNet-Lite0 backbone (pooled 1280-d embedding of the animal crop and of
the whole frame), the posture head, the cat/dog keypoint head, the person-pet
interaction geometry model, and YAMNet (521 class scores plus the 1024-d
embedding). No label of the clip is used: the animal is the largest cat or dog
detection of any species, as at inference time.

Stored per clip (``.npz``; embeddings float16):

* video, sampled at ``fps``: ``v_t``; ``v_whole`` [T,1280] (scene / environment);
  ``v_crop`` [T,1280] (animal appearance and body configuration, zeros without
  an animal); ``v_box`` [T,4]; ``v_pet`` [T,4] = (present, detector score,
  verifier score or -1, species 0 cat / 1 dog); ``v_posture`` [T,3]
  (lying/sitting/standing scores); ``v_kp`` [T,24,3] (keypoint x, y relative to
  the box, visibility); ``v_person`` [T,4] = (people, best person score,
  best interaction score, best IoU with the animal); ``v_objects`` [T,C] (best
  score per COCO class); ``v_motion`` [T] (mean absolute grey-level change
  from the previous frame, 0..1);
* audio: ``a_t`` [W]; ``a_emb`` [W,1024]; ``a_scores`` [W,521]; ``a_rms_db``
  [W]; ``audio_state`` (present / absent / silent).

Optional ``frame_fn`` / ``audio_fn`` transforms degrade the inputs for the
robustness experiments (they are applied before any model sees the data).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import Settings, load_settings

YAMNET_SR = 16000
SILENT_DBFS = -60.0


@dataclass
class Components:
    settings: Settings
    detector: object
    verifier: object | None
    backbone: object
    posture: object | None
    pose: object | None
    interaction: object | None
    yamnet: object | None
    object_classes: list[str]
    versions: dict


def load_components(num_threads: int = 2, with_audio: bool = True) -> Components:
    from ..pipeline import AnalysisEngine
    from ..registry import model_dir

    s = load_settings(overrides={"audio": {"enabled": False}, "pose": {"provider": "catdog_keypoints_awa_v1", "head_file": "artifacts/keypoints/catdog_keypoints_head_awa_v1.npz"}, "posture": {"enabled": True},
                                 "runtime": {"num_threads": num_threads}})
    eng = AnalysisEngine(s).load()
    if eng.detector is None or eng._backbone is None:
        raise RuntimeError("detector or backbone unavailable; run: petperception fetch-models")
    yam = None
    if with_audio:
        from ..audio.yamnet import YamnetBackend

        yam = YamnetBackend(model_dir(s, s.audio.yamnet.model_id), num_threads, with_embeddings=True)
    objects = [lab for lab in eng.detector.labels if lab not in ("???", "cat", "dog", "person")]
    versions = {k: (v.model_id, v.version, v.state) for k, v in eng.status.items()}
    versions["yamnet"] = s.audio.yamnet.model_id if with_audio else None
    return Components(s, eng.detector, eng.verifier, eng._backbone, eng.posture, eng.pose if eng.pose.available else None, eng.interaction, yam,
                      objects, versions)


def decode_frames(path: Path, fps: float, max_side: int = 480) -> list[np.ndarray]:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=60)
    try:
        w0, h0 = (int(v) for v in probe.stdout.strip().split(",")[:2])
    except ValueError:
        return []
    k = min(1.0, max_side / max(w0, h0))
    w, h = max(2, int(round(w0 * k / 2)) * 2), max(2, int(round(h0 * k / 2)) * 2)
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "pipe:1"], capture_output=True, timeout=300)
    n = len(r.stdout) // (w * h * 3)
    return [np.frombuffer(r.stdout[i * w * h * 3:(i + 1) * w * h * 3], np.uint8).reshape(h, w, 3) for i in range(n)]


def decode_audio16k(path: Path) -> np.ndarray | None:
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(YAMNET_SR), "-f", "f32le", "pipe:1"],
                       capture_output=True, timeout=300)
    if r.returncode != 0 or not r.stdout:
        return None
    return np.frombuffer(r.stdout, np.float32).copy()


def audio_features(c: Components, y: np.ndarray | None, hop_s: float = 0.48) -> dict:
    if y is None or y.size == 0:
        return {"audio_state": np.array("absent")}
    y = np.clip(y, -1.0, 1.0)
    ws = c.yamnet.score_waveform16k(y, 0.0, hop_s)
    win = int(0.96 * YAMNET_SR)
    hop = int(round(hop_s * YAMNET_SR))
    rms = []
    for k in range(len(ws.t_start)):
        seg = y[k * hop: k * hop + win]
        rms.append(20 * np.log10(max(1e-6, float(np.sqrt(np.mean(seg ** 2))) if seg.size else 1e-6)))
    rms = np.array(rms, np.float32)
    state = "silent" if float(np.max(rms)) < SILENT_DBFS else "present"
    out = {"a_t": ws.t_start.astype(np.float32), "a_scores": ws.scores.astype(np.float16), "a_rms_db": rms, "audio_state": np.array(state)}
    if ws.embeddings is not None:
        out["a_emb"] = ws.embeddings.astype(np.float16)
    return out


def empty_video(T: int, n_objects: int, fps: float) -> dict:
    return {"v_t": np.arange(T, dtype=np.float32) / fps, "v_whole": np.zeros((T, 1280), np.float16), "v_crop": np.zeros((T, 1280), np.float16),
            "v_box": np.zeros((T, 4), np.float32), "v_pet": np.zeros((T, 4), np.float32), "v_posture": np.zeros((T, 3), np.float32),
            "v_kp": np.zeros((T, 24, 3), np.float32), "v_person": np.zeros((T, 4), np.float32),
            "v_objects": np.zeros((T, n_objects), np.float16), "v_motion": np.zeros(T, np.float32)}


def frame_features(c: Components, rgb: np.ndarray, t_s: float, prev_gray: np.ndarray | None, out: dict, i: int) -> np.ndarray:
    """Fill row i of the per-frame arrays in ``out`` for one RGB frame; returns this frame's 64x48 grey image."""
    from ..vision.detector import Detection
    from ..vision.interaction import _iou
    from ..vision.keypoints import square_crop
    from ..vision.posture import posture_features

    s = c.settings
    hw = rgb.shape[:2]
    g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    out["v_motion"][i] = float(np.mean(np.abs(g - prev_gray))) if prev_gray is not None else 0.0
    out["v_whole"][i] = c.backbone.features(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA))["pooled"]
    dets = c.detector.detect(rgb, ["cat", "dog", "person"] + c.object_classes, s.vision.score_threshold, s.vision.nms_iou, 25)
    for d in dets:
        if d.label in c.object_classes:
            j = c.object_classes.index(d.label)
            out["v_objects"][i, j] = max(float(out["v_objects"][i, j]), d.score)
    pets = [d for d in dets if d.label in ("cat", "dog")]
    people = [d for d in dets if d.label == "person" and d.score >= s.interaction.min_person_score]
    if people:
        out["v_person"][i, :2] = [len(people), max(p.score for p in people)]
    if not pets:
        return g
    a = max(pets, key=lambda d: d.area)
    vs = -1.0
    if c.verifier is not None:
        vr = c.verifier.verify(rgb, Detection(a.label, a.score, a.box, getattr(a, "class_scores", None)))
        vs = vr[0] if vr is not None else -1.0
    out["v_pet"][i] = [1.0, a.score, vs, 1.0 if a.label == "dog" else 0.0]
    out["v_box"][i] = a.box
    feats = c.backbone.features(square_crop(rgb, a.box)[0])
    out["v_crop"][i] = feats["pooled"]
    if c.posture is not None:
        out["v_posture"][i] = c.posture.scores(posture_features(feats["pooled"], a.box, hw, a.label))
    if c.pose is not None:
        kp = c.pose.estimate(rgb, a.box, t_s, feats)
        bw, bh = max(a.box[2] - a.box[0], 1e-4), max(a.box[3] - a.box[1], 1e-4)
        out["v_kp"][i, :, 0] = (kp.xy[:, 0] - a.box[0]) / bw
        out["v_kp"][i, :, 1] = (kp.xy[:, 1] - a.box[1]) / bh
        out["v_kp"][i, :, 2] = kp.visibility
    if people and c.interaction is not None:
        out["v_person"][i, 2] = max(c.interaction.score(p.box, a.box, hw, a.label) for p in people)
        out["v_person"][i, 3] = max(_iou(p.box, a.box) for p in people)
    return g


def video_features(c: Components, frames: list[np.ndarray], fps: float) -> dict:
    out = empty_video(len(frames), len(c.object_classes), fps)
    prev = None
    for i, rgb in enumerate(frames):
        prev = frame_features(c, rgb, float(out["v_t"][i]), prev, out, i)
    return out


def extract_clip(c: Components, path: Path, fps: float = 2.0, frame_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
                 audio_fn: Callable[[np.ndarray], np.ndarray] | None = None, video: bool = True, audio: bool = True) -> dict:
    out: dict = {}
    if audio:
        y = decode_audio16k(path)
        if y is not None and audio_fn is not None:
            y = audio_fn(y)
        out.update(audio_features(c, y))
    if video:
        frames = decode_frames(path, fps)
        if frame_fn is not None:
            frames = [frame_fn(f, i) for i, f in enumerate(frames)]
        out.update(video_features(c, frames, fps))
        out["video_state"] = np.array("present" if frames else "absent")
    return out


def extract_many(c: Components, items: list[tuple[str, Path]], cache_dir: Path, log_every: int = 25, **kw) -> dict[str, Path]:
    """items: (clip key, path). Resumable: clips with a cache file are skipped."""
    import time

    cache_dir.mkdir(parents=True, exist_ok=True)
    done: dict[str, Path] = {}
    t0 = time.perf_counter()
    todo = [(k, p) for k, p in items if not (cache_dir / f"{k}.npz").exists()]
    for n, (k, p) in enumerate(todo, 1):
        f = extract_clip(c, p, **kw)
        tmp = cache_dir / f"{k}.tmp.npz"
        np.savez(tmp, **f)
        tmp.replace(cache_dir / f"{k}.npz")
        if n % log_every == 0 or n == len(todo):
            el = time.perf_counter() - t0
            print(f"{cache_dir.name}: {n}/{len(todo)} clips, {el / n:.1f} s/clip", flush=True)
    for k, _ in items:
        done[k] = cache_dir / f"{k}.npz"
    return done
