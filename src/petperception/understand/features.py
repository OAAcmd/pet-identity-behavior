"""Per-clip evidence for behaviour understanding: the target animal's track, motion over time, and frame embeddings.

The user names the species, so the detector is used only to *locate* an
animal (cat or dog boxes are both accepted; the detector's species vote is
reported, never used to reject the user's answer). Stored per clip (.npz):

* ``t`` [T] sample times at ``MOTION_FPS``; ``box`` [T,4] (normalized, NaN
  where no animal); ``det`` [T,3] = (detector score, 1 if the detector said
  dog, 1 if interpolated);
* motion per step (``MOTION_FPS``) inside the animal's region: ``flow`` [T,5] =
  (mean speed, angular velocity about the region centre in rad/s, mean dx,
  mean dy, share of moving pixels), all relative to the region size;
* ``app`` [T,576]: 24x24 grey appearance of the region (z-scored), for
  periodicity;
* ``emb_t`` [E], ``emb`` [E,768]: SigLIP embeddings of the animal crop at
  ``EMB_FPS``; ``scene`` [768]: SigLIP embedding of the middle frame;
* audio: ``a_t`` [W], ``a_scores`` [W,521] (YAMNet), when the clip has sound.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MOTION_FPS = 6.0
EMB_FPS = 2.0
MAX_SECONDS = 60.0
MAX_SIDE = 480
SIGLIP_DIR = "var/models/siglip_base_patch16_224"
SIGLIP_REVISION = "google/siglip-base-patch16-224@7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"


def decode(path: Path, fps: float = MOTION_FPS, max_seconds: float = MAX_SECONDS, max_side: int = MAX_SIDE) -> tuple[list[np.ndarray], np.ndarray]:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:stream_side_data=rotation",
                            "-of", "json", str(path)], capture_output=True, text=True, timeout=60)
    import json

    try:
        st = json.loads(probe.stdout)["streams"][0]
        w0, h0 = int(st["width"]), int(st["height"])
        rot = abs(int(float((st.get("side_data_list") or [{}])[0].get("rotation", 0) or 0)))
        if rot in (90, 270):
            w0, h0 = h0, w0
    except (KeyError, IndexError, ValueError):
        return [], np.zeros(0)
    k = min(1.0, max_side / max(w0, h0))
    w, h = max(2, int(round(w0 * k / 2)) * 2), max(2, int(round(h0 * k / 2)) * 2)
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-t", str(max_seconds), "-vf", f"fps={fps},scale={w}:{h}",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, timeout=900)
    n = len(r.stdout) // (w * h * 3)
    frames = [np.frombuffer(r.stdout[i * w * h * 3:(i + 1) * w * h * 3], np.uint8).reshape(h, w, 3) for i in range(n)]
    return frames, np.arange(n, dtype=np.float32) / fps


def decode_audio16k(path: Path, max_seconds: float = MAX_SECONDS) -> np.ndarray | None:
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-t", str(max_seconds), "-vn", "-ac", "1", "-ar", "16000",
                        "-f", "f32le", "pipe:1"], capture_output=True, timeout=300)
    if r.returncode != 0 or len(r.stdout) < 16000 * 4 // 2:
        return None
    return np.frombuffer(r.stdout, np.float32).copy()


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def locate(detector, frames: list[np.ndarray], score: float = 0.25, max_gap: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Single-target track: per frame the cat/dog box that best continues the previous one (else the largest).
    Gaps up to ``max_gap`` samples are interpolated."""
    T = len(frames)
    box = np.full((T, 4), np.nan, np.float32)
    det = np.zeros((T, 3), np.float32)
    prev = None
    for i, rgb in enumerate(frames):
        d = [x for x in detector.detect(rgb, ["cat", "dog"], score, 0.5, 5)]
        if not d:
            continue
        if prev is not None:
            best = max(d, key=lambda x: (_iou(prev, x.box), x.area))
            if _iou(prev, best.box) == 0.0:
                best = max(d, key=lambda x: x.area)
        else:
            best = max(d, key=lambda x: x.area)
        box[i] = best.box
        det[i, :2] = [best.score, 1.0 if best.label == "dog" else 0.0]
        prev = best.box
    ok = ~np.isnan(box[:, 0])
    idx = np.where(ok)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            for j in range(a + 1, b):
                f = (j - a) / (b - a)
                box[j] = (1 - f) * box[a] + f * box[b]
                det[j, 2] = 1.0
    return box, det


def _region(box: np.ndarray, hw: tuple[int, int], scale: float = 1.3) -> tuple[int, int, int, int]:
    h, w = hw
    cx, cy = (box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h
    side = max((box[2] - box[0]) * w, (box[3] - box[1]) * h, 16) * scale
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    return x0, y0, max(x0 + 8, x1), max(y0 + 8, y1)


def motion(frames: list[np.ndarray], box: np.ndarray, fps: float = MOTION_FPS) -> tuple[np.ndarray, np.ndarray]:
    """flow [T,5] (step t-1 -> t, in the region of frame t-1's box) and appearance descriptors [T,576]."""
    T = len(frames)
    flow = np.full((T, 5), np.nan, np.float32)
    app = np.full((T, 576), np.nan, np.float32)
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32)
    rx, ry = xx - 47.5, yy - 47.5
    r2 = rx ** 2 + ry ** 2 + 25.0
    for i in range(T):
        if np.isnan(box[i, 0]):
            continue
        x0, y0, x1, y1 = _region(box[i], grays[i].shape)
        a = cv2.resize(grays[i][y0:y1, x0:x1], (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        app[i] = (a - a.mean()) / (a.std() + 1e-6)
        if i == 0 or np.isnan(box[i - 1, 0]):
            continue
        x0, y0, x1, y1 = _region(box[i - 1], grays[i].shape)
        p = cv2.resize(grays[i - 1][y0:y1, x0:x1], (96, 96), interpolation=cv2.INTER_AREA)
        q = cv2.resize(grays[i][y0:y1, x0:x1], (96, 96), interpolation=cv2.INTER_AREA)
        f = cv2.calcOpticalFlowFarneback(p, q, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vx, vy = f[..., 0], f[..., 1]
        mag = np.hypot(vx, vy)
        moving = mag > 0.5
        wgt = mag * moving
        omega = float(np.sum(wgt * (rx * vy - ry * vx) / r2) / max(np.sum(wgt), 1e-6))  # rad per step (image y down)
        flow[i] = [float(mag.mean()) / 96 * fps, omega * fps, float(vx.mean()) / 96 * fps, float(vy.mean()) / 96 * fps, float(moving.mean())]
    return flow, app


class SiglipEncoder:
    def __init__(self, model_dir: str | Path = SIGLIP_DIR, threads: int = 2):
        import torch
        from transformers import AutoModel, AutoProcessor

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = AutoModel.from_pretrained(str(model_dir)).eval()
        self.proc = AutoProcessor.from_pretrained(str(model_dir))
        self.scale = float(self.model.logit_scale.exp())
        self.bias = float(self.model.logit_bias)

    def images(self, imgs: list[np.ndarray], batch: int = 16) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(imgs), batch):
                x = self.proc(images=imgs[i:i + batch], return_tensors="pt")
                e = self.model.get_image_features(**x)
                out.append(self.torch.nn.functional.normalize(e, dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            y = self.proc(text=texts, padding="max_length", return_tensors="pt")
            e = self.model.get_text_features(**y)
            return self.torch.nn.functional.normalize(e, dim=-1).numpy()


def square_crop(rgb: np.ndarray, box: np.ndarray, scale: float = 1.2) -> np.ndarray:
    x0, y0, x1, y1 = _region(box, rgb.shape[:2], scale)
    return cv2.resize(rgb[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_AREA)


@dataclass
class Extractor:
    detector: object
    encoder: SiglipEncoder
    yamnet: object | None = None

    @classmethod
    def load(cls, threads: int = 2, detector_path: str | None = None) -> Extractor:
        from ..intent import av_features as AF

        comp = AF.load_components(threads, with_audio=True)
        det = comp.detector
        if detector_path:
            from ..vision.detector import EfficientDetDetector

            det = EfficientDetDetector(Path(detector_path), threads)
        return cls(det, SiglipEncoder(threads=threads), comp.yamnet)

    def extract(self, path: Path, with_audio: bool = True) -> dict:
        frames, t = decode(path)
        out: dict = {"t": t, "frames": np.array(len(frames))}
        if not frames:
            out["video_state"] = np.array("absent")
            return out
        out["video_state"] = np.array("present")
        out["hw"] = np.array(frames[0].shape[:2])
        box, det = locate(self.detector, frames)
        flow, app = motion(frames, box)
        step = int(round(MOTION_FPS / EMB_FPS))
        idx = [i for i in range(0, len(frames), step) if not np.isnan(box[i, 0])]
        crops = [square_crop(frames[i], box[i]) for i in idx]
        out.update({"box": box, "det": det, "flow": flow, "app": app.astype(np.float16), "emb_t": t[idx] if idx else np.zeros(0, np.float32),
                    "emb": self.encoder.images(crops).astype(np.float16) if crops else np.zeros((0, 768), np.float16),
                    "scene": self.encoder.images([cv2.resize(frames[len(frames) // 2], (224, 224), interpolation=cv2.INTER_AREA)])[0].astype(np.float16)})
        if with_audio and self.yamnet is not None:
            y = decode_audio16k(path)
            if y is not None and float(np.sqrt(np.mean(y ** 2))) > 1e-4:
                ws = self.yamnet.score_waveform16k(np.clip(y, -1, 1), 0.0, 0.48)
                out["a_t"] = ws.t_start.astype(np.float32)
                out["a_scores"] = ws.scores.astype(np.float16)
                out["audio_state"] = np.array("present")
            else:
                out["audio_state"] = np.array("absent" if y is None else "silent")
        return out
