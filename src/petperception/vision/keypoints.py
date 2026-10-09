"""Dog keypoints: frozen EfficientNet-Lite0 features + a small trained head.

Backbone: ``efficientnet_lite0_mediapipe_v1`` (Apache-2.0 per embedded
metadata), run unchanged; three intermediate feature maps (28x28x40,
14x14x112, 7x7x320) are read from the interpreter. Head (trained here by
``eval/e9_dog_keypoints.py`` on the Ultralytics dog-pose data, Research-Only):
per-cell 1x1 projection, one 3x3 convolution, then per keypoint a spatial
softmax over the 14x14 grid, a sub-cell offset and a visibility logit.

Keypoints follow the dog-pose vocabulary (24 slots; the eye, withers and
throat slots are never annotated in this release, so 20 points are learned
and reported: paws, knees, elbows, tail start/end, ear base/tip, nose, chin). There is no cat keypoint model: no cat keypoint dataset
was available to this project (see docs/DATA_AND_MODEL_REGISTER.md).

Outputs are image-normalized coordinates with two unitless scores per point:
``peak`` (spatial-softmax mass near the argmax) and ``visible`` (sigmoid of the
visibility logit). Neither is a calibrated probability.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .pose import KeypointSet
from .tflite_io import dequantize, encode_image
from .tflite_meta import input_normalization

DOG_KEYPOINTS = [
    "front_left_paw", "front_left_knee", "front_left_elbow", "rear_left_paw", "rear_left_knee", "rear_left_elbow",
    "front_right_paw", "front_right_knee", "front_right_elbow", "rear_right_paw", "rear_right_knee", "rear_right_elbow",
    "tail_start", "tail_end", "left_ear_base", "right_ear_base", "nose", "chin", "left_ear_tip", "right_ear_tip",
    "left_eye", "right_eye", "withers", "throat",
]
# Never annotated in the dog-pose release used here (0% label rate): not reported.
UNANNOTATED = ("left_eye", "right_eye", "withers", "throat")
REPORTED_KEYPOINTS = tuple(k for k in DOG_KEYPOINTS if k not in UNANNOTATED)
# Index pairs swapped by a horizontal flip.
FLIP_PAIRS = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (14, 15), (18, 19), (20, 21)]
FEATURE_TENSORS = {"f28": (28, 40), "f14": (14, 112), "f7": (7, 320)}
GRID = 14
CROP_SCALE = 1.15


def square_crop(rgb: np.ndarray, box: np.ndarray, scale: float = CROP_SCALE, size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    """Square crop around a normalized box (padded with gray). Returns the crop and its
    (x0, y0, side) in pixels, used to map crop coordinates back to the image."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, 8.0) * scale
    x0, y0 = cx - side / 2, cy - side / 2
    m = np.array([[size / side, 0, -x0 * size / side], [0, size / side, -y0 * size / side]], dtype=np.float64)
    crop = cv2.warpAffine(rgb, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
    return crop, np.array([x0, y0, side])


class Backbone:
    """EfficientNet-Lite0 with intermediate feature maps exposed."""

    def __init__(self, model_path: Path, num_threads: int = 2):
        from ai_edge_litert.interpreter import Interpreter

        data = model_path.read_bytes()
        norm = input_normalization(data)
        if norm is None:
            raise RuntimeError("input NormalizationOptions missing from backbone metadata")
        self.mean, self.std = float(norm[0][0]), float(norm[1][0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.interp = Interpreter(model_path=str(model_path), num_threads=num_threads, experimental_preserve_all_tensors=True)
        self.interp.allocate_tensors()
        self.in_detail = self.interp.get_input_details()[0]
        self.in_idx = self.in_detail["index"]
        self.quant = {d["index"]: d["quantization"] for d in self.interp.get_tensor_details()}
        # Pick the last tensor of each required shape (block outputs feeding the next stage).
        self.idx: dict[str, int] = {}
        for name, (g, c) in FEATURE_TENSORS.items():
            cands = [d["index"] for d in self.interp.get_tensor_details()
                     if tuple(int(v) for v in d["shape"]) == (1, g, g, c) and ("Add" in d["name"] or "blocks_15" in d["name"])]
            if not cands:
                raise RuntimeError(f"backbone tensor {name} {g}x{g}x{c} not found")
            self.idx[name] = max(cands)

        heads = [d["index"] for d in self.interp.get_tensor_details() if tuple(int(v) for v in d["shape"]) == (1, 7, 7, 1280)]
        if not heads:
            raise RuntimeError("backbone head tensor 7x7x1280 not found")
        self.head_idx = max(heads)

    def features(self, crop_rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Keypoint feature maps plus the pooled 1280-d head embedding ('pooled')."""
        self.interp.set_tensor(self.in_idx, encode_image(crop_rgb, self.in_detail, self.mean, self.std))
        self.interp.invoke()
        out = {k: dequantize(self.interp.get_tensor(i)[0], self.quant[i]).copy() for k, i in self.idx.items()}
        out["pooled"] = dequantize(self.interp.get_tensor(self.head_idx)[0], self.quant[self.head_idx]).mean(axis=(0, 1))
        return out


def assemble(f: dict[str, np.ndarray]) -> np.ndarray:
    """Stack the three maps on the 14x14 grid: f14 | upsampled f7 | space-to-depth f28 -> [14, 14, 592]."""
    f7 = np.repeat(np.repeat(f["f7"], 2, axis=0), 2, axis=1)
    f28 = f["f28"].reshape(GRID, 2, GRID, 2, -1).transpose(0, 2, 1, 3, 4).reshape(GRID, GRID, -1)
    return np.concatenate([f["f14"], f7, f28], axis=-1)


def head_forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy forward pass (mirrors the JAX training graph). x: [N, 14, 14, 592]."""
    z = (x - params["mean"]) / params["std"]
    h = np.maximum(z @ params["W1"] + params["b1"], 0.0)
    n, g, _, c = h.shape
    hp = np.pad(h, ((0, 0), (1, 1), (1, 1), (0, 0)))
    conv = np.zeros((n, g, g, params["Wc"].shape[-1]), dtype=h.dtype)
    for dy in range(3):
        for dx in range(3):
            conv += hp[:, dy:dy + g, dx:dx + g, :] @ params["Wc"][dy, dx]
    h2 = np.maximum(conv + params["bc"], 0.0) + h
    out = h2 @ params["W2"] + params["b2"]  # [N, g, g, K + 2K]
    k = params["bv"].shape[0]
    heat = out[..., :k]
    offs = out[..., k:].reshape(n, g, g, k, 2)
    vis = h2.mean(axis=(1, 2)) @ params["Wv"] + params["bv"]
    return heat, offs, vis


def decode(heat: np.ndarray, offs: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> xy in crop units [N, K, 2] (0..1), peak mass [N, K], visibility score [N, K]."""
    n, g, _, k = heat.shape
    flat = heat.reshape(n, g * g, k)
    flat = flat - flat.max(axis=1, keepdims=True)
    p = np.exp(flat)
    p /= p.sum(axis=1, keepdims=True)
    am = p.argmax(axis=1)  # [N, K]
    cy, cx = am // g, am % g
    ni, ki = np.meshgrid(np.arange(n), np.arange(k), indexing="ij")
    o = offs[ni, cy, cx, ki]  # [N, K, 2] in cell units, (dx, dy)
    xy = np.stack([(cx + 0.5 + o[..., 0]) / g, (cy + 0.5 + o[..., 1]) / g], axis=-1)
    pm = p.reshape(n, g, g, k)
    peak = np.zeros((n, k))
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            yy, xx = np.clip(cy + di, 0, g - 1), np.clip(cx + dj, 0, g - 1)
            ok = ((cy + di) == yy) & ((cx + dj) == xx)
            peak += np.where(ok, pm[ni, yy, xx, ki], 0.0)
    return xy, peak, 1.0 / (1.0 + np.exp(-vis))


@dataclass
class DogKeypointModel:
    backbone: Backbone
    params: dict[str, np.ndarray]
    version: str

    @classmethod
    def load(cls, backbone_path: Path, head_path: Path, num_threads: int = 2) -> DogKeypointModel:
        with np.load(head_path, allow_pickle=False) as z:
            params = {k: z[k].astype(np.float32) for k in z.files if k != "version"}
            version = str(z["version"]) if "version" in z.files else head_path.stem
        return cls(Backbone(backbone_path, num_threads), params, version)

    def predict(self, rgb: np.ndarray, box: np.ndarray, feats: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keypoints in image-normalized coordinates [K, 2], peak [K], visible [K].

        ``feats`` may be passed when the backbone already ran on ``square_crop(rgb, box)``."""
        crop, (x0, y0, side) = square_crop(rgb, box)
        heat, offs, vis = head_forward(self.params, assemble(feats if feats is not None else self.backbone.features(crop))[None])
        xy, peak, v = decode(heat, offs, vis)
        h, w = rgb.shape[:2]
        img = np.stack([(x0 + xy[0, :, 0] * side) / w, (y0 + xy[0, :, 1] * side) / h], axis=-1)
        return img, peak[0], v[0]


@dataclass
class DogPoseProvider:
    """PoseProvider for dogs backed by DogKeypointModel (research profile only)."""

    model: DogKeypointModel
    name: str = "dog_keypoints_effnetlite0_head"
    species: tuple[str, ...] = ("dog",)
    keypoint_vocabulary: tuple[str, ...] = REPORTED_KEYPOINTS
    available: bool = True
    unavailable_reason: str | None = None
    # A point is reported visible when both scores pass these a-priori gates
    # (chosen on the development split; see results/e9).
    min_visible: float = 0.5
    min_peak: float = 0.2

    def estimate(self, rgb: np.ndarray, box: np.ndarray, t_s: float, feats: dict[str, np.ndarray] | None = None) -> KeypointSet | None:
        xy, peak, vis = self.model.predict(rgb, box, feats)
        conf = np.where((vis >= self.min_visible) & (peak >= self.min_peak), np.minimum(vis, 1.0), 0.0)
        conf[[DOG_KEYPOINTS.index(k) for k in UNANNOTATED]] = 0.0
        return KeypointSet(t_s=t_s, names=list(DOG_KEYPOINTS), xy=xy, visibility=conf)
