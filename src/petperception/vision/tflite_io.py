"""Input encoding and output decoding shared by float and quantized TFLite models.

The float input of an image model is defined by its metadata
NormalizationOptions: x = (pixel - mean) / std. A quantized (uint8/int8) input
tensor holds the quantized encoding of that same float input, using the
tensor's own scale and zero point. Quantized outputs and intermediate tensors
are dequantized as (q - zero_point) * scale. Float models are unaffected.
"""

from __future__ import annotations

import numpy as np


def encode_image(img: np.ndarray, detail: dict, mean: float, std: float) -> np.ndarray:
    """Batch of one model input from an RGB uint8 image already resized to the model size."""
    x = (img.astype(np.float32) - mean) / std
    dt = np.dtype(detail["dtype"])
    if dt.kind == "f":
        return x[None].astype(dt)
    scale, zero = detail["quantization"]
    if not scale:
        raise RuntimeError(f"quantized input {detail.get('name')} has no quantization parameters")
    info = np.iinfo(dt)
    return np.clip(np.round(x / scale + zero), info.min, info.max).astype(dt)[None]


def dequantize(arr: np.ndarray, quantization: tuple[float, int]) -> np.ndarray:
    scale, zero = quantization
    if arr.dtype.kind in "iu" and scale:
        return (arr.astype(np.float32) - np.float32(zero)) * np.float32(scale)
    return arr
