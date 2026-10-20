"""Input degradations for the robustness experiments (applied before any model sees the data).

* Audio noise: an ESC-50 indoor-appliance or office recording (vacuum
  cleaner, washing machine, clock tick, keyboard typing, mouse click; no
  animal, water, door, vehicle or outdoor sound, so the noise is not itself a
  cue for any label) or white noise, looped and mixed at a target
  signal-to-noise ratio relative to the clip's RMS. The noise file is chosen
  deterministically from the clip key.
* Poor visual conditions: brightness x0.35, Gaussian pixel noise (sigma 10)
  and a 5x5 Gaussian blur, deterministic per clip.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

NOISE_CATEGORIES = ("vacuum_cleaner", "washing_machine", "clock_tick", "keyboard_typing", "mouse_click")


def _seed(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def noise_pool(esc50_root: Path, sr: int = 16000) -> list[tuple[str, np.ndarray]]:
    from ..ingest.media import resample

    pool = []
    with open(esc50_root / "meta" / "esc50.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["category"] in NOISE_CATEGORIES:
                x, s = sf.read(esc50_root / "audio" / r["filename"], dtype="float32", always_2d=False)
                x = x.mean(axis=1) if x.ndim > 1 else x
                x = resample(x, s, sr)
                if np.sqrt(np.mean(x ** 2)) > 1e-4:
                    pool.append((r["filename"], x.astype(np.float32)))
    pool.sort()
    return pool


def mix_at_snr(y: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if y.size == 0:
        return y
    reps = int(np.ceil(len(y) / max(1, len(noise))))
    n = np.tile(noise, reps)[: len(y)]
    ps = float(np.mean(y ** 2)) + 1e-12
    pn = float(np.mean(n ** 2)) + 1e-12
    out = y + n * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.99 if peak > 1.0 else out).astype(np.float32)


def audio_noise_fn(key: str, snr_db: float, pool: list[tuple[str, np.ndarray]] | None):
    rng = np.random.default_rng(_seed(key))

    def f(y: np.ndarray) -> np.ndarray:
        if pool:
            noise = pool[_seed(key) % len(pool)][1]
            off = int(rng.integers(0, max(1, len(noise) // 2)))
            noise = np.roll(noise, -off)
        else:
            noise = rng.standard_normal(len(y)).astype(np.float32)
        return mix_at_snr(y, noise, snr_db)

    return f


def poor_visual_fn(key: str):
    rng = np.random.default_rng(_seed(key) + 1)

    def f(rgb: np.ndarray, i: int) -> np.ndarray:
        x = rgb.astype(np.float32) * 0.35 + rng.normal(0, 10, rgb.shape)
        x = cv2.GaussianBlur(np.clip(x, 0, 255).astype(np.uint8), (5, 5), 0)
        return x

    return f
