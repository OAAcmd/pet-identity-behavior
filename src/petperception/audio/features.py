"""Lightweight CPU MFCC / spectral summary features.

The feature configuration is part of every persisted model so preprocessing
is reproducible. Features are computed at ``sample_rate`` (audio is resampled
to it); ``fmax`` must not exceed the Nyquist frequency of the *original*
recording, otherwise the upper mel bands would only see resampling artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..ingest.media import resample


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 40
    n_mfcc: int = 20
    fmin: float = 50.0
    fmax: float = 7500.0
    trim_silence_db: float | None = 40.0  # trim leading/trailing silence relative to peak
    version: str = "mfcc-stats-v1"

    def to_dict(self) -> dict:
        return asdict(self)


def feature_names(cfg: FeatureConfig) -> list[str]:
    names = []
    for stat in ("mean", "std"):
        names += [f"mfcc{i}_{stat}" for i in range(cfg.n_mfcc)]
        names += [f"dmfcc{i}_{stat}" for i in range(cfg.n_mfcc)]
        names += [f"{n}_{stat}" for n in ("centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms_db")]
    names += ["active_duration_s"]
    return names


def extract(x: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    import librosa  # imported lazily: numba JIT start-up is slow

    y = resample(np.asarray(x, np.float32), sr, cfg.sample_rate)
    if cfg.trim_silence_db is not None and y.size:
        yt, _ = librosa.effects.trim(y, top_db=cfg.trim_silence_db, frame_length=cfg.n_fft, hop_length=cfg.hop_length)
        if yt.size >= cfg.n_fft:
            y = yt
    if y.size < cfg.n_fft:
        y = np.pad(y, (0, cfg.n_fft - y.size))
    fmax = min(cfg.fmax, cfg.sample_rate / 2)
    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length, center=True)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=cfg.sample_rate, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=fmax)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel + 1e-10), n_mfcc=cfg.n_mfcc)
    width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1)
    dmfcc = librosa.feature.delta(mfcc, width=width) if width >= 3 else np.zeros_like(mfcc)
    mag = np.sqrt(S)
    cent = librosa.feature.spectral_centroid(S=mag, sr=cfg.sample_rate)
    bw = librosa.feature.spectral_bandwidth(S=mag, sr=cfg.sample_rate)
    roll = librosa.feature.spectral_rolloff(S=mag, sr=cfg.sample_rate)
    flat = librosa.feature.spectral_flatness(S=mag)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)
    rms = librosa.feature.rms(S=mag, frame_length=cfg.n_fft)
    rms_db = 20 * np.log10(rms + 1e-10)
    blocks = [mfcc, dmfcc, np.vstack([cent / 1000.0, bw / 1000.0, roll / 1000.0, flat, zcr, rms_db / 10.0])]
    means = [b.mean(axis=1) for b in blocks]
    stds = [b.std(axis=1) for b in blocks]
    # Order matches feature_names(): (mfcc, dmfcc, spectral) means, then stds.
    m = np.concatenate(means)
    s = np.concatenate(stds)
    vec = np.concatenate([m, s, np.array([y.size / cfg.sample_rate])])
    return vec.astype(np.float32)
