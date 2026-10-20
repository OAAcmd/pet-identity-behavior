"""End-to-end intent analysis of one media file (features -> trained channels -> combined inference)."""

from __future__ import annotations

from pathlib import Path

from . import av_features as AF
from . import inference as I
from . import kinetics as K


class IntentAnalyzer:
    def __init__(self, num_threads: int = 2, situation_variant: str = "full"):
        self.components = AF.load_components(num_threads)
        self.models = I.IntentModels(variant=situation_variant)
        self.labels = self.components.yamnet.labels

    def features(self, path: Path, modalities: tuple[str, ...] = ("audio", "video"), **kw) -> tuple[dict, dict]:
        f = AF.extract_clip(self.components, Path(path), video="video" in modalities, audio="audio" in modalities, **kw)
        if "audio" not in modalities:
            f["audio_state"] = "disabled"
        if "video" not in modalities:
            f["video_state"] = "disabled"
        return K.pool(_NpzLike(f))

    def analyze(self, path: Path, modalities: tuple[str, ...] = ("audio", "video"), environment: str | None = None) -> I.IntentResult:
        br, cues = self.features(path, modalities)
        return I.infer(self.models, {b: (None if v is None else v[None, :]) for b, v in br.items()}, cues, self.labels, environment)


class _NpzLike:
    """Gives a feature dict the ``.files`` / item access of an opened .npz (as stored by extract_many)."""

    def __init__(self, d: dict):
        import numpy as np

        self._d = {k: (v if isinstance(v, np.ndarray) else np.array(v)) for k, v in d.items()}
        self.files = list(self._d)

    def __getitem__(self, k):
        return self._d[k]
