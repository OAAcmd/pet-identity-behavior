"""One command: locate the animal, recognize behaviour over time, and compose a source-grounded answer to the user's question.

``petperception explain MEDIA --species cat --scene "..." --question "..."``
The species comes from the user and is never overridden; the detector only
locates a cat or dog. The deployed recognizer (``artifacts/understand/``,
written by E31) is used when present, the fixed zero-shot defaults otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import REPO_ROOT
from . import explain as X
from . import question as Qn
from . import recognize as R
from .features import Extractor
from .head import BehaviorHead

ARTIFACTS = REPO_ROOT / "artifacts" / "understand"


class _Z(dict):
    @property
    def files(self):
        return list(self.keys())


@dataclass
class Explainer:
    extractor: Extractor
    zs: R.ZeroShot
    config: R.RecognizerConfig
    head: BehaviorHead | None

    @classmethod
    def load(cls, threads: int = 2, detector_path: str | None = None, artifacts: Path = ARTIFACTS) -> Explainer:
        ex = Extractor.load(threads, detector_path)
        cfg_path = artifacts / "recognizer.json"
        cfg = R.RecognizerConfig.load(cfg_path) if cfg_path.exists() else R.RecognizerConfig()
        head = BehaviorHead.load(artifacts / "behavior_head") if cfg.method != "zero_shot" and (artifacts / "behavior_head.json").exists() else None
        if cfg.method != "zero_shot" and head is None:
            cfg = R.RecognizerConfig()  # configuration without its head: fall back to the fixed zero-shot defaults
        return cls(ex, R.ZeroShot(ex.encoder), cfg, head)

    def explain(self, media: Path, species: str, scene: str = "", question: str = "") -> X.Answer:
        if species not in ("cat", "dog"):
            raise ValueError("species must be 'cat' or 'dog'")
        f = self.extractor.extract(Path(media))
        z = _Z(f)
        q = Qn.parse(species, scene, question)
        notes = []
        if str(f.get("video_state")) == "absent":
            notes.append("No video frames could be decoded; only the question was used.")
        audio = str(f.get("audio_state", "absent"))
        if audio != "present":
            notes.append("No usable sound in the recording, so vocalizations could not be checked." if audio == "absent" else
                         "The recording is silent, so vocalizations could not be checked.")
        yam = self.extractor.yamnet.labels if self.extractor.yamnet is not None and "a_scores" in z else None
        readings, pattern = R.recognize(z, species, self.zs, yam, self.head, self.config)
        return X.compose(q, readings, pattern, self.config, notes)
