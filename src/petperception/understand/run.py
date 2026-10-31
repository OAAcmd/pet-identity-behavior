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
    circling: object | None = None
    matcher: object | None = None
    places: object | None = None

    @classmethod
    def load(cls, threads: int = 2, detector_path: str | None = None, artifacts: Path = ARTIFACTS, version: int = 2) -> Explainer:
        """version 2 (default): E32 system (feature version 2, turning-in-place model, sound evidence, recording context, parser v2);
        version 1: the E31 system."""
        ex = Extractor.load(threads, detector_path, version=version)
        cfg_path = artifacts / "recognizer.json"
        cfg = R.RecognizerConfig.load(cfg_path) if cfg_path.exists() else R.RecognizerConfig()
        head = BehaviorHead.load(artifacts / "behavior_head") if cfg.method != "zero_shot" and (artifacts / "behavior_head.json").exists() else None
        if cfg.method != "zero_shot" and head is None:
            cfg = R.RecognizerConfig()  # configuration without its head: fall back to the fixed zero-shot defaults
        circ = matcher = places = None
        if version >= 2:
            from . import context as CX
            from . import question as Qn
            from . import temporal as TM

            if (artifacts / "recognizer_v2.json").exists():
                cfg = R.RecognizerConfig.load(artifacts / "recognizer_v2.json")
                head = None
            if (artifacts / "circling_v1.json").exists():
                circ = TM.CirclingModel.load(artifacts / "circling_v1.json")
            try:
                matcher = Qn.SemanticMatcher()
            except OSError:
                matcher = None  # sentence model not fetched: keyword parsing only
            places = CX.PlaceModel(ex.encoder)
        return cls(ex, R.ZeroShot(ex.encoder), cfg, head, circ, matcher, places)

    def explain(self, media: Path, species: str, scene: str = "", question: str = "") -> X.Answer:
        if species not in ("cat", "dog"):
            raise ValueError("species must be 'cat' or 'dog'")
        f = self.extractor.extract(Path(media))
        z = _Z(f)
        return self.answer(z, species, scene, question)

    def answer(self, z, species: str, scene: str = "", question: str = "") -> X.Answer:
        """Answer from already extracted evidence (``Extractor.extract`` output or a stored feature file)."""
        f = z
        v2 = self.extractor.version >= 2
        q = Qn.parse_v2(species, scene, question, self.matcher) if v2 else Qn.parse(species, scene, question)
        notes = []
        if str(f["video_state"]) == "absent" if "video_state" in f.files else False:
            notes.append("No video frames could be decoded; only the question was used.")
        audio = str(f["audio_state"]) if "audio_state" in f.files else "absent"
        if audio != "present":
            notes.append("No usable sound in the recording, so vocalizations could not be checked." if audio == "absent" else
                         "The recording is silent, so vocalizations could not be checked.")
        yam = self.extractor.yamnet.labels if self.extractor.yamnet is not None and "a_scores" in z.files else None
        if not v2:
            readings, pattern = R.recognize(z, species, self.zs, yam, self.head, self.config)
            return X.compose(q, readings, pattern, self.config, notes)
        from . import context as CX

        readings, pattern = R.recognize_v2(z, species, self.zs, yam, self.config, self.circling)
        items, cues = CX.recording_context(z, yam, self.places)
        return X.compose(q, readings, pattern, self.config, notes, items, cues, CX.vocal_types(z, species, yam))
