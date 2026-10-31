"""One command: locate the animal, recognize behaviour over time, and compose a source-grounded answer to the user's question.

``petperception explain MEDIA --species cat --scene "..." --question "..."``
The species comes from the user and is never overridden; the detector only
locates a cat or dog. Version 1 (default) is the frozen E31 system
(``understand/e31``, thresholds ``artifacts/understand/recognizer.json``);
version 2 is the E32 system (``--system-version 2``: feature version 2, the
turning-in-place model, recording context, parser v2; ``recognizer_v2.json``,
``circling_v1.json``). Results: ``results/e31``, ``results/e32``.
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
# E32 decision (pre-registered, results/e32/metrics.json "decision"): v2 was better than v1 on every paired measure but
# missed the absolute bar C4 >= 0.9 (0.85), so the default went back to the E31 system; version 2 stays available.
DEFAULT_VERSION = 1


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
    def load(cls, threads: int = 2, detector_path: str | None = None, artifacts: Path = ARTIFACTS, version: int = DEFAULT_VERSION) -> Explainer:
        """version 1 (default): the E31 system; version 2: the E32 system (feature version 2, turning-in-place model, sound evidence,
        recording context, parser v2)."""
        ex = Extractor.load(threads, detector_path, version=version)
        if version == 1:  # the frozen E31 system (understand/e31)
            from .e31 import recognize as R1

            cfg_path = artifacts / "recognizer.json"
            cfg = R1.RecognizerConfig.load(cfg_path) if cfg_path.exists() else R1.RecognizerConfig()
            head = BehaviorHead.load(artifacts / "behavior_head") if cfg.method != "zero_shot" and (artifacts / "behavior_head.json").exists() else None
            if cfg.method != "zero_shot" and head is None:
                cfg = R1.RecognizerConfig()  # configuration without its head: fall back to the fixed zero-shot defaults
            return cls(ex, R1.ZeroShot(ex.encoder), cfg, head)
        from . import context as CX
        from . import temporal as TM

        cfg = R.RecognizerConfig.load(artifacts / "recognizer_v2.json") if (artifacts / "recognizer_v2.json").exists() else R.RecognizerConfig()
        circ = TM.CirclingModel.load(artifacts / "circling_v1.json") if (artifacts / "circling_v1.json").exists() else None
        try:
            matcher = Qn.SemanticMatcher()
        except OSError:
            matcher = None  # sentence model not fetched: keyword parsing only
        return cls(ex, R.ZeroShot(ex.encoder), cfg, None, circ, matcher, CX.PlaceModel(ex.encoder))

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
        notes = []
        if str(f["video_state"]) == "absent" if "video_state" in f.files else False:
            notes.append("No video frames could be decoded; only the question was used.")
        audio = str(f["audio_state"]) if "audio_state" in f.files else "absent"
        if audio != "present":
            notes.append("No usable sound in the recording, so vocalizations could not be checked." if audio == "absent" else
                         "The recording is silent, so vocalizations could not be checked.")
        yam = self.extractor.yamnet.labels if self.extractor.yamnet is not None and "a_scores" in z.files else None
        if not v2:
            from .e31 import explain as X1
            from .e31 import question as Q1
            from .e31 import recognize as R1

            readings, pattern = R1.recognize(z, species, self.zs, yam, self.head, self.config)
            return X1.compose(Q1.parse(species, scene, question), readings, pattern, self.config, notes)
        from . import context as CX

        q = Qn.parse_v2(species, scene, question, self.matcher)
        readings, pattern = R.recognize_v2(z, species, self.zs, yam, self.config, self.circling)
        items, cues = CX.recording_context(z, yam, self.places)
        return X.compose(q, readings, pattern, self.config, notes, items, cues, CX.vocal_types(z, species, yam))
