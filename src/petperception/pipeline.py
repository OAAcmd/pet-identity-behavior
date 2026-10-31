"""Offline media analysis engine shared by the CLI and the API."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import __version__
from .audio.events import (
    CONTEXT_SOURCE,
    SoundEventRaw,
    build_groups,
    extract_events,
    load_context_thresholds,
    load_subtype_validation,
    load_thresholds,
    shown_subtypes,
    vocal_conflict_tags,
    vocal_tags,
)
from .audio.yamnet import YamnetBackend
from .config import REPO_ROOT, Settings
from .context.zones import ZoneConfig, occupancy
from .explain.templates import explain_hypothesis, suggestions_for
from .fusion.engine import Ev, FusionContext, FusionEngine
from .ingest import media as M
from .intent import av_features as AF
from .registry import check_model, model_dir, model_terms_summary, release_decision, resource
from .schemas import analysis as S
from .utils import StageTimer, sha256_file, utc_now
from .vision import behavior as BH
from .vision.detector import Detection, EfficientDetDetector, PostprocessedDetector, detect_at_operating_point, load_detector
from .vision.pose import UnavailablePoseProvider, compute_features
from .vision.quality import detection_flags, frame_quality
from .vision.tracking import Track, estimate_camera_motion, make_tracker, merge_species, movement_summary
from .vision.verifier import CropClassifier, DetectionVerifier

log = logging.getLogger("petperception.pipeline")

STATIC_LIMITATIONS = [
    "Research engine: outputs are observations plus unvalidated heuristics. They are not translations of animal "
    "language, readings of thoughts or emotions, or medical assessments.",
    "Track IDs are per-analysis labels for linked detections, not verified pet identities.",
    "Hypothesis scores are uncalibrated; multimodal accuracy has not been evaluated (no paired household validation data).",
    "Sound events use raw YAMNet scores with thresholds selected on ESC-50 clips, which are not household recordings.",
    "Cat/dog detection uses a COCO-trained EfficientDet-Lite model; a second-stage verifier reduces but does not remove "
    "false detections (E7, web photos).",
    "Behavior segments (stillness, movement, zone visits, back-and-forth, startle candidates) use fixed thresholds checked "
    "only on synthetic and semi-synthetic sequences; they are not validated against labelled household behavior.",
    "For veterinary or behavioral concerns, consult a qualified professional; this system does not diagnose.",
]


class AnalysisCancelled(Exception):
    """Raised inside ``analyze`` when the caller requested cancellation."""


class AnalysisEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.audio_backend: YamnetBackend | None = None
        self.detector: EfficientDetDetector | PostprocessedDetector | None = None
        self.verifier: DetectionVerifier | None = None
        self.pose = UnavailablePoseProvider()
        self.posture = None
        self.interaction = None
        self.intent = None  # (IntentModels, av_features.Components) when settings.intent.enabled
        self.object_relation = None
        self._backbone = None
        self.release_notes: list[str] = []
        self.fusion = FusionEngine(settings.fusion)
        self.status: dict[str, S.ComponentStatus] = {}
        self.thresholds: dict[str, float] = {}
        self.threshold_source: dict[str, str] = {}
        self.subtype_validation: dict[str, str] = {}
        self.context_thresholds: dict[str, float] = {}
        self.model_load_ms = 0.0
        self._groups = None

    # -- model loading -----------------------------------------------------
    def load(self) -> AnalysisEngine:
        t0 = time.perf_counter()
        s = self.settings
        if not s.audio.enabled or s.audio.event_backend == "none":
            self.status["audio_event"] = S.ComponentStatus(name="audio_event", kind="audio_event", state="disabled", reason="disabled by configuration")
        else:
            mid = s.audio.yamnet.model_id
            r = check_model(s, mid)
            rd = release_decision(s.release_profile, mid)
            if r.ready and not rd.allowed:
                self.status["audio_event"] = S.ComponentStatus(name="audio_event", kind="audio_event", state="unavailable", model_id=None, reason=rd.reason)
            elif r.ready:
                self.audio_backend = YamnetBackend(model_dir(s, mid), s.runtime.num_threads)
                self._groups = build_groups(self.audio_backend)
                th_path = s.audio.yamnet.thresholds_file
                td = release_decision(s.release_profile, "audio_event_thresholds_esc50")
                if th_path and not td.allowed:
                    self.release_notes.append(f"ESC-50-derived thresholds not used under the '{s.release_profile}' profile; unfitted defaults apply.")
                    th_path = None
                sv = s.audio.yamnet.subtype_validation_file
                self.subtype_validation = load_subtype_validation((REPO_ROOT / sv) if sv and not Path(sv).is_absolute() else (Path(sv) if sv else None))
                ct = s.audio.yamnet.context_thresholds_file
                self.context_thresholds = load_context_thresholds((REPO_ROOT / ct) if ct and not Path(ct).is_absolute() else (Path(ct) if ct else None))
                self.thresholds, self.threshold_source = load_thresholds(
                    (REPO_ROOT / th_path) if th_path and not Path(th_path).is_absolute() else (Path(th_path) if th_path else None),
                    s.audio.yamnet.default_thresholds)
                self.status["audio_event"] = S.ComponentStatus(name="audio_event", kind="audio_event", state="ok", model_id=mid, version="yamnet-mediapipe-v1")
            else:
                self.status["audio_event"] = S.ComponentStatus(name="audio_event", kind="audio_event", state="unavailable", model_id=mid, reason=r.reason)
        if not s.vision.enabled:
            self.status["visual_detector"] = S.ComponentStatus(name="visual_detector", kind="visual_detector", state="disabled", reason="disabled by configuration")
        else:
            mid = s.vision.detector
            r = check_model(s, mid)
            rd = release_decision(s.release_profile, mid)
            if r.ready and rd.allowed:
                fname = next(iter(r.files.values()))
                self.detector = load_detector(Path(fname), s.runtime.num_threads, decode=resource(s.vision.detector).get("decode"))
                self.status["visual_detector"] = S.ComponentStatus(name="visual_detector", kind="visual_detector", state="ok", model_id=mid, version=mid)
            else:
                self.status["visual_detector"] = S.ComponentStatus(name="visual_detector", kind="visual_detector", state="unavailable",
                                                                   model_id=None if r.ready else mid, reason=r.reason if not r.ready else rd.reason)
        self._load_verifier()
        tm = s.vision.tracker.method
        self.status["tracker"] = S.ComponentStatus(name=f"{tm}_tracker", kind="tracker", state="ok" if self.detector else "unavailable",
                                                   version="iou-hungarian-v1" if tm == "iou" else "kalman-cv-sort-v1",
                                                   reason=None if self.detector else "requires the visual detector")
        self._load_pose()
        self._load_posture()
        self._load_interaction()
        self._load_object_relation()
        self._load_intent()
        self.status["behavior"] = S.ComponentStatus(name="behavior_segments", kind="behavior", state="ok" if (self.detector and s.behavior.enabled) else
                                                    ("disabled" if not s.behavior.enabled else "unavailable"), version="behavior-rules-v1",
                                                    reason="fixed thresholds; validated on synthetic/semi-synthetic sequences only (E8)")
        self.status["fusion"] = S.ComponentStatus(name="fusion", kind="fusion", state="ok", version=self.fusion.scorer.revision,
                                                  reason="experimental; not validated on paired data")
        self.status["vlm"] = S.ComponentStatus(name="vlm", kind="vlm", state="disabled", reason="optional local VLM adapter is disabled by default (Phase 1)")
        self.model_load_ms = (time.perf_counter() - t0) * 1000.0
        return self

    def _load_verifier(self) -> None:
        s, v = self.settings, self.settings.vision.verifier
        name = "detection_verifier"
        if not v.enabled:
            self.status["visual_verifier"] = S.ComponentStatus(name=name, kind="visual_verifier", state="disabled", reason="disabled by configuration")
            return
        if self.detector is None:
            self.status["visual_verifier"] = S.ComponentStatus(name=name, kind="visual_verifier", state="unavailable", reason="requires the visual detector")
            return
        params_path = REPO_ROOT / v.params_file(s.vision.detector)
        rc = check_model(s, v.classifier)
        blocked = [d for d in (release_decision(s.release_profile, v.classifier), release_decision(s.release_profile, "detection_verifier_v1")) if not d.allowed]
        reason = None
        if not params_path.exists():
            reason = f"no verifier parameters for detector '{s.vision.detector}' ({params_path.name}); run: petperception evaluate --experiment e7"
        elif not rc.ready:
            reason = rc.reason
        elif blocked:
            reason = blocked[0].reason
        if reason:
            self.status["visual_verifier"] = S.ComponentStatus(name=name, kind="visual_verifier", state="unavailable", reason=reason)
            return
        import json as _json

        params = _json.loads(params_path.read_text())
        clf = CropClassifier(Path(next(iter(rc.files.values()))), s.runtime.num_threads)
        self.verifier = DetectionVerifier(self.detector, clf, params, s.vision.target_classes)
        self.status["visual_verifier"] = S.ComponentStatus(name=name, kind="visual_verifier", state="ok", model_id=v.classifier,
                                                           version=f"verifier-v1:{s.vision.detector}",
                                                           reason="track-level median of per-detection rejection scores; thresholds from E7 tuning")

    def _load_pose(self) -> None:
        s = self.settings
        if s.pose.provider == "none":
            self.status["pose"] = S.ComponentStatus(name="pose", kind="pose", state="unavailable", reason=self.pose.unavailable_reason)
            return
        from .vision.keypoints import DOG_KEYPOINTS, DogKeypointModel, DogPoseProvider

        head = REPO_ROOT / s.pose.head_file
        head_id = {"dog_keypoints_v1": "dog_keypoints_head_v1", "catdog_keypoints_awa_v1": "catdog_keypoints_head_awa_v1"}[s.pose.provider]
        rb = check_model(s, s.pose.backbone)
        blocked = [d for d in (release_decision(s.release_profile, s.pose.backbone), release_decision(s.release_profile, head_id)) if not d.allowed]
        reason = None if head.exists() else f"{head.name} not found; run: petperception evaluate --experiment e9"
        reason = reason or (rb.reason if not rb.ready else None) or (blocked[0].reason if blocked else None)
        if reason:
            self.pose = UnavailablePoseProvider(unavailable_reason=reason)
            self.status["pose"] = S.ComponentStatus(name="pose", kind="pose", state="unavailable", reason=reason)
            return
        import json as _json

        meta_path = head.with_suffix(".json")
        gates = _json.loads(meta_path.read_text()).get("gates", {}) if meta_path.exists() else {}
        model = DogKeypointModel.load(Path(next(iter(rb.files.values()))), head, s.runtime.num_threads)
        self._backbone = model.backbone
        if s.pose.provider == "catdog_keypoints_awa_v1":
            self.pose = DogPoseProvider(model, name="catdog_keypoints_awa_effnetlite0_head", species=("cat", "dog"), keypoint_vocabulary=tuple(DOG_KEYPOINTS),
                                        min_visible=float(gates.get("min_visible", 0.5)), min_peak=float(gates.get("min_peak", 0.2)), suppressed=())
            why = "cats and dogs (research profile; AwA-Pose, E24: cat ears PCK@0.1 0.91, tails 0.51)"
        else:
            self.pose = DogPoseProvider(model, min_visible=float(gates.get("min_visible", 0.5)), min_peak=float(gates.get("min_peak", 0.2)))
            why = "dogs only (research profile); cats need pose.provider catdog_keypoints_awa_v1"
        self.status["pose"] = S.ComponentStatus(name="pose", kind="pose", state="ok", model_id=s.pose.backbone, version=model.version, reason=why)

    def _load_posture(self) -> None:
        s = self.settings
        name = "posture"
        if not s.posture.enabled:
            self.status["posture"] = S.ComponentStatus(name=name, kind="pose", state="disabled", reason="disabled by configuration")
            return
        if self.detector is None:
            self.status["posture"] = S.ComponentStatus(name=name, kind="pose", state="unavailable", reason="requires the visual detector")
            return
        from .vision.posture import PostureModel

        path = REPO_ROOT / s.posture.model_file
        rb = check_model(s, s.posture.backbone)
        blocked = [d for d in (release_decision(s.release_profile, s.posture.backbone), release_decision(s.release_profile, "posture_v1")) if not d.allowed]
        reason = None if path.exists() else f"{path.name} not found; run: petperception evaluate --experiment e10"
        reason = reason or (rb.reason if not rb.ready else None) or (blocked[0].reason if blocked else None)
        if reason:
            self.status["posture"] = S.ComponentStatus(name=name, kind="pose", state="unavailable", reason=reason)
            return
        self.posture = PostureModel.load(path)
        if self._backbone is None:
            from .vision.keypoints import Backbone

            self._backbone = Backbone(Path(next(iter(rb.files.values()))), s.runtime.num_threads)
        self.status["posture"] = S.ComponentStatus(name=name, kind="pose", state="ok", model_id=s.posture.backbone, version=self.posture.version,
                                                   reason="coarse lying/sitting/standing from weak caption labels (E10); uncalibrated")

    def _load_interaction(self) -> None:
        s = self.settings
        name = "person_interaction"
        if not s.interaction.enabled:
            self.status["interaction"] = S.ComponentStatus(name=name, kind="behavior", state="disabled", reason="disabled by configuration")
            return
        path = REPO_ROOT / s.interaction.model_file
        decision = release_decision(s.release_profile, "person_pet_interaction_v1")
        reason = ("requires the visual detector" if self.detector is None else
                  f"{path.name} not found; run: petperception evaluate --experiment e13" if not path.exists() else
                  None if decision.allowed else decision.reason)
        if reason:
            self.status["interaction"] = S.ComponentStatus(name=name, kind="behavior", state="unavailable", reason=reason)
            return
        from .vision.interaction import InteractionModel

        self.interaction = InteractionModel.load(path)
        self.status["interaction"] = S.ComponentStatus(
            name=name, kind="behavior", state="ok", model_id=s.vision.detector, version=self.interaction.version,
            reason="person-pet interaction from box geometry (E13, still images, dogs evaluated); people are detected but not stored")

    def _load_object_relation(self) -> None:
        s = self.settings
        name = "object_relation"
        if not s.object_relation.enabled:
            self.status["object_relation"] = S.ComponentStatus(name=name, kind="behavior", state="disabled", reason="disabled by configuration")
            return
        path = REPO_ROOT / s.object_relation.model_file
        decision = release_decision(s.release_profile, s.object_relation.resource_id)
        reason = ("requires the visual detector" if self.detector is None else
                  f"{path.name} not found; run: petperception evaluate --experiment e15" if not path.exists() else
                  None if decision.allowed else decision.reason)
        if reason:
            self.status["object_relation"] = S.ComponentStatus(name=name, kind="behavior", state="unavailable", reason=reason)
            return
        from .vision.object_relation import ObjectRelationModel

        self.object_relation = ObjectRelationModel.load(path)
        self.status["object_relation"] = S.ComponentStatus(
            name=name, kind="behavior", state="ok", model_id=s.vision.detector, version=self.object_relation.version,
            reason=OBJECT_RELATION_REASON)

    def _load_intent(self) -> None:
        s = self.settings
        name = "intent"
        if not s.intent.enabled:
            return  # no component entry: results stay identical to engines without the intent stage
        from .intent import inference as IN

        need = ["intent_situation_kinetics_v1", "intent_dog_audio_activity_v1", "intent_cat_audio_context_v1"]
        if s.intent.variant == "full":
            need.append("intent_dog_bark_context_v1")
        blocked = [d for d in (release_decision(s.release_profile, r) for r in need) if not d.allowed]
        reason = ("requires the visual detector and the EfficientNet-Lite0 backbone (enable posture or pose)" if self.detector is None or self._backbone is None
                  else "requires the YAMNet audio backend" if self.audio_backend is None
                  else blocked[0].reason if blocked
                  else "the full variant needs pose.provider catdog_keypoints_awa_v1" if s.intent.variant == "full" and not self.pose.available
                  else None)
        art = IN.ART / "kinetics_situation_v1" / s.intent.variant / "late_fusion.json"
        if reason is None and not art.exists():
            reason = f"{art.relative_to(REPO_ROOT)} missing; run: petperception evaluate --experiment e28"
        if reason:
            self.status["intent"] = S.ComponentStatus(name=name, kind="fusion", state="unavailable", reason=reason)
            return
        from .intent.av_features import Components

        objects = [lab for lab in self.detector.labels if lab not in ("???", "cat", "dog", "person")]
        yam = self.audio_backend
        if s.intent.variant == "full":
            yam = YamnetBackend(model_dir(s, s.audio.yamnet.model_id), s.runtime.num_threads, with_embeddings=True)
        comp = Components(s, self.detector, self.verifier, self._backbone, self.posture, self.pose if s.intent.variant == "full" else None,
                          self.interaction, yam, objects, {})
        self.intent = (IN.IntentModels(variant=s.intent.variant), comp)
        self.status["intent"] = S.ComponentStatus(name=name, kind="fusion", state="ok", version=f"{IN.INFERENCE_VERSION}:{s.intent.variant}",
                                                  reason="trained channels (E26-E28); the cross-channel intent combination is not validated")

    def _intent_output(self, audio: dict | None, audio_state: str, rows: list[dict], video_state: str) -> S.IntentOutput:
        from .intent import kinetics as K
        from .intent.inference import infer as IN_infer
        from .intent.run import _NpzLike

        models, comp = self.intent
        z: dict = {"audio_state": np.array(audio["audio_state"] if audio else ("absent" if audio_state in ("absent", "disabled") else audio_state))}
        if audio:
            z.update({k: v for k, v in audio.items() if k != "audio_state"})
        if rows:
            z.update({k: np.concatenate([r[k] for r in rows]) for k in rows[0]})
        z["video_state"] = np.array("present" if rows else video_state)
        br, cues = K.pool(_NpzLike(z))
        labels = comp.yamnet.labels if comp.yamnet is not None else []
        res = IN_infer(models, {b: (None if v is None else v[None, :].astype(np.float64)) for b, v in br.items()}, cues, labels)
        return S.IntentOutput(**res.to_dict())

    def capabilities(self) -> dict:
        s = self.settings
        return {
            "engine_version": __version__,
            "schema_version": S.SCHEMA_VERSION,
            "config_profile": s.profile,
            "components": [c.model_dump() for c in self.status.values()],
            "modalities": {"audio": self.audio_backend is not None, "video": self.detector is not None},
            "detector_classes": s.vision.target_classes,
            "context_object_classes": s.vision.context_object_classes,
            "fusion": {"mode": "experimental", "method": s.fusion.method, "validated": False, "multimodal_accuracy": "not_evaluated"},
            "limits": s.limits.model_dump(),
            "release_profile": s.release_profile,
            "research_only": s.release_profile == "research",
        }

    # -- analysis ------------------------------------------------------------
    def analyze(self, path: str | Path, zones: ZoneConfig | None = None, modalities: tuple[str, ...] = ("audio", "video"),
                analysis_id: str | None = None, display_name: str | None = None,
                progress: Callable[[str, float], None] | None = None,
                should_cancel: Callable[[], bool] | None = None) -> S.AnalysisResult:
        """Analyze one media file.

        ``progress(stage, fraction)`` is called with a monotonically increasing
        fraction in [0, 1]; ``should_cancel()`` is polled between stages and per
        sampled frame, and a True answer raises :class:`AnalysisCancelled`.
        """
        s = self.settings
        timer = StageTimer()
        aid = analysis_id or str(uuid.uuid4())

        def report(stage: str, frac: float) -> None:
            if should_cancel is not None and should_cancel():
                raise AnalysisCancelled(stage)
            if progress is not None:
                progress(stage, max(0.0, min(1.0, frac)))

        report("probe", 0.0)
        with timer.stage("probe"):
            mp = M.probe(path, s.limits)
        with timer.stage("hash"):
            digest = sha256_file(path)
        qflags: list[S.QualityFlag] = []
        components = dict(self.status)

        # ---------------- audio ----------------
        audio_state: str
        intent_audio: dict | None = None
        track_audio: M.AudioTrack | None = None
        raw_events: list[SoundEventRaw] = []
        audio_summary = S.AudioSummary(state="disabled")
        bandwidth_limited = False
        if "audio" not in modalities or not s.audio.enabled:
            audio_state = "disabled"
        elif mp.audio is None:
            audio_state = "absent"
        else:
            with timer.stage("decode_audio"):
                track_audio = M.decode_audio(mp, s.limits, s.audio.silence_dbfs)
            audio_state = track_audio.state
            if audio_state == "present" and track_audio.duration_s < 0.1:
                audio_state = "too_short"
            if track_audio.decode_warnings:
                qflags.append(S.QualityFlag(code="audio_decode_warnings", severity="warning", detail="; ".join(track_audio.decode_warnings[:3])))
            if audio_state == "corrupt":
                qflags.append(S.QualityFlag(code="audio_corrupt", severity="limiting", detail="audio stream could not be decoded"))
            if audio_state == "silent":
                qflags.append(S.QualityFlag(code="audio_silent", severity="info", detail="audio is silent; silence is not evidence of relaxation or of absence of need"))
            if track_audio.clipping_fraction and track_audio.clipping_fraction > s.audio.clipping_fraction:
                qflags.append(S.QualityFlag(code="audio_clipping", severity="warning", detail=f"{track_audio.clipping_fraction:.1%} of samples at full scale"))
            if track_audio.sample_rate_hz and track_audio.sample_rate_hz < s.audio.analysis_sample_rate:
                bandwidth_limited = True
                qflags.append(S.QualityFlag(code="bandwidth_limited_audio", severity="warning",
                                            detail=f"{track_audio.sample_rate_hz} Hz input: no content above {track_audio.sample_rate_hz / 2:.0f} Hz; resampling does not restore it"))
            audio_summary = S.AudioSummary(state=audio_state, rms_dbfs=track_audio.rms_dbfs, peak_dbfs=track_audio.peak_dbfs,
                                           silent_fraction=track_audio.silent_fraction, clipping_fraction=track_audio.clipping_fraction,
                                           effective_bandwidth_hz=(min(track_audio.sample_rate_hz, s.audio.analysis_sample_rate) / 2) if track_audio.sample_rate_hz else None)
            report("audio", 0.10)
            if audio_state == "present" and self.audio_backend is not None:
                with timer.stage("audio_inference"):
                    ws = self.audio_backend.score_track(track_audio, s.audio.yamnet.hop_s)
                    raw_events = extract_events(ws, self.audio_backend, self._groups, self.thresholds,
                                                s.audio.yamnet.context_tag_threshold, s.audio.yamnet.min_event_gap_s, self.context_thresholds)
                if self.intent is not None:
                    with timer.stage("intent_audio"):
                        intent_audio = AF.audio_features(self.intent[1], M.resample(np.clip(track_audio.samples, -1.0, 1.0), track_audio.sample_rate_hz,
                                                                                     AF.YAMNET_SR), s.audio.yamnet.hop_s)
                audio_summary.windows_scored = int(ws.scores.shape[0])
                if ws.padded_windows:
                    qflags.append(S.QualityFlag(code="audio_end_padded", severity="info", detail="final analysis window zero-padded"))

        # ---------------- video ----------------
        video_state: str
        tracks: list[Track] = []
        persons: list[tuple[float, np.ndarray, float]] = []
        camera_shift: dict[float, tuple[float, float]] = {}
        camera_moving_times: list[float] = []
        camera_unknown = 0
        frame_size = (0, 0)
        frame_report = M.FrameSamplingReport(s.vision.sample_fps)
        intent_rows: list[dict] = []
        intent_s = 0.0
        low_light_times: list[float] = []
        blurry_times: list[float] = []
        n_frames = 0
        if "video" not in modalities or not s.vision.enabled:
            video_state = "disabled"
        elif mp.video is None:
            video_state = "absent"
        elif self.detector is None:
            video_state = "present"
            qflags.append(S.QualityFlag(code="visual_model_unavailable", severity="limiting", detail=self.status["visual_detector"].reason))
        else:
            tracker = make_tracker(s.vision.tracker.method, s.vision.tracker.iou_threshold, s.vision.tracker.max_gap_s)
            verify_s = pose_s = 0.0
            object_classes = self.object_relation.detector_classes if self.object_relation is not None else []
            classes = list(dict.fromkeys(s.vision.target_classes + s.vision.context_object_classes + (["person"] if self.interaction else [])
                                         + object_classes))
            prev_gray, prev_t, prev_boxes = None, None, []
            detect_s = quality_s = 0.0
            intent_gray = None
            report("video", 0.30)
            expected = max(1, min(s.limits.max_video_frames, int((mp.duration_s or 0.0) * s.vision.sample_fps) + 1))
            with timer.stage("video_decode_detect_track"):
                for fr in M.iter_frames(mp, s.vision.sample_fps, s.vision.max_side, s.limits, frame_report):
                    n_frames += 1
                    report("video", 0.30 + 0.60 * min(1.0, n_frames / expected))
                    tq = time.perf_counter()
                    fq, gray = frame_quality(fr.t_s, fr.rgb, s.vision.quality)
                    quality_s += time.perf_counter() - tq
                    if fq.low_light:
                        low_light_times.append(fr.t_s)
                    if fq.blurry:
                        blurry_times.append(fr.t_s)
                    td = time.perf_counter()
                    dets = detect_at_operating_point(self.detector, fr.rgb, classes, s.vision)
                    detect_s += time.perf_counter() - td
                    animal = [d for d in dets if d.label in s.vision.target_classes]
                    for d in dets:
                        if d.label == "person" and "person" in s.vision.context_object_classes:
                            persons.append((fr.t_s, d.box, d.score))
                    adets = merge_species(animal, fr.t_s, fr.index)
                    detection_flags(adets, s.vision.quality)
                    if self.interaction is not None:
                        people = [d.box for d in dets if d.label == "person" and d.score >= s.interaction.min_person_score]
                        for a in adets:
                            if people:
                                a.interaction = max(self.interaction.score(pb, a.box, fr.rgb.shape[:2], a.species) for pb in people)
                    if self.object_relation is not None:
                        objects = [(d.label, d.box) for d in dets if d.label in object_classes and d.score >= s.object_relation.min_object_score]
                        for a in adets:
                            if objects:
                                a.on_furniture = self.object_relation.best(objects, a.box, fr.rgb.shape[:2], a.species).get("on")
                    if self.verifier is not None:
                        tv = time.perf_counter()
                        for a in sorted(adets, key=lambda x: -x.score)[: s.vision.verifier.max_checks_per_frame]:
                            cs = a.source.class_scores if a.source is not None else None
                            vr = self.verifier.verify(fr.rgb, Detection(a.species, a.score, a.box, cs))
                            a.verify_score = vr[0] if vr is not None else None
                        verify_s += time.perf_counter() - tv
                    if self._backbone is not None and (self.pose.available or self.posture is not None):
                        from .vision.keypoints import square_crop

                        tp = time.perf_counter()
                        lo = min(s.pose.min_detection_score, s.posture.min_detection_score)
                        for a in sorted((x for x in adets if x.score >= lo), key=lambda x: -x.score)[: max(s.pose.max_per_frame, s.posture.max_per_frame)]:
                            need_kp = self.pose.available and a.species in self.pose.species and a.score >= s.pose.min_detection_score
                            need_po = self.posture is not None and a.score >= s.posture.min_detection_score
                            if not (need_kp or need_po):
                                continue
                            feats = self._backbone.features(square_crop(fr.rgb, a.box)[0])
                            if need_kp:
                                a.keypoints = self.pose.estimate(fr.rgb, a.box, fr.t_s, feats)
                            if need_po:
                                a.posture = self.posture.predict(feats["pooled"], a.box, fr.rgb.shape[:2], a.species)
                        pose_s += time.perf_counter() - tp
                    shift = (0.0, 0.0)
                    if prev_gray is not None and not mp.video.is_still_image:
                        tq = time.perf_counter()
                        cm = estimate_camera_motion(prev_gray, gray, prev_boxes + [a.box for a in adets], prev_t, fr.t_s)
                        if cm is None or not cm.reliable:
                            camera_unknown += 1
                        else:
                            moving = np.hypot(cm.dx, cm.dy) > s.vision.quality.camera_motion_threshold or abs(cm.scale - 1) > 0.02
                            if moving:
                                camera_moving_times.append(fr.t_s)
                                if s.vision.camera_motion_compensation:
                                    shift = (cm.dx, cm.dy)
                                    camera_shift[fr.t_s] = shift
                        quality_s += time.perf_counter() - tq
                    tracker.update(fr.t_s, adets, shift)
                    prev_gray, prev_t, prev_boxes = gray, fr.t_s, [a.box for a in adets]
                    if self.intent is not None:
                        ti = time.perf_counter()
                        row = AF.empty_video(1, len(self.intent[1].object_classes), s.vision.sample_fps)
                        row["v_t"][0] = fr.t_s
                        intent_gray = AF.frame_features(self.intent[1], fr.rgb, fr.t_s, intent_gray, row, 0)
                        intent_rows.append(row)
                        intent_s += time.perf_counter() - ti
                    frame_size = (fr.rgb.shape[1], fr.rgb.shape[0])
            timer.timings_ms["video_detector_inference"] = detect_s * 1000.0
            timer.timings_ms["video_verifier"] = verify_s * 1000.0
            timer.timings_ms["video_pose"] = pose_s * 1000.0
            timer.timings_ms["video_quality_and_camera_motion"] = quality_s * 1000.0
            min_hits = s.vision.tracker.min_hits if n_frames >= 2 * s.vision.tracker.min_hits else 1
            tracks = tracker.tracks(min_hits)
            if self.verifier is not None:
                kept, rejected = [], []
                for tr in tracks:
                    vs = [d.verify_score for d in tr.detections if d.verify_score is not None]
                    cv = self.verifier.per_class.get(tr.species)
                    if cv is None or not vs:
                        kept.append(tr)
                        continue
                    med = float(np.median(vs))
                    tr.verification = (len(vs), med, cv.tau, med >= cv.tau)
                    (kept if med >= cv.tau else rejected).append(tr)
                tracks = kept
                if rejected:
                    qflags.append(S.QualityFlag(code="detections_rejected_by_verifier", severity="info",
                                                t_start_s=min(t.first_t for t in rejected), t_end_s=max(t.last_t for t in rejected),
                                                detail=f"{len(rejected)} animal-like track(s) did not pass the second-stage check and are not reported as animals"))
            video_state = "present" if n_frames > 0 else ("corrupt" if frame_report.decode_warnings else "no_decodable_frames")
            if frame_report.decode_warnings:
                qflags.append(S.QualityFlag(code="video_decode_warnings", severity="warning", detail="; ".join(frame_report.decode_warnings[:3])))
            if frame_report.truncated_at_limit:
                qflags.append(S.QualityFlag(code="frame_limit_reached", severity="limiting", detail=f"stopped after {s.limits.max_video_frames} sampled frames"))
            with timer.stage("frame_gap_check"):
                for a, b in M.frame_gaps(M.packet_timestamps(mp, s.limits), s.vision.quality.frame_gap_factor):
                    qflags.append(S.QualityFlag(code="dropped_or_missing_frames", severity="warning", t_start_s=a, t_end_s=b))
            for code, times, det in (("low_light", low_light_times, "mean luma below threshold"), ("blurry_frames", blurry_times, "low Laplacian variance")):
                if times:
                    qflags.append(S.QualityFlag(code=code, severity="warning", t_start_s=min(times), t_end_s=max(times), detail=f"{det} in {len(times)}/{n_frames} sampled frames"))
            if camera_moving_times:
                qflags.append(S.QualityFlag(code="camera_motion", severity="warning", t_start_s=min(camera_moving_times), t_end_s=max(camera_moving_times),
                                            detail="global background motion detected; " + ("translation-compensated (approximate)" if s.vision.camera_motion_compensation else "not compensated")))
            if camera_unknown and n_frames > 1:
                qflags.append(S.QualityFlag(code="camera_motion_unknown", severity="info", detail=f"background motion could not be estimated for {camera_unknown} frame pairs"))
            if mp.video.is_still_image:
                qflags.append(S.QualityFlag(code="still_image", severity="info", detail="single frame: movement and dwell time cannot be measured"))

        # ---------------- observations & evidence ----------------
        evidence: list[Ev] = []
        track_models: list[S.AnimalTrack] = []
        is_still = bool(mp.video and mp.video.is_still_image)
        bparams = BH.BehaviorParams(**s.behavior.model_dump(exclude={"enabled"}))
        step_series: dict[str, BH.StepSeries] = {}
        tb = time.perf_counter()
        media_start = mp.video.start_time_s if mp.video else 0.0
        media_end = (mp.video.start_time_s + mp.video.duration_s) if (mp.video and mp.video.duration_s) else mp.duration_s
        for tr in tracks:
            tid = tr.track_id
            flags = sorted({f for d in tr.detections for f in d.flags})
            mv = movement_summary(tr, frame_size[0], frame_size[1], camera_shift if s.vision.camera_motion_compensation else None)
            span_camera = any(tr.first_t <= t <= tr.last_t for t in camera_moving_times)
            span_dark = any(tr.first_t <= t <= tr.last_t for t in low_light_times)
            mv_rel = "low" if (not mv.get("available") or len(tr.detections) < 3 or span_dark) else ("medium" if span_camera or camera_unknown else "high")
            zone_occ = occupancy([d.t_s for d in tr.detections], [d.box for d in tr.detections], zones, s.vision.tracker.max_gap_s) if zones else []
            kp_series = [d.keypoints for d in tr.detections if d.keypoints is not None]
            if tr.species in self.pose.species:
                pose_feats = compute_features(self.pose, kp_series, s.pose.min_keypoint_visibility)
            else:
                why = (f"no {tr.species} keypoint model is available" if self.pose.available else self.pose.unavailable_reason)
                pose_feats = compute_features(UnavailablePoseProvider(unavailable_reason=why), [], s.pose.min_keypoint_visibility)
            segs: list[BH.Segment] = []
            if s.behavior.enabled and not is_still and len(tr.detections) >= 2 and frame_size[0]:
                ts_, bx_ = [d.t_s for d in tr.detections], [d.box for d in tr.detections]
                ss = BH.step_series(ts_, bx_, frame_size[0], frame_size[1], camera_shift if s.vision.camera_motion_compensation else None,
                                    s.vision.tracker.max_gap_s)
                step_series[tid] = ss
                brel = "low" if (span_dark or len(tr.detections) < 3) else "medium"
                segs = (BH.motion_segments(ss, bparams, tid, brel) + BH.pacing_segments(ss, bparams, tid)
                        + BH.zone_segments(ts_, bx_, zones, frame_size[0], frame_size[1], bparams, s.vision.tracker.max_gap_s, tid, brel)
                        + BH.view_segments(ts_, bx_, media_start, media_end, bparams, tid, brel))
                if self.interaction is not None:
                    segs += BH.interaction_segments(ts_, [d.interaction for d in tr.detections], self.interaction.threshold,
                                                    s.interaction.min_frames, s.vision.tracker.max_gap_s, tid)
                if self.object_relation is not None and "on" in self.object_relation.heads:
                    segs += BH.on_furniture_segments(ts_, [d.on_furniture for d in tr.detections], self.object_relation.heads["on"].threshold,
                                                     s.object_relation.min_frames, s.vision.tracker.max_gap_s, tid)
                segs.sort(key=lambda g: (g.t_start_s, g.kind))
            mean_score = float(np.mean([d.score for d in tr.detections]))
            track_rel = "high" if mean_score >= 0.6 and not ({"tiny_subject", "species_ambiguous"} & set(flags)) else ("medium" if mean_score >= 0.45 else "low")
            if span_dark:
                track_rel = "low"  # out-of-domain lighting lowers every vision-derived item
            ev_ids = []
            e_track = S.EvidenceItem(
                evidence_id=f"ev-{tid}", kind="animal_track", t_start_s=tr.first_t, t_end_s=tr.last_t, track_id=tid,
                summary=f"{tr.species} detected ({tid}) from {tr.first_t:.2f}s to {tr.last_t:.2f}s in {len(tr.detections)} sampled frames",
                value=mean_score, value_kind="mean_detector_score_raw", reliability=track_rel,
                provenance=S.Provenance(component="visual_detector", model_id=s.vision.detector, root="vision:detector", method="efficientdet+iou_tracker"))
            evidence.append(Ev(e_track, "vision:detector", frozenset({"animal_visible", f"species:{tr.species}"}), 1.0))
            ev_ids.append(e_track.evidence_id)
            if mv.get("available"):
                tags = set()
                if mv["moving_fraction"] >= 0.5 and mv["p90_speed"] >= 0.15:
                    tags.add("movement:high")
                elif mv["moving_fraction"] <= 0.1 and (tr.last_t - tr.first_t) >= 3.0:
                    tags.add("movement:low")
                desc = "high" if "movement:high" in tags else ("low" if "movement:low" in tags else "moderate")
                e_mv = S.EvidenceItem(
                    evidence_id=f"ev-move-{tid}", kind="movement", t_start_s=tr.first_t, t_end_s=tr.last_t, track_id=tid,
                    summary=(f"{desc} apparent movement of {tid} (median {mv['median_speed']:.3f} image diagonals/s; "
                             f"moving in {round(mv['moving_fraction'] * mv['steps'])} of {mv['steps']} sampled steps)"),
                    value=mv["median_speed"], value_kind="image_diagonals_per_second", reliability=mv_rel,
                    provenance=S.Provenance(component="tracker", model_id=s.vision.detector, root="vision:detector", method="centroid displacement, camera-translation compensated when estimated"))
                evidence.append(Ev(e_mv, "vision:detector", frozenset(tags), 1.0))
                ev_ids.append(e_mv.evidence_id)
            for z in zone_occ:
                ok = z["dwell_s"] >= s.fusion.min_zone_dwell_s or is_still
                e_z = S.EvidenceItem(
                    evidence_id=f"ev-zone-{tid}-{z['zone_id']}", kind="zone_occupancy",
                    t_start_s=z["intervals_s"][0][0], t_end_s=z["intervals_s"][-1][1], track_id=tid,
                    summary=f"{tid} in owner-marked zone '{z['label']}' ({z['semantic_tag']}) for {z['dwell_s']:.1f}s",
                    value=z["dwell_s"], value_kind="dwell_seconds_sampled", reliability="low" if (is_still or span_dark) else "medium",
                    provenance=S.Provenance(component="context", root="context:zones", method=f"box {z['anchor']} point in owner polygon"))
                evidence.append(Ev(e_z, "context:zones", frozenset({f"zone:{z['semantic_tag']}"}) if ok else frozenset(), 1.0))
                ev_ids.append(e_z.evidence_id)
            track_models.append(S.AnimalTrack(
                track_id=tid, species=tr.species, species_scores=tr.species_summary(), first_seen_s=tr.first_t, last_seen_s=tr.last_t,
                duration_s=tr.last_t - tr.first_t, n_detections=len(tr.detections), missed_sampled_frames=tr.missed_frames,
                detections=[S.Detection(t_s=d.t_s, box_xyxy_norm=tuple(float(v) for v in d.box), species_scores=d.species_scores, flags=d.flags) for d in tr.detections],
                movement=S.MovementSummary(**{k: v for k, v in mv.items()}, reliability=mv_rel),
                zones=[S.ZoneOccupancy(**z) for z in zone_occ],
                pose_features=[S.PoseFeature(name=f.name, available=f.available, value=f.value, units=f.units, reason=f.reason) for f in pose_feats],
                quality_flags=flags, evidence_ids=ev_ids,
                behavior_segments=[_segment(g, f"beh-{tid}-{k}") for k, g in enumerate(segs, start=1)],
                keypoints=[S.KeypointFrame(t_s=ks.t_s, provider=self.pose.name,
                                           points={n: (round(float(ks.xy[i][0]), 4), round(float(ks.xy[i][1]), 4), round(float(ks.visibility[i]), 3))
                                                   for i, n in enumerate(ks.names) if n in self.pose.keypoint_vocabulary})
                           for ks in kp_series[:120]],
                posture=[S.PostureFrame(t_s=d.t_s, label=d.posture[0], score=round(d.posture[1], 3)) for d in tr.detections if d.posture is not None],
                verification=(S.TrackVerification(verifier_id=self.status["visual_verifier"].version or "verifier", checked_detections=tr.verification[0],
                                                  median_score=tr.verification[1], threshold=tr.verification[2], accepted=tr.verification[3])
                              if tr.verification else None)))
        if persons:
            ts = sorted(t for t, _, _ in persons)
            e_p = S.EvidenceItem(evidence_id="ev-person-1", kind="context_object", t_start_s=ts[0], t_end_s=ts[-1],
                                 summary=f"person visible in {len(ts)} sampled frames", value=float(np.mean([p[2] for p in persons])),
                                 value_kind="mean_detector_score_raw", reliability="medium",
                                 provenance=S.Provenance(component="visual_detector", model_id=s.vision.detector, root="vision:detector", method="COCO person class"))
            evidence.append(Ev(e_p, "vision:detector", frozenset({"person_visible"}), 1.0))

        sound_models: list[S.SoundEvent] = []
        context_models: list[S.SoundEvent] = []
        audio_rel = "medium" if bandwidth_limited else "high"
        video_ok = video_state == "present" and self.detector is not None
        for k, ev in enumerate(raw_events, start=1):
            eid = f"ev-snd-{k}"
            is_context = ev.label.startswith("context:")
            if is_context:
                att = S.Attribution(status="unknown", note="environmental sound; not attributed to an animal")
            elif not video_ok:
                att = S.Attribution(status="video_unavailable", note="no usable video to relate the sound to a visible animal")
            else:
                w = s.fusion.co_occurrence_window_s
                cands = [t for t in tracks if t.first_t - w <= ev.end_s and t.last_t + w >= ev.start_s]
                species_needed = {"cat_vocalization": "cat", "dog_vocalization": "dog"}.get(ev.label)
                if not cands:
                    att = S.Attribution(status="no_visible_animal", note="no animal visible near this time; the source may be off-screen or a TV/speaker")
                elif len(cands) == 1 and (species_needed is None or cands[0].species == species_needed):
                    att = S.Attribution(status="single_visible_candidate", candidate_track_ids=[cands[0].track_id],
                                        note="exactly one species-consistent animal visible; attribution is unverified")
                else:
                    note = ("several animals visible; no evaluated audio-visual association method, so attribution is unknown"
                            if len(cands) > 1 else "the only visible animal's species does not match the sound; source may be off-screen")
                    att = S.Attribution(status="unknown", candidate_track_ids=[c.track_id for c in cands], note=note)
            label_txt = ev.label.replace("context:", "").replace("_", " ")
            if ev.label == "growl" and "growl" not in shown_subtypes(self.subtype_validation):
                label_txt = "possible growl (unvalidated sound type)"
            sm = S.SoundEvent(event_id=f"snd-{k}", label=ev.label, t_start_s=ev.start_s, t_end_s=ev.end_s, peak_raw_score=ev.peak_score,
                              mean_raw_score=ev.mean_score, threshold=ev.threshold, threshold_source=self.threshold_source.get(ev.label, "config default (not selected on data)") if not is_context else (CONTEXT_SOURCE if ev.label.split(":", 1)[1] in self.context_thresholds else "config context_tag_threshold (not selected on data)"),
                              subtypes=ev.subtypes, subtype_validation={k2: self.subtype_validation.get(k2, "not_evaluated") for k2 in ev.subtypes},
                              top_audioset_labels=ev.top_audioset_labels, attribution=att, evidence_id=eid)
            (context_models if is_context else sound_models).append(sm)
            tags: set[str] = set()
            shown = shown_subtypes(self.subtype_validation)  # unvalidated subtypes stay in the JSON, not in the text
            blocking: set[str] = set()
            if is_context:
                tags.add(ev.label.split(":", 1)[1])
            else:
                tags |= vocal_tags(ev.label, ev.subtypes, shown)
                blocking = vocal_conflict_tags(ev.label, ev.subtypes, shown)
            sub_txt = ", ".join([f"{k2} {v:.2f}" for k2, v in sorted(ev.subtypes.items(), key=lambda t: -t[1]) if k2 in shown][:3])
            item = S.EvidenceItem(
                evidence_id=eid, kind="context_sound" if is_context else "sound_event", t_start_s=ev.start_s, t_end_s=ev.end_s,
                track_id=None, summary=f"{label_txt} sound {ev.start_s:.2f}-{ev.end_s:.2f}s (peak raw score {ev.peak_score:.2f}{'; ' + sub_txt if sub_txt else ''})",
                value=ev.peak_score, value_kind="yamnet_sigmoid_raw", reliability=audio_rel,
                provenance=S.Provenance(component="audio_event", model_id=s.audio.yamnet.model_id, root="audio:yamnet", method="YAMNet tags, ontology group max, thresholded"))
            attributed = att.candidate_track_ids[0] if att.status == "single_visible_candidate" else None
            e = Ev(item, "audio:yamnet", frozenset(tags), 1.0, attributed_track=attributed, conflict_tags=frozenset(blocking))
            evidence.append(e)

        behavior_events: list[S.BehaviorSegment] = []
        if s.behavior.enabled:
            vocal = [(m.label, m.t_start_s, m.t_end_s) for m in sound_models]
            onsets = [(m.label.replace("context:", ""), m.t_start_s) for m in context_models]
            bev = BH.vocal_bouts(vocal, bparams) + BH.startle_candidates(step_series, onsets, bparams)
            bev.sort(key=lambda g: (g.t_start_s, g.kind))
            behavior_events = [_segment(g, f"beh-{k}") for k, g in enumerate(bev, start=1)]
        timer.timings_ms["behavior_segments"] = (time.perf_counter() - tb) * 1000.0

        # Scene-level evidence (person visible, environmental sounds) applies to every subject.
        scene = [e for e in evidence if e.item.kind in ("context_object", "context_sound")]
        expanded: list[Ev] = []
        for e in evidence:
            if e in scene:
                for tm in track_models:
                    expanded.append(Ev(e.item, e.root, e.tags, e.strength, attributed_track=tm.track_id))
                expanded.append(e)
            else:
                expanded.append(e)

        # ---------------- fusion ----------------
        qcodes = {q.code for q in qflags}
        gates = {"audio:yamnet": 0.5 if bandwidth_limited else 1.0,
                 "vision:detector": 0.6 if ({"low_light", "camera_motion"} & qcodes) else 1.0,
                 "context:zones": 1.0}
        fctx = FusionContext(audio_available=audio_state == "present" and self.audio_backend is not None, audio_state=audio_state, video_available=video_ok,
                             zones_configured=bool(zones and zones.zones), person_detection_enabled="person" in s.vision.context_object_classes,
                             n_tracks=len(tracks), gates=gates)
        report("fusion", 0.95)
        with timer.stage("fusion"):
            hyps = self.fusion.evaluate(expanded, fctx)
            ev_index = {e.item.evidence_id: e.item for e in evidence}
            for h in hyps:
                h.explanation = explain_hypothesis(h, ev_index)

        # ---------------- decision ----------------
        reasons: list[str] = []
        pet_sound = bool(sound_models)
        if audio_state in ("absent", "disabled") and "audio" in modalities:
            reasons.append("missing_modality_audio")
        if audio_state == "corrupt":
            reasons.append("audio_corrupt")
        if audio_state == "silent":
            reasons.append("audio_silent")
        if video_state in ("absent", "disabled") and "video" in modalities:
            reasons.append("missing_modality_video")
        if video_state in ("no_decodable_frames", "corrupt"):
            reasons.append("no_decodable_video_frames")
        if (audio_state == "present" and self.audio_backend is None) or (mp.video is not None and "video" in modalities and s.vision.enabled and self.detector is None):
            reasons.append("model_unavailable")
        if video_ok and not tracks:
            reasons.append("no_pet_detected")
        if audio_state == "present" and self.audio_backend is not None and not pet_sound:
            reasons.append("no_pet_sound_detected")
        if len(tracks) > 1:
            reasons.append("multiple_animals_visible")
        experimental = [h for h in hyps if h.status == "experimental_hypothesis"]
        if experimental:
            state, interp = "experimental_hypothesis", "experimental_hypotheses_present"
            summary = f"{len(experimental)} experimental hypothesis(es) with co-occurring independent evidence; scores are uncalibrated."
        elif tracks or pet_sound:
            state = "supported_observation"
            interp = "insufficient_evidence" if hyps else "observations_only"
            summary = (f"Observations only: {len(tracks)} animal track(s), {len(sound_models)} pet sound event(s). "
                       + ("Candidate interpretations lacked sufficient independent evidence." if hyps else "No interpretation was attempted."))
        else:
            state, interp = "abstention", "unknown"
            summary = "No cat or dog was detected in the available modalities; no interpretation is offered."
            if not reasons:
                reasons.append("no_pet_detected")
        decision = S.Decision(state=state, overall_interpretation=interp, reason_codes=list(dict.fromkeys(reasons)), summary=summary)
        suggestions = suggestions_for(hyps, qcodes, len(tracks), pet_sound)

        limitations = list(STATIC_LIMITATIONS)
        if self.pose.available:
            limitations.append("Dog keypoints come from a research-only model evaluated on web photos (E9), not household video; "
                               "no cat keypoint model is available, so cat ear/tail features are unavailable.")
        else:
            limitations.append(f"Ear, tail and keypoint-based features are unavailable for this analysis: {self.pose.unavailable_reason}.")
        limitations += self.release_notes
        if is_still:
            limitations.append("Single still image: movement, dwell time and temporal co-occurrence cannot be measured.")
        if bandwidth_limited:
            limitations.append("Low-sample-rate audio: frequency content above the Nyquist limit is missing.")
        models = [S.ModelInfo(**{k: (v if k != "terms" else {kk: str(vv) for kk, vv in v.items()}) for k, v in model_terms_summary(c.model_id).items()})
                  for c in {c.model_id: c for c in components.values() if c.model_id and c.state == "ok"}.values()]
        av = []
        if audio_state == "present":
            av.append("audio")
        if video_ok and n_frames:
            av.append("video")
        intent_out = None
        if self.intent is not None:
            ti = time.perf_counter()
            intent_out = self._intent_output(intent_audio, audio_state, intent_rows, video_state)
            timer.timings_ms["intent_video_features"] = intent_s * 1000.0
            timer.timings_ms["intent_inference"] = (time.perf_counter() - ti) * 1000.0
        result = S.AnalysisResult(
            analysis_id=aid, created_at=utc_now(), engine_version=__version__, config_profile=s.profile, config_hash=s.config_hash(),
            media=S.MediaInfo(
                analysis_filename=_safe_name(display_name or Path(path).name), size_bytes=mp.size_bytes, sha256=digest, container=mp.format_name,
                duration_s=mp.duration_s, start_time_s=mp.start_time_s,
                audio=S.AudioStream(codec=mp.audio.codec, sample_rate_hz=mp.audio.sample_rate_hz, channels=mp.audio.channels,
                                    first_sample_time_s=track_audio.start_s if track_audio else mp.audio.start_time_s,
                                    decoded_duration_s=track_audio.duration_s if track_audio else 0.0) if mp.audio else None,
                video=S.VideoStream(codec=mp.video.codec, width=mp.video.width, height=mp.video.height, avg_fps=mp.video.avg_fps,
                                    start_time_s=mp.video.start_time_s, duration_s=mp.video.duration_s, rotation_deg=mp.video.rotation_deg,
                                    is_still_image=mp.video.is_still_image) if mp.video else None),
            modalities=S.ModalityStatus(requested=[m for m in ("audio", "video") if m in modalities], available=av,
                                        audio_state=audio_state, video_state=video_state),
            components=list(components.values()),
            observations=S.Observations(animal_tracks=track_models, sound_events=sound_models, context_sounds=context_models,
                                        audio_summary=audio_summary, evidence=[e.item for e in evidence], sampled_frames=n_frames,
                                        sample_fps=s.vision.sample_fps if n_frames else None, behavior_events=behavior_events),
            quality_flags=qflags, hypotheses=hyps, decision=decision, suggestions=suggestions, limitations=limitations,
            fusion=S.FusionInfo(method=s.fusion.method, note="Experimental fusion; weights/temperature are unfitted configuration values."),
            timings_ms={**{k: round(v, 2) for k, v in timer.timings_ms.items()}, "model_load_ms_engine_startup": round(self.model_load_ms, 2)},
            models=models,
            usage_restrictions=S.UsageRestrictions(research_only=s.release_profile == "research", notices=[
                f"Release profile '{s.release_profile}': only assets permitted for this profile were loaded (manifests/resources.json).",
                "Model weights and evaluation data carry third-party terms (see docs/DATA_AND_MODEL_REGISTER.md).",
                *(["Audio-event thresholds derive from ESC-50 (CC BY-NC 3.0): non-commercial use."] if self.thresholds and s.release_profile != "commercial" else []),
                "Do not assume commercial permission for any asset without a recorded legal approval."]),
            intent=intent_out,
        )
        return result


OBJECT_RELATION_REASON = ("pet on bed / couch / chair / table: at least half of the pet's box inside the furniture box (E15 rule, still images; "
                          "the learned head showed no advantage); furniture is detected but not stored; under-furniture and inside-car did not pass the E15 gate")


def _segment(g: BH.Segment, sid: str) -> S.BehaviorSegment:
    return S.BehaviorSegment(segment_id=sid, kind=g.kind, track_id=g.track_id, t_start_s=round(g.t_start_s, 3), t_end_s=round(g.t_end_s, 3),
                             reliability=g.reliability, detail=g.detail, measurements={k: round(float(v), 4) for k, v in g.measurements.items()},
                             method=g.method, validation_status=g.validation, related_object=g.related_object)


def _safe_name(name: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(name).name)
    return keep[:120] or "upload"
