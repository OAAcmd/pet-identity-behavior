"""Offline media analysis engine shared by the CLI and the API."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import numpy as np

from . import __version__
from .audio.events import SoundEventRaw, build_groups, extract_events, load_thresholds
from .audio.yamnet import YamnetBackend
from .config import REPO_ROOT, Settings
from .context.zones import ZoneConfig, occupancy
from .explain.templates import explain_hypothesis, suggestions_for
from .fusion.engine import Ev, FusionContext, FusionEngine
from .ingest import media as M
from .registry import check_model, model_dir, model_terms_summary
from .schemas import analysis as S
from .utils import StageTimer, sha256_file, utc_now
from .vision.detector import EfficientDetDetector
from .vision.pose import UnavailablePoseProvider, compute_features
from .vision.quality import detection_flags, frame_quality
from .vision.tracking import IouTracker, Track, estimate_camera_motion, merge_species, movement_summary

log = logging.getLogger("petperception.pipeline")

REQUEST_LIKE = {"meow", "caterwaul", "bark", "yip", "bow_wow", "howl", "whimper_dog"}
SUBTYPE_MIN = 0.3
STATIC_LIMITATIONS = [
    "Phase 1 research engine: outputs are observations plus unvalidated heuristics. They are not translations of animal "
    "language, readings of thoughts or emotions, or medical assessments.",
    "Track IDs are per-analysis labels for linked detections, not verified pet identities.",
    "Hypothesis scores are uncalibrated; multimodal accuracy has not been evaluated (no paired household validation data).",
    "Sound events use raw YAMNet scores with thresholds selected on ESC-50 clips, which are not household recordings.",
    "Cat/dog detection uses a COCO-trained EfficientDet-Lite model; ear, tail and posture features are unavailable "
    "(no verified animal pose model).",
    "For veterinary or behavioral concerns, consult a qualified professional; this system does not diagnose.",
]


class AnalysisEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.audio_backend: YamnetBackend | None = None
        self.detector: EfficientDetDetector | None = None
        self.pose = UnavailablePoseProvider()
        self.fusion = FusionEngine(settings.fusion)
        self.status: dict[str, S.ComponentStatus] = {}
        self.thresholds: dict[str, float] = {}
        self.threshold_source: dict[str, str] = {}
        self.model_load_ms = 0.0
        self._groups = None

    # -- model loading -----------------------------------------------------
    def load(self) -> "AnalysisEngine":
        t0 = time.perf_counter()
        s = self.settings
        if not s.audio.enabled or s.audio.event_backend == "none":
            self.status["audio_event"] = S.ComponentStatus(name="audio_event", kind="audio_event", state="disabled", reason="disabled by configuration")
        else:
            mid = s.audio.yamnet.model_id
            r = check_model(s, mid)
            if r.ready:
                self.audio_backend = YamnetBackend(model_dir(s, mid), s.runtime.num_threads)
                self._groups = build_groups(self.audio_backend)
                th_path = s.audio.yamnet.thresholds_file
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
            if r.ready:
                fname = next(iter(r.files.values()))
                self.detector = EfficientDetDetector(Path(fname), s.runtime.num_threads)
                self.status["visual_detector"] = S.ComponentStatus(name="visual_detector", kind="visual_detector", state="ok", model_id=mid, version=mid)
            else:
                self.status["visual_detector"] = S.ComponentStatus(name="visual_detector", kind="visual_detector", state="unavailable", model_id=mid, reason=r.reason)
        self.status["tracker"] = S.ComponentStatus(name="iou_tracker", kind="tracker", state="ok" if self.detector else "unavailable",
                                                   version="iou-hungarian-v1", reason=None if self.detector else "requires the visual detector")
        self.status["pose"] = S.ComponentStatus(name="pose", kind="pose", state="unavailable", reason=self.pose.unavailable_reason)
        self.status["fusion"] = S.ComponentStatus(name="fusion", kind="fusion", state="ok", version=self.fusion.scorer.revision,
                                                  reason="experimental; not validated on paired data")
        self.status["vlm"] = S.ComponentStatus(name="vlm", kind="vlm", state="disabled", reason="optional local VLM adapter is disabled by default (Phase 1)")
        self.model_load_ms = (time.perf_counter() - t0) * 1000.0
        return self

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
            "research_only": True,
        }

    # -- analysis ------------------------------------------------------------
    def analyze(self, path: str | Path, zones: ZoneConfig | None = None, modalities: tuple[str, ...] = ("audio", "video"),
                analysis_id: str | None = None, display_name: str | None = None) -> S.AnalysisResult:
        s = self.settings
        timer = StageTimer()
        aid = analysis_id or str(uuid.uuid4())
        with timer.stage("probe"):
            mp = M.probe(path, s.limits)
        with timer.stage("hash"):
            digest = sha256_file(path)
        qflags: list[S.QualityFlag] = []
        components = dict(self.status)

        # ---------------- audio ----------------
        audio_state: str
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
            if audio_state == "present" and self.audio_backend is not None:
                with timer.stage("audio_inference"):
                    ws = self.audio_backend.score_track(track_audio, s.audio.yamnet.hop_s)
                    raw_events = extract_events(ws, self.audio_backend, self._groups, self.thresholds,
                                                s.audio.yamnet.context_tag_threshold, s.audio.yamnet.min_event_gap_s)
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
            tracker = IouTracker(s.vision.tracker.iou_threshold, s.vision.tracker.max_gap_s)
            classes = list(dict.fromkeys(s.vision.target_classes + s.vision.context_object_classes))
            prev_gray, prev_t, prev_boxes = None, None, []
            detect_s = quality_s = 0.0
            with timer.stage("video_decode_detect_track"):
                for fr in M.iter_frames(mp, s.vision.sample_fps, s.vision.max_side, s.limits, frame_report):
                    n_frames += 1
                    tq = time.perf_counter()
                    fq, gray = frame_quality(fr.t_s, fr.rgb, s.vision.quality)
                    quality_s += time.perf_counter() - tq
                    if fq.low_light:
                        low_light_times.append(fr.t_s)
                    if fq.blurry:
                        blurry_times.append(fr.t_s)
                    td = time.perf_counter()
                    dets = self.detector.detect(fr.rgb, classes, s.vision.score_threshold, s.vision.nms_iou, s.vision.max_detections_per_frame)
                    detect_s += time.perf_counter() - td
                    animal = [d for d in dets if d.label in s.vision.target_classes]
                    for d in dets:
                        if d.label == "person" and "person" in s.vision.context_object_classes:
                            persons.append((fr.t_s, d.box, d.score))
                    adets = merge_species(animal, fr.t_s, fr.index)
                    detection_flags(adets, s.vision.quality)
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
                    frame_size = (fr.rgb.shape[1], fr.rgb.shape[0])
            timer.timings_ms["video_detector_inference"] = detect_s * 1000.0
            timer.timings_ms["video_quality_and_camera_motion"] = quality_s * 1000.0
            min_hits = s.vision.tracker.min_hits if n_frames >= 2 * s.vision.tracker.min_hits else 1
            tracks = tracker.tracks(min_hits)
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
        for tr in tracks:
            tid = tr.track_id
            flags = sorted({f for d in tr.detections for f in d.flags})
            mv = movement_summary(tr, frame_size[0], frame_size[1], camera_shift if s.vision.camera_motion_compensation else None)
            span_camera = any(tr.first_t <= t <= tr.last_t for t in camera_moving_times)
            span_dark = any(tr.first_t <= t <= tr.last_t for t in low_light_times)
            mv_rel = "low" if (not mv.get("available") or len(tr.detections) < 3 or span_dark) else ("medium" if span_camera or camera_unknown else "high")
            zone_occ = occupancy([d.t_s for d in tr.detections], [d.box for d in tr.detections], zones, s.vision.tracker.max_gap_s) if zones else []
            pose_feats = compute_features(self.pose, [], s.pose.min_keypoint_visibility)
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
                quality_flags=flags, evidence_ids=ev_ids))
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
            sm = S.SoundEvent(event_id=f"snd-{k}", label=ev.label, t_start_s=ev.start_s, t_end_s=ev.end_s, peak_raw_score=ev.peak_score,
                              mean_raw_score=ev.mean_score, threshold=ev.threshold, threshold_source=self.threshold_source.get(ev.label, "config default (not selected on data)") if not is_context else "config context_tag_threshold (not selected on data)",
                              subtypes=ev.subtypes, top_audioset_labels=ev.top_audioset_labels, attribution=att, evidence_id=eid)
            (context_models if is_context else sound_models).append(sm)
            tags: set[str] = set()
            if is_context:
                tags.add(ev.label.split(":", 1)[1])
            else:
                strong = {k2 for k2, v in ev.subtypes.items() if v >= SUBTYPE_MIN}
                tags |= strong & {"hiss", "growl", "purr", "bark", "howl"}
                # Request-like vocalization: a request-like subtype, or a species group
                # event without any confident subtype. Hiss/growl/purr alone are not requests.
                if ev.label != "growl" and (strong & REQUEST_LIKE or not strong):
                    tags.add("vocalization")
                if ev.label == "growl":
                    tags.add("growl")
            sub_txt = ", ".join(f"{k2} {v:.2f}" for k2, v in sorted(ev.subtypes.items(), key=lambda t: -t[1])[:3])
            item = S.EvidenceItem(
                evidence_id=eid, kind="context_sound" if is_context else "sound_event", t_start_s=ev.start_s, t_end_s=ev.end_s,
                track_id=None, summary=f"{label_txt} sound {ev.start_s:.2f}-{ev.end_s:.2f}s (peak raw score {ev.peak_score:.2f}{'; ' + sub_txt if sub_txt else ''})",
                value=ev.peak_score, value_kind="yamnet_sigmoid_raw", reliability=audio_rel,
                provenance=S.Provenance(component="audio_event", model_id=s.audio.yamnet.model_id, root="audio:yamnet", method="YAMNet tags, ontology group max, thresholded"))
            attributed = att.candidate_track_ids[0] if att.status == "single_visible_candidate" else None
            e = Ev(item, "audio:yamnet", frozenset(tags), 1.0, attributed_track=attributed)
            evidence.append(e)

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
        if is_still:
            limitations.append("Single still image: movement, dwell time and temporal co-occurrence cannot be measured.")
        if bandwidth_limited:
            limitations.append("Low-sample-rate audio: frequency content above the Nyquist limit is missing.")
        models = [S.ModelInfo(**{k: (v if k != "terms" else {kk: str(vv) for kk, vv in v.items()}) for k, v in model_terms_summary(c.model_id).items()})
                  for c in components.values() if c.model_id and c.state == "ok"]
        av = []
        if audio_state == "present":
            av.append("audio")
        if video_ok and n_frames:
            av.append("video")
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
                                        sample_fps=s.vision.sample_fps if n_frames else None),
            quality_flags=qflags, hypotheses=hyps, decision=decision, suggestions=suggestions, limitations=limitations,
            fusion=S.FusionInfo(method=s.fusion.method, note="Experimental fusion; weights/temperature are unfitted configuration values."),
            timings_ms={**{k: round(v, 2) for k, v in timer.timings_ms.items()}, "model_load_ms_engine_startup": round(self.model_load_ms, 2)},
            models=models,
            usage_restrictions=S.UsageRestrictions(research_only=True, notices=[
                "Research profile: model weights and evaluation data carry third-party terms (see docs/DATA_AND_MODEL_REGISTER.md).",
                "Audio-event thresholds derive from ESC-50 (CC BY-NC 3.0): non-commercial research use.",
                "Do not assume commercial permission for any bundled model or dataset."]),
        )
        return result


def _safe_name(name: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(name).name)
    return keep[:120] or "upload"
