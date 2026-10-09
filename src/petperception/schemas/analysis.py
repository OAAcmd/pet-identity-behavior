"""Versioned analysis result schema shared by the CLI and the API.

Layering (never collapsed):
1. ``observations`` - what was detected/measured, with timestamps, quality
   flags and provenance;
2. ``hypotheses`` - possible needs/activities/states, each tied to supporting
   and conflicting evidence IDs, with explicitly uncalibrated scores;
3. ``suggestions`` - restrained, contextual guidance that cites evidence.

Score fields never claim to be probabilities unless ``calibration_status`` is
``calibrated`` AND ``calibrated_confidence`` is set; in schema 1.x that never
happens for hypotheses because no paired household validation data exists.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.2.0"
# 1.1.0 is additive (behavior segments, keypoints, verification); 1.2.0 adds person_interaction_candidate.
SchemaVersion = Literal["1.0.0", "1.1.0", "1.2.0"]

Modality = Literal["audio", "video"]
ComponentState = Literal["ok", "unavailable", "disabled", "failed", "skipped", "not_applicable"]
DecisionState = Literal["supported_observation", "experimental_hypothesis", "abstention"]
HypothesisName = Literal["food", "outside", "attention", "play", "alert", "tense", "warning", "relaxed"]
SemanticType = Literal["need", "activity", "state", "communicative_signal"]
HypothesisStatus = Literal["experimental_hypothesis", "insufficient_evidence", "abstained"]
CalibrationStatus = Literal["uncalibrated", "calibrated", "not_applicable"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AudioStream(_Model):
    codec: str
    sample_rate_hz: int
    channels: int
    first_sample_time_s: float
    decoded_duration_s: float


class VideoStream(_Model):
    codec: str
    width: int
    height: int
    avg_fps: float | None
    start_time_s: float
    duration_s: float | None
    rotation_deg: int
    is_still_image: bool


class MediaInfo(_Model):
    analysis_filename: str = Field(description="Sanitized basename; the server never echoes client paths")
    size_bytes: int
    sha256: str
    container: str
    duration_s: float | None
    start_time_s: float
    audio: AudioStream | None
    video: VideoStream | None
    timeline: Literal["container_pts_seconds"] = "container_pts_seconds"


class ModalityStatus(_Model):
    requested: list[Modality]
    available: list[Modality]
    audio_state: Literal["present", "absent", "silent", "corrupt", "disabled", "too_short"]
    video_state: Literal["present", "absent", "no_decodable_frames", "corrupt", "disabled"]


class ComponentStatus(_Model):
    name: str
    kind: Literal["ingest", "audio_event", "visual_detector", "visual_verifier", "tracker", "pose", "behavior", "context", "fusion", "explanation", "vlm"]
    state: ComponentState
    model_id: str | None = None
    version: str | None = None
    reason: str | None = None


class QualityFlag(_Model):
    code: str
    severity: Literal["info", "warning", "limiting"]
    t_start_s: float | None = None
    t_end_s: float | None = None
    detail: str | None = None
    track_id: str | None = None


class Provenance(_Model):
    component: str
    model_id: str | None = None
    root: str = Field(description="Provenance root; evidence sharing a root is not independent")
    method: str


class EvidenceItem(_Model):
    evidence_id: str
    kind: Literal["animal_track", "sound_event", "zone_occupancy", "movement", "pose_feature", "context_sound", "context_object", "audio_absence"]
    t_start_s: float
    t_end_s: float
    summary: str
    track_id: str | None = None
    value: float | None = None
    value_kind: str | None = None
    reliability: Literal["high", "medium", "low"]
    provenance: Provenance


class Detection(_Model):
    t_s: float
    box_xyxy_norm: tuple[float, float, float, float]
    species_scores: dict[str, float]
    flags: list[str] = Field(default_factory=list)


class MovementSummary(_Model):
    available: bool
    units: str
    reason: str | None = None
    median_speed: float | None = None
    p90_speed: float | None = None
    path_length_diagonals: float | None = None
    moving_fraction: float | None = None
    moving_threshold: float | None = None
    median_speed_box_units_per_second: float | None = None
    box_area_change_ratio: float | None = None
    camera_compensated_steps: int | None = None
    steps: int | None = None
    reliability: Literal["high", "medium", "low"] = "low"


class ZoneOccupancy(_Model):
    zone_id: str
    semantic_tag: str
    label: str
    intervals_s: list[tuple[float, float]]
    dwell_s: float
    fraction_of_track_detections: float
    anchor: str


class PoseFeature(_Model):
    name: str
    available: bool
    value: float | bool | None = None
    units: str | None = None
    reason: str | None = None


BehaviorKind = Literal["stationary", "moving", "active_bout", "pacing_candidate", "approach_zone", "enter_zone", "dwell_zone", "leave_zone",
                       "entered_view", "left_view", "vocal_bout", "startle_candidate", "person_interaction_candidate"]


class BehaviorSegment(_Model):
    """Operationally defined, timestamped observation (never a mental state)."""

    segment_id: str
    kind: BehaviorKind
    track_id: str | None = None
    t_start_s: float
    t_end_s: float
    reliability: Literal["high", "medium", "low"]
    detail: str
    measurements: dict[str, float] = Field(default_factory=dict)
    method: str
    validation_status: Literal["synthetic_and_semi_synthetic_only", "per_frame_model_public_still_images"] = "synthetic_and_semi_synthetic_only"


class KeypointFrame(_Model):
    t_s: float
    points: dict[str, tuple[float, float, float]] = Field(description="name -> (x, y, visibility score); normalized image coords; score uncalibrated")
    provider: str


class PostureFrame(_Model):
    t_s: float
    label: Literal["lying", "sitting", "standing", "uncertain"]
    score: float
    score_kind: Literal["posture_softmax_uncalibrated"] = "posture_softmax_uncalibrated"


class TrackVerification(_Model):
    verifier_id: str
    checked_detections: int
    median_score: float | None
    threshold: float
    accepted: bool
    score_kind: Literal["verifier_rejection_score_uncalibrated"] = "verifier_rejection_score_uncalibrated"


class AnimalTrack(_Model):
    track_id: str
    identity_note: Literal["per-analysis track label; not a verified pet identity"] = "per-analysis track label; not a verified pet identity"
    species: Literal["cat", "dog"]
    species_scores: dict[str, dict[str, float]]
    score_kind: Literal["detector_sigmoid_raw"] = "detector_sigmoid_raw"
    first_seen_s: float
    last_seen_s: float
    duration_s: float
    n_detections: int
    missed_sampled_frames: int
    detections: list[Detection]
    movement: MovementSummary
    zones: list[ZoneOccupancy] = Field(default_factory=list)
    pose_features: list[PoseFeature] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    behavior_segments: list[BehaviorSegment] = Field(default_factory=list)
    keypoints: list[KeypointFrame] = Field(default_factory=list)
    posture: list[PostureFrame] = Field(default_factory=list)
    verification: TrackVerification | None = None


class Attribution(_Model):
    status: Literal["unknown", "no_visible_animal", "single_visible_candidate", "video_unavailable"]
    candidate_track_ids: list[str] = Field(default_factory=list)
    method: Literal["none_evaluated"] = "none_evaluated"
    note: str


class SoundEvent(_Model):
    event_id: str
    label: str
    t_start_s: float
    t_end_s: float
    peak_raw_score: float
    mean_raw_score: float
    threshold: float
    threshold_source: str
    score_kind: Literal["yamnet_sigmoid_raw"] = "yamnet_sigmoid_raw"
    calibration_status: Literal["uncalibrated"] = "uncalibrated"
    calibrated_probability: None = None
    subtypes: dict[str, float] = Field(default_factory=dict)
    top_audioset_labels: list[tuple[str, float]] = Field(default_factory=list)
    attribution: Attribution
    evidence_id: str


class AudioSummary(_Model):
    state: str
    rms_dbfs: float | None = None
    peak_dbfs: float | None = None
    silent_fraction: float | None = None
    clipping_fraction: float | None = None
    effective_bandwidth_hz: float | None = None
    windows_scored: int = 0


class Observations(_Model):
    animal_tracks: list[AnimalTrack]
    sound_events: list[SoundEvent]
    context_sounds: list[SoundEvent]
    audio_summary: AudioSummary
    evidence: list[EvidenceItem]
    sampled_frames: int
    sample_fps: float | None
    behavior_events: list[BehaviorSegment] = Field(default_factory=list)


class Scorer(_Model):
    name: str
    revision: str
    kind: Literal["rule_baseline", "weighted_average", "product_of_experts"]


class Hypothesis(_Model):
    hypothesis_id: str
    name: HypothesisName
    semantic_type: SemanticType
    status: HypothesisStatus
    track_id: str | None = None
    t_start_s: float | None = None
    t_end_s: float | None = None
    supporting_evidence_ids: list[str]
    conflicting_evidence_ids: list[str]
    independent_provenance_roots: list[str]
    score: float | None = Field(None, description="Experimental unitless support score; NOT a probability")
    score_kind: Literal["heuristic_support_0_1", "fusion_experimental_score_0_1", "none"]
    scorer: Scorer
    calibration_status: CalibrationStatus = "uncalibrated"
    calibrated_confidence: float | None = None
    reason_codes: list[str] = Field(default_factory=list)
    explanation: str


class Decision(_Model):
    state: DecisionState
    overall_interpretation: Literal["observations_only", "experimental_hypotheses_present", "unknown", "insufficient_evidence"]
    reason_codes: list[str]
    summary: str


class Suggestion(_Model):
    suggestion_id: str
    text: str
    based_on_evidence_ids: list[str]
    kind: Literal["observation_followup", "safety_general", "data_quality"]


class FusionInfo(_Model):
    mode: Literal["experimental"] = "experimental"
    method: str
    validated: Literal[False] = False
    multimodal_accuracy: Literal["not_evaluated"] = "not_evaluated"
    weights_fitted: Literal[False] = False
    note: str


class ModelInfo(_Model):
    model_id: str
    name: str | None
    revision: str
    sha256: dict[str, str]
    terms: dict[str, str]
    use_restrictions: list[str]


class UsageRestrictions(_Model):
    research_only: bool
    notices: list[str]


class AnalysisResult(_Model):
    schema_version: SchemaVersion = SCHEMA_VERSION
    analysis_id: str
    created_at: datetime
    engine_version: str
    config_profile: str
    config_hash: str
    media: MediaInfo
    modalities: ModalityStatus
    components: list[ComponentStatus]
    observations: Observations
    quality_flags: list[QualityFlag]
    hypotheses: list[Hypothesis]
    decision: Decision
    suggestions: list[Suggestion]
    limitations: list[str]
    fusion: FusionInfo
    timings_ms: dict[str, float]
    models: list[ModelInfo]
    usage_restrictions: UsageRestrictions


class AnalysisError(_Model):
    schema_version: SchemaVersion = SCHEMA_VERSION
    analysis_id: str | None = None
    error_code: str
    message: str
    components: list[ComponentStatus] = Field(default_factory=list)
