"""Configuration loading.

Configuration lives in YAML files under ``configs/``. Every path and optional
endpoint is configurable; nothing about infrastructure is hard-coded. Values
of the form ``${ENV_VAR:-default}`` are expanded from the environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Project root holding configs/, manifests/ and artifacts/. Overridable for containers.
REPO_ROOT = Path(os.environ.get("PETPERCEPTION_HOME", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cpu.yaml"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class _Strict(BaseModel):
    """Configuration models reject unknown keys, so a misspelled option fails loudly."""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    data_root: str = "var/data"
    models_root: str = "var/models"
    results_root: str = "results"
    artifacts_root: str = "artifacts"
    manifests_root: str = "manifests"
    jobs_root: str = "var/jobs"

    def resolve(self, name: str) -> Path:
        p = Path(getattr(self, name)).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p)


class LimitsConfig(_Strict):
    max_file_bytes: int = 200 * 1024 * 1024
    max_duration_s: float = 600.0
    max_video_frames: int = 1200
    max_audio_channels: int = 8
    ffprobe_timeout_s: float = 30.0
    decode_timeout_s: float = 300.0


class YamnetConfig(_Strict):
    model_id: str = "yamnet_mediapipe_v1"
    hop_s: float = 0.48
    # Per-group decision thresholds on raw (uncalibrated) sigmoid scores.
    # Defaults are replaced by values selected on ESC-50 development folds
    # (see results/e1 and artifacts/audio_event_thresholds.json).
    thresholds_file: str | None = "artifacts/audio_event_thresholds.json"
    default_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"cat_vocalization": 0.5, "dog_vocalization": 0.5, "growl": 0.5}
    )
    context_tag_threshold: float = 0.3
    min_event_gap_s: float = 0.5
    # Which vocalization subtypes passed the pre-set validation gate (E21, FSD50K eval); only those
    # are shown or used as interpretation triggers. None: no subtype counts as validated.
    subtype_validation_file: str | None = "artifacts/audio_subtype_validation.json"
    # Per-tag context thresholds adopted in E21b (selected on FSD50K dev CC0 / CC BY clips); other tags use context_tag_threshold.
    context_thresholds_file: str | None = "artifacts/audio_context_thresholds.json"


class AudioConfig(_Strict):
    enabled: bool = True
    event_backend: Literal["yamnet", "none"] = "yamnet"
    yamnet: YamnetConfig = Field(default_factory=YamnetConfig)
    silence_dbfs: float = -60.0
    clipping_fraction: float = 0.01
    analysis_sample_rate: int = 16000


class TrackerConfig(_Strict):
    method: Literal["iou", "kalman"] = "kalman"  # selected on E8 development seeds
    iou_threshold: float = 0.3
    max_gap_s: float = 1.5
    min_hits: int = 2


class VisionQualityConfig(_Strict):
    low_light_luma: float = 40.0
    blur_laplacian_var: float = 25.0
    tiny_area_fraction: float = 0.004
    camera_motion_threshold: float = 0.01
    frame_gap_factor: float = 2.5


E7_CLASSIFIER = "efficientnet_lite0_mediapipe_v1"


class VerifierConfig(_Strict):
    """Second-stage false-positive rejection (E7). Parameters are per detector."""

    enabled: bool = True
    classifier: str = "efficientnet_lite0_mediapipe_v1"
    params_dir: str = "artifacts/verifier"  # verifier_<detector id>[__<classifier id>].json, exported by E7 / E12
    max_checks_per_frame: int = 3

    def params_file(self, detector: str) -> str:
        """Verifier parameters are specific to the (detector, classifier) pair they were fitted on.
        E7 fitted with the float32 classifier (no suffix); other classifiers carry a suffix (E12)."""
        suffix = "" if self.classifier == E7_CLASSIFIER else f"__{self.classifier}"
        return f"{self.params_dir}/verifier_{detector}{suffix}.json"


class OnDeviceConfig(_Strict):
    """Model builds exported into the phone bundle (export-ondevice).

    Default: the float32 builds, i.e. the configuration E4-E10 and engine parity were
    measured with. The int8 builds (efficientdet_lite0_mediapipe_int8_v1 +
    efficientnet_lite0_mediapipe_int8_v1, with their refitted verifier) are an evaluated
    compact alternative (E12: within the replacement rule on held-out photos, 14.8 MB
    instead of 37.2 MB), but individual verifier decisions can diverge strongly from
    float32, so they are opt-in until real-phone and household evaluation."""

    detector: str = "efficientdet_lite0_mediapipe_v1"
    backbone: str = "efficientnet_lite0_mediapipe_v1"
    posture_model: str = "artifacts/posture/posture_v1.npz"


class VisionConfig(_Strict):
    enabled: bool = True
    detector: str = "efficientdet_lite0_mediapipe_v1"
    sample_fps: float = 2.0
    max_side: int = 640
    score_threshold: float = 0.35
    nms_iou: float = 0.5
    max_detections_per_frame: int = 10
    target_classes: list[str] = Field(default_factory=lambda: ["cat", "dog"])
    # Generic COCO context objects (e.g. "person"). Off by default: the COCO
    # "bowl" class, for example, is not validated for pet food bowls.
    context_object_classes: list[str] = Field(default_factory=list)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    quality: VisionQualityConfig = Field(default_factory=VisionQualityConfig)
    camera_motion_compensation: bool = True
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)


class PoseConfig(_Strict):
    # "dog_keypoints_v1": E9 head on EfficientNet-Lite0 features (dogs only, research profile).
    # "catdog_keypoints_awa_v1": E24 joint head (AwA-Pose), cats and dogs, research profile;
    #   set head_file to artifacts/keypoints/catdog_keypoints_head_awa_v1.npz with it.
    provider: Literal["none", "dog_keypoints_v1", "catdog_keypoints_awa_v1"] = "none"
    min_keypoint_visibility: float = 0.5
    head_file: str = "artifacts/keypoints/dog_keypoints_head_v1.npz"
    backbone: str = "efficientnet_lite0_mediapipe_v1"
    max_per_frame: int = 2
    min_detection_score: float = 0.45


class InteractionConfig(_Strict):
    """Person-pet interaction candidates from box geometry (E13). Persons are detected on
    the analysing device with the same detector; no person box is stored in results."""

    enabled: bool = True
    model_file: str = "artifacts/interaction/person_pet_interaction_v1.npz"
    min_person_score: float = 0.35
    min_frames: int = 2  # consecutive sampled frames above the threshold (2 fps: >= 0.5 s)


class ObjectRelationConfig(_Strict):
    """Pet-on-furniture candidates from box geometry (E15). Furniture (bed, couch, chair,
    dining table) is detected on the analysing device with the same detector; object boxes
    are not stored in results. Only relations that passed the E15 gate are in the model."""

    enabled: bool = True
    model_file: str = "artifacts/object_relation/pet_object_relation_v2.npz"
    resource_id: str = "pet_object_relation_v2"  # v2: containment rule (shipped); v1: learned E15 head (evaluated alternative)
    min_object_score: float = 0.35
    min_frames: int = 2


class IntentConfig(_Strict):
    """Trained behaviour/intent channels and their combination (src/petperception/intent/, E26-E28).
    Off by default: the extra per-frame passes (whole-frame embedding, all COCO object classes, largest-animal
    crop) roughly double the video cost. "phone_now" uses only what the phone bundle computes (YAMNet class
    scores, no research-only keypoints); "full" adds the YAMNet embedding and the cat/dog keypoint head."""

    enabled: bool = False
    variant: Literal["phone_now", "full"] = "phone_now"


class PostureConfig(_Strict):
    """Coarse posture from weak caption labels (E10); cats and dogs."""

    enabled: bool = True
    model_file: str = "artifacts/posture/posture_v1.npz"
    backbone: str = "efficientnet_lite0_mediapipe_v1"
    max_per_frame: int = 2
    min_detection_score: float = 0.45


class BehaviorConfig(_Strict):
    """A-priori thresholds for vision/behavior.py (checked on synthetic data only, E8)."""

    enabled: bool = True
    moving_speed: float = 0.05
    active_speed: float = 0.25
    min_stationary_s: float = 2.0
    min_moving_s: float = 1.0
    min_active_s: float = 1.0
    pacing_min_amplitude: float = 0.08
    pacing_min_reversals: int = 3
    pacing_max_leg_s: float = 10.0
    pacing_min_span_s: float = 4.0
    approach_min_drop: float = 0.05
    approach_min_s: float = 1.0
    approach_lookback_s: float = 10.0
    view_edge_margin_s: float = 1.0
    vocal_max_gap_s: float = 2.0
    startle_max_lag_s: float = 1.5
    startle_baseline_s: float = 3.0
    startle_ratio: float = 3.0


class FusionConfig(_Strict):
    mode: Literal["experimental"] = "experimental"
    method: Literal["rule_baseline", "weighted_average", "product_of_experts"] = "rule_baseline"
    # Expert weights; NOT fitted (no paired development data exists).
    weights: dict[str, float] = Field(
        default_factory=lambda: {"audio": 1.0, "vision": 1.0, "context": 0.5, "pose": 1.0}
    )
    temperature: float = 2.0
    smoothing_epsilon: float = 0.02
    prior_logit: float = -1.0
    max_evidence_age_s: float = 10.0
    co_occurrence_window_s: float = 3.0
    min_zone_dwell_s: float = 1.0
    abstain_below_score: float = 0.5


class VlmConfig(_Strict):
    enabled: bool = False
    endpoint: str | None = None

    @field_validator("endpoint")
    @classmethod
    def _local_only(cls, v: str | None) -> str | None:
        if v and not re.match(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?/", v):
            raise ValueError("VLM endpoint must be an owner-configured loopback URL")
        return v


class ServeConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = 8000
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".m4a", ".aac",
                                 ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".jpg", ".jpeg", ".png"]
    )
    # Static PWA build served from the same origin (no CORS needed).
    static_dir: str = "web/dist"
    # On-device model bundle published at /ondevice/ (petperception export-ondevice).
    ondevice_dir: str = "var/ondevice/bundle"
    # Extra Content-Security-Policy connect-src entries (none by default).
    trusted_proxy_hops: int = 0


class AuthConfig(_Strict):
    registration: Literal["first_user_only", "open", "closed"] = "first_user_only"
    session_ttl_hours: float = 24.0 * 14
    session_idle_hours: float = 24.0 * 3
    cookie_name: str = "pp_session"
    cookie_secure: bool = True  # set false only for plain-http localhost development
    min_password_length: int = 10
    login_max_failures: int = 5
    login_window_s: float = 900.0


class JobsConfig(_Strict):
    workers: int = 1
    lease_s: float = 60.0
    heartbeat_s: float = 10.0
    max_attempts: int = 3
    retry_backoff_s: float = 5.0
    job_timeout_s: float = 900.0
    poll_interval_s: float = 0.5
    max_active_per_user: int = 3
    max_queued_total: int = 50


class StorageConfig(_Strict):
    database_path: str = "${PETPERCEPTION_DB:-var/app/petperception.sqlite3}"
    media_root: str = "${PETPERCEPTION_MEDIA_ROOT:-var/app/media}"
    max_upload_bytes: int = 100 * 1024 * 1024
    max_chunk_bytes: int = 8 * 1024 * 1024
    max_open_uploads_per_user: int = 5
    upload_expiry_hours: float = 24.0
    user_quota_bytes: int = 2 * 1024 * 1024 * 1024
    max_retain_media_days: int = 30
    min_free_disk_bytes: int = 512 * 1024 * 1024
    cleanup_interval_s: float = 300.0

    def resolve(self, value: str) -> Path:
        p = Path(_expand_env(value)).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p)


class AppConfig(_Strict):
    auth: AuthConfig = Field(default_factory=AuthConfig)
    jobs: JobsConfig = Field(default_factory=JobsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    log_format: Literal["json", "text"] = "json"
    log_level: str = "INFO"


class RuntimeConfig(_Strict):
    num_threads: int = 2
    accelerator: Literal["cpu"] = "cpu"


SUPPORTED_CONFIG_VERSION = 2


class Settings(_Strict):
    config_version: int = SUPPORTED_CONFIG_VERSION
    profile: str = "cpu"
    # Release profile decides which assets may be loaded (see manifests/resources.json):
    # "research" allows research-only assets; "personal_noncommercial" and
    # "commercial" refuse assets whose terms do not permit that use.
    release_profile: Literal["research", "personal_noncommercial", "commercial"] = "research"
    research_profile: bool = True
    paths: PathsConfig = Field(default_factory=PathsConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    posture: PostureConfig = Field(default_factory=PostureConfig)
    interaction: InteractionConfig = Field(default_factory=InteractionConfig)
    object_relation: ObjectRelationConfig = Field(default_factory=ObjectRelationConfig)
    intent: IntentConfig = Field(default_factory=IntentConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    ondevice: OnDeviceConfig = Field(default_factory=OnDeviceConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @field_validator("config_version")
    @classmethod
    def _supported_version(cls, v: int) -> int:
        if v != SUPPORTED_CONFIG_VERSION:
            raise ValueError(f"config_version {v} is not supported (expected {SUPPORTED_CONFIG_VERSION}); see docs/CONFIGURATION.md")
        return v

    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_settings(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    """Load settings from YAML (default ``configs/cpu.yaml``) with env expansion."""
    cfg_path = Path(path) if path else Path(os.environ.get("PETPERCEPTION_CONFIG", DEFAULT_CONFIG))
    data: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    data = _expand_env(data)
    if overrides:
        data = _deep_merge(data, overrides)
    return Settings.model_validate(data)


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
