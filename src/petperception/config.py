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
from pydantic import BaseModel, Field, field_validator

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


class PathsConfig(BaseModel):
    data_root: str = "var/data"
    models_root: str = "var/models"
    results_root: str = "results"
    artifacts_root: str = "artifacts"
    manifests_root: str = "manifests"
    jobs_root: str = "var/jobs"

    def resolve(self, name: str) -> Path:
        p = Path(getattr(self, name)).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p)


class LimitsConfig(BaseModel):
    max_file_bytes: int = 200 * 1024 * 1024
    max_duration_s: float = 600.0
    max_video_frames: int = 1200
    max_audio_channels: int = 8
    ffprobe_timeout_s: float = 30.0
    decode_timeout_s: float = 300.0


class YamnetConfig(BaseModel):
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


class AudioConfig(BaseModel):
    enabled: bool = True
    event_backend: Literal["yamnet", "none"] = "yamnet"
    yamnet: YamnetConfig = Field(default_factory=YamnetConfig)
    silence_dbfs: float = -60.0
    clipping_fraction: float = 0.01
    analysis_sample_rate: int = 16000


class TrackerConfig(BaseModel):
    iou_threshold: float = 0.3
    max_gap_s: float = 1.5
    min_hits: int = 2


class VisionQualityConfig(BaseModel):
    low_light_luma: float = 40.0
    blur_laplacian_var: float = 25.0
    tiny_area_fraction: float = 0.004
    camera_motion_threshold: float = 0.01
    frame_gap_factor: float = 2.5


class VisionConfig(BaseModel):
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


class PoseConfig(BaseModel):
    provider: Literal["none"] = "none"
    min_keypoint_visibility: float = 0.5


class FusionConfig(BaseModel):
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


class VlmConfig(BaseModel):
    enabled: bool = False
    endpoint: str | None = None

    @field_validator("endpoint")
    @classmethod
    def _local_only(cls, v: str | None) -> str | None:
        if v and not re.match(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?/", v):
            raise ValueError("VLM endpoint must be an owner-configured loopback URL")
        return v


class ServeConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    auth_token_env: str = "PETPERCEPTION_API_TOKEN"
    max_upload_bytes: int = 100 * 1024 * 1024
    job_ttl_s: float = 3600.0
    max_jobs: int = 32
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".wav", ".flac", ".ogg", ".mp3", ".m4a", ".mp4", ".mov", ".mkv", ".webm", ".avi"]
    )


class RuntimeConfig(BaseModel):
    num_threads: int = 2
    accelerator: Literal["cpu"] = "cpu"


class Settings(BaseModel):
    profile: str = "cpu"
    research_profile: bool = True
    paths: PathsConfig = Field(default_factory=PathsConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

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
