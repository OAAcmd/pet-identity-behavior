"""Resource register: model/dataset metadata, readiness checks and explicit fetching.

``manifests/resources.json`` is the machine-readable register. Normal inference
only *checks* files (presence + SHA-256); downloads happen exclusively through
``petperception fetch-models`` / ``fetch-data``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, Settings
from .utils import DownloadError, download, read_json, sha256_file

RESOURCES_PATH = REPO_ROOT / "manifests" / "resources.json"


@lru_cache(maxsize=1)
def load_resources() -> dict[str, dict[str, Any]]:
    data = read_json(RESOURCES_PATH)
    return {r["id"]: r for r in data["resources"]}


def resource(resource_id: str) -> dict[str, Any]:
    res = load_resources()
    if resource_id not in res:
        raise KeyError(f"unknown resource '{resource_id}'")
    return res[resource_id]


@dataclass
class ModelReadiness:
    model_id: str
    status: str  # ready | missing | checksum_mismatch | blocked | unknown_model
    files: dict[str, str]
    reason: str | None = None

    @property
    def ready(self) -> bool:
        return self.status == "ready"


_checksum_cache: dict[tuple[str, float, int], str] = {}


def _cached_sha256(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_mtime, st.st_size)
    if key not in _checksum_cache:
        _checksum_cache[key] = sha256_file(path)
    return _checksum_cache[key]


def model_dir(settings: Settings, model_id: str) -> Path:
    return settings.paths.resolve("models_root") / model_id


def check_model(settings: Settings, model_id: str, verify_checksum: bool = True) -> ModelReadiness:
    try:
        r = resource(model_id)
    except KeyError:
        return ModelReadiness(model_id, "unknown_model", {}, f"'{model_id}' is not in manifests/resources.json")
    if r["status"] in ("blocked", "blocked_pending_review"):
        return ModelReadiness(model_id, "blocked", {}, "; ".join(r.get("errors", [])) or "blocked")
    files: dict[str, str] = {}
    d = model_dir(settings, model_id)
    for f in r.get("files", []):
        p = _file_path(d, f)
        files[f["name"]] = str(p)
        if not p.exists():
            return ModelReadiness(model_id, "missing", files, f"{p} not found; run: uv run petperception fetch-models --only {model_id}")
        if verify_checksum and _cached_sha256(p) != f["sha256"]:
            return ModelReadiness(model_id, "checksum_mismatch", files, f"{p} does not match the pinned SHA-256")
    return ModelReadiness(model_id, "ready", files)


def _file_path(d: Path, f: dict) -> Path:
    """Models trained in this project are committed and referenced by ``repo_path``; downloaded ones live in the model directory."""
    return REPO_ROOT / f["repo_path"] if f.get("repo_path") else d / f["name"]


def fetch_model(settings: Settings, model_id: str) -> dict[str, Any]:
    r = resource(model_id)
    if r["kind"] != "model":
        raise ValueError(f"{model_id} is not a model")
    if r["status"] in ("blocked", "blocked_pending_review"):
        return {"model_id": model_id, "status": "blocked", "errors": r.get("errors", [])}
    d = model_dir(settings, model_id)
    out = []
    for f in r["files"]:
        p = _file_path(d, f)
        if p.exists() and sha256_file(p) == f["sha256"]:
            out.append({"name": f["name"], "status": "already_present"})
            continue
        if f.get("repo_path"):
            out.append({"name": f["name"], "status": "error", "error": f"{p} is missing or modified (part of this repository)"})
            continue
        try:
            rec = download(f["url"], p, expected_sha256=f["sha256"])
            out.append({"name": f["name"], "status": "downloaded", **rec})
        except DownloadError as exc:
            out.append({"name": f["name"], "status": "error", "error": str(exc)})
    ok = all(x["status"] in ("downloaded", "already_present") for x in out)
    return {"model_id": model_id, "status": "ready" if ok else "error", "files": out}


def default_model_ids(settings: Settings) -> list[str]:
    """Every model the configured engine loads (detector, verifier classifier, backbones, YAMNet)."""
    return [rid for rid in configured_resources(settings) if load_resources().get(rid, {}).get("kind") == "model"]


def model_terms_summary(model_id: str) -> dict[str, Any]:
    r = resource(model_id)
    return {
        "model_id": model_id,
        "name": r.get("name"),
        "revision": "; ".join(f.get("revision", "") for f in r.get("files", [])),
        "sha256": {f["name"]: f["sha256"] for f in r.get("files", [])},
        "terms": r.get("terms", {}),
        "use_restrictions": r.get("use_restrictions", []),
    }


# ---------------------------------------------------------------------------
# Release profiles: which assets may be loaded for a given release.
# ---------------------------------------------------------------------------
APPROVALS_PATH = REPO_ROOT / "configs" / "release_approvals.yaml"


@dataclass
class ReleaseDecision:
    resource_id: str
    profile: str
    allowed: bool
    status: str  # suitable | approved | requires_legal_review | not_permitted | unknown
    reason: str | None = None


def load_approvals(path: Path | None = None) -> dict[str, Any]:
    """Legal approvals recorded by a person with authority (never written by code)."""
    p = path or APPROVALS_PATH
    if not p.exists():
        return {}
    import yaml

    data = yaml.safe_load(p.read_text()) or {}
    return data.get("approvals", {}) if isinstance(data, dict) else {}


def release_decision(profile: str, resource_id: str, approvals: dict[str, Any] | None = None) -> ReleaseDecision:
    try:
        r = resource(resource_id)
    except KeyError:
        return ReleaseDecision(resource_id, profile, False, "unknown", f"'{resource_id}' is not in the register")
    st = r.get("release_suitability", {}).get(profile)
    note = r.get("release_note", "")
    if st == "suitable":
        return ReleaseDecision(resource_id, profile, True, "suitable")
    if st == "requires_legal_review":
        appr = (approvals if approvals is not None else load_approvals()).get(resource_id, {}).get(profile, {})
        if isinstance(appr, dict) and appr.get("approved_by") and appr.get("reference"):
            return ReleaseDecision(resource_id, profile, True, "approved", f"approved by {appr['approved_by']} ({appr['reference']})")
        return ReleaseDecision(resource_id, profile, False, "requires_legal_review",
                               f"{resource_id} requires legal review for the '{profile}' profile: {note} Record an approval in configs/release_approvals.yaml.")
    if st == "not_permitted":
        return ReleaseDecision(resource_id, profile, False, "not_permitted", f"{resource_id} is not permitted under the '{profile}' profile: {note}")
    return ReleaseDecision(resource_id, profile, False, "unknown", f"{resource_id} has no release assessment for the '{profile}' profile")


def configured_resources(settings: Settings) -> list[str]:
    """Resources the analysis engine would load with this configuration."""
    out = []
    if settings.audio.enabled and settings.audio.event_backend == "yamnet":
        out.append(settings.audio.yamnet.model_id)
        if settings.audio.yamnet.thresholds_file:
            out.append("audio_event_thresholds_esc50")
    if settings.vision.enabled:
        out.append(settings.vision.detector)
        if settings.vision.verifier.enabled:
            out += [settings.vision.verifier.classifier, "detection_verifier_v1"]
    if settings.pose.provider == "dog_keypoints_v1":
        out += [settings.pose.backbone, "dog_keypoints_head_v1"]
    if settings.posture.enabled:
        out += [settings.posture.backbone, "posture_v1"]
    return list(dict.fromkeys(out))


def release_report(settings: Settings) -> dict[str, Any]:
    approvals = load_approvals()
    decisions = [release_decision(settings.release_profile, rid, approvals) for rid in configured_resources(settings)]
    return {"release_profile": settings.release_profile, "all_allowed": all(d.allowed for d in decisions),
            "decisions": [d.__dict__ for d in decisions]}
