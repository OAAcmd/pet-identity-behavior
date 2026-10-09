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
        p = d / f["name"]
        files[f["name"]] = str(p)
        if not p.exists():
            return ModelReadiness(model_id, "missing", files, f"{p} not found; run: uv run petperception fetch-models --only {model_id}")
        if verify_checksum and _cached_sha256(p) != f["sha256"]:
            return ModelReadiness(model_id, "checksum_mismatch", files, f"{p} does not match the pinned SHA-256")
    return ModelReadiness(model_id, "ready", files)


def fetch_model(settings: Settings, model_id: str) -> dict[str, Any]:
    r = resource(model_id)
    if r["kind"] != "model":
        raise ValueError(f"{model_id} is not a model")
    if r["status"] in ("blocked", "blocked_pending_review"):
        return {"model_id": model_id, "status": "blocked", "errors": r.get("errors", [])}
    d = model_dir(settings, model_id)
    out = []
    for f in r["files"]:
        p = d / f["name"]
        if p.exists() and sha256_file(p) == f["sha256"]:
            out.append({"name": f["name"], "status": "already_present"})
            continue
        try:
            rec = download(f["url"], p, expected_sha256=f["sha256"])
            out.append({"name": f["name"], "status": "downloaded", **rec})
        except DownloadError as exc:
            out.append({"name": f["name"], "status": "error", "error": str(exc)})
    ok = all(x["status"] in ("downloaded", "already_present") for x in out)
    return {"model_id": model_id, "status": "ready" if ok else "error", "files": out}


def default_model_ids(settings: Settings) -> list[str]:
    ids = []
    if settings.audio.enabled and settings.audio.event_backend == "yamnet":
        ids.append(settings.audio.yamnet.model_id)
    if settings.vision.enabled:
        ids.append(settings.vision.detector)
    return ids


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
