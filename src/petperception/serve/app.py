"""FastAPI service exposing the same AnalysisEngine as the CLI.

Security posture for Phase 1 (local research service):
* binds to 127.0.0.1 by default; ``petperception serve`` refuses a
  non-loopback bind unless a bearer token is configured;
* if ``$PETPERCEPTION_API_TOKEN`` (configurable) is set, every /v1 endpoint
  except /v1/health requires ``Authorization: Bearer <token>``; without a
  token, requests from non-loopback clients are rejected;
* uploads only (no server-side URL fetching), extension allow-list,
  size cap enforced while streaming, ffprobe validation before queueing;
* private 0700 job directories, media deleted right after analysis,
  results deleted on DELETE or after a TTL.
TLS and production authentication are Phase 2 items.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .. import __version__
from ..config import Settings
from ..context.zones import ZoneConfig
from ..ingest.media import MediaError, probe
from ..pipeline import AnalysisEngine
from ..schemas.analysis import SCHEMA_VERSION, AnalysisError, AnalysisResult
from .jobs import JobStore

API_PREFIX = "/v1"


class ErrorBody(BaseModel):
    error_code: str
    message: str


class SubmitResponse(BaseModel):
    analysis_id: str
    status: Literal["queued"]
    status_url: str
    result_url: str


class StatusResponse(BaseModel):
    analysis_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    updated_at: str
    error: AnalysisError | None = None


class DeleteResponse(BaseModel):
    analysis_id: str
    status: Literal["deleted", "deletion_scheduled"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    schema_version: str
    engine_version: str
    engine_loaded: bool


def _is_loopback(host: str | None) -> bool:
    if host in (None, "testclient", "localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_app(settings: Settings, engine: AnalysisEngine | None = None) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        eng = engine or AnalysisEngine(settings).load()
        state["engine"] = eng
        state["jobs"] = JobStore(settings.paths.resolve("jobs_root"), eng.analyze, settings.serve.job_ttl_s, settings.serve.max_jobs)
        yield
        state["jobs"].shutdown()

    app = FastAPI(
        title="Pet Perception Research API",
        version=__version__,
        description=("Observation-first cat/dog behavior evidence engine (Phase 1, research use). Results separate observations, "
                     "experimental uncalibrated hypotheses and restrained suggestions. Not a translator, emotion reader or diagnostic tool."),
        lifespan=lifespan,
    )

    def auth(request: Request) -> None:
        token = os.environ.get(settings.serve.auth_token_env)
        if token:
            header = request.headers.get("authorization", "")
            if not (header.startswith("Bearer ") and hmac.compare_digest(header[7:], token)):
                raise HTTPException(status_code=401, detail={"error_code": "unauthorized", "message": "missing or invalid bearer token"})
        elif not _is_loopback(request.client.host if request.client else None):
            raise HTTPException(status_code=403, detail={"error_code": "forbidden", "message": "unauthenticated access is limited to loopback clients"})

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        body = exc.detail if isinstance(exc.detail, dict) else {"error_code": "http_error", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", schema_version=SCHEMA_VERSION, engine_version=__version__, engine_loaded="engine" in state)

    @app.get(f"{API_PREFIX}/capabilities", tags=["service"], dependencies=[Depends(auth)])
    def capabilities() -> dict[str, Any]:
        caps = state["engine"].capabilities()
        caps["upload"] = {"max_bytes": settings.serve.max_upload_bytes, "allowed_extensions": settings.serve.allowed_extensions,
                          "url_fetching": False, "media_retained_after_analysis": False, "result_ttl_s": settings.serve.job_ttl_s}
        return caps

    @app.get(f"{API_PREFIX}/schema/analysis-result", tags=["service"], dependencies=[Depends(auth)])
    def analysis_schema() -> dict[str, Any]:
        return AnalysisResult.model_json_schema()

    @app.post(f"{API_PREFIX}/analyses", status_code=202, response_model=SubmitResponse, tags=["analyses"],
              responses={401: {"model": ErrorBody}, 403: {"model": ErrorBody}, 413: {"model": ErrorBody}, 415: {"model": ErrorBody},
                         422: {"model": ErrorBody}, 503: {"model": ErrorBody}},
              dependencies=[Depends(auth)])
    async def submit(request: Request, file: UploadFile = File(..., description="Audio or video file (no URLs)"),
                     zones: str | None = Form(None, description="Optional zone configuration JSON (petperception.zones/1)"),
                     modalities: str = Form("audio,video", description="Comma-separated subset of audio,video")) -> SubmitResponse:
        cap = settings.serve.max_upload_bytes
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > cap + 1_000_000:
            raise HTTPException(413, {"error_code": "upload_too_large", "message": f"limit is {cap} bytes"})
        ext = Path(file.filename or "").suffix.lower()
        if ext not in settings.serve.allowed_extensions:
            raise HTTPException(415, {"error_code": "unsupported_extension", "message": f"allowed: {settings.serve.allowed_extensions}"})
        mods = tuple(m.strip() for m in modalities.split(",") if m.strip())
        if not mods or any(m not in ("audio", "video") for m in mods):
            raise HTTPException(422, {"error_code": "invalid_modalities", "message": "use a non-empty subset of audio,video"})
        zcfg = None
        if zones:
            try:
                zcfg = ZoneConfig.model_validate(json.loads(zones))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise HTTPException(422, {"error_code": "invalid_zones", "message": str(exc)[:300]})
        jobs: JobStore = state["jobs"]
        aid, workdir = jobs.new_workdir()
        dest = workdir / f"input{ext}"
        size = 0
        try:
            with open(dest, "wb") as out:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > cap:
                        raise HTTPException(413, {"error_code": "upload_too_large", "message": f"limit is {cap} bytes"})
                    out.write(chunk)
            if size == 0:
                raise HTTPException(422, {"error_code": "empty_upload", "message": "empty file"})
            try:
                probe(dest, settings.limits)
            except MediaError as exc:
                raise HTTPException(422, {"error_code": exc.code, "message": str(exc)[:300]})
            job = jobs.submit(aid, workdir, dest, file.filename or f"upload{ext}", zcfg, mods)
        except HTTPException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        except OverflowError as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(503, {"error_code": "job_store_full", "message": str(exc)})
        return SubmitResponse(analysis_id=job.analysis_id, status="queued", status_url=f"{API_PREFIX}/analyses/{aid}",
                              result_url=f"{API_PREFIX}/analyses/{aid}/result")

    def _job(aid: str):
        j = state["jobs"].get(aid)
        if j is None:
            raise HTTPException(404, {"error_code": "not_found", "message": "unknown or deleted analysis"})
        return j

    @app.get(f"{API_PREFIX}/analyses/{{analysis_id}}", response_model=StatusResponse, tags=["analyses"],
             responses={404: {"model": ErrorBody}}, dependencies=[Depends(auth)])
    def status(analysis_id: str) -> StatusResponse:
        j = _job(analysis_id)
        return StatusResponse(analysis_id=j.analysis_id, status=j.status, created_at=j.created_at, updated_at=j.updated_at, error=j.error)

    @app.get(f"{API_PREFIX}/analyses/{{analysis_id}}/result", response_model=AnalysisResult, tags=["analyses"],
             responses={404: {"model": ErrorBody}, 409: {"model": ErrorBody}, 422: {"model": AnalysisError}}, dependencies=[Depends(auth)])
    def result(analysis_id: str):
        j = _job(analysis_id)
        if j.status in ("queued", "running"):
            raise HTTPException(409, {"error_code": "not_ready", "message": f"analysis is {j.status}"})
        if j.status == "failed":
            return JSONResponse(status_code=422, content=j.error.model_dump(mode="json"))
        return j.result

    @app.delete(f"{API_PREFIX}/analyses/{{analysis_id}}", response_model=DeleteResponse, tags=["analyses"],
                responses={404: {"model": ErrorBody}}, dependencies=[Depends(auth)])
    def delete(analysis_id: str) -> DeleteResponse:
        st = state["jobs"].delete(analysis_id)
        if st == "not_found":
            raise HTTPException(404, {"error_code": "not_found", "message": "unknown or deleted analysis"})
        return DeleteResponse(analysis_id=analysis_id, status=st)

    return app
