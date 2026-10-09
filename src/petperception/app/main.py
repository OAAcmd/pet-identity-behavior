"""Application factory: API + persistent jobs + static PWA on one origin."""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..config import REPO_ROOT, Settings
from ..registry import check_model, default_model_ids
from .api import build_router
from .db import Database
from .errors import ApiError, ErrorBody, ErrorDetail
from .logs import configure, request_id_var, user_ref
from .store import Store
from .worker import JobRunner

log = logging.getLogger("petperception.http")

SECURITY_HEADERS = {
    "Content-Security-Policy": ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
                                "media-src 'self' blob:; connect-src 'self'; worker-src 'self'; manifest-src 'self'; "
                                "frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(self), microphone=(self), geolocation=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


def _engine_factory(settings: Settings) -> Callable[[], object]:
    def make():
        from ..pipeline import AnalysisEngine

        return AnalysisEngine(settings).load()

    return make


def create_app(settings: Settings, engine_factory: Callable[[], object] | None = None, start_workers: bool | None = None) -> FastAPI:
    configure(settings.app.log_format, settings.app.log_level)
    db = Database(settings.app.storage.resolve(settings.app.storage.database_path))
    db.migrate()
    store = Store(settings, db)
    factory = engine_factory or _engine_factory(settings)
    state: dict = {"runners": [], "caps_engine": None}
    n_workers = settings.app.jobs.workers if start_workers is None else (settings.app.jobs.workers if start_workers else 0)

    def caps_engine():
        if state["caps_engine"] is None:
            state["caps_engine"] = factory()
        return state["caps_engine"]

    def engine_caps() -> dict:
        eng = caps_engine()
        return eng.capabilities() if hasattr(eng, "capabilities") else {"engine_version": __version__}

    def readiness() -> dict:
        checks: dict[str, dict] = {}
        try:
            db.one("SELECT 1")
            checks["database"] = {"ok": True, "schema_version": db.schema_version()}
        except Exception as exc:  # noqa: BLE001
            checks["database"] = {"ok": False, "error": type(exc).__name__}
        models = {mid: check_model(settings, mid) for mid in default_model_ids(settings)}
        checks["models"] = {"ok": all(m.ready for m in models.values()), "detail": {k: m.status for k, m in models.items()}}
        free = shutil.disk_usage(store.media_root).free
        checks["disk"] = {"ok": free >= settings.app.storage.min_free_disk_bytes, "free_bytes": free}
        if n_workers > 0 or settings.app.jobs.workers > 0:
            hb = db.one("SELECT max(last_seen) AS t FROM worker_heartbeats")
            age = time.time() - hb["t"] if hb and hb["t"] else None
            checks["worker"] = {"ok": age is not None and age < 3 * settings.app.jobs.heartbeat_s + 5, "last_heartbeat_age_s": age}
        queued = db.one("SELECT count(*) AS n FROM analyses WHERE status = 'queued'")["n"]
        checks["queue"] = {"ok": queued < settings.app.jobs.max_queued_total, "queued": queued}
        return {"ready": all(c["ok"] for c in checks.values()), "checks": checks, "engine_version": __version__}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for i in range(n_workers):
            runner = JobRunner(settings, db, store, factory, worker_id=f"inproc-{uuid.uuid4().hex[:8]}-{i}")
            runner.heartbeat()
            runner.start()
            state["runners"].append(runner)
        log.info("startup", extra={"workers": n_workers, "db_schema": db.schema_version(), "release_profile": settings.release_profile})
        yield
        for runner in state["runners"]:
            runner.stop()
        log.info("shutdown")

    app = FastAPI(title="Pet Perception API", version=__version__, lifespan=lifespan,
                  description=("Observation-first cat/dog behavior evidence service. Results separate observations, experimental "
                               "uncalibrated hypotheses and restrained suggestions. Not a translator, emotion reader or diagnostic tool."),
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.state.store = store
    app.state.db = db
    app.state.runners = state["runners"]

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        rid = rid if rid and len(rid) <= 64 and rid.replace("-", "").isalnum() else uuid.uuid4().hex
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            route = request.scope.get("route")
            log.info("request", extra={"method": request.method, "path": getattr(route, "path", "unmatched"), "status": status,
                                       "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
                                       "user": user_ref(getattr(request.state, "user_id", None))})
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    def err(status: int, code: str, message: str, action: str | None = None, details: dict | None = None, headers: dict | None = None):
        body = ErrorBody(error=ErrorDetail(code=code, message=message, action=action, details=details))
        return JSONResponse(status_code=status, content=body.model_dump(), headers=headers or {})

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status, content=exc.body.model_dump(), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        details = [{"loc": [str(x) for x in e.get("loc", [])], "msg": e.get("msg")} for e in exc.errors()][:20]
        return err(422, "invalid_request", "Some fields are missing or invalid.", "Correct the highlighted fields.", {"errors": details})

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return err(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled_error")
        return err(500, "internal_error", "Something went wrong on the server.", f"Retry; report request ID {request_id_var.get()} to the operator.")

    app.include_router(build_router(store, engine_caps, readiness), prefix="/api/v1")

    static = Path(settings.serve.static_dir)
    static = static if static.is_absolute() else REPO_ROOT / static

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        if path.startswith("api/"):
            return err(404, "not_found", "Unknown API path.")
        if not (static / "index.html").exists():
            return err(503, "frontend_not_built", "The web app has not been built.", "Run `npm --prefix web ci && npm --prefix web run build`.")
        target = (static / path).resolve()
        if path and target.is_file() and str(target).startswith(str(static.resolve())):
            headers = {"Cache-Control": "public, max-age=31536000, immutable"} if "/assets/" in f"/{path}" else {"Cache-Control": "no-cache"}
            if path == "sw.js":
                headers["Service-Worker-Allowed"] = "/"
            return FileResponse(target, headers=headers)
        return FileResponse(static / "index.html", headers={"Cache-Control": "no-cache"})

    return app
