"""HTTP API (mounted at /api/v1)."""

from __future__ import annotations

import hmac
import io
import json
import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..context.zones import ZoneConfig
from ..schemas.analysis import SCHEMA_VERSION
from . import security
from .db import now_iso
from .errors import ApiError, ErrorBody
from .store import Principal, Store

MEDIA_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska",
               ".avi": "video/x-msvideo", ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg", ".oga": "audio/ogg",
               ".opus": "audio/ogg", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".aac": "audio/aac", ".jpg": "image/jpeg",
               ".jpeg": "image/jpeg", ".png": "image/png"}
ERRORS: dict[int | str, dict[str, Any]] = {c: {"model": ErrorBody} for c in (400, 401, 403, 404, 409, 413, 415, 422, 429, 503, 507)}


# ------------------------------------------------------------------- models
class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class Me(BaseModel):
    id: str
    username: str
    is_admin: bool
    csrf_token: str
    retain_media_days: int
    feedback_research_consent: bool


class PasswordChange(BaseModel):
    current_password: str = Field(max_length=256)
    new_password: str = Field(max_length=256)


class AccountDelete(BaseModel):
    password: str = Field(max_length=256)
    confirm: Literal["DELETE"]


class SettingsIn(BaseModel):
    retain_media_days: int | None = None
    feedback_research_consent: bool | None = None


class PetIn(BaseModel):
    name: str = Field(max_length=60)
    species: Literal["cat", "dog"]
    birth_year: int | None = None
    notes: str = Field("", max_length=1000)


class PetPatch(BaseModel):
    name: str | None = Field(None, max_length=60)
    species: Literal["cat", "dog"] | None = None
    birth_year: int | None = None
    notes: str | None = Field(None, max_length=1000)


class Pet(BaseModel):
    id: str
    name: str
    species: str
    birth_year: int | None
    notes: str
    created_at: str
    updated_at: str


class PlaceIn(BaseModel):
    name: str = Field(max_length=60)
    zones: ZoneConfig


class PlacePatch(BaseModel):
    name: str | None = Field(None, max_length=60)
    zones: ZoneConfig | None = None


class UploadIn(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    size_bytes: int
    content_type: str = Field("application/octet-stream", max_length=100)
    sha256: str | None = None


class Upload(BaseModel):
    id: str
    filename: str
    size_bytes: int
    received_bytes: int
    status: str
    max_chunk_bytes: int
    expires_at: float


class AnalysisIn(BaseModel):
    upload_id: str
    subject_mode: Literal["unknown", "single_known", "multiple_known"] = "unknown"
    pet_ids: list[str] = Field(default_factory=list, max_length=20)
    place_id: str | None = None
    zones: ZoneConfig | None = None
    modalities: list[Literal["audio", "video"]] = Field(default_factory=lambda: ["audio", "video"])
    note: str = Field("", max_length=500)


class DeviceReportIn(BaseModel):
    report: dict[str, Any]


# Opt-in backup of on-device results (JSON only).
SYNC_MAX_RESULTS = 2000
SYNC_MAX_RESULT_BYTES = 3_000_000


class SyncedResultIn(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    created_at: str = Field(min_length=1, max_length=40)
    bundle_version: str | None = Field(None, max_length=64)
    result: dict[str, Any]
    feedback: list[dict[str, Any]] = Field(default_factory=list, max_length=200)


class FeedbackIn(BaseModel):
    target_type: Literal["overall", "hypothesis", "observation", "sound_event", "track"]
    target_id: str | None = Field(None, max_length=64)
    verdict: Literal["agree", "disagree", "unsure"]
    actual_behavior: str | None = Field(None, max_length=120)
    note: str = Field("", max_length=1000)
    consent_research_use: bool = False


def build_router(store: Store, engine_caps, readiness) -> APIRouter:
    s = store.s
    r = APIRouter(responses=ERRORS)

    # ------------------------------------------------------------- deps
    def principal(request: Request) -> Principal:
        p = store.resolve_session(request.cookies.get(s.app.auth.cookie_name))
        if p is None:
            raise ApiError(401, "not_signed_in", "You are not signed in or your session expired.", "Sign in again.")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            sent = request.headers.get("x-csrf-token", "")
            if not hmac.compare_digest(sent, p.csrf_token):
                raise ApiError(403, "csrf_failed", "The request is missing a valid CSRF token.", "Reload the page and try again.")
        request.state.user_id = p.user_id
        return p

    def admin(p: Principal = Depends(principal)) -> Principal:
        if not p.is_admin:
            raise ApiError(403, "forbidden", "Administrator rights are required.")
        return p

    def client_ip(request: Request) -> str:
        hops = s.serve.trusted_proxy_hops
        if hops > 0:
            fwd = [x.strip() for x in request.headers.get("x-forwarded-for", "").split(",") if x.strip()]
            if len(fwd) >= hops:
                return fwd[-hops]
        return request.client.host if request.client else "unknown"

    def set_cookie(resp: Response, token: str) -> None:
        resp.set_cookie(s.app.auth.cookie_name, token, max_age=int(s.app.auth.session_ttl_hours * 3600), httponly=True,
                        secure=s.app.auth.cookie_secure, samesite="strict", path="/")

    def me_payload(p: Principal) -> Me:
        u = store.get_user(p.user_id)
        return Me(id=u["id"], username=u["username"], is_admin=bool(u["is_admin"]), csrf_token=p.csrf_token,
                  retain_media_days=u["retain_media_days"], feedback_research_consent=bool(u["feedback_research_consent"]))

    # ----------------------------------------------------------- service
    @r.get("/health", tags=["service"])
    def health() -> dict:
        return {"status": "ok", "engine_version": __version__, "schema_version": SCHEMA_VERSION}

    @r.get("/ready", tags=["service"])
    def ready(response: Response) -> dict:
        out = readiness()
        if not out["ready"]:
            response.status_code = 503
        return out

    @r.get("/capabilities", tags=["service"])
    def capabilities(p: Principal = Depends(principal)) -> dict:
        caps = engine_caps()
        st = s.app.storage
        caps["upload"] = {"max_bytes": st.max_upload_bytes, "max_chunk_bytes": st.max_chunk_bytes, "allowed_extensions": s.serve.allowed_extensions,
                          "max_duration_s": s.limits.max_duration_s, "url_fetching": False, "resumable": True}
        caps["retention"] = {"max_retain_media_days": st.max_retain_media_days, "default_retain_media_days": 0}
        caps["registration"] = s.app.auth.registration
        return caps

    # -------------------------------------------------------------- auth
    @r.get("/auth/status", tags=["auth"])
    def auth_status(request: Request) -> dict:
        p = store.resolve_session(request.cookies.get(s.app.auth.cookie_name))
        mode = s.app.auth.registration
        can_register = mode == "open" or (mode == "first_user_only" and store.user_count() == 0)
        return {"signed_in": p is not None, "registration_open": can_register}

    @r.post("/auth/register", tags=["auth"], status_code=201, response_model=Me)
    def register(body: Credentials, response: Response) -> Me:
        mode = s.app.auth.registration
        first = store.user_count() == 0
        if mode == "closed" or (mode == "first_user_only" and not first):
            raise ApiError(403, "registration_closed", "Registration is closed.", "Ask the administrator to create an account for you.")
        u = store.create_user(body.username, body.password, is_admin=first)
        token, csrf = store.create_session(u["id"])
        set_cookie(response, token)
        return me_payload(Principal(u["id"], u["username"], bool(u["is_admin"]), csrf, ""))

    @r.post("/auth/login", tags=["auth"], response_model=Me)
    def login(body: Credentials, request: Request, response: Response) -> Me:
        u = store.authenticate(body.username, body.password, client_ip(request))
        token, csrf = store.create_session(u["id"])
        set_cookie(response, token)
        return me_payload(Principal(u["id"], u["username"], bool(u["is_admin"]), csrf, ""))

    @r.post("/auth/logout", tags=["auth"], status_code=204)
    def logout(response: Response, p: Principal = Depends(principal)) -> Response:
        store.delete_session(p.session_hash)
        response.delete_cookie(s.app.auth.cookie_name, path="/")
        response.status_code = 204
        return response

    @r.get("/auth/me", tags=["auth"], response_model=Me)
    def me(p: Principal = Depends(principal)) -> Me:
        return me_payload(p)

    @r.post("/auth/password", tags=["auth"], status_code=204)
    def change_password(body: PasswordChange, request: Request, p: Principal = Depends(principal)) -> Response:
        store.authenticate(p.username, body.current_password, client_ip(request))
        store.set_password(p.user_id, body.new_password, keep_session_hash=p.session_hash)
        return Response(status_code=204)

    @r.post("/auth/delete-account", tags=["auth"], status_code=204)
    def delete_account(body: AccountDelete, request: Request, response: Response, p: Principal = Depends(principal)) -> Response:
        store.authenticate(p.username, body.password, client_ip(request))
        store.delete_account(p.user_id)
        response.delete_cookie(s.app.auth.cookie_name, path="/")
        response.status_code = 204
        return response

    @r.get("/admin/users", tags=["admin"])
    def admin_users(p: Principal = Depends(admin)) -> list[dict]:
        return store.list_users()

    @r.post("/admin/users", tags=["admin"], status_code=201)
    def admin_create_user(body: Credentials, p: Principal = Depends(admin)) -> dict:
        u = store.create_user(body.username, body.password, is_admin=False)
        return {"id": u["id"], "username": u["username"]}

    # ---------------------------------------------------------- settings
    @r.patch("/settings", tags=["settings"], response_model=Me)
    def update_settings(body: SettingsIn, p: Principal = Depends(principal)) -> Me:
        store.update_settings(p.user_id, body.retain_media_days, body.feedback_research_consent)
        return me_payload(p)

    @r.get("/export", tags=["settings"])
    def export(p: Principal = Depends(principal)) -> Response:
        data = store.export_user(p.user_id)
        return Response(json.dumps(data, indent=2), media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="petperception-export.json"'})

    # -------------------------------------------------------------- pets
    @r.get("/pets", tags=["pets"], response_model=list[Pet])
    def list_pets(p: Principal = Depends(principal)) -> list[dict]:
        return store.list_pets(p.user_id)

    @r.post("/pets", tags=["pets"], status_code=201, response_model=Pet)
    def create_pet(body: PetIn, p: Principal = Depends(principal)) -> dict:
        return store.create_pet(p.user_id, body.name, body.species, body.birth_year, body.notes)

    @r.get("/pets/{pet_id}", tags=["pets"], response_model=Pet)
    def get_pet(pet_id: str, p: Principal = Depends(principal)) -> dict:
        return store.get_pet(p.user_id, pet_id)

    @r.patch("/pets/{pet_id}", tags=["pets"], response_model=Pet)
    def update_pet(pet_id: str, body: PetPatch, p: Principal = Depends(principal)) -> dict:
        return store.update_pet(p.user_id, pet_id, **body.model_dump())

    @r.delete("/pets/{pet_id}", tags=["pets"], status_code=204)
    def delete_pet(pet_id: str, p: Principal = Depends(principal)) -> Response:
        store.delete_pet(p.user_id, pet_id)
        return Response(status_code=204)

    # ------------------------------------------------------------ places
    @r.get("/places", tags=["places"])
    def list_places(p: Principal = Depends(principal)) -> list[dict]:
        return store.list_places(p.user_id)

    @r.post("/places", tags=["places"], status_code=201)
    def create_place(body: PlaceIn, p: Principal = Depends(principal)) -> dict:
        return store.create_place(p.user_id, body.name, body.zones)

    @r.patch("/places/{place_id}", tags=["places"])
    def update_place(place_id: str, body: PlacePatch, p: Principal = Depends(principal)) -> dict:
        return store.update_place(p.user_id, place_id, body.name, body.zones)

    @r.delete("/places/{place_id}", tags=["places"], status_code=204)
    def delete_place(place_id: str, p: Principal = Depends(principal)) -> Response:
        store.delete_place(p.user_id, place_id)
        return Response(status_code=204)

    # ----------------------------------------------------------- uploads
    def upload_out(u: dict) -> Upload:
        return Upload(id=u["id"], filename=u["filename"], size_bytes=u["size_bytes"], received_bytes=u["received_bytes"], status=u["status"],
                      max_chunk_bytes=s.app.storage.max_chunk_bytes, expires_at=u["expires_at"])

    @r.post("/uploads", tags=["uploads"], status_code=201, response_model=Upload)
    def create_upload(body: UploadIn, p: Principal = Depends(principal)) -> Upload:
        return upload_out(store.create_upload(p.user_id, body.filename, body.size_bytes, body.content_type, body.sha256))

    @r.get("/uploads/{upload_id}", tags=["uploads"], response_model=Upload)
    def get_upload(upload_id: str, p: Principal = Depends(principal)) -> Upload:
        return upload_out(store.get_upload(p.user_id, upload_id))

    @r.put("/uploads/{upload_id}", tags=["uploads"], response_model=Upload,
           openapi_extra={"requestBody": {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}, "required": True}})
    async def put_chunk(upload_id: str, request: Request, offset: int = Query(..., ge=0), p: Principal = Depends(principal)) -> Upload:
        cl = request.headers.get("content-length")
        length = int(cl) if cl and cl.isdigit() else None
        if length is not None and length > s.app.storage.max_chunk_bytes:
            raise ApiError(413, "chunk_too_large", f"Chunks may be at most {s.app.storage.max_chunk_bytes} bytes.", "Send smaller chunks.")
        buf = bytearray()
        async for part in request.stream():  # bounded read: never buffers more than one chunk
            buf.extend(part)
            if len(buf) > s.app.storage.max_chunk_bytes:
                raise ApiError(413, "chunk_too_large", f"Chunks may be at most {s.app.storage.max_chunk_bytes} bytes.", "Send smaller chunks.")
        return upload_out(store.append_chunk(p.user_id, upload_id, offset, io.BytesIO(bytes(buf)), len(buf)))

    @r.delete("/uploads/{upload_id}", tags=["uploads"], status_code=204)
    def abort_upload(upload_id: str, p: Principal = Depends(principal)) -> Response:
        store.abort_upload(p.user_id, upload_id)
        return Response(status_code=204)

    # ---------------------------------------------------------- analyses
    @r.post("/analyses", tags=["analyses"], status_code=202)
    def create_analysis(body: AnalysisIn, p: Principal = Depends(principal)) -> dict:
        return store.create_analysis(p.user_id, body.upload_id, body.pet_ids, body.subject_mode, body.place_id, body.zones, list(body.modalities), body.note)

    @r.get("/analyses", tags=["analyses"])
    def list_analyses(pet_id: str | None = None, limit: int = Query(30, ge=1, le=100), before: str | None = None,
                      p: Principal = Depends(principal)) -> list[dict]:
        return store.list_analyses(p.user_id, pet_id, limit, before)

    @r.get("/analyses/{analysis_id}", tags=["analyses"])
    def get_analysis(analysis_id: str, include_result: bool = True, p: Principal = Depends(principal)) -> dict:
        return store.get_analysis(p.user_id, analysis_id, with_result=include_result)

    @r.post("/analyses/{analysis_id}/cancel", tags=["analyses"])
    def cancel_analysis(analysis_id: str, p: Principal = Depends(principal)) -> dict:
        return store.cancel_analysis(p.user_id, analysis_id)

    @r.post("/analyses/{analysis_id}/retry", tags=["analyses"])
    def retry_analysis(analysis_id: str, p: Principal = Depends(principal)) -> dict:
        return store.retry_analysis(p.user_id, analysis_id)

    @r.delete("/analyses/{analysis_id}", tags=["analyses"], status_code=204)
    def delete_analysis(analysis_id: str, p: Principal = Depends(principal)) -> Response:
        store.delete_analysis(p.user_id, analysis_id)
        return Response(status_code=204)

    @r.get("/analyses/{analysis_id}/media", tags=["analyses"], response_class=FileResponse)
    def get_media(analysis_id: str, p: Principal = Depends(principal)):
        path, ext = store.media_file(p.user_id, analysis_id)
        return FileResponse(path, media_type=MEDIA_TYPES.get(ext, "application/octet-stream"), headers={"Cache-Control": "private, no-store"})

    @r.delete("/analyses/{analysis_id}/media", tags=["analyses"])
    def delete_media(analysis_id: str, p: Principal = Depends(principal)) -> dict:
        return store.delete_media(p.user_id, analysis_id)

    @r.get("/analyses/{analysis_id}/feedback", tags=["feedback"])
    def list_feedback(analysis_id: str, p: Principal = Depends(principal)) -> list[dict]:
        return store.list_feedback(p.user_id, analysis_id)

    @r.post("/analyses/{analysis_id}/feedback", tags=["feedback"], status_code=201)
    def add_feedback(analysis_id: str, body: FeedbackIn, p: Principal = Depends(principal)) -> dict:
        return store.add_feedback(p.user_id, analysis_id, body.target_type, body.target_id, body.verdict, body.actual_behavior, body.note,
                                  body.consent_research_use)

    @r.post("/device-reports", tags=["device"], status_code=201)
    def add_device_report(body: DeviceReportIn, p: Principal = Depends(principal)) -> dict:
        """Store a device-check report the user chose to send (no media; size-limited)."""
        blob = json.dumps(body.report)
        if len(blob) > 200_000:
            raise ApiError(413, "report_too_large", "The report is too large.", "Send the report file instead.")
        rid = security.new_id()
        rep = body.report
        env = rep.get("environment") if isinstance(rep.get("environment"), dict) else {}
        with store.db.tx(immediate=True) as c:
            c.execute("INSERT INTO device_reports (id, user_id, created_at, bundle_version, passed, user_agent, report_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (rid, p.user_id, now_iso(), str(rep.get("bundle_version", ""))[:64], int(bool(rep.get("passed"))), str(env.get("user_agent", ""))[:400], blob))
        return {"id": rid}

    @r.get("/device-reports", tags=["device"])
    def list_device_reports(p: Principal = Depends(principal)) -> list[dict]:
        rows = store.db.query("SELECT id, created_at, bundle_version, passed, user_agent FROM device_reports WHERE user_id = ? ORDER BY created_at DESC LIMIT 50",
                            (p.user_id,))
        return [dict(r) | {"passed": bool(r["passed"])} for r in rows]

    @r.put("/synced-results/{local_id}", tags=["sync"])
    def put_synced_result(local_id: str, body: SyncedResultIn, p: Principal = Depends(principal)) -> dict:
        """Back up an on-device analysis result (opt-in). Result JSON and feedback only - the
        recording never leaves the device. The result must validate against the result schema."""
        from pydantic import ValidationError

        from ..schemas.analysis import AnalysisResult

        if not re.fullmatch(r"dev[0-9a-f]{24}", local_id):
            raise ApiError(422, "invalid_id", "Unknown on-device analysis id format.")
        try:
            res = AnalysisResult.model_validate(body.result)
        except ValidationError as exc:
            raise ApiError(422, "invalid_result", "The result does not match the result schema.", None, {"errors": len(exc.errors())}) from None
        if res.analysis_id != local_id:
            raise ApiError(422, "id_mismatch", "The result belongs to a different analysis.")
        blob, fb = res.model_dump_json(), json.dumps(body.feedback)
        if len(blob) > SYNC_MAX_RESULT_BYTES or len(fb) > 100_000:
            raise ApiError(413, "result_too_large", "This result is too large to back up.")
        with store.db.tx(immediate=True) as c:
            n = c.execute("SELECT count(*) FROM synced_results WHERE user_id = ? AND id != ?", (p.user_id, local_id)).fetchone()[0]
            if n >= SYNC_MAX_RESULTS:
                raise ApiError(507, "sync_quota_exceeded", f"At most {SYNC_MAX_RESULTS} results can be backed up.", "Delete old backed-up results first.")
            c.execute("INSERT INTO synced_results (id, user_id, created_at, synced_at, filename, bundle_version, decision_state, result_json, feedback_json, bytes) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(user_id, id) DO UPDATE SET synced_at = excluded.synced_at, filename = excluded.filename, "
                      "bundle_version = excluded.bundle_version, decision_state = excluded.decision_state, result_json = excluded.result_json, "
                      "feedback_json = excluded.feedback_json, bytes = excluded.bytes",
                      (local_id, p.user_id, body.created_at, now_iso(), body.filename, body.bundle_version, res.decision.state, blob, fb, len(blob) + len(fb)))
        return {"id": local_id, "synced": True}

    @r.get("/synced-results", tags=["sync"])
    def list_synced_results(p: Principal = Depends(principal)) -> list[dict]:
        rows = store.db.query("SELECT id, created_at, synced_at, filename, bundle_version, decision_state, bytes FROM synced_results "
                              "WHERE user_id = ? ORDER BY created_at DESC", (p.user_id,))
        return [dict(r) for r in rows]

    @r.get("/synced-results/{local_id}", tags=["sync"])
    def get_synced_result(local_id: str, p: Principal = Depends(principal)) -> dict:
        row = store.db.one("SELECT * FROM synced_results WHERE user_id = ? AND id = ?", (p.user_id, local_id))
        if row is None:
            raise ApiError(404, "not_found", "No backed-up result with this id.")
        return {"id": row["id"], "created_at": row["created_at"], "synced_at": row["synced_at"], "filename": row["filename"],
                "bundle_version": row["bundle_version"], "result": json.loads(row["result_json"]), "feedback": json.loads(row["feedback_json"])}

    @r.delete("/synced-results/{local_id}", tags=["sync"], status_code=204)
    def delete_synced_result(local_id: str, p: Principal = Depends(principal)) -> Response:
        if store.db.execute("DELETE FROM synced_results WHERE user_id = ? AND id = ?", (p.user_id, local_id)) == 0:
            raise ApiError(404, "not_found", "No backed-up result with this id.")
        return Response(status_code=204)

    @r.delete("/feedback/{feedback_id}", tags=["feedback"], status_code=204)
    def delete_feedback(feedback_id: str, p: Principal = Depends(principal)) -> Response:
        store.delete_feedback(p.user_id, feedback_id)
        return Response(status_code=204)

    return r
