"""Repository/service layer. Every user-owned query is scoped by ``user_id``.

Foreign objects are indistinguishable from missing ones (404), so IDs of
other users' data cannot be probed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from ..config import Settings
from ..context.zones import ZoneConfig
from ..ingest.media import MediaError, probe
from . import security
from .db import Database, now_iso, row
from .errors import MEDIA_ERROR_TEXT, ApiError, not_found
from .logs import user_ref

log = logging.getLogger("petperception.store")

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
RETRYABLE = {"internal_error", "worker_lost", "analysis_timeout"}
ACTIVE = ("queued", "running")


@dataclass
class Principal:
    user_id: str
    username: str
    is_admin: bool
    csrf_token: str
    session_hash: str


class Store:
    def __init__(self, settings: Settings, db: Database):
        self.s = settings
        self.db = db
        self.media_root = settings.app.storage.resolve(settings.app.storage.media_root)
        self.media_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.media_root, 0o700)
        self.research_root = settings.app.storage.resolve(settings.app.storage.research_root)
        self.research_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.research_root, 0o700)
        self._upload_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------ audit
    def audit(self, user_id: str | None, action: str, object_type: str | None = None, object_id: str | None = None) -> None:
        self.db.execute("INSERT INTO audit_log (user_id, action, object_type, object_id, at) VALUES (?, ?, ?, ?, ?)",
                        (user_ref(user_id), action, object_type, object_id, now_iso()))

    # ------------------------------------------------------------------ users
    def user_count(self) -> int:
        return self.db.one("SELECT count(*) AS n FROM users")["n"]

    def create_user(self, username: str, password: str, is_admin: bool = False) -> dict:
        if not USERNAME_RE.match(username):
            raise ApiError(422, "invalid_username", "Username must be 3-32 characters: letters, digits, '.', '_' or '-'.")
        if len(password) < self.s.app.auth.min_password_length:
            raise ApiError(422, "weak_password", f"Password must have at least {self.s.app.auth.min_password_length} characters.")
        if len(password) > 256:
            raise ApiError(422, "weak_password", "Password is too long (max 256 characters).")
        uid, ts = security.new_id(), now_iso()
        try:
            with self.db.tx() as c:
                if c.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
                    raise ApiError(409, "username_taken", "That username is already in use.", "Choose another username.")
                c.execute("INSERT INTO users (id, username, password_hash, is_admin, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                          (uid, username, security.hash_password(password), int(is_admin), ts, ts))
        except ApiError:
            raise
        self.audit(uid, "user_created")
        return self.get_user(uid)

    def get_user(self, user_id: str) -> dict:
        r = row(self.db.one("SELECT id, username, is_admin, disabled, retain_media_days, feedback_research_consent, created_at FROM users WHERE id = ?", (user_id,)))
        if r is None:
            raise not_found("user")
        return r

    def list_users(self) -> list[dict]:
        return [dict(r) for r in self.db.query("SELECT id, username, is_admin, disabled, created_at FROM users ORDER BY created_at")]

    def set_disabled(self, username: str, disabled: bool) -> None:
        with self.db.tx() as c:
            r = c.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if r is None:
                raise not_found("user")
            c.execute("UPDATE users SET disabled = ?, updated_at = ? WHERE id = ?", (int(disabled), now_iso(), r["id"]))
            if disabled:
                c.execute("DELETE FROM sessions WHERE user_id = ?", (r["id"],))

    def set_password(self, user_id: str, new_password: str, keep_session_hash: str | None = None) -> None:
        if len(new_password) < self.s.app.auth.min_password_length:
            raise ApiError(422, "weak_password", f"Password must have at least {self.s.app.auth.min_password_length} characters.")
        with self.db.tx() as c:
            c.execute("UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?", (security.hash_password(new_password), now_iso(), user_id))
            c.execute("DELETE FROM sessions WHERE user_id = ? AND token_hash IS NOT ?", (user_id, keep_session_hash))
        self.audit(user_id, "password_changed")

    def _rate_key(self, kind: str, value: str) -> str:
        return f"{kind}:{hashlib.sha256(value.lower().encode()).hexdigest()[:24]}"

    def check_login_allowed(self, username: str, client_ip: str) -> None:
        a = self.s.app.auth
        since = time.time() - a.login_window_s
        for key in (self._rate_key("u", username), self._rate_key("ip", client_ip)):
            n = self.db.one("SELECT count(*) AS n FROM login_failures WHERE key = ? AND at > ?", (key, since))["n"]
            if n >= a.login_max_failures:
                raise ApiError(429, "too_many_attempts", "Too many failed sign-in attempts.",
                               f"Wait {int(a.login_window_s // 60)} minutes and try again.", headers={"Retry-After": str(int(a.login_window_s))})

    def authenticate(self, username: str, password: str, client_ip: str) -> dict:
        self.check_login_allowed(username, client_ip)
        r = row(self.db.one("SELECT * FROM users WHERE username = ?", (username,)))
        ok = security.verify_password(password, r["password_hash"] if r else security.DUMMY_HASH)
        if not r or not ok or r["disabled"]:
            now = time.time()
            for key in (self._rate_key("u", username), self._rate_key("ip", client_ip)):
                self.db.execute("INSERT INTO login_failures (key, at) VALUES (?, ?)", (key, now))
            self.audit(r["id"] if r else None, "login_failed")
            raise ApiError(401, "invalid_credentials", "Username or password is incorrect.")
        self.db.execute("DELETE FROM login_failures WHERE key = ?", (self._rate_key("u", username),))
        self.audit(r["id"], "login")
        return r

    # --------------------------------------------------------------- sessions
    def create_session(self, user_id: str) -> tuple[str, str]:
        token, csrf = security.new_token(), security.new_token()
        now = time.time()
        self.db.execute("INSERT INTO sessions (token_hash, user_id, csrf_token, created_at, expires_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (security.token_hash(token), user_id, csrf, now, now + self.s.app.auth.session_ttl_hours * 3600, now))
        return token, csrf

    def resolve_session(self, token: str | None) -> Principal | None:
        if not token or len(token) > 128:
            return None
        h = security.token_hash(token)
        r = self.db.one("SELECT s.*, u.username, u.is_admin, u.disabled FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?", (h,))
        if r is None:
            return None
        now = time.time()
        if r["expires_at"] < now or now - r["last_seen_at"] > self.s.app.auth.session_idle_hours * 3600 or r["disabled"]:
            self.db.execute("DELETE FROM sessions WHERE token_hash = ?", (h,))
            return None
        if now - r["last_seen_at"] > 60:
            self.db.execute("UPDATE sessions SET last_seen_at = ? WHERE token_hash = ?", (now, h))
        return Principal(r["user_id"], r["username"], bool(r["is_admin"]), r["csrf_token"], h)

    def delete_session(self, session_hash: str) -> None:
        self.db.execute("DELETE FROM sessions WHERE token_hash = ?", (session_hash,))

    # --------------------------------------------------------------- settings
    def update_settings(self, user_id: str, retain_media_days: int | None, consent: bool | None) -> dict:
        if retain_media_days is not None and not (0 <= retain_media_days <= self.s.app.storage.max_retain_media_days):
            raise ApiError(422, "invalid_retention", f"Retention must be between 0 and {self.s.app.storage.max_retain_media_days} days.")
        with self.db.tx() as c:
            if retain_media_days is not None:
                c.execute("UPDATE users SET retain_media_days = ?, updated_at = ? WHERE id = ?", (retain_media_days, now_iso(), user_id))
            if consent is not None:
                c.execute("UPDATE users SET feedback_research_consent = ?, updated_at = ? WHERE id = ?", (int(consent), now_iso(), user_id))
        return self.get_user(user_id)

    def delete_account(self, user_id: str) -> None:
        with self.db.tx() as c:
            c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        shutil.rmtree(self.user_dir(user_id), ignore_errors=True)
        shutil.rmtree(self.research_root / user_id, ignore_errors=True)
        self.audit(user_id, "account_deleted")

    # ------------------------------------------------------------------- pets
    def _pet_payload(self, name: str, species: str, birth_year: int | None, notes: str) -> tuple:
        name = name.strip()
        if not 1 <= len(name) <= 60:
            raise ApiError(422, "invalid_name", "Name must be 1-60 characters.")
        if species not in ("cat", "dog"):
            raise ApiError(422, "invalid_species", "Species must be 'cat' or 'dog'.")
        if birth_year is not None and not (1980 <= birth_year <= time.gmtime().tm_year):
            raise ApiError(422, "invalid_birth_year", "Birth year is out of range.")
        if len(notes) > 1000:
            raise ApiError(422, "invalid_notes", "Notes are limited to 1000 characters.")
        return name, species, birth_year, notes

    def create_pet(self, user_id: str, name: str, species: str, birth_year: int | None, notes: str) -> dict:
        n, sp, by, nt = self._pet_payload(name, species, birth_year, notes)
        if self.db.one("SELECT count(*) AS n FROM pets WHERE user_id = ?", (user_id,))["n"] >= 50:
            raise ApiError(422, "too_many_pets", "A maximum of 50 pet profiles is supported.")
        pid, ts = security.new_id(), now_iso()
        self.db.execute("INSERT INTO pets (id, user_id, name, species, birth_year, notes, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (pid, user_id, n, sp, by, nt, ts, ts))
        return self.get_pet(user_id, pid)

    def get_pet(self, user_id: str, pet_id: str) -> dict:
        r = row(self.db.one("SELECT id, name, species, birth_year, notes, created_at, updated_at FROM pets WHERE id = ? AND user_id = ?", (pet_id, user_id)))
        if r is None:
            raise not_found("pet")
        return r

    def list_pets(self, user_id: str) -> list[dict]:
        return [dict(r) for r in self.db.query("SELECT id, name, species, birth_year, notes, created_at, updated_at FROM pets WHERE user_id = ? ORDER BY name", (user_id,))]

    def update_pet(self, user_id: str, pet_id: str, **fields: Any) -> dict:
        cur = self.get_pet(user_id, pet_id)
        merged = {**cur, **{k: v for k, v in fields.items() if v is not None}}
        n, sp, by, nt = self._pet_payload(merged["name"], merged["species"], merged["birth_year"], merged["notes"] or "")
        self.db.execute("UPDATE pets SET name = ?, species = ?, birth_year = ?, notes = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                        (n, sp, by, nt, now_iso(), pet_id, user_id))
        return self.get_pet(user_id, pet_id)

    def delete_pet(self, user_id: str, pet_id: str) -> None:
        self.get_pet(user_id, pet_id)
        self.db.execute("DELETE FROM pets WHERE id = ? AND user_id = ?", (pet_id, user_id))

    # ----------------------------------------------------------------- places
    def create_place(self, user_id: str, name: str, zones: ZoneConfig) -> dict:
        name = name.strip()
        if not 1 <= len(name) <= 60:
            raise ApiError(422, "invalid_name", "Name must be 1-60 characters.")
        pid, ts = security.new_id(), now_iso()
        self.db.execute("INSERT INTO places (id, user_id, name, zones_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (pid, user_id, name, zones.model_dump_json(by_alias=True), ts, ts))
        return self.get_place(user_id, pid)

    def get_place(self, user_id: str, place_id: str) -> dict:
        r = row(self.db.one("SELECT id, name, zones_json, created_at, updated_at FROM places WHERE id = ? AND user_id = ?", (place_id, user_id)))
        if r is None:
            raise not_found("place")
        r["zones"] = json.loads(r.pop("zones_json"))
        return r

    def list_places(self, user_id: str) -> list[dict]:
        out = []
        for r in self.db.query("SELECT id FROM places WHERE user_id = ? ORDER BY name", (user_id,)):
            out.append(self.get_place(user_id, r["id"]))
        return out

    def update_place(self, user_id: str, place_id: str, name: str | None, zones: ZoneConfig | None) -> dict:
        cur = self.get_place(user_id, place_id)
        nm = (name or cur["name"]).strip()
        if not 1 <= len(nm) <= 60:
            raise ApiError(422, "invalid_name", "Name must be 1-60 characters.")
        zj = zones.model_dump_json(by_alias=True) if zones is not None else json.dumps(cur["zones"])
        self.db.execute("UPDATE places SET name = ?, zones_json = ?, updated_at = ? WHERE id = ? AND user_id = ?", (nm, zj, now_iso(), place_id, user_id))
        return self.get_place(user_id, place_id)

    def delete_place(self, user_id: str, place_id: str) -> None:
        self.get_place(user_id, place_id)
        self.db.execute("DELETE FROM places WHERE id = ? AND user_id = ?", (place_id, user_id))

    # ---------------------------------------------------------------- storage
    def user_dir(self, user_id: str) -> Path:
        d = self.media_root / user_id
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        return d

    def used_bytes(self, user_id: str) -> int:
        a = self.db.one("SELECT coalesce(sum(media_size), 0) AS n FROM analyses WHERE user_id = ? AND media_path IS NOT NULL", (user_id,))["n"]
        u = self.db.one("SELECT coalesce(sum(size_bytes), 0) AS n FROM uploads WHERE user_id = ? AND status IN ('receiving', 'complete')", (user_id,))["n"]
        return int(a) + int(u)

    def disk_ok(self, needed: int) -> bool:
        return shutil.disk_usage(self.media_root).free - needed >= self.s.app.storage.min_free_disk_bytes

    # ---------------------------------------------------------------- uploads
    def _lock_for(self, upload_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._upload_locks.setdefault(upload_id, threading.Lock())

    def create_upload(self, user_id: str, filename: str, size_bytes: int, content_type: str, sha256: str | None) -> dict:
        st = self.s.app.storage
        ext = Path(filename).suffix.lower()
        if ext not in self.s.serve.allowed_extensions:
            raise ApiError(415, "unsupported_file_type", f"Files of type '{ext or 'unknown'}' are not supported.",
                           "Use MP4, WebM, MOV, WAV, M4A, MP3, OGG or JPEG/PNG.", {"allowed": self.s.serve.allowed_extensions})
        if size_bytes <= 0:
            raise ApiError(422, "empty_upload", "The file is empty.", "Record or choose a non-empty file.")
        if size_bytes > st.max_upload_bytes:
            raise ApiError(413, "upload_too_large", f"The file exceeds the {st.max_upload_bytes // (1024 * 1024)} MiB limit.",
                           "Record a shorter clip or trim the file.")
        if sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ApiError(422, "invalid_checksum", "sha256 must be 64 lowercase hex characters.")
        n_open = self.db.one("SELECT count(*) AS n FROM uploads WHERE user_id = ? AND status = 'receiving'", (user_id,))["n"]
        if n_open >= st.max_open_uploads_per_user:
            raise ApiError(429, "too_many_open_uploads", "Too many unfinished uploads.", "Finish or cancel an existing upload first.")
        if self.used_bytes(user_id) + size_bytes > st.user_quota_bytes:
            raise ApiError(507, "quota_exceeded", "Your storage quota is full.", "Delete retained recordings in History or Settings.")
        if not self.disk_ok(size_bytes):
            raise ApiError(507, "server_storage_full", "The server is low on storage.", "Try again later; the operator has been notified in the logs.")
        uid, ts = security.new_id(), now_iso()
        path = self.user_dir(user_id) / f"upload-{uid}.part"
        path.touch(mode=0o600)
        self.db.execute("INSERT INTO uploads (id, user_id, filename, ext, content_type, size_bytes, received_bytes, sha256_expected, status, path, created_at, updated_at, expires_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, 0, ?, 'receiving', ?, ?, ?, ?)",
                        (uid, user_id, Path(filename).name[:200], ext, content_type[:100], size_bytes, sha256, str(path), ts, ts,
                         time.time() + st.upload_expiry_hours * 3600))
        return self.get_upload(user_id, uid)

    def get_upload(self, user_id: str, upload_id: str) -> dict:
        r = row(self.db.one("SELECT id, filename, ext, content_type, size_bytes, received_bytes, status, created_at, updated_at, expires_at, path, sha256_expected "
                            "FROM uploads WHERE id = ? AND user_id = ?", (upload_id, user_id)))
        if r is None:
            raise not_found("upload")
        return r

    def append_chunk(self, user_id: str, upload_id: str, offset: int, stream: BinaryIO, length: int | None) -> dict:
        st = self.s.app.storage
        with self._lock_for(upload_id):
            up = self.get_upload(user_id, upload_id)
            if up["status"] != "receiving":
                raise ApiError(409, "upload_not_receiving", f"This upload is {up['status']}.", "Start a new upload if needed.")
            if offset != up["received_bytes"]:
                raise ApiError(409, "upload_offset_mismatch", "The chunk does not start where the server expects.",
                               f"Resume from byte {up['received_bytes']}.", {"expected_offset": up["received_bytes"]})
            remaining = up["size_bytes"] - offset
            limit = min(st.max_chunk_bytes, remaining)
            if length is not None and length > limit:
                raise ApiError(413, "chunk_too_large", f"Chunks may be at most {limit} bytes here.", "Send smaller chunks.", {"max_chunk_bytes": limit})
            written = 0
            with open(up["path"], "r+b") as fh:
                fh.seek(offset)
                while True:
                    b = stream.read(min(1 << 20, limit - written + 1))
                    if not b:
                        break
                    written += len(b)
                    if written > limit:
                        fh.truncate(offset)
                        raise ApiError(413, "chunk_too_large", f"Chunks may be at most {limit} bytes here.", "Send smaller chunks.", {"max_chunk_bytes": limit})
                    fh.write(b)
                fh.flush()
                os.fsync(fh.fileno())
            new_off = offset + written
            status = "receiving"
            if new_off == up["size_bytes"]:
                if up["sha256_expected"]:
                    digest = _sha256(Path(up["path"]))
                    if digest != up["sha256_expected"]:
                        self._abort_upload_row(user_id, upload_id, up["path"])
                        raise ApiError(422, "checksum_mismatch", "The uploaded data does not match the expected checksum.",
                                       "The upload was discarded; upload the file again.")
                status = "complete"
            self.db.execute("UPDATE uploads SET received_bytes = ?, status = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                            (new_off, status, now_iso(), upload_id, user_id))
            return self.get_upload(user_id, upload_id)

    def _abort_upload_row(self, user_id: str, upload_id: str, path: str) -> None:
        Path(path).unlink(missing_ok=True)
        self.db.execute("UPDATE uploads SET status = 'aborted', updated_at = ? WHERE id = ? AND user_id = ?", (now_iso(), upload_id, user_id))

    def abort_upload(self, user_id: str, upload_id: str) -> None:
        with self._lock_for(upload_id):
            up = self.get_upload(user_id, upload_id)
            if up["status"] in ("receiving", "complete"):
                self._abort_upload_row(user_id, upload_id, up["path"])

    # --------------------------------------------------------------- analyses
    def create_analysis(self, user_id: str, upload_id: str, pet_ids: list[str], subject_mode: str, place_id: str | None,
                        zones: ZoneConfig | None, modalities: list[str], note: str, question: dict | None = None) -> dict:
        """``question``: {species, scene, question, understanding, strategy, retain_for_research, research_consent_version}."""
        j = self.s.app.jobs
        if question is not None:
            if question["understanding"] == "cloud" and not self.s.cloud.enabled:
                raise ApiError(422, "cloud_disabled", "Cloud-assisted understanding is not enabled on this server.", "Choose local understanding.")
            if question["retain_for_research"] and not question.get("research_consent_version"):
                raise ApiError(422, "research_consent_missing", "Keeping the recording as a research sample needs your explicit consent.",
                               "Accept the research-sample consent or leave the option off.")
        if subject_mode == "unknown" and pet_ids:
            raise ApiError(422, "subject_mismatch", "Pets were selected but the subject is 'unknown'.", "Choose 'single' or 'multiple', or clear the pets.")
        if subject_mode == "single_known" and len(pet_ids) != 1:
            raise ApiError(422, "subject_mismatch", "Select exactly one pet for a single known animal.")
        if subject_mode == "multiple_known" and len(pet_ids) < 2:
            raise ApiError(422, "subject_mismatch", "Select at least two pets for multiple known animals.")
        if not modalities or any(m not in ("audio", "video") for m in modalities):
            raise ApiError(422, "invalid_modalities", "Modalities must be a non-empty subset of audio and video.")
        if len(note) > 500:
            raise ApiError(422, "invalid_note", "Notes are limited to 500 characters.")
        for pid in pet_ids:
            self.get_pet(user_id, pid)
        zones_json = None
        if zones is not None:
            zones_json = zones.model_dump_json(by_alias=True)
        elif place_id:
            zones_json = json.dumps(self.get_place(user_id, place_id)["zones"])
        if place_id:
            self.get_place(user_id, place_id)
        with self._lock_for(upload_id):
            up = self.get_upload(user_id, upload_id)
            if up["status"] != "complete":
                raise ApiError(409, "upload_incomplete", f"The upload is {up['status']} ({up['received_bytes']}/{up['size_bytes']} bytes).",
                               "Finish the upload before starting the analysis.")
            active = self.db.one("SELECT count(*) AS n FROM analyses WHERE user_id = ? AND status IN ('queued', 'running')", (user_id,))["n"]
            if active >= j.max_active_per_user:
                raise ApiError(429, "too_many_active_analyses", f"You already have {active} analyses in progress.",
                               "Wait for one to finish or cancel one.")
            if self.db.one("SELECT count(*) AS n FROM analyses WHERE status = 'queued'")["n"] >= j.max_queued_total:
                raise ApiError(503, "queue_full", "The server queue is full.", "Try again in a few minutes.", headers={"Retry-After": "120"})
            try:
                mp = probe(up["path"], self.s.limits)
            except MediaError as exc:
                msg, action = MEDIA_ERROR_TEXT.get(exc.code, (str(exc), None))
                self._abort_upload_row(user_id, upload_id, up["path"])
                raise ApiError(422, exc.code, msg, action) from exc
            aid, ts = security.new_id(), now_iso()
            media = self.user_dir(user_id) / f"analysis-{aid}{up['ext']}"
            os.replace(up["path"], media)
            with self.db.tx() as c:
                c.execute("UPDATE uploads SET status = 'consumed', updated_at = ? WHERE id = ?", (ts, upload_id))
                c.execute("INSERT INTO analyses (id, user_id, place_id, subject_mode, modalities, zones_json, status, stage, progress, attempts, max_attempts, "
                          "next_attempt_at, media_path, media_ext, media_size, original_filename, duration_s, note, created_at, updated_at) "
                          "VALUES (?, ?, ?, ?, ?, ?, 'queued', 'queued', 0, 0, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (aid, user_id, place_id, subject_mode, ",".join(modalities), zones_json, j.max_attempts, str(media), up["ext"],
                           up["size_bytes"], up["filename"], mp.duration_s, note, ts, ts))
                for pid in pet_ids:
                    c.execute("INSERT INTO analysis_pets (analysis_id, pet_id) VALUES (?, ?)", (aid, pid))
                if question is not None:
                    c.execute("INSERT INTO analysis_questions (analysis_id, species, scene, question, understanding, strategy, retain_for_research, "
                              "research_consent_version, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                              (aid, question["species"], question["scene"], question["question"], question["understanding"], question["strategy"],
                               int(question["retain_for_research"]), question.get("research_consent_version"), ts, ts))
        log.info("analysis_queued", extra={"analysis_id": aid, "user": user_ref(user_id)})
        return self.get_analysis(user_id, aid)

    def _analysis_row(self, user_id: str, analysis_id: str) -> dict:
        r = row(self.db.one("SELECT * FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id)))
        if r is None:
            raise not_found("analysis")
        return r

    def _public(self, r: dict) -> dict:
        pets = [x["pet_id"] for x in self.db.query("SELECT pet_id FROM analysis_pets WHERE analysis_id = ?", (r["id"],))]
        retained = r["media_path"] is not None and r["status"] in ("succeeded", "failed", "cancelled")
        return {
            "id": r["id"], "status": r["status"], "stage": r["stage"], "progress": r["progress"], "attempts": r["attempts"],
            "max_attempts": r["max_attempts"], "subject_mode": r["subject_mode"], "pet_ids": pets, "place_id": r["place_id"],
            "modalities": r["modalities"].split(","), "original_filename": r["original_filename"], "duration_s": r["duration_s"],
            "note": r["note"], "decision_state": r["decision_state"], "summary": json.loads(r["summary_json"]) if r["summary_json"] else None,
            "error": {"code": r["error_code"], "message": r["error_message"], "action": r["error_action"]} if r["error_code"] and r["status"] in ("failed", "queued") else None,
            "cancel_requested": bool(r["cancel_requested"]),
            "media_available": retained and Path(r["media_path"]).exists(),
            "media_retained_until": r["media_retained_until"],
            "created_at": r["created_at"], "updated_at": r["updated_at"], "started_at": r["started_at"], "finished_at": r["finished_at"],
        }

    def get_analysis(self, user_id: str, analysis_id: str, with_result: bool = False) -> dict:
        out = self._public(self._analysis_row(user_id, analysis_id))
        q = self.question(analysis_id)
        if q is not None:
            out["behavior_question"] = {k: q[k] for k in ("species", "scene", "question", "understanding", "strategy", "status")} | {
                "retain_for_research": bool(q["retain_for_research"]), "answer": json.loads(q["answer_json"]) if (with_result and q["answer_json"]) else None}
        if with_result:
            rr = self.db.one("SELECT result_json FROM analysis_results WHERE analysis_id = ?", (analysis_id,))
            out["result"] = json.loads(rr["result_json"]) if rr else None
        return out

    # ------------------------------------------------------ behaviour questions and research samples (E35)
    def question(self, analysis_id: str) -> dict | None:
        return row(self.db.one("SELECT * FROM analysis_questions WHERE analysis_id = ?", (analysis_id,)))

    def store_answer(self, analysis_id: str, status: str, answer: dict | None) -> None:
        self.db.execute("UPDATE analysis_questions SET status = ?, answer_json = ?, updated_at = ? WHERE analysis_id = ?",
                        (status, json.dumps(answer) if answer is not None else None, now_iso(), analysis_id))

    def donate_sample(self, analysis_id: str, model_annotation: dict | None) -> str | None:
        """Copy the analysed recording into the research store when the owner opted in for this analysis."""
        q = self.question(analysis_id)
        a = row(self.db.one("SELECT user_id, media_path, media_ext FROM analyses WHERE id = ?", (analysis_id,)))
        if q is None or not q["retain_for_research"] or a is None or not a["media_path"] or not Path(a["media_path"]).exists():
            return None
        sid, ts = security.new_id(), now_iso()
        d = self.research_root / a["user_id"]
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        dst = d / f"sample-{sid}{a['media_ext']}"
        shutil.copyfile(a["media_path"], dst)
        sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        with self.db.tx() as c:
            c.execute("INSERT INTO research_samples (id, user_id, analysis_id, media_path, media_sha256, species, scene, question, consent_version, "
                      "consent_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (sid, a["user_id"], analysis_id, str(dst), sha, q["species"], q["scene"], q["question"], q["research_consent_version"], q["created_at"], ts))
            if model_annotation is not None:
                c.execute("INSERT INTO sample_annotations (id, sample_id, source, status, author_id, payload_json, created_at) VALUES (?, ?, 'model', 'provisional', NULL, ?, ?)",
                          (security.new_id(), sid, json.dumps(model_annotation), ts))
        self.audit(a["user_id"], "research_sample_donated", "research_sample", sid)
        return sid

    def _sample(self, user_id: str | None, sample_id: str) -> dict:
        r = row(self.db.one("SELECT * FROM research_samples WHERE id = ?" + (" AND user_id = ?" if user_id else ""),
                            (sample_id, user_id) if user_id else (sample_id,)))
        if r is None:
            raise not_found("research sample")
        return r

    def _sample_public(self, r: dict) -> dict:
        ann = [dict(x) for x in self.db.query("SELECT id, source, status, payload_json, created_at FROM sample_annotations WHERE sample_id = ? ORDER BY created_at", (r["id"],))]
        return {k: r[k] for k in ("id", "analysis_id", "species", "scene", "question", "consent_version", "consent_at", "created_at", "media_sha256")} | {
            "annotations": [{**{k: x[k] for k in ("id", "source", "status", "created_at")}, "payload": json.loads(x["payload_json"])} for x in ann]}

    def list_samples(self, user_id: str) -> list[dict]:
        return [self._sample_public(dict(r)) for r in self.db.query("SELECT * FROM research_samples WHERE user_id = ? ORDER BY created_at DESC", (user_id,))]

    def get_sample(self, user_id: str, sample_id: str) -> dict:
        return self._sample_public(self._sample(user_id, sample_id))

    def delete_sample(self, user_id: str, sample_id: str) -> None:
        r = self._sample(user_id, sample_id)
        with self.db.tx() as c:
            c.execute("DELETE FROM research_samples WHERE id = ?", (sample_id,))
        Path(r["media_path"]).unlink(missing_ok=True)
        self.audit(user_id, "research_sample_deleted", "research_sample", sample_id)

    def annotate_sample(self, user_id: str | None, sample_id: str, source: str, payload: dict) -> dict:
        """source 'owner' (owner-reported) or 'reviewer' (human-confirmed; administrators only, enforced by the API)."""
        status = {"owner": "owner_reported", "reviewer": "human_confirmed"}[source]
        r = self._sample(user_id if source == "owner" else None, sample_id)
        if len(json.dumps(payload)) > 8000:
            raise ApiError(422, "annotation_too_large", "Feedback is limited to 8,000 characters.")
        aid = security.new_id()
        self.db.execute("INSERT INTO sample_annotations (id, sample_id, source, status, author_id, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (aid, r["id"], source, status, user_id, json.dumps(payload), now_iso()))
        return self._sample_public(self._sample(None, sample_id))

    def list_analyses(self, user_id: str, pet_id: str | None, limit: int, before: str | None) -> list[dict]:
        limit = max(1, min(100, limit))
        sql = "SELECT a.* FROM analyses a WHERE a.user_id = ?"
        params: list[Any] = [user_id]
        if pet_id:
            sql += " AND EXISTS (SELECT 1 FROM analysis_pets p WHERE p.analysis_id = a.id AND p.pet_id = ?)"
            params.append(pet_id)
        if before:
            sql += " AND a.created_at < ?"
            params.append(before)
        sql += " ORDER BY a.created_at DESC LIMIT ?"
        params.append(limit)
        return [self._public(dict(r)) for r in self.db.query(sql, tuple(params))]

    def cancel_analysis(self, user_id: str, analysis_id: str) -> dict:
        with self.db.tx() as c:
            r = c.execute("SELECT status FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id)).fetchone()
            if r is None:
                raise not_found("analysis")
            if r["status"] == "queued":
                c.execute("UPDATE analyses SET status = 'cancelled', stage = 'cancelled', finished_at = ?, updated_at = ? WHERE id = ?",
                          (now_iso(), now_iso(), analysis_id))
            elif r["status"] == "running":
                c.execute("UPDATE analyses SET cancel_requested = 1, updated_at = ? WHERE id = ?", (now_iso(), analysis_id))
            else:
                raise ApiError(409, "not_cancellable", f"The analysis already {r['status']}.")
        self.apply_retention_after_terminal(analysis_id)
        return self.get_analysis(user_id, analysis_id)

    def retry_analysis(self, user_id: str, analysis_id: str) -> dict:
        r = self._analysis_row(user_id, analysis_id)
        if r["status"] not in ("failed", "cancelled"):
            raise ApiError(409, "not_retryable", f"The analysis is {r['status']}.")
        if r["status"] == "failed" and r["error_code"] not in RETRYABLE:
            raise ApiError(409, "not_retryable", "This failure is caused by the input, so retrying will not help.", r["error_action"])
        if not r["media_path"] or not Path(r["media_path"]).exists():
            raise ApiError(409, "media_gone", "The recording was already deleted after processing.", "Upload the recording again.")
        if self.db.one("SELECT count(*) AS n FROM analyses WHERE user_id = ? AND status IN ('queued', 'running')", (user_id,))["n"] >= self.s.app.jobs.max_active_per_user:
            raise ApiError(429, "too_many_active_analyses", "Too many analyses in progress.", "Wait for one to finish.")
        self.db.execute("UPDATE analyses SET status = 'queued', stage = 'queued', progress = 0, attempts = 0, cancel_requested = 0, error_code = NULL, "
                        "error_message = NULL, error_action = NULL, next_attempt_at = 0, media_retained_until = NULL, updated_at = ? WHERE id = ?",
                        (now_iso(), analysis_id))
        return self.get_analysis(user_id, analysis_id)

    def delete_analysis(self, user_id: str, analysis_id: str) -> None:
        r = self._analysis_row(user_id, analysis_id)
        with self.db.tx() as c:
            c.execute("DELETE FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id))
        if r["media_path"]:
            Path(r["media_path"]).unlink(missing_ok=True)
        self.audit(user_id, "analysis_deleted", "analysis", analysis_id)

    def delete_media(self, user_id: str, analysis_id: str) -> dict:
        r = self._analysis_row(user_id, analysis_id)
        if r["status"] in ACTIVE:
            raise ApiError(409, "analysis_active", "The recording is still being processed.", "Cancel the analysis first.")
        if r["media_path"]:
            Path(r["media_path"]).unlink(missing_ok=True)
            self.db.execute("UPDATE analyses SET media_path = NULL, media_retained_until = NULL, updated_at = ? WHERE id = ?", (now_iso(), analysis_id))
            self.audit(user_id, "media_deleted", "analysis", analysis_id)
        return self.get_analysis(user_id, analysis_id)

    def media_file(self, user_id: str, analysis_id: str) -> tuple[Path, str]:
        r = self._analysis_row(user_id, analysis_id)
        if r["status"] in ACTIVE or not r["media_path"] or not Path(r["media_path"]).exists():
            raise ApiError(404, "media_not_retained", "The recording is not retained on the server.",
                           "Enable media retention in Settings before analyzing to replay recordings.")
        return Path(r["media_path"]), r["media_ext"]

    def apply_retention_after_terminal(self, analysis_id: str) -> None:
        """Delete media now (retention 0) or schedule deletion (retention N days)."""
        r = self.db.one("SELECT a.status, a.media_path, a.error_code, u.retain_media_days FROM analyses a JOIN users u ON u.id = a.user_id WHERE a.id = ?", (analysis_id,))
        if r is None or r["status"] in ACTIVE or not r["media_path"]:
            return
        keep_for_retry = r["status"] == "failed" and r["error_code"] in RETRYABLE
        days = r["retain_media_days"]
        if days <= 0 and not keep_for_retry:
            Path(r["media_path"]).unlink(missing_ok=True)
            self.db.execute("UPDATE analyses SET media_path = NULL, media_retained_until = NULL WHERE id = ?", (analysis_id,))
        else:
            until = time.time() + max(days, 1) * 86400
            self.db.execute("UPDATE analyses SET media_retained_until = ? WHERE id = ?", (until, analysis_id))

    # --------------------------------------------------------------- feedback
    def add_feedback(self, user_id: str, analysis_id: str, target_type: str, target_id: str | None, verdict: str,
                     actual_behavior: str | None, note: str, consent: bool) -> dict:
        a = self.get_analysis(user_id, analysis_id)
        if a["status"] != "succeeded":
            raise ApiError(409, "analysis_not_finished", "Feedback can be given once the analysis has finished.")
        if len(note) > 1000 or (actual_behavior and len(actual_behavior) > 120):
            raise ApiError(422, "invalid_feedback", "Feedback text is too long.")
        fid, ts = security.new_id(), now_iso()
        self.db.execute("INSERT INTO feedback (id, user_id, analysis_id, target_type, target_id, verdict, actual_behavior, note, consent_research_use, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (fid, user_id, analysis_id, target_type, target_id, verdict, actual_behavior, note, int(consent), ts))
        return self.list_feedback(user_id, analysis_id)[-1]

    def list_feedback(self, user_id: str, analysis_id: str) -> list[dict]:
        self._analysis_row(user_id, analysis_id)
        return [dict(r) | {"consent_research_use": bool(r["consent_research_use"])} for r in self.db.query(
            "SELECT id, analysis_id, target_type, target_id, verdict, actual_behavior, note, consent_research_use, created_at FROM feedback "
            "WHERE analysis_id = ? AND user_id = ? ORDER BY created_at", (analysis_id, user_id))]

    def delete_feedback(self, user_id: str, feedback_id: str) -> None:
        if self.db.execute("DELETE FROM feedback WHERE id = ? AND user_id = ?", (feedback_id, user_id)) == 0:
            raise not_found("feedback")

    # ----------------------------------------------------------------- export
    def export_user(self, user_id: str) -> dict:
        analyses = []
        for a in self.list_analyses(user_id, None, 100000, None):
            full = self.get_analysis(user_id, a["id"], with_result=True)
            full["feedback"] = self.list_feedback(user_id, a["id"])
            analyses.append(full)
        u = self.get_user(user_id)
        synced = [{"id": r["id"], "created_at": r["created_at"], "synced_at": r["synced_at"], "filename": r["filename"],
                   "result": json.loads(r["result_json"]), "feedback": json.loads(r["feedback_json"])}
                  for r in self.db.query("SELECT * FROM synced_results WHERE user_id = ? ORDER BY created_at", (user_id,))]
        return {"format": "petperception.export/1", "exported_at": now_iso(), "user": {k: u[k] for k in ("username", "retain_media_days", "feedback_research_consent", "created_at")},
                "pets": self.list_pets(user_id), "places": self.list_places(user_id), "analyses": analyses, "on_device_results_backed_up": synced,
                "research_samples": self.list_samples(user_id),
                "note": "Recordings are not included; download retained recordings individually. On-device recordings never reach the server."}

    # ---------------------------------------------------------------- cleanup
    def cleanup(self) -> dict[str, int]:
        now = time.time()
        stats = {"uploads_expired": 0, "media_expired": 0, "sessions_expired": 0}
        for r in self.db.query("SELECT id, user_id, path FROM uploads WHERE status IN ('receiving', 'complete') AND expires_at < ?", (now,)):
            self._abort_upload_row(r["user_id"], r["id"], r["path"])
            stats["uploads_expired"] += 1
        for r in self.db.query("SELECT id, media_path FROM analyses WHERE media_path IS NOT NULL AND media_retained_until IS NOT NULL AND media_retained_until < ? "
                               "AND status NOT IN ('queued', 'running')", (now,)):
            Path(r["media_path"]).unlink(missing_ok=True)
            self.db.execute("UPDATE analyses SET media_path = NULL, media_retained_until = NULL WHERE id = ?", (r["id"],))
            stats["media_expired"] += 1
        stats["sessions_expired"] = self.db.execute("DELETE FROM sessions WHERE expires_at < ? OR last_seen_at < ?",
                                                    (now, now - self.s.app.auth.session_idle_hours * 3600))
        self.db.execute("DELETE FROM login_failures WHERE at < ?", (now - self.s.app.auth.login_window_s,))
        self.db.execute("DELETE FROM uploads WHERE status IN ('consumed', 'aborted') AND expires_at < ?", (now - 86400,))
        # Orphan files (e.g. left by a crash between file move and commit).
        known = {r["media_path"] for r in self.db.query("SELECT media_path FROM analyses WHERE media_path IS NOT NULL")}
        known |= {r["path"] for r in self.db.query("SELECT path FROM uploads WHERE status IN ('receiving', 'complete')")}
        for f in self.media_root.glob("*/*"):
            if f.is_file() and str(f) not in known and now - f.stat().st_mtime > 3600:
                f.unlink(missing_ok=True)
        return stats


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
