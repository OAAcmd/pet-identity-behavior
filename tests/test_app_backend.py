"""Product backend tests: auth, isolation, uploads, jobs, recovery, retention.

Most tests use a stub engine (marked ``synthetic``) so job-state logic is
tested deterministically; ``integration`` tests run the real engine.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from petperception import __version__
from petperception.app.main import create_app
from petperception.app.worker import JobRunner
from petperception.ingest.media import MediaError
from petperception.pipeline import AnalysisCancelled
from petperception.schemas import analysis as S

PW = "correct horse battery"


def minimal_result(aid: str) -> S.AnalysisResult:
    return S.AnalysisResult(
        analysis_id=aid, created_at=datetime.now(timezone.utc), engine_version=__version__, config_profile="test", config_hash="x",
        media=S.MediaInfo(analysis_filename="x.wav", size_bytes=1, sha256="0" * 64, container="wav", duration_s=1.0, start_time_s=0.0, audio=None, video=None),
        modalities=S.ModalityStatus(requested=["audio"], available=[], audio_state="silent", video_state="absent"),
        components=[], observations=S.Observations(animal_tracks=[], sound_events=[], context_sounds=[], audio_summary=S.AudioSummary(state="silent"),
                                                     evidence=[], sampled_frames=0, sample_fps=None),
        quality_flags=[], hypotheses=[], decision=S.Decision(state="abstention", overall_interpretation="unknown", reason_codes=["audio_silent"], summary="stub"),
        suggestions=[], limitations=[], fusion=S.FusionInfo(method="rule_baseline", note="stub"), timings_ms={}, models=[],
        usage_restrictions=S.UsageRestrictions(research_only=True, notices=[]))


class StubEngine:
    """Synthetic engine with programmable behavior."""

    def __init__(self, mode: str = "ok", fail_times: int = 0):
        self.mode, self.fail_times, self.calls = mode, fail_times, 0
        self.started = threading.Event()

    def capabilities(self):
        return {"engine_version": "stub", "components": []}

    def analyze(self, path, zones=None, modalities=("audio", "video"), analysis_id=None, display_name=None, progress=None, should_cancel=None):
        self.calls += 1
        self.started.set()
        if progress:
            progress("audio", 0.5)
        if self.mode == "input_error":
            raise MediaError("corrupt_input", "bad")
        if self.calls <= self.fail_times:
            raise RuntimeError("transient")
        if self.mode in ("block", "slow"):
            t0 = time.time()
            while time.time() - t0 < 20:
                if should_cancel and should_cancel():
                    raise AnalysisCancelled("test")
                time.sleep(0.05)
        return minimal_result(analysis_id)


def make(app_settings, engine=None):
    eng = engine or StubEngine()
    app = create_app(app_settings, engine_factory=lambda: eng, start_workers=False)
    runner = JobRunner(app_settings, app.state.db, app.state.store, lambda: eng, worker_id="test-worker")
    return app, runner, eng


def register(c: TestClient, username="alice") -> str:
    r = c.post("/api/v1/auth/register", json={"username": username, "password": PW})
    assert r.status_code == 201, r.text
    return r.json()["csrf_token"]


def login(c: TestClient, username: str) -> str:
    r = c.post("/api/v1/auth/login", json={"username": username, "password": PW})
    assert r.status_code == 200, r.text
    return r.json()["csrf_token"]


def wav_bytes(seconds=1.0, sr=16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, (0.1 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * sr)) / sr)).astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def upload(c: TestClient, csrf: str, data: bytes, name="clip.wav", chunk=None, sha=True) -> str:
    body = {"filename": name, "size_bytes": len(data), "content_type": "audio/wav"}
    if sha:
        body["sha256"] = hashlib.sha256(data).hexdigest()
    r = c.post("/api/v1/uploads", json=body, headers={"X-CSRF-Token": csrf})
    assert r.status_code == 201, r.text
    uid = r.json()["id"]
    step = chunk or len(data)
    for off in range(0, len(data), step):
        r = c.put(f"/api/v1/uploads/{uid}?offset={off}", content=data[off:off + step], headers={"X-CSRF-Token": csrf, "Content-Type": "application/octet-stream"})
        assert r.status_code == 200, r.text
    return uid


def start(c, csrf, data=None, **kw) -> str:
    uid = upload(c, csrf, data or wav_bytes())
    r = c.post("/api/v1/analyses", json={"upload_id": uid, **kw}, headers={"X-CSRF-Token": csrf})
    assert r.status_code == 202, r.text
    return r.json()["id"]


pytestmark = pytest.mark.synthetic


def test_registration_login_csrf_and_rate_limit(app_settings):
    app, _, _ = make(app_settings)
    with TestClient(app) as c:
        assert c.get("/api/v1/auth/status").json() == {"signed_in": False, "registration_open": True}
        csrf = register(c)
        assert c.get("/api/v1/auth/me").json()["is_admin"] is True
        assert c.post("/api/v1/pets", json={"name": "Milo", "species": "cat"}).status_code == 403  # no CSRF header
        assert c.post("/api/v1/pets", json={"name": "Milo", "species": "cat"}, headers={"X-CSRF-Token": csrf}).status_code == 201
        c.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf})
        assert c.get("/api/v1/pets").status_code == 401
        assert c.post("/api/v1/auth/register", json={"username": "bob", "password": PW}).json()["error"]["code"] == "registration_closed"
        for _ in range(5):
            assert c.post("/api/v1/auth/login", json={"username": "alice", "password": "wrong-password"}).status_code == 401
        r = c.post("/api/v1/auth/login", json={"username": "alice", "password": PW})
        assert r.status_code == 429 and "Retry-After" in r.headers


def test_weak_password_and_bad_username_rejected(app_settings):
    app, _, _ = make(app_settings)
    with TestClient(app) as c:
        assert c.post("/api/v1/auth/register", json={"username": "al", "password": PW}).json()["error"]["code"] == "invalid_username"
        assert c.post("/api/v1/auth/register", json={"username": "alice", "password": "short"}).json()["error"]["code"] == "weak_password"


def two_users(c: TestClient) -> tuple[str, str]:
    csrf_a = register(c, "alice")
    assert c.post("/api/v1/admin/users", json={"username": "bob", "password": PW}, headers={"X-CSRF-Token": csrf_a}).status_code == 201
    return csrf_a, "bob"


def test_user_data_isolation(app_settings):
    app, runner, _ = make(app_settings)
    with TestClient(app) as c:
        csrf_a, bob = two_users(c)
        pet = c.post("/api/v1/pets", json={"name": "Milo", "species": "cat"}, headers={"X-CSRF-Token": csrf_a}).json()
        aid = start(c, csrf_a, subject_mode="single_known", pet_ids=[pet["id"]])
        runner.run_pending()
        up_a = c.post("/api/v1/uploads", json={"filename": "a.wav", "size_bytes": 10}, headers={"X-CSRF-Token": csrf_a}).json()["id"]
        c.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf_a})
        csrf_b = login(c, bob)
        assert c.get("/api/v1/pets").json() == [] and c.get("/api/v1/analyses").json() == []
        for path in (f"/api/v1/pets/{pet['id']}", f"/api/v1/analyses/{aid}", f"/api/v1/analyses/{aid}/feedback", f"/api/v1/uploads/{up_a}"):
            assert c.get(path).status_code == 404, path
        h = {"X-CSRF-Token": csrf_b}
        assert c.delete(f"/api/v1/analyses/{aid}", headers=h).status_code == 404
        assert c.delete(f"/api/v1/pets/{pet['id']}", headers=h).status_code == 404
        assert c.put(f"/api/v1/uploads/{up_a}?offset=0", content=b"x" * 10, headers=h).status_code == 404
        r = c.post("/api/v1/analyses", json={"upload_id": up_a}, headers=h)
        assert r.status_code == 404
        # Bob cannot attach Alice's pet to his own analysis either.
        uid = upload(c, csrf_b, wav_bytes())
        r = c.post("/api/v1/analyses", json={"upload_id": uid, "subject_mode": "single_known", "pet_ids": [pet["id"]]}, headers=h)
        assert r.status_code == 404
        assert c.get("/api/v1/admin/users").status_code == 403


def test_resumable_upload_interruption_and_validation(app_settings):
    app, _, _ = make(app_settings)
    with TestClient(app) as c:
        csrf = register(c)
        h = {"X-CSRF-Token": csrf}
        data = wav_bytes(2.0)
        sha = hashlib.sha256(data).hexdigest()
        uid = c.post("/api/v1/uploads", json={"filename": "rec.wav", "size_bytes": len(data), "sha256": sha}, headers=h).json()["id"]
        half = len(data) // 2
        assert c.put(f"/api/v1/uploads/{uid}?offset=0", content=data[:half], headers=h).json()["received_bytes"] == half
        # "Interruption": the client lost track; it asks the server where to resume.
        st = c.get(f"/api/v1/uploads/{uid}").json()
        assert st["status"] == "receiving" and st["received_bytes"] == half
        r = c.put(f"/api/v1/uploads/{uid}?offset=0", content=data[:100], headers=h)
        assert r.status_code == 409 and r.json()["error"]["details"]["expected_offset"] == half
        assert c.put(f"/api/v1/uploads/{uid}?offset={half}", content=data[half:], headers=h).json()["status"] == "complete"
        # Checksum mismatch discards the upload.
        bad = c.post("/api/v1/uploads", json={"filename": "b.wav", "size_bytes": len(data), "sha256": "0" * 64}, headers=h).json()["id"]
        r = c.put(f"/api/v1/uploads/{bad}?offset=0", content=data, headers=h)
        assert r.status_code == 422 and r.json()["error"]["code"] == "checksum_mismatch"
        assert c.get(f"/api/v1/uploads/{bad}").json()["status"] == "aborted"
        # Type, size and chunk limits.
        assert c.post("/api/v1/uploads", json={"filename": "x.exe", "size_bytes": 10}, headers=h).status_code == 415
        assert c.post("/api/v1/uploads", json={"filename": "x.wav", "size_bytes": 10**10}, headers=h).status_code == 413
        big = c.post("/api/v1/uploads", json={"filename": "x.wav", "size_bytes": 20_000_000}, headers=h).json()["id"]
        r = c.put(f"/api/v1/uploads/{big}?offset=0", content=b"0" * (app_settings.app.storage.max_chunk_bytes + 1), headers=h)
        assert r.status_code == 413 and r.json()["error"]["code"] == "chunk_too_large"


def test_invalid_media_rejected_with_actionable_error(app_settings):
    app, _, _ = make(app_settings)
    with TestClient(app) as c:
        csrf = register(c)
        uid = upload(c, csrf, b"this is definitely not audio", name="fake.wav")
        r = c.post("/api/v1/analyses", json={"upload_id": uid}, headers={"X-CSRF-Token": csrf})
        body = r.json()["error"]
        assert r.status_code == 422 and body["code"] == "corrupt_input" and body["action"]
        assert c.get(f"/api/v1/uploads/{uid}").json()["status"] == "aborted"


def test_analysis_lifecycle_retention_and_deletion(app_settings, tmp_path):
    app, runner, _ = make(app_settings)
    with TestClient(app) as c:
        csrf = register(c)
        h = {"X-CSRF-Token": csrf}
        aid = start(c, csrf)
        assert c.get(f"/api/v1/analyses/{aid}").json()["status"] == "queued"
        assert runner.run_pending() == 1
        a = c.get(f"/api/v1/analyses/{aid}").json()
        assert a["status"] == "succeeded" and a["result"]["schema_version"] == "1.0.0" and a["decision_state"] == "abstention"
        assert a["media_available"] is False  # default retention 0: media deleted right after processing
        assert not list((tmp_path / "media").rglob("analysis-*"))
        # With retention, media can be replayed with range requests, then deleted on demand.
        assert c.patch("/api/v1/settings", json={"retain_media_days": 7}, headers=h).json()["retain_media_days"] == 7
        aid2 = start(c, csrf)
        runner.run_pending()
        assert c.get(f"/api/v1/analyses/{aid2}").json()["media_available"] is True
        r = c.get(f"/api/v1/analyses/{aid2}/media", headers={"Range": "bytes=0-99"})
        assert r.status_code == 206 and len(r.content) == 100
        assert c.delete(f"/api/v1/analyses/{aid2}/media", headers=h).json()["media_available"] is False
        assert c.get(f"/api/v1/analyses/{aid2}/media").status_code == 404
        # Feedback, then deterministic deletion of analysis + feedback.
        fb = c.post(f"/api/v1/analyses/{aid}/feedback", json={"target_type": "overall", "verdict": "unsure", "consent_research_use": True}, headers=h)
        assert fb.status_code == 201 and fb.json()["consent_research_use"] is True
        assert c.delete(f"/api/v1/analyses/{aid}", headers=h).status_code == 204
        assert c.get(f"/api/v1/analyses/{aid}").status_code == 404
        db = app.state.db
        assert db.one("SELECT count(*) AS n FROM feedback WHERE analysis_id = ?", (aid,))["n"] == 0
        assert db.one("SELECT count(*) AS n FROM analysis_results WHERE analysis_id = ?", (aid,))["n"] == 0
        export = c.get("/api/v1/export").json()
        assert export["format"] == "petperception.export/1" and len(export["analyses"]) == 1


def test_transient_failures_are_retried_with_bound(app_settings):
    eng = StubEngine(fail_times=1)
    app, runner, _ = make(app_settings, eng)
    with TestClient(app) as c:
        csrf = register(c)
        aid = start(c, csrf)
        runner.run_pending(max_jobs=1)
        a = c.get(f"/api/v1/analyses/{aid}").json()
        assert a["status"] == "queued" and a["attempts"] == 1 and a["error"]["code"] == "internal_error"
        runner.run_pending()
        a = c.get(f"/api/v1/analyses/{aid}").json()
        assert a["status"] == "succeeded" and a["attempts"] == 2
    eng2 = StubEngine(fail_times=99)
    app_settings.app.storage.database_path = app_settings.app.storage.database_path + ".2"
    app2, runner2, _ = make(app_settings, eng2)
    with TestClient(app2) as c:
        csrf = register(c)
        aid = start(c, csrf)
        for _ in range(5):
            runner2.run_pending()
        a = c.get(f"/api/v1/analyses/{aid}").json()
        assert a["status"] == "failed" and a["attempts"] == app_settings.app.jobs.max_attempts and a["error"]["code"] == "internal_error"
        assert eng2.calls == app_settings.app.jobs.max_attempts
        # Media is kept for retryable failures so the user can retry.
        assert c.post(f"/api/v1/analyses/{aid}/retry", headers={"X-CSRF-Token": csrf}).json()["status"] == "queued"


def test_input_errors_are_not_retried(app_settings):
    eng = StubEngine(mode="input_error")
    app, runner, _ = make(app_settings, eng)
    with TestClient(app) as c:
        csrf = register(c)
        aid = start(c, csrf)
        runner.run_pending()
        a = c.get(f"/api/v1/analyses/{aid}").json()
        assert a["status"] == "failed" and a["error"]["code"] == "corrupt_input" and eng.calls == 1
        assert c.post(f"/api/v1/analyses/{aid}/retry", headers={"X-CSRF-Token": csrf}).status_code == 409


def test_cancel_queued_and_running(app_settings):
    eng = StubEngine(mode="block")
    app, runner, _ = make(app_settings, eng)
    with TestClient(app) as c:
        csrf = register(c)
        h = {"X-CSRF-Token": csrf}
        q = start(c, csrf)
        assert c.post(f"/api/v1/analyses/{q}/cancel", headers=h).json()["status"] == "cancelled"
        r = start(c, csrf)
        th = threading.Thread(target=runner.run_pending)
        th.start()
        assert eng.started.wait(10)
        assert c.post(f"/api/v1/analyses/{r}/cancel", headers=h).json()["cancel_requested"] is True
        th.join(timeout=15)
        assert c.get(f"/api/v1/analyses/{r}").json()["status"] == "cancelled"
        assert c.post(f"/api/v1/analyses/{r}/cancel", headers=h).status_code == 409


def test_crash_recovery_requeues_then_fails_after_max_attempts(app_settings):
    app, runner, _ = make(app_settings)
    db = app.state.db
    with TestClient(app) as c:
        csrf = register(c)
        aid = start(c, csrf)
        job = runner.claim()  # a worker claims the job ... and dies (no finish, no heartbeat)
        assert job["id"] == aid and c.get(f"/api/v1/analyses/{aid}").json()["status"] == "running"
        db.execute("UPDATE analyses SET lease_expires_at = 0 WHERE id = ?", (aid,))
        new_worker = JobRunner(app_settings, db, app.state.store, lambda: StubEngine(), worker_id="replacement")
        assert new_worker.recover_expired() == 1
        a = c.get(f"/api/v1/analyses/{aid}").json()
        assert a["status"] == "queued" and a["attempts"] == 1 and a["error"]["code"] == "worker_lost"
        new_worker.run_pending()
        assert c.get(f"/api/v1/analyses/{aid}").json()["status"] == "succeeded"
        # A job whose worker keeps dying ends as failed(worker_lost).
        aid2 = start(c, csrf)
        for _ in range(app_settings.app.jobs.max_attempts):
            assert runner.claim()["id"] == aid2
            db.execute("UPDATE analyses SET lease_expires_at = 0 WHERE id = ?", (aid2,))
            new_worker.recover_expired()
        a2 = c.get(f"/api/v1/analyses/{aid2}").json()
        assert a2["status"] == "failed" and a2["error"]["code"] == "worker_lost"


def test_processing_timeout(app_settings):
    app_settings.app.jobs.job_timeout_s = 0.5
    app, runner, _ = make(app_settings, StubEngine(mode="slow"))
    with TestClient(app) as c:
        csrf = register(c)
        aid = start(c, csrf)
        runner.run_pending()
        a = c.get(f"/api/v1/analyses/{aid}").json()
        assert a["status"] == "failed" and a["error"]["code"] == "analysis_timeout"


def test_per_user_active_limit_and_subject_validation(app_settings):
    app_settings.app.jobs.max_active_per_user = 2
    app, _, _ = make(app_settings)
    with TestClient(app) as c:
        csrf = register(c)
        h = {"X-CSRF-Token": csrf}
        start(c, csrf)
        start(c, csrf)
        uid = upload(c, csrf, wav_bytes())
        r = c.post("/api/v1/analyses", json={"upload_id": uid}, headers=h)
        assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_active_analyses"
        r = c.post("/api/v1/analyses", json={"upload_id": uid, "subject_mode": "multiple_known", "pet_ids": []}, headers=h)
        assert r.json()["error"]["code"] == "subject_mismatch"


def test_account_deletion_removes_all_data(app_settings, tmp_path):
    app_settings.app.storage.media_root = str(tmp_path / "media")
    app, runner, _ = make(app_settings)
    with TestClient(app) as c:
        csrf = register(c)
        h = {"X-CSRF-Token": csrf}
        c.patch("/api/v1/settings", json={"retain_media_days": 3}, headers=h)
        start(c, csrf)
        runner.run_pending()
        assert list((tmp_path / "media").rglob("analysis-*"))
        r = c.post("/api/v1/auth/delete-account", json={"password": PW, "confirm": "DELETE"}, headers=h)
        assert r.status_code == 204
        assert c.get("/api/v1/auth/me").status_code == 401
        db = app.state.db
        for table in ("users", "pets", "analyses", "analysis_results", "uploads", "sessions"):
            assert db.one(f"SELECT count(*) AS n FROM {table}")["n"] == 0, table
        assert not list((tmp_path / "media").rglob("*.*"))


def test_cleanup_expires_uploads_and_media(app_settings):
    app, runner, _ = make(app_settings)
    store, db = app.state.store, app.state.db
    with TestClient(app) as c:
        csrf = register(c)
        h = {"X-CSRF-Token": csrf}
        c.patch("/api/v1/settings", json={"retain_media_days": 1}, headers=h)
        aid = start(c, csrf)
        runner.run_pending()
        up = c.post("/api/v1/uploads", json={"filename": "a.wav", "size_bytes": 100}, headers=h).json()["id"]
        db.execute("UPDATE uploads SET expires_at = 0 WHERE id = ?", (up,))
        db.execute("UPDATE analyses SET media_retained_until = 1 WHERE id = ?", (aid,))
        stats = store.cleanup()
        assert stats["uploads_expired"] == 1 and stats["media_expired"] == 1
        assert c.get(f"/api/v1/analyses/{aid}").json()["media_available"] is False


def test_health_ready_headers_and_log_hygiene(app_settings, caplog, tmp_path):
    app_settings.paths.models_root = str(tmp_path / "no-models")
    app, _, _ = make(app_settings)
    with TestClient(app) as c, caplog.at_level(logging.INFO):
        r = c.get("/api/v1/health")
        assert r.status_code == 200 and r.headers["X-Request-ID"] and "default-src 'self'" in r.headers["Content-Security-Policy"]
        assert "camera=(self)" in r.headers["Permissions-Policy"]
        rd = c.get("/api/v1/ready")
        assert rd.status_code == 503 and rd.json()["checks"]["models"]["ok"] is False
        csrf = register(c)
        upload(c, csrf, wav_bytes(), name="secret-family-name.wav")
        assert c.get("/").json()["error"]["code"] == "frontend_not_built"
    text = "\n".join(r.getMessage() + json.dumps({k: str(v) for k, v in r.__dict__.items()}) for r in caplog.records)
    assert "secret-family-name" not in text and PW not in text
