"""API contract tests: auth, upload validation, job lifecycle, deterministic deletion."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from petperception.serve.app import create_app

TOKEN_ENV = "PETPERCEPTION_API_TOKEN"


class StubEngine:
    """Synthetic engine: blocks until released so in-flight deletion can be tested."""

    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()

    def capabilities(self):
        return {"engine_version": "stub", "components": [], "research_only": True}

    def analyze(self, path, **kw):
        self.started.set()
        self.release.wait(10)
        raise RuntimeError("stub engine produces no results")


@pytest.fixture
def api_settings(settings, tmp_path):
    s = settings.model_copy(deep=True)
    s.paths.jobs_root = str(tmp_path / "jobs")
    s.serve.max_upload_bytes = 200_000
    return s


def _wav(tmp_path, seconds=1.0, sr=16000):
    p = tmp_path / "tone.wav"
    sf.write(p, (0.2 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * sr)) / sr)).astype(np.float32), sr)
    return p


def _wait(client, aid, want=("succeeded", "failed"), timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get(f"/v1/analyses/{aid}").json()
        if st.get("status") in want:
            return st
        time.sleep(0.1)
    raise TimeoutError(aid)


@pytest.mark.synthetic
def test_health_and_auth(api_settings, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "s3cret")
    with TestClient(create_app(api_settings, engine=StubEngine())) as c:
        assert c.get("/v1/health").json()["status"] == "ok"
        assert c.get("/v1/capabilities").status_code == 401
        r = c.get("/v1/capabilities", headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200 and r.json()["upload"]["url_fetching"] is False


@pytest.mark.synthetic
def test_non_loopback_without_token_is_forbidden(api_settings, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with TestClient(create_app(api_settings, engine=StubEngine()), client=("203.0.113.7", 40000)) as c:
        assert c.get("/v1/capabilities").status_code == 403
        assert c.get("/v1/health").status_code == 200


@pytest.mark.synthetic
def test_invalid_uploads(api_settings, tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with TestClient(create_app(api_settings, engine=StubEngine())) as c:
        r = c.post("/v1/analyses", files={"file": ("x.exe", b"MZ....", "application/octet-stream")})
        assert r.status_code == 415 and r.json()["error_code"] == "unsupported_extension"
        r = c.post("/v1/analyses", files={"file": ("big.wav", b"0" * 300_000, "audio/wav")})
        assert r.status_code == 413
        r = c.post("/v1/analyses", files={"file": ("bad.wav", b"not really a wav file", "audio/wav")})
        assert r.status_code == 422 and r.json()["error_code"] in ("corrupt_input", "unsupported_input")
        r = c.post("/v1/analyses", files={"file": ("empty.wav", b"", "audio/wav")})
        assert r.status_code == 422
        wav = _wav(tmp_path).read_bytes()
        r = c.post("/v1/analyses", files={"file": ("t.wav", wav, "audio/wav")}, data={"zones": "{not json"})
        assert r.status_code == 422 and r.json()["error_code"] == "invalid_zones"
        r = c.post("/v1/analyses", files={"file": ("t.wav", wav, "audio/wav")}, data={"modalities": "smell"})
        assert r.status_code == 422 and r.json()["error_code"] == "invalid_modalities"
        assert not any((tmp_path / "jobs").iterdir()), "rejected uploads must leave no files behind"


@pytest.mark.synthetic
def test_delete_while_running_is_deterministic(api_settings, tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    eng = StubEngine()
    with TestClient(create_app(api_settings, engine=eng)) as c:
        aid = c.post("/v1/analyses", files={"file": ("t.wav", _wav(tmp_path).read_bytes(), "audio/wav")}).json()["analysis_id"]
        assert eng.started.wait(10)
        assert c.get(f"/v1/analyses/{aid}/result").status_code == 409
        r = c.delete(f"/v1/analyses/{aid}")
        assert r.status_code == 200 and r.json()["status"] == "deletion_scheduled"
        assert c.get(f"/v1/analyses/{aid}").status_code == 404  # never readable again
        eng.release.set()
        t0 = time.time()
        while any((tmp_path / "jobs").iterdir()) and time.time() - t0 < 10:
            time.sleep(0.05)
        assert not any((tmp_path / "jobs").iterdir())
        assert c.delete(f"/v1/analyses/{aid}").status_code == 404


@pytest.mark.integration
def test_full_lifecycle_with_real_engine(api_settings, engine, tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with TestClient(create_app(api_settings, engine=engine)) as c:
        r = c.post("/v1/analyses", files={"file": ("tone.wav", _wav(tmp_path, 2.0).read_bytes(), "audio/wav")})
        assert r.status_code == 202
        aid = r.json()["analysis_id"]
        assert _wait(c, aid)["status"] == "succeeded"
        res = c.get(f"/v1/analyses/{aid}/result").json()
        assert res["schema_version"] == "1.0.0" and res["analysis_id"] == aid
        assert res["modalities"]["video_state"] == "absent" and res["fusion"]["multimodal_accuracy"] == "not_evaluated"
        assert res["observations"]["sound_events"] == []  # a pure tone is not a cat/dog sound
        jobdir = next((tmp_path / "jobs").iterdir())
        assert not list(jobdir.glob("input*")), "uploaded media must be deleted after analysis"
        assert c.delete(f"/v1/analyses/{aid}").json()["status"] == "deleted"
        assert not jobdir.exists()
        assert c.get(f"/v1/analyses/{aid}/result").status_code == 404
