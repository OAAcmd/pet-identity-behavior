"""Product backend integration tests with the REAL engine (real models, real media)."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient
from test_app_backend import login, make, register, start

from petperception.app.main import create_app
from petperception.config import REPO_ROOT


@pytest.mark.integration
def test_real_engine_end_to_end_with_inprocess_worker(app_settings, models_ready, esc50_root):
    if not models_ready:
        pytest.skip("models not fetched")
    app = create_app(app_settings, start_workers=True)  # real engine, real worker thread
    with TestClient(app) as c:
        csrf = register(c)
        aid = start(c, csrf, data=(esc50_root / "audio" / "1-34094-A-5.wav").read_bytes())
        t0 = time.time()
        while time.time() - t0 < 120:
            a = c.get(f"/api/v1/analyses/{aid}").json()
            if a["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.25)
        assert a["status"] == "succeeded", a
        assert any(e["label"] == "cat_vocalization" for e in a["result"]["observations"]["sound_events"])
        assert c.get("/api/v1/ready").json()["checks"]["worker"]["ok"] is True


@pytest.mark.integration
def test_separate_worker_process(app_settings, tmp_path, models_ready):
    if not models_ready:
        pytest.skip("models not fetched")
    app, _, _ = make(app_settings)
    with TestClient(app) as c:
        csrf = register(c)
        aid = start(c, csrf)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(json.dumps(app_settings.model_dump(mode="json")))
    out = subprocess.run([sys.executable, "-m", "petperception.cli", "--config", str(cfg), "worker", "--once"], capture_output=True, text=True,
                         cwd=REPO_ROOT, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1])["processed"] == 1
    with TestClient(app) as c:
        login(c, "alice")
        assert c.get(f"/api/v1/analyses/{aid}").json()["status"] == "succeeded"
