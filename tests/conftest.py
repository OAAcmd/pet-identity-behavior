"""Shared fixtures.

Markers (see pyproject.toml):
* ``synthetic`` - synthetic or mocked inputs; NOT evidence of accuracy.
* ``integration`` - real models and/or real public media; skipped when the
  required artifacts have not been fetched.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from petperception.config import REPO_ROOT, load_settings
from petperception.registry import check_model


@pytest.fixture(scope="session")
def settings():
    return load_settings(REPO_ROOT / "configs" / "cpu.yaml")


@pytest.fixture(scope="session")
def models_ready(settings) -> bool:
    return check_model(settings, settings.audio.yamnet.model_id).ready and check_model(settings, settings.vision.detector).ready


@pytest.fixture(scope="session")
def engine(settings, models_ready):
    if not models_ready:
        pytest.skip("models not fetched: uv run petperception fetch-models")
    from petperception.pipeline import AnalysisEngine

    return AnalysisEngine(settings).load()


@pytest.fixture(scope="session")
def esc50_root(settings) -> Path:
    root = settings.paths.resolve("data_root") / "esc50"
    if not (root / "meta" / "esc50.csv").exists():
        pytest.skip("ESC-50 not fetched: uv run petperception fetch-data esc50")
    return root


@pytest.fixture(scope="session")
def oi_images(settings) -> Path:
    d = settings.paths.resolve("data_root") / "openimages" / "images"
    if not d.exists():
        pytest.skip("Open Images subset not fetched: uv run petperception fetch-data openimages-catdog")
    return d


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def synth_media(tmp_path_factory) -> dict[str, Path]:
    """Synthetic media generated with FFmpeg (marked synthetic in tests using it)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg missing")
    d = tmp_path_factory.mktemp("synth")
    out: dict[str, Path] = {}
    # Flash at t=1.00 s in video; 1 kHz click at t=1.00 s in audio.
    ffmpeg("-f", "lavfi", "-i", "color=black:s=160x120:r=25:d=3", "-f", "lavfi",
           "-i", "aevalsrc='if(between(t,1,1.01),0.9*sin(2*PI*1000*t),0)':s=48000:d=3",
           "-vf", "drawbox=color=white:t=fill:enable='between(t,1,1.039)'", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", str(d / "sync.mp4"))
    out["sync_mp4"] = d / "sync.mp4"
    ffmpeg("-itsoffset", "0.75", "-i", str(d / "sync.mp4"), "-map", "0", "-c", "copy", str(d / "sync_offset.mkv"))
    out["sync_offset_mkv"] = d / "sync_offset.mkv"
    ffmpeg("-i", str(d / "sync.mp4"), "-c:v", "libvpx-vp9", "-b:v", "200k", "-c:a", "libopus", str(d / "sync.webm"))
    out["sync_webm"] = d / "sync.webm"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "noaudio.mp4"))
    out["noaudio"] = d / "noaudio.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "silent.mp4"))
    out["silent"] = d / "silent.mp4"
    (d / "corrupt.mp4").write_bytes((d / "sync.mp4").read_bytes()[:3000])
    out["corrupt"] = d / "corrupt.mp4"
    sf.write(d / "tone8k.wav", 0.3 * np.sin(2 * np.pi * 440 * np.arange(12000) / 8000).astype(np.float32), 8000)
    out["tone8k"] = d / "tone8k.wav"
    sf.write(d / "silence.wav", np.zeros(32000, np.float32), 16000)
    out["silence_wav"] = d / "silence.wav"
    ffmpeg("-f", "lavfi", "-i", "color=gray:s=64x64:d=1", "-frames:v", "1", str(d / "cover.png"))
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=300:sample_rate=22050:duration=1", "-i", str(d / "cover.png"), "-map", "0:a", "-map", "1:v",
           "-c:a", "libmp3lame", "-c:v", "png", "-disposition:v", "attached_pic", str(d / "with_cover.mp3"))
    out["mp3_cover"] = d / "with_cover.mp3"
    (d / "notmedia.wav").write_text("this is not audio")
    out["notmedia"] = d / "notmedia.wav"
    return out


@pytest.fixture
def app_settings(settings, tmp_path):
    s = settings.model_copy(deep=True)
    s.app.storage.database_path = str(tmp_path / "db.sqlite3")
    s.app.storage.media_root = str(tmp_path / "media")
    s.app.storage.min_free_disk_bytes = 0
    s.app.auth.cookie_secure = False
    s.app.jobs.retry_backoff_s = 0.0
    s.serve.static_dir = str(tmp_path / "no-web")
    return s
