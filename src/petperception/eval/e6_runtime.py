"""E6: runtime on the actual workspace hardware.

Separates (a) cold start = interpreter start + imports + model loading,
measured in a fresh subprocess, from (b) warm per-file analysis with the
engine already loaded (median of repeated runs). Reports per-stage timings,
real-time factor, and peak resident memory (``ru_maxrss``).
"""

from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

from ..config import Settings
from ..pipeline import AnalysisEngine
from ..utils import utc_now, write_json

COLD_SNIPPET = """
import json, time, resource
t0 = time.perf_counter()
from petperception.config import load_settings
from petperception.pipeline import AnalysisEngine
t1 = time.perf_counter()
eng = AnalysisEngine(load_settings()).load()
t2 = time.perf_counter()
print(json.dumps({"import_s": t1 - t0, "model_load_s": t2 - t1, "engine_reported_load_ms": eng.model_load_ms,
                  "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}))
"""


def hardware() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count_logical": os.cpu_count()}
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                info["mem_total_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except OSError:
        pass
    info["gpu"] = "none detected (no /dev/nvidia*, no nvidia-smi)" if not any(Path("/dev").glob("nvidia*")) else "present"
    try:
        info["ffmpeg"] = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    except OSError:
        info["ffmpeg"] = "missing"
    return info


def run(settings: Settings, out_dir: Path, repeats: int = 3) -> dict:
    media_dir = settings.paths.resolve("data_root").parent / "examples"
    base = media_dir / "s01_food_zone_meow.mp4"
    if not base.exists():
        raise FileNotFoundError("scenario media missing; run: uv run petperception evaluate e5")
    long = media_dir / "e6_long_64s.mp4"
    if not long.exists():
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-stream_loop", "7", "-i", str(base), "-c", "copy", str(long)], check=True)
    inputs = {
        "video_8s_640x480_with_audio": base,
        "video_64s_640x480_with_audio": long,
        "audio_only_5s_44k": media_dir / "s07_audio_only_meow.wav",
        "still_image_jpeg": settings.paths.resolve("data_root") / "openimages" / "images" / "d60e1c25e87d8b45.jpg",
    }
    cold = json.loads(subprocess.run([sys.executable, "-c", COLD_SNIPPET], capture_output=True, text=True, check=True,
                                     env={**os.environ, "PYTHONWARNINGS": "ignore"}).stdout.strip().splitlines()[-1])
    eng = AnalysisEngine(settings).load()
    rows = {}
    for name, path in inputs.items():
        walls, stages, ser = [], [], []
        res = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            res = eng.analyze(path)
            t1 = time.perf_counter()
            res.model_dump_json()
            ser.append((time.perf_counter() - t1) * 1000.0)
            walls.append(time.perf_counter() - t0)
            stages.append(res.timings_ms)
        dur = res.media.duration_s or 0.0
        keys = sorted({k for s in stages for k in s if k != "model_load_ms_engine_startup"})
        rows[name] = {
            "media_duration_s": dur,
            "sampled_frames": res.observations.sampled_frames,
            "audio_windows": res.observations.audio_summary.windows_scored,
            "wall_s_median": statistics.median(walls),
            "wall_s_all": walls,
            "real_time_factor": (statistics.median(walls) / dur) if dur and not (res.media.video and res.media.video.is_still_image) else None,
            "stage_ms_median": {k: statistics.median(s.get(k, 0.0) for s in stages) for k in keys},
            "output_json_serialization_ms_median": statistics.median(ser),
        }
    out = {
        "experiment": "E6",
        "hardware": hardware(),
        "config": {"profile": settings.profile, "num_threads": settings.runtime.num_threads, "sample_fps": settings.vision.sample_fps,
                   "detector": settings.vision.detector, "audio_backend": settings.audio.yamnet.model_id},
        "cold_start_fresh_process": cold,
        "warm_runs": rows,
        "peak_rss_mb_this_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "notes": ["Output serialization is included in 'wall_s' but not in stage timings.",
                  "Video decode, detection and tracking share one streaming stage ('video_decode_detect_track')."],
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    write_json(out_dir / "metrics.json", out)
    return out
