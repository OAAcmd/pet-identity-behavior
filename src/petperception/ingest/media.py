"""Media probing, decoding and synchronization on one shared timeline.

Timeline convention
-------------------
All timestamps are seconds on the *container presentation timeline* reported
by FFmpeg (``pts_time``). Audio sample ``i`` of a stream whose first decoded
sample has ``start_time = s`` lies at ``s + i / sample_rate``. Video frames
keep their original ``pts_time`` (frames are *selected*, never re-timed).
Audio is decoded at its native sample rate; resampling happens only inside the
model adapter that needs a specific rate (YAMNet: 16 kHz). Upsampling an
8 kHz recording does not recreate content above 4 kHz; this is reported as a
bandwidth limitation.

Supported inputs: anything the installed FFmpeg can demux/decode. Tested
containers/codecs: WAV (PCM), FLAC, OGG/Vorbis, MP3, MP4/MOV (H.264 + AAC),
MKV/WebM (VP8/VP9 + Opus/Vorbis), AVI (MJPEG/PCM), and still images
(JPEG/PNG) treated as a single frame at t = 0.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.signal import resample_poly

from ..config import LimitsConfig

AudioState = Literal["present", "absent", "silent", "corrupt", "disabled", "too_short"]
VideoState = Literal["present", "absent", "no_decodable_frames", "corrupt", "disabled"]

IMAGE_CODECS = {"mjpeg", "png", "webp", "bmp", "tiff"}
IMAGE_FORMATS = {"image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe", "tiff_pipe"}


class MediaError(Exception):
    """Input-level failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class AudioStreamInfo:
    index: int
    codec: str
    sample_rate_hz: int
    channels: int
    start_time_s: float
    duration_s: float | None


@dataclass
class VideoStreamInfo:
    index: int
    codec: str
    width: int
    height: int
    avg_fps: float | None
    start_time_s: float
    duration_s: float | None
    nb_frames: int | None
    rotation_deg: int
    is_still_image: bool


@dataclass
class MediaProbe:
    path: str
    size_bytes: int
    format_name: str
    duration_s: float | None
    start_time_s: float
    audio: AudioStreamInfo | None
    video: VideoStreamInfo | None
    n_audio_streams: int
    n_video_streams: int


def _f(x: object, default: float | None = None) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _rate(s: str | None) -> float | None:
    if not s or s in ("0/0", "0/1"):
        return None
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b) if float(b) else None
    return _f(s)


def probe(path: str | Path, limits: LimitsConfig) -> MediaProbe:
    """Probe a file with ffprobe before any decoding; enforce size/duration limits."""
    p = Path(path)
    if not p.is_file():
        raise MediaError("not_found", f"{p} is not a file")
    size = p.stat().st_size
    if size == 0:
        raise MediaError("corrupt_input", "empty file")
    if size > limits.max_file_bytes:
        raise MediaError("file_too_large", f"{size} bytes exceeds limit {limits.max_file_bytes}")
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(p)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=limits.ffprobe_timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise MediaError("probe_timeout", "ffprobe timed out") from exc
    if r.returncode != 0:
        raise MediaError("corrupt_input", f"ffprobe failed: {r.stderr.strip()[:300]}")
    info = json.loads(r.stdout or "{}")
    fmt = info.get("format", {})
    streams = info.get("streams", [])
    if not streams:
        raise MediaError("corrupt_input", "no decodable streams found")
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    video_streams = [
        s for s in streams
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic", 0)
    ]
    fmt_name = fmt.get("format_name", "")
    a = None
    if audio_streams:
        s = audio_streams[0]
        a = AudioStreamInfo(
            index=int(s["index"]), codec=s.get("codec_name", "unknown"),
            sample_rate_hz=int(s.get("sample_rate", 0) or 0), channels=int(s.get("channels", 0) or 0),
            start_time_s=_f(s.get("start_time"), 0.0) or 0.0, duration_s=_f(s.get("duration")),
        )
        if a.channels > limits.max_audio_channels:
            raise MediaError("unsupported_input", f"{a.channels} audio channels exceeds limit")
    v = None
    if video_streams:
        s = video_streams[0]
        rot = 0
        for sd in s.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(round(float(sd["rotation"])))
        if "rotate" in (s.get("tags") or {}):
            rot = int(s["tags"]["rotate"])
        codec = s.get("codec_name", "unknown")
        still = fmt_name in IMAGE_FORMATS or (codec in IMAGE_CODECS and _f(s.get("duration")) in (None, 0.0) and fmt_name.endswith("_pipe"))
        v = VideoStreamInfo(
            index=int(s["index"]), codec=codec, width=int(s.get("width", 0) or 0), height=int(s.get("height", 0) or 0),
            avg_fps=_rate(s.get("avg_frame_rate")), start_time_s=_f(s.get("start_time"), 0.0) or 0.0,
            duration_s=_f(s.get("duration")), nb_frames=int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None,
            rotation_deg=rot, is_still_image=bool(still),
        )
    duration = _f(fmt.get("duration"))
    if duration is not None and duration > limits.max_duration_s:
        raise MediaError("duration_too_long", f"duration {duration:.1f}s exceeds limit {limits.max_duration_s}s")
    if a is None and v is None:
        raise MediaError("unsupported_input", "no audio or video stream")
    return MediaProbe(
        path=str(p), size_bytes=size, format_name=fmt_name, duration_s=duration,
        start_time_s=_f(fmt.get("start_time"), 0.0) or 0.0, audio=a, video=v,
        n_audio_streams=len(audio_streams), n_video_streams=len(video_streams),
    )


@dataclass
class AudioTrack:
    state: AudioState
    samples: np.ndarray  # float32 mono at native rate
    sample_rate_hz: int
    start_s: float
    channels_original: int = 0
    decode_warnings: list[str] = field(default_factory=list)
    silent_fraction: float | None = None
    rms_dbfs: float | None = None
    peak_dbfs: float | None = None
    clipping_fraction: float | None = None

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate_hz if self.sample_rate_hz else 0.0

    def time_of(self, sample_index: int) -> float:
        return self.start_s + sample_index / self.sample_rate_hz


def decode_audio(mp: MediaProbe, limits: LimitsConfig, silence_dbfs: float = -60.0) -> AudioTrack:
    """Decode the first audio stream to float32 mono at its native sample rate."""
    if mp.audio is None:
        return AudioTrack("absent", np.zeros(0, np.float32), 0, 0.0)
    a = mp.audio
    sr = a.sample_rate_hz
    if sr <= 0:
        return AudioTrack("corrupt", np.zeros(0, np.float32), 0, a.start_time_s, a.channels, ["unknown sample rate"])
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", mp.path, "-map", f"0:{a.index}",
        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=limits.decode_timeout_s)
    except subprocess.TimeoutExpired:
        return AudioTrack("corrupt", np.zeros(0, np.float32), sr, a.start_time_s, a.channels, ["decode timeout"])
    warnings = [ln for ln in r.stderr.decode(errors="replace").splitlines() if ln.strip()][:20]
    x = np.frombuffer(r.stdout, dtype="<f4").astype(np.float32, copy=True)
    if r.returncode != 0 and x.size == 0:
        return AudioTrack("corrupt", x, sr, a.start_time_s, a.channels, warnings or ["ffmpeg failed"])
    if x.size == 0:
        return AudioTrack("corrupt", x, sr, a.start_time_s, a.channels, warnings or ["no samples decoded"])
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        warnings.append("non-finite samples replaced with 0")
    track = AudioTrack("present", x, sr, first_decoded_audio_pts(mp, limits), a.channels, warnings)
    _audio_levels(track, silence_dbfs)
    if track.silent_fraction is not None and track.silent_fraction >= 0.999:
        track.state = "silent"
    return track


def first_decoded_audio_pts(mp: MediaProbe, limits: LimitsConfig) -> float:
    """Presentation time of the first *decoded* audio sample.

    The stream ``start_time`` can include encoder priming / Opus pre-skip that
    the decoder discards, so the first decoded frame's timestamp is the anchor
    for sample-to-time mapping. Falls back to the stream start time.
    """
    assert mp.audio is not None
    cmd = ["ffprobe", "-v", "error", "-select_streams", str(mp.audio.index), "-show_frames",
           "-read_intervals", "%+#4", "-show_entries", "frame=best_effort_timestamp_time,pts_time",
           "-of", "csv=p=0", mp.path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=limits.ffprobe_timeout_s)
        for ln in r.stdout.splitlines():
            for tok in ln.split(","):
                v = _f(tok.strip())
                if v is not None:
                    return v
    except subprocess.TimeoutExpired:
        pass
    return mp.audio.start_time_s


def _audio_levels(track: AudioTrack, silence_dbfs: float, frame_s: float = 0.05) -> None:
    x = track.samples
    n = max(1, int(track.sample_rate_hz * frame_s))
    m = len(x) // n
    eps = 1e-12
    if m == 0:
        frames = x[None, :]
    else:
        frames = x[: m * n].reshape(m, n)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + eps)
    db = 20 * np.log10(rms + eps)
    track.silent_fraction = float(np.mean(db < silence_dbfs))
    track.rms_dbfs = float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps)))
    track.peak_dbfs = float(20 * np.log10(np.max(np.abs(x)) + eps))
    track.clipping_fraction = float(np.mean(np.abs(x) >= 0.999))


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling (scipy) used only by model adapters."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


@dataclass
class VideoFrame:
    index: int
    t_s: float  # original presentation timestamp
    rgb: np.ndarray  # HxWx3 uint8


@dataclass
class FrameSamplingReport:
    requested_fps: float
    frames_returned: int = 0
    decode_warnings: list[str] = field(default_factory=list)
    truncated_at_limit: bool = False
    output_size: tuple[int, int] = (0, 0)


def _output_size(v: VideoStreamInfo, max_side: int) -> tuple[int, int]:
    w, h = v.width, v.height
    if abs(v.rotation_deg) % 180 == 90:
        w, h = h, w
    scale = min(1.0, max_side / max(w, h)) if max(w, h) > 0 else 1.0
    ow, oh = max(2, int(round(w * scale / 2)) * 2), max(2, int(round(h * scale / 2)) * 2)
    return ow, oh


_PTS_RE = re.compile(r"pts_time:\s*([-0-9.eE+]+)")


def iter_frames(mp: MediaProbe, sample_fps: float, max_side: int, limits: LimitsConfig,
                report: FrameSamplingReport | None = None) -> Iterator[VideoFrame]:
    """Yield source frames spaced at least 1/sample_fps apart, with original pts."""
    if mp.video is None:
        return
    rep = report or FrameSamplingReport(sample_fps)
    ow, oh = _output_size(mp.video, max_side)
    rep.output_size = (ow, oh)
    interval = 1.0 / sample_fps
    vf = (
        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval:.6f})',"
        f"scale={ow}:{oh}:flags=area,showinfo"
    )
    cmd = [
        # -copyts keeps original presentation timestamps inside the filter graph
        # (otherwise FFmpeg re-bases them to the container start time).
        "ffmpeg", "-nostdin", "-v", "info", "-copyts", "-i", mp.path, "-map", f"0:{mp.video.index}",
        "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pts: list[float] = []
    errors: list[str] = []
    cond = threading.Condition()

    def read_stderr() -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.decode(errors="replace")
            m = _PTS_RE.search(line) if "Parsed_showinfo" in line else None
            with cond:
                if m:
                    pts.append(float(m.group(1)))
                elif "error" in line.lower() or "invalid" in line.lower():
                    errors.append(line.strip()[:200])
                cond.notify_all()
        with cond:
            pts.append(float("nan"))  # sentinel: stderr closed
            cond.notify_all()

    th = threading.Thread(target=read_stderr, daemon=True)
    th.start()
    frame_bytes = ow * oh * 3
    i = 0
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            with cond:
                cond.wait_for(lambda: len(pts) > i, timeout=10.0)
                t = pts[i] if len(pts) > i else float("nan")
            if not np.isfinite(t):
                # Timestamp unavailable: fall back to the sampling grid and flag it.
                t = mp.video.start_time_s + i * interval
                errors.append("missing pts for a sampled frame; grid timestamp used")
            frame = np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)
            yield VideoFrame(i, t, frame)
            i += 1
            if i >= limits.max_video_frames:
                rep.truncated_at_limit = True
                break
    finally:
        proc.kill()
        proc.wait()
        th.join(timeout=5)
        rep.frames_returned = i
        rep.decode_warnings = errors[:20]


def packet_timestamps(mp: MediaProbe, limits: LimitsConfig) -> np.ndarray:
    """Video packet pts (seconds, sorted) without decoding, for dropped-frame checks."""
    if mp.video is None or mp.video.is_still_image:
        return np.zeros(0)
    cmd = ["ffprobe", "-v", "error", "-select_streams", str(mp.video.index), "-show_entries",
           "packet=pts_time", "-of", "csv=p=0", mp.path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=limits.ffprobe_timeout_s)
    except subprocess.TimeoutExpired:
        return np.zeros(0)
    vals = [_f(ln.strip().rstrip(",")) for ln in r.stdout.splitlines() if ln.strip()]
    arr = np.array([v for v in vals if v is not None], dtype=np.float64)
    return np.sort(arr)


def frame_gaps(pts: np.ndarray, factor: float) -> list[tuple[float, float]]:
    """Intervals where consecutive frames are > factor x the median spacing apart."""
    if pts.size < 3:
        return []
    d = np.diff(pts)
    med = float(np.median(d[d > 0])) if np.any(d > 0) else 0.0
    if med <= 0:
        return []
    idx = np.where(d > factor * med)[0]
    return [(float(pts[i]), float(pts[i + 1])) for i in idx]
