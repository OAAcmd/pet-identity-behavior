"""Ingestion, timestamp alignment and input-state tests (synthetic media)."""

import numpy as np
import pytest

from petperception.config import LimitsConfig
from petperception.ingest import media as M

pytestmark = pytest.mark.synthetic
L = LimitsConfig()


@pytest.mark.parametrize("key", ["sync_mp4", "sync_offset_mkv", "sync_webm"])
def test_audio_video_timestamp_alignment(synth_media, key):
    """A flash and a click authored at the same instant land within 30 ms on the shared timeline."""
    mp = M.probe(synth_media[key], L)
    a = M.decode_audio(mp, L)
    click = a.time_of(int(np.argmax(np.abs(a.samples) > 0.3)))
    frames = list(M.iter_frames(mp, 100.0, 160, L))
    flash = next(f.t_s for f in frames if f.rgb.mean() > 128)
    assert abs(click - flash) < 0.03, (click, flash)
    if key == "sync_offset_mkv":  # original timestamps preserved, not re-based to zero
        assert flash == pytest.approx(1.75, abs=0.03)


def test_missing_audio_is_absent(synth_media):
    mp = M.probe(synth_media["noaudio"], L)
    assert mp.audio is None
    assert M.decode_audio(mp, L).state == "absent"


def test_silent_audio_is_silent_not_absent(synth_media):
    mp = M.probe(synth_media["silent"], L)
    a = M.decode_audio(mp, L)
    assert a.state == "silent" and a.silent_fraction >= 0.999


def test_corrupt_input_raises_stable_code(synth_media):
    with pytest.raises(M.MediaError) as e:
        M.probe(synth_media["corrupt"], L)
    assert e.value.code == "corrupt_input"


def test_non_media_file_rejected(synth_media):
    with pytest.raises(M.MediaError) as e:
        M.probe(synth_media["notmedia"], L)
    assert e.value.code in ("corrupt_input", "unsupported_input")


def test_size_and_duration_limits(synth_media):
    with pytest.raises(M.MediaError) as e:
        M.probe(synth_media["sync_mp4"], LimitsConfig(max_file_bytes=100))
    assert e.value.code == "file_too_large"
    with pytest.raises(M.MediaError) as e:
        M.probe(synth_media["sync_mp4"], LimitsConfig(max_duration_s=1.0))
    assert e.value.code == "duration_too_long"


def test_cover_art_is_not_video(synth_media):
    mp = M.probe(synth_media["mp3_cover"], L)
    assert mp.video is None and mp.audio is not None


def test_native_sample_rate_preserved(synth_media):
    mp = M.probe(synth_media["tone8k"], L)
    a = M.decode_audio(mp, L)
    assert a.sample_rate_hz == 8000 and a.state == "present"
    assert len(M.resample(a.samples, 8000, 16000)) == 2 * len(a.samples)


def test_frame_sampling_respects_limit(synth_media):
    rep = M.FrameSamplingReport(25.0)
    frames = list(M.iter_frames(M.probe(synth_media["sync_mp4"], L), 25.0, 160, LimitsConfig(max_video_frames=5), rep))
    assert len(frames) == 5 and rep.truncated_at_limit
    assert all(b.t_s > a.t_s for a, b in zip(frames, frames[1:]))


def test_frame_gap_detection():
    pts = np.array([0.0, 0.04, 0.08, 0.12, 0.40, 0.44, 0.48])
    gaps = M.frame_gaps(pts, factor=2.5)
    assert gaps == [(0.12, 0.40)]
