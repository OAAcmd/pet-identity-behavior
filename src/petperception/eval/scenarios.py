"""Synthetic-pairing scenarios for integration tests (E5, E6, offline examples).

SYNTHETIC: every scenario pairs *real* public media (Open Images photos, ESC-50
clips) that were never recorded together. Detection and sound tagging run for
real on real pixels/waveforms, but the pairing, timing, camera motion and
compositing are constructed here. These scenarios test behavior (abstention,
attribution, conflict handling, missing modalities); they are NOT evidence of
multimodal accuracy and must never be reported as such.

Generated media are written under ``var/examples`` (git-ignored) and can be
regenerated deterministically from the fetched datasets.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from ..config import Settings
from ..context.zones import ZoneConfig

W, H, FPS, DUR = 640, 480, 10, 8.0
AUDIO_SR = 44100

IMAGES = {
    "cat": "d60e1c25e87d8b45",
    "dog": "1b7e7fe61e350040",
    "cat_and_dog": "bdf391a6f4b1840a",
    "no_pet": "35e7f8550ba0d7e7",
    "background": "c1a0d7aed52c0c2c",
}
CLIPS = {
    "meow": "4-133047-A-5.wav",
    "cat_growl": "3-146972-A-5.wav",
    "bark": "4-192236-A-0.wav",
    "vacuum": "5-182012-A-36.wav",
}


@dataclass
class Scenario:
    scenario_id: str
    description: str
    media: Path
    zones: ZoneConfig | None
    expect: dict[str, object]
    sources: list[dict[str, str]] = field(default_factory=list)
    synthetic: bool = True
    modalities: tuple[str, ...] = ("audio", "video")


def _image_attr(manifest_dir: Path, image_id: str) -> dict[str, str]:
    with open(manifest_dir / "openimages_catdog_images.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["image_id"] == image_id:
                return {"type": "image", "dataset": "Open Images V7 validation", "id": image_id, "author": r["author"],
                        "license": r["license"], "url": r["original_landing_url"], "title": r["title"]}
    raise KeyError(image_id)


def _clip_attr(manifest_dir: Path, filename: str) -> dict[str, str]:
    with open(manifest_dir / "esc50_files.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["filename"] == filename:
                return {"type": "audio", "dataset": "ESC-50 @ 33c8ce9e", "id": filename, "author": r["author"],
                        "license": r["source_license"], "url": r["source_url"], "category": r["category"]}
    raise KeyError(filename)


def _gt_box(manifest_dir: Path, image_id: str, label: str) -> np.ndarray:
    with open(manifest_dir / "openimages_catdog_boxes.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["image_id"] == image_id and r["label"] == label:
                return np.array([float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])])
    raise KeyError((image_id, label))


def _fit(img: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Letterbox an RGB image into W x H; returns frame and (scale_x, scale_y, off_x, off_y) in normalized units."""
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = np.full((H, W, 3), 96, np.uint8)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, (nw / W, nh / H, ox / W, oy / H)


def _map_box(b: np.ndarray, t: tuple[float, float, float, float]) -> np.ndarray:
    sx, sy, ox, oy = t
    return np.array([b[0] * sx + ox, b[1] * sy + oy, b[2] * sx + ox, b[3] * sy + oy])


def _zone(box: np.ndarray, tag: str, pad: float = 0.08) -> ZoneConfig:
    x1, y1, x2, y2 = np.clip([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], 0, 1)
    return ZoneConfig.model_validate({"schema": "petperception.zones/1", "zones": [
        {"zone_id": tag, "label": f"owner-marked {tag.replace('_', ' ')} (synthetic test zone)", "semantic_tag": tag,
         "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}]})


def _audio_track(clip: np.ndarray | None, offset_s: float = 2.0) -> np.ndarray:
    track = np.zeros(int(DUR * AUDIO_SR), np.float32)
    if clip is not None:
        a = int(offset_s * AUDIO_SR)
        n = min(len(clip), len(track) - a)
        track[a:a + n] = clip[:n]
    return track


def _write_video(frames: list[np.ndarray], audio: np.ndarray | None, out: Path, work: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0"]
    if audio is not None:
        wav = work / (out.stem + "_audio.wav")
        sf.write(wav, audio, AUDIO_SR, subtype="PCM_16")
        cmd += ["-i", str(wav), "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", "-shortest" if audio is not None else "-an", str(out)]
    cmd = [c for c in cmd if c]
    p = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-500:])


def build(settings: Settings, out_dir: Path) -> list[Scenario]:
    data = settings.paths.resolve("data_root")
    man = settings.paths.resolve("manifests_root")
    img_dir = data / "openimages" / "images"
    esc = data / "esc50" / "audio"
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    def load_img(key: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(img_dir / f"{IMAGES[key]}.jpg")), cv2.COLOR_BGR2RGB)

    def load_clip(key: str) -> np.ndarray:
        x, sr = sf.read(esc / CLIPS[key], dtype="float32")
        if sr != AUDIO_SR:
            raise RuntimeError("unexpected ESC-50 sample rate")
        return x

    n_frames = int(DUR * FPS)
    cat_frame, tcat = _fit(load_img("cat"))
    cat_box = _map_box(_gt_box(man, IMAGES["cat"], "cat"), tcat)
    dog_frame, _ = _fit(load_img("dog"))
    both_frame, _ = _fit(load_img("cat_and_dog"))
    neg_frame, _ = _fit(load_img("no_pet"))
    src = {k: _image_attr(man, v) for k, v in IMAGES.items()}
    asrc = {k: _clip_attr(man, v) for k, v in CLIPS.items()}
    sc: list[Scenario] = []

    def static(frame: np.ndarray) -> list[np.ndarray]:
        return [frame] * n_frames

    # S1: meow while the cat is in an owner-marked food area.
    p = out_dir / "s01_food_zone_meow.mp4"
    _write_video(static(cat_frame), _audio_track(load_clip("meow")), p, work)
    sc.append(Scenario("s01_food_zone_meow", "Static cat photo inside a food_area zone; meow clip at 2-7 s.", p, _zone(cat_box, "food_area"),
                       {"food_experimental_rule_baseline": True, "attribution": "single_visible_candidate"}, [src["cat"], asrc["meow"]]))
    # S2: same scene, silent audio track: context alone must not create a food hypothesis.
    p = out_dir / "s02_food_zone_silent.mp4"
    _write_video(static(cat_frame), _audio_track(None), p, work)
    sc.append(Scenario("s02_food_zone_silent", "Cat in food_area zone with a silent audio track (context only).", p, _zone(cat_box, "food_area"),
                       {"food_experimental": False, "audio_state": "silent"}, [src["cat"]]))
    # S3: two animals visible and a bark: attribution must stay unknown.
    p = out_dir / "s03_two_animals_bark.mp4"
    _write_video(static(both_frame), _audio_track(load_clip("bark")), p, work)
    sc.append(Scenario("s03_two_animals_bark", "Cat and dog visible; dog bark clip at 2-7 s.", p, None,
                       {"min_tracks": 2, "attribution": "unknown", "no_experimental_with_audio": True}, [src["cat_and_dog"], asrc["bark"]]))
    # S4: no pet visible, vacuum cleaner sound.
    p = out_dir / "s04_no_pet_vacuum.mp4"
    _write_video(static(neg_frame), _audio_track(load_clip("vacuum")), p, work)
    sc.append(Scenario("s04_no_pet_vacuum", "Verified no-cat/no-dog scene with a vacuum cleaner clip.", p, None,
                       {"decision_state": "abstention", "reason": "no_pet_detected"}, [src["no_pet"], asrc["vacuum"]]))
    # S5: only a dog is visible but the sound is a meow (off-screen source / media audio).
    p = out_dir / "s05_dog_visible_meow.mp4"
    _write_video(static(dog_frame), _audio_track(load_clip("meow")), p, work)
    sc.append(Scenario("s05_dog_visible_meow", "Dog visible, cat meow audio: species mismatch, attribution unknown.", p, None,
                       {"attribution": "unknown", "no_experimental_with_audio": True}, [src["dog"], asrc["meow"]]))
    # S6: cat resting in rest_area zone while a growl-like sound occurs -> conflicting evidence for 'relaxed'.
    p = out_dir / "s06_rest_zone_growl.mp4"
    _write_video(static(cat_frame), _audio_track(load_clip("cat_growl")), p, work)
    sc.append(Scenario("s06_rest_zone_growl", "Still cat in rest_area zone; growl-like cat clip at 2-7 s.", p, _zone(cat_box, "rest_area"),
                       {"relaxed_experimental": False, "relaxed_conflict": True}, [src["cat"], asrc["cat_growl"]]))
    # S7: audio only (missing video).
    p = out_dir / "s07_audio_only_meow.wav"
    sf.write(p, load_clip("meow"), AUDIO_SR, subtype="PCM_16")
    sc.append(Scenario("s07_audio_only_meow", "Meow clip only (no video stream).", p, None,
                       {"video_state": "absent", "any_experimental": False}, [asrc["meow"]]))
    # S8: video only (no audio stream).
    p = out_dir / "s08_video_only_cat.mp4"
    _write_video(static(cat_frame), None, p, work)
    sc.append(Scenario("s08_video_only_cat", "Cat in food_area zone, no audio stream.", p, _zone(cat_box, "food_area"),
                       {"audio_state": "absent", "food_experimental": False}, [src["cat"]]))
    # S9: synthetic camera pan over the still cat photo: apparent motion is camera motion.
    big = cv2.resize(cat_frame, (int(W * 1.5), int(H * 1.5)), interpolation=cv2.INTER_LINEAR)
    pan = []
    for i in range(n_frames):
        x0 = int((big.shape[1] - W) * i / (n_frames - 1))
        pan.append(np.ascontiguousarray(big[(big.shape[0] - H) // 2:(big.shape[0] - H) // 2 + H, x0:x0 + W]))
    p = out_dir / "s09_camera_pan_cat.mp4"
    _write_video(pan, _audio_track(None), p, work)
    sc.append(Scenario("s09_camera_pan_cat", "Horizontal synthetic camera pan over a still cat (no real animal motion).", p, None,
                       {"camera_motion_flag": True}, [src["cat"]]))
    # S10: cat crop moving across a real no-pet background inside a play_area zone.
    crop = load_img("cat")
    gb = _gt_box(man, IMAGES["cat"], "cat")
    hh, ww = crop.shape[:2]
    crop = crop[int(gb[1] * hh):int(gb[3] * hh), int(gb[0] * ww):int(gb[2] * ww)]
    ch = int(H * 0.55)
    cw = int(crop.shape[1] * ch / crop.shape[0])
    crop = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)
    bg_frame, _ = _fit(load_img("background"))
    moving = []
    for i in range(n_frames):
        f = bg_frame.copy()
        x = int(10 + (W - cw - 20) * (0.5 - 0.5 * np.cos(2 * np.pi * i / n_frames * 1.5)))
        y = int(H - ch - 12 + 8 * np.sin(2 * np.pi * i / n_frames * 3))
        f[y:y + ch, x:x + cw] = crop
        moving.append(f)
    p = out_dir / "s10_moving_cat_play_zone.mp4"
    _write_video(moving, _audio_track(None), p, work)
    play_zone = ZoneConfig.model_validate({"schema": "petperception.zones/1", "zones": [
        {"zone_id": "play", "label": "owner-marked play area (synthetic test zone)", "semantic_tag": "play_area",
         "polygon": [[0.0, 0.3], [1.0, 0.3], [1.0, 1.0], [0.0, 1.0]]}]})
    sc.append(Scenario("s10_moving_cat_play_zone", "Composite: real cat crop moved across a real no-pet background inside a play_area zone.", p,
                       play_zone, {"movement_high": True}, [src["cat"], src["background"]]))
    # S11: corrupt file.
    p = out_dir / "s11_corrupt.mp4"
    p.write_bytes((out_dir / "s01_food_zone_meow.mp4").read_bytes()[:4096])
    sc.append(Scenario("s11_corrupt", "Truncated MP4 (first 4 KiB).", p, None, {"error_code": "corrupt_input"}, []))
    # S12: 8 kHz audio-only meow.
    p = out_dir / "s12_meow_8khz.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(esc / CLIPS["meow"]), "-ar", "8000", str(p)], check=True)
    sc.append(Scenario("s12_meow_8khz", "Meow clip downsampled to 8 kHz (bandwidth-limited input).", p, None,
                       {"quality_flag": "bandwidth_limited_audio"}, [asrc["meow"]]))
    # S13: low-light version of the cat scene.
    rng = np.random.default_rng(0)
    dark = [np.clip(cat_frame.astype(np.float32) * 0.12 + rng.normal(0, 2.0, cat_frame.shape), 0, 255).astype(np.uint8) for _ in range(n_frames)]
    p = out_dir / "s13_low_light_cat.mp4"
    _write_video(dark, _audio_track(load_clip("meow")), p, work)
    sc.append(Scenario("s13_low_light_cat", "Cat scene darkened to 12% brightness with sensor-like noise; meow audio.", p, _zone(cat_box, "food_area"),
                       {"quality_flag": "low_light"}, [src["cat"], asrc["meow"]]))
    # S14: still cat in rest_area with silent audio: relaxed may only come from vision+context, never from silence.
    p = out_dir / "s14_rest_zone_silent.mp4"
    _write_video(static(cat_frame), _audio_track(None), p, work)
    sc.append(Scenario("s14_rest_zone_silent", "Still cat in rest_area zone, silent audio.", p, _zone(cat_box, "rest_area"),
                       {"relaxed_uses_no_audio": True}, [src["cat"]]))
    return sc
