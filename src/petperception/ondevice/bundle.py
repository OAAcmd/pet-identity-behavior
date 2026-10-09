"""On-device model bundle: everything the phone needs to analyse offline.

``export_bundle`` writes a directory that the server publishes unchanged at
``/ondevice/`` (static files) and the PWA downloads once, verifies (SHA-256)
and caches:

* ``manifest.json`` - bundle version (content hash), release profile, every
  file with SHA-256 and size, runtime requirements. Fetched with no-cache.
* ``models/*.tflite`` - unmodified model files, except the EfficientNet-Lite0
  backbone, which is patched to expose four intermediate feature maps as
  extra outputs (weights untouched; parity verified bit-exactly).
* ``engine_assets.json`` - label maps, YAMNet group indices, detector
  anchors and decoding parameters, verifier/posture parameters, thresholds,
  behavior/fusion configuration.

Only assets permitted for the requested release profile are included. The
detector is the raw-anchor MediaPipe EfficientDet build: the TF Hub
post-processed build crashes in LiteRT.js 2.5.3 (results/ondevice).
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .. import __version__
from ..config import REPO_ROOT, Settings
from ..registry import check_model, model_dir, release_decision
from ..utils import sha256_file, utc_now
from ..vision.tflite_meta import associated_files, input_normalization, mediapipe_detector_options
from .tflite_patch import add_outputs

BUNDLE_SCHEMA = "petperception.ondevice_bundle/1"
LITERT_JS_VERSION = "2.5.3"
BACKBONE_OUTPUTS = [118, 140, 158, 159]  # 28x28x40, 14x14x112, 7x7x320, 7x7x1280 (see vision/keypoints.py)
WEB_DETECTORS = {"efficientdet_lite0_mediapipe_v1", "efficientdet_lite2_mediapipe_v1"}


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def _audio_assets(settings: Settings, notes: list[str]) -> dict:
    from ..audio.events import CONTEXT_TAG_MIDS, SUBTYPE_MIDS, build_groups, load_thresholds
    from ..audio.yamnet import YamnetBackend

    mid = settings.audio.yamnet.model_id
    be = YamnetBackend(model_dir(settings, mid), 1)
    groups = build_groups(be)
    th_path = settings.audio.yamnet.thresholds_file
    if th_path and not release_decision(settings.release_profile, "audio_event_thresholds_esc50").allowed:
        notes.append("ESC-50-derived thresholds excluded for this release profile; unfitted defaults apply.")
        th_path = None
    th, src = load_thresholds((REPO_ROOT / th_path) if th_path else None, settings.audio.yamnet.default_thresholds)
    return {
        "model_id": mid, "sample_rate": 16000, "window_samples": 15600, "hop_s": settings.audio.yamnet.hop_s,
        "labels": be.labels,
        "groups": {g: spec.indices for g, spec in groups.items()},
        "subtypes": {k: be.mid_to_index[m] for k, m in SUBTYPE_MIDS.items() if m in be.mid_to_index},
        "context_tags": {k: be.mid_to_index[m] for k, m in CONTEXT_TAG_MIDS.items() if m in be.mid_to_index},
        "thresholds": th, "threshold_source": src, "context_tag_threshold": settings.audio.yamnet.context_tag_threshold,
        "min_event_gap_s": settings.audio.yamnet.min_event_gap_s, "silence_dbfs": settings.audio.silence_dbfs,
        "clipping_fraction": settings.audio.clipping_fraction,
    }


def _detector_assets(path: Path, settings: Settings, mid: str) -> dict:
    data = path.read_bytes()
    dec = mediapipe_detector_options(data)
    norm = input_normalization(data)
    labels = [x.strip() for x in associated_files(data)["labels.txt"].decode().splitlines()]
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_content=data)
    inp = it.get_input_details()[0]
    outs = it.get_output_details()
    return {
        "model_id": mid, "input_hw": [int(inp["shape"][1]), int(inp["shape"][2])],
        "norm_mean": float(norm[0][0]), "norm_std": float(norm[1][0]), "labels": labels,
        "output_order": ["scores" if o["shape"][-1] == dec.num_classes else "boxes" for o in outs],
        "decoding": {"anchors_b64": _b64(dec.anchors), "num_boxes": dec.num_boxes, "num_classes": dec.num_classes,
                     "x_scale": dec.x_scale, "y_scale": dec.y_scale, "w_scale": dec.w_scale, "h_scale": dec.h_scale,
                     "apply_exponential_on_box_size": dec.apply_exponential_on_box_size, "sigmoid_score": dec.sigmoid_score},
        "score_threshold": settings.vision.score_threshold, "nms_iou": settings.vision.nms_iou,
        "max_detections_per_frame": settings.vision.max_detections_per_frame, "target_classes": settings.vision.target_classes,
    }


def export_bundle(settings: Settings, out: Path) -> dict:
    profile = settings.release_profile
    notes: list[str] = []
    if out.exists():
        shutil.rmtree(out)
    (out / "models").mkdir(parents=True)
    files: list[dict] = []

    def add(rel: str, kind: str, resource_id: str | None) -> None:
        p = out / rel
        files.append({"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size, "kind": kind, "resource_id": resource_id})

    def allowed(rid: str) -> bool:
        d = release_decision(profile, rid)
        if not d.allowed:
            notes.append(d.reason or rid)
        return d.allowed

    assets: dict = {"schema": "petperception.ondevice_assets/1", "engine_version": __version__, "release_profile": profile}
    # Audio
    ymid = settings.audio.yamnet.model_id
    if settings.audio.enabled and allowed(ymid) and check_model(settings, ymid).ready:
        shutil.copy(model_dir(settings, ymid) / "yamnet.tflite", out / "models" / "yamnet.tflite")
        add("models/yamnet.tflite", "model", ymid)
        assets["audio"] = {**_audio_assets(settings, notes), "model_file": "models/yamnet.tflite"}
    # Detector (raw-anchor build only on the web runtime)
    det = settings.vision.detector
    if det not in WEB_DETECTORS:
        notes.append(f"{det} is not supported by the web runtime (in-graph post-processing crashes LiteRT.js {LITERT_JS_VERSION}); "
                     "no detector in this bundle")
    elif allowed(det) and check_model(settings, det).ready:
        src = Path(next(iter(check_model(settings, det).files.values())))
        shutil.copy(src, out / "models" / src.name)
        add(f"models/{src.name}", "model", det)
        assets["detector"] = {**_detector_assets(src, settings, det), "model_file": f"models/{src.name}"}
    # Backbone (verifier classifier + posture/keypoint features)
    bb = settings.vision.verifier.classifier
    if "detector" in assets and allowed(bb) and check_model(settings, bb).ready:
        src = Path(next(iter(check_model(settings, bb).files.values())))
        (out / "models" / "efficientnet_lite0_features_v1.tflite").write_bytes(add_outputs(src.read_bytes(), BACKBONE_OUTPUTS))
        add("models/efficientnet_lite0_features_v1.tflite", "model", bb)
        from ..vision.verifier import _ANIMAL_END, _RANGES, _TEDDY, GROUPS

        data = src.read_bytes()
        norm = input_normalization(data)
        assets["backbone"] = {"model_file": "models/efficientnet_lite0_features_v1.tflite", "source_sha256": sha256_file(src),
                              "norm_mean": float(norm[0][0]), "norm_std": float(norm[1][0]), "input_hw": [224, 224],
                              "outputs": ["softmax_1000", "f28", "f14", "f7", "head_7x7x1280"]}
        vp = REPO_ROOT / settings.vision.verifier.params_dir / f"verifier_{det}.json"
        if settings.vision.verifier.enabled and vp.exists() and allowed("detection_verifier_v1"):
            v = json.loads(vp.read_text())
            assets["verifier"] = {"params": v, "groups": GROUPS, "ranges": {g: [a, b] for g, (a, b, _, _) in _RANGES.items()},
                                  "teddy_index": _TEDDY[0], "animal_end": _ANIMAL_END, "margin": 0.15, "eps": 1e-4,
                                  "max_checks_per_frame": settings.vision.verifier.max_checks_per_frame}
        pm = REPO_ROOT / settings.posture.model_file
        if settings.posture.enabled and pm.exists() and allowed("posture_v1"):
            with np.load(pm) as z:
                assets["posture"] = {"mean_b64": _b64(z["mean"]), "scale_b64": _b64(z["scale"]), "coef_b64": _b64(z["coef"]),
                                     "coef_shape": list(z["coef"].shape), "intercept": z["intercept"].astype(float).tolist(),
                                     "gate": float(z["gate"]), "version": str(z["version"]), "labels": ["lying", "sitting", "standing"],
                                     "min_detection_score": settings.posture.min_detection_score, "max_per_frame": settings.posture.max_per_frame}
        if settings.pose.provider == "dog_keypoints_v1" and allowed("dog_keypoints_head_v1"):
            notes.append("dog keypoints are research-only; not bundled for on-device use in this version")
    assets["vision"] = {"sample_fps": settings.vision.sample_fps, "max_side": settings.vision.max_side, "quality": settings.vision.quality.model_dump(),
                        "tracker": settings.vision.tracker.model_dump(), "camera_motion_compensation": settings.vision.camera_motion_compensation}
    assets["behavior"] = settings.behavior.model_dump()
    assets["fusion"] = settings.fusion.model_dump()
    assets["limits"] = {"max_duration_s": settings.limits.max_duration_s, "max_video_frames": settings.limits.max_video_frames,
                        "max_file_bytes": settings.limits.max_file_bytes}
    assets["notes"] = notes
    (out / "engine_assets.json").write_text(json.dumps(assets))
    add("engine_assets.json", "assets", None)
    version = hashlib.sha256("".join(f["sha256"] for f in sorted(files, key=lambda f: f["path"])).encode()).hexdigest()[:16]
    manifest = {"schema": BUNDLE_SCHEMA, "bundle_version": version, "engine_version": __version__, "release_profile": profile,
                "created_at": utc_now().isoformat(timespec="seconds"), "runtime": {"litertjs_core": LITERT_JS_VERSION, "accelerator": "wasm"},
                "files": sorted(files, key=lambda f: f["path"]), "total_bytes": sum(f["bytes"] for f in files), "notes": notes}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
