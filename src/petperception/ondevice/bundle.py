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

from ..config import REPO_ROOT, Settings
from ..registry import check_model, model_dir, release_decision, resource
from ..utils import sha256_file, utc_now
from ..vision.detector import label_list
from ..vision.tflite_io import dequantize
from ..vision.tflite_meta import associated_files, input_normalization, mediapipe_detector_options
from .tflite_patch import expose_outputs

BUNDLE_SCHEMA = "petperception.ondevice_bundle/1"
# Format of engine_assets.json as consumed by the TypeScript engine. Change the
# major.minor only together with SUPPORTED_BUNDLE_ENGINE in web/src/engine/modelstore.ts:
# the app refuses bundles whose major.minor it does not support.
ENGINE_ASSET_FORMAT = "0.2.3"  # 0.2.3: optional intent heads (intent.enabled); 0.2: quantized model inputs/outputs (input_quant, output_quant); 0.2.1: optional object_relation;
# 0.2.2: audio.subtype_validation (absent in older bundles: no subtype counts as validated) and optional
# audio.context_thresholds (per tag; absent: context_tag_threshold for every tag)
LITERT_JS_VERSION = "2.5.3"
BACKBONE_OUTPUTS = [118, 140, 158, 159]  # 28x28x40, 14x14x112, 7x7x320, 7x7x1280 (see vision/keypoints.py)
WEB_DETECTORS = ("efficientdet_lite0_mediapipe_v1", "efficientdet_lite0_mediapipe_int8_v1", "efficientdet_lite0_mediapipe_f16_v1",
                 "efficientdet_lite0_tfhub_rawdecode_v1")  # raw-anchor builds; browser parity measured


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def _audio_assets(settings: Settings, notes: list[str]) -> dict:
    from ..audio.events import CONTEXT_TAG_MIDS, SUBTYPE_MIDS, build_groups, load_context_thresholds, load_subtype_validation, load_thresholds
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
        "subtype_validation": load_subtype_validation((REPO_ROOT / sv) if (sv := settings.audio.yamnet.subtype_validation_file) else None),
        "context_thresholds": load_context_thresholds((REPO_ROOT / ct) if (ct := settings.audio.yamnet.context_thresholds_file) else None),
        "clipping_fraction": settings.audio.clipping_fraction,
    }


def _quant(d: dict) -> list | None:
    scale, zero = d["quantization"]
    return [float(scale), int(zero)] if np.dtype(d["dtype"]).kind in "iu" and scale else None


def _quant_io(inp: dict, outs: list[dict]) -> dict:
    """Quantization of the input and of each output, in the model's output order (null = float)."""
    return {"input_quant": _quant(inp), "output_quant": [_quant(o) for o in outs]}


def _detector_assets(path: Path, settings: Settings, mid: str) -> dict:
    data = path.read_bytes()
    dec = mediapipe_detector_options(data)
    norm = input_normalization(data)
    labels = label_list(associated_files(data)["labels.txt"])
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_content=data)
    inp = it.get_input_details()[0]
    outs = it.get_output_details()
    return {
        "model_id": mid, "input_hw": [int(inp["shape"][1]), int(inp["shape"][2])],
        "norm_mean": float(norm[0][0]), "norm_std": float(norm[1][0]), **_quant_io(inp, outs), "labels": labels,
        "output_order": ["scores" if o["shape"][-1] == dec.num_classes else "boxes" for o in outs],
        "decoding": {"anchors_b64": _b64(dec.anchors), "num_boxes": dec.num_boxes, "num_classes": dec.num_classes,
                     "x_scale": dec.x_scale, "y_scale": dec.y_scale, "w_scale": dec.w_scale, "h_scale": dec.h_scale,
                     "apply_exponential_on_box_size": dec.apply_exponential_on_box_size, "sigmoid_score": dec.sigmoid_score},
        "score_threshold": settings.vision.score_threshold, "nms_iou": settings.vision.nms_iou,
        "max_detections_per_frame": settings.vision.max_detections_per_frame, "target_classes": settings.vision.target_classes,
    }


def _detector_assets_from(d, settings: Settings, mid: str) -> dict:
    """Assets for a detector object built in Python (e.g. the raw-decode TF Hub build)."""
    dec = d.decoding
    outs = d.interp.get_output_details()
    return {
        "model_id": mid, "input_hw": [d.in_h, d.in_w], "norm_mean": d.norm_mean, "norm_std": d.norm_std, **_quant_io(d.in_detail, outs),
        "labels": d.labels, "output_order": ["scores" if o["shape"][-1] == dec.num_classes else "boxes" for o in outs],
        "decoding": {"anchors_b64": _b64(dec.anchors), "num_boxes": dec.num_boxes, "num_classes": dec.num_classes,
                     "x_scale": dec.x_scale, "y_scale": dec.y_scale, "w_scale": dec.w_scale, "h_scale": dec.h_scale,
                     "apply_exponential_on_box_size": dec.apply_exponential_on_box_size, "sigmoid_score": dec.sigmoid_score},
        "score_threshold": settings.vision.score_threshold, "nms_iou": settings.vision.nms_iou,
        "max_detections_per_frame": settings.vision.max_detections_per_frame, "target_classes": settings.vision.target_classes,
    }


def _head_assets(path: Path) -> dict:
    d = json.loads(path.read_text())
    coef = np.asarray(d["coef"], np.float32)
    return {"classes": d["classes"], "mean_b64": _b64(np.asarray(d["standardize_mean"], np.float32)),
            "scale_b64": _b64(np.asarray(d["standardize_scale"], np.float32)), "coef_b64": _b64(coef), "coef_shape": list(coef.shape),
            "intercept": [float(v) for v in d["intercept"]], "temperature": float(d["temperature"]), "abstain_below": float(d["abstain_below"]),
            **({"branches": d["branches"]} if "branches" in d else {})}


def _intent_assets(allowed) -> dict | None:
    from ..intent import inference as I
    from ..intent import taxonomy as T
    from ..intent.av_features import load_components

    art = REPO_ROOT / "artifacts" / "intent"
    sd = art / "kinetics_situation_v1" / "phone_now"
    if not (sd / "late_fusion.json").exists() or not allowed("intent_situation_kinetics_v1"):
        return None
    pol = json.loads((sd / "policy.json").read_text())
    out: dict = {"version": I.INFERENCE_VERSION, "taxonomy_version": T.TAXONOMY_VERSION, "variant": "phone_now", "vocal_heard": I.VOCAL_HEARD,
                 "dog_sounds": list(I.DOG_SOUND), "cat_sounds": list(I.CAT_SOUND),
                 "object_classes": load_components(1, with_audio=False).object_classes,
                 "situation": {"heads": {b: _head_assets(sd / f"head_{b}.json") for b in pol["branches"]},
                               "fusion": _head_assets(sd / "late_fusion.json"), "visual_only_abstain_below": pol["visual_only_abstain_below"],
                               "conflict_js_threshold": pol.get("conflict_js_threshold"),
                               "degraded_video_rule": pol.get("degraded_video_rule"),
                               "species": I.SITUATION_SPECIES,
                               "qualifier": {lab.raw: lab.qualifier for lab in T.DATASETS["kinetics_pets"].labels},
                               "context_for": {lab.raw: [h for h, r in lab.hypotheses.items() if r == "context_for"] for lab in T.DATASETS["kinetics_pets"].labels}},
                 "dog_activity_hypothesis": I.DOG_ACTIVITY_HYP}
    da = art / "dog_audio_activity_v1" / "head_audio_scores.json"
    if da.exists() and allowed("intent_dog_audio_activity_v1"):
        out["dog_audio"] = _head_assets(da)
    ca = art / "cat_audio_context_v1" / "head_audio_scores.json"
    if ca.exists() and allowed("intent_cat_audio_context_v1"):
        out["cat_context"] = {**_head_assets(ca), "evidence_gate_passed": False,
                              "context_for": {lab.raw: [h for h, r in lab.hypotheses.items() if r == "context_for"] for lab in T.DATASETS["catmeows"].labels}}
    return out


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

    assets: dict = {"schema": "petperception.ondevice_assets/1", "engine_version": ENGINE_ASSET_FORMAT, "release_profile": profile}
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
        if resource(det).get("decode") == "strip_postprocess":
            # In-graph post-processing removed; raw outputs decoded by the TypeScript detector.
            from ..vision.detector import EfficientDetDetector

            d = EfficientDetDetector.from_postprocessed(src, 1)
            name = "efficientdet_lite0_rawdecode.tflite"
            (out / "models" / name).write_bytes(d.model_bytes)
            assets["detector"] = {**_detector_assets_from(d, settings, det), "model_file": f"models/{name}"}
        else:
            name = src.name
            shutil.copy(src, out / "models" / name)
            assets["detector"] = {**_detector_assets(src, settings, det), "model_file": f"models/{name}"}
        add(f"models/{name}", "model", det)
    # Backbone (verifier classifier + posture/keypoint features)
    bb = settings.vision.verifier.classifier
    if "detector" in assets and allowed(bb) and check_model(settings, bb).ready:
        src = Path(next(iter(check_model(settings, bb).files.values())))
        patched = expose_outputs(src.read_bytes(), BACKBONE_OUTPUTS)
        (out / "models" / "efficientnet_lite0_features_v1.tflite").write_bytes(patched)
        from ai_edge_litert.interpreter import Interpreter

        pit = Interpreter(model_content=patched)
        add("models/efficientnet_lite0_features_v1.tflite", "model", bb)
        from ..vision.verifier import _ANIMAL_END, _RANGES, _TEDDY, GROUPS

        data = src.read_bytes()
        norm = input_normalization(data)
        assets["backbone"] = {"model_file": "models/efficientnet_lite0_features_v1.tflite", "source_sha256": sha256_file(src),
                              "norm_mean": float(norm[0][0]), "norm_std": float(norm[1][0]), "input_hw": [224, 224],
                              **_quant_io(pit.get_input_details()[0], pit.get_output_details()),
                              "outputs": ["softmax_1000", "f28", "f14", "f7", "head_7x7x1280"]}
        vp = REPO_ROOT / settings.vision.verifier.params_file(det)
        if settings.vision.verifier.enabled and vp.exists() and allowed("detection_verifier_v1"):
            v = json.loads(vp.read_text())
            assets["verifier"] = {"params": v, "groups": GROUPS, "ranges": {g: [a, b] for g, (a, b, _, _) in _RANGES.items()},
                                  "teddy_index": _TEDDY[0], "animal_end": _ANIMAL_END, "margin": 0.15, "eps": 1e-4,
                                  "max_checks_per_frame": settings.vision.verifier.max_checks_per_frame}
        pm = REPO_ROOT / settings.posture.model_file
        if settings.posture.backbone != bb:
            notes.append(f"posture model fitted on {settings.posture.backbone} features, bundle backbone is {bb}: posture not bundled")
        elif settings.posture.enabled and pm.exists() and allowed("posture_v1"):
            with np.load(pm) as z:
                assets["posture"] = {"mean_b64": _b64(z["mean"]), "scale_b64": _b64(z["scale"]), "coef_b64": _b64(z["coef"]),
                                     "coef_shape": list(z["coef"].shape), "intercept": z["intercept"].astype(float).tolist(),
                                     "gate": float(z["gate"]), "version": str(z["version"]), "labels": ["lying", "sitting", "standing"],
                                     "min_detection_score": settings.posture.min_detection_score, "max_per_frame": settings.posture.max_per_frame}
        if settings.pose.provider == "dog_keypoints_v1" and allowed("dog_keypoints_head_v1"):
            notes.append("dog keypoints are research-only; not bundled for on-device use in this version")
    # Person-pet interaction (E13): box geometry only, needs the detector but not the backbone.
    im = REPO_ROOT / settings.interaction.model_file
    if "detector" in assets and settings.interaction.enabled and im.exists() and allowed("person_pet_interaction_v1"):
        with np.load(im) as z:
            assets["interaction"] = {"mean_b64": _b64(z["mean"]), "scale_b64": _b64(z["scale"]), "coef_b64": _b64(z["coef"]),
                                     "intercept": float(np.ravel(z["intercept"])[0]), "threshold": float(z["threshold"]),
                                     "version": str(z["version"]), "min_person_score": settings.interaction.min_person_score,
                                     "min_frames": settings.interaction.min_frames}
    # Pet on furniture (E15): box geometry only; only the relations that passed the E15 gate are in the file.
    om = REPO_ROOT / settings.object_relation.model_file
    if "detector" in assets and settings.object_relation.enabled and om.exists() and allowed(settings.object_relation.resource_id):
        from ..vision.object_relation import DETECTOR_CLASS, GROUPS

        with np.load(om) as z:
            heads = {r: {"groups": [str(g) for g in z[f"{r}_groups"]], "mean_b64": _b64(z[f"{r}_mean"]), "scale_b64": _b64(z[f"{r}_scale"]),
                         "coef_b64": _b64(z[f"{r}_coef"]), "intercept": float(np.ravel(z[f"{r}_intercept"])[0]), "threshold": float(z[f"{r}_threshold"])}
                     for r in [str(v) for v in z["relations"]]}
            assets["object_relation"] = {"version": str(z["version"]), "groups": list(GROUPS), "detector_class": DETECTOR_CLASS, "heads": heads,
                                         "min_object_score": settings.object_relation.min_object_score,
                                         "min_frames": settings.object_relation.min_frames}
    # Behaviour / intent heads, "phone_now" variant (YAMNet class scores, no keypoints): situation fusion (E28),
    # dog vocal activity (E27), cat meow context (E2 data). Each part follows its own release decision.
    if settings.intent.enabled and all(k in assets for k in ("audio", "detector", "backbone")):
        ia = _intent_assets(allowed)
        if ia:
            assets["intent"] = ia
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
    manifest = {"schema": BUNDLE_SCHEMA, "bundle_version": version, "engine_version": ENGINE_ASSET_FORMAT, "release_profile": profile,
                "created_at": utc_now().isoformat(timespec="seconds"), "runtime": {"litertjs_core": LITERT_JS_VERSION, "accelerator": "wasm"},
                "files": sorted(files, key=lambda f: f["path"]), "total_bytes": sum(f["bytes"] for f in files), "notes": notes}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def add_device_check(settings: Settings, out: Path) -> dict:
    """Reference outputs for the on-device numerical self-check and a reference clip
    (MP4/H.264 and WebM/VP9) with the Python engine's expected summary."""
    import csv
    import subprocess

    from ..eval import e8_behavior_sequences as E8
    from ..pipeline import AnalysisEngine
    from .engine_parity import reference_settings
    from .parity import CASES, _run

    dc = out / "device_check"
    dc.mkdir(parents=True, exist_ok=True)
    m = json.loads((out / "manifest.json").read_text())
    by_res = {f["resource_id"]: f["path"] for f in m["files"] if f["kind"] == "model"}
    refs = {}
    from ai_edge_litert.interpreter import Interpreter

    def quantized(path: Path) -> bool:
        return np.dtype(Interpreter(model_path=str(path)).get_input_details()[0]["dtype"]).kind in "iu"

    # Keys are model roles; "input" names the deterministic input generator (parity.CASES,
    # web/src/engine/parity.ts), which depends on the build (float or quantized input).
    pick = {"yamnet": (settings.audio.yamnet.model_id, 521, ("yamnet", "yamnet")), "detector_float": (settings.vision.detector, None, ("detector_float", "detector_int8")),
            "backbone": (settings.vision.verifier.classifier, 1000, ("backbone", "backbone_int8"))}
    for role, (rid, size, (case_f, case_q)) in pick.items():
        if rid not in by_res:
            continue
        case = case_q if quantized(out / by_res[rid]) else case_f
        model = (out / by_res[rid]).read_bytes()
        outs = _run(model, CASES[case]())
        # Quantized outputs are compared dequantized, as the app's runModel returns them.
        qs = [d["quantization"] for d in Interpreter(model_content=model).get_output_details()]
        if size is None:  # detector: the box tensor
            size = next(o.size for o in outs if o.shape[-1] == 4)
        k = next(i for i, x in enumerate(outs) if x.size == size)
        o = dequantize(outs[k], qs[k]).astype(np.float32).ravel()
        refs[role] = {"input": case, "elements": int(o.size), "first": [float(v) for v in o[:64]], "mean_abs": float(np.abs(o).mean()), "argmax": int(o.argmax())}
    (dc / "reference_outputs.json").write_text(json.dumps(refs))
    # Reference clip: a real Open Images cat crop walking across a real indoor background (CC BY 2.0 sources).
    data, man = settings.paths.resolve("data_root"), settings.paths.resolve("manifests_root")
    from ..data import openimages as OI

    # Backgrounds: E8's pinned indoor images (fetched here if absent; checksums in e8_backgrounds_sha256.csv).
    OI.fetch_images(data, man, image_dir=data / "openimages_test" / "images", images_csv="e8_backgrounds.csv", sums_csv="e8_backgrounds_sha256.csv")
    imgs = {r["image_id"]: r for r in csv.DictReader(open(man / "oi_test_tuning_images.csv", newline="", encoding="utf-8"))}
    bgs = list(csv.DictReader(open(man / "e8_backgrounds.csv", newline="", encoding="utf-8")))
    boxes = [r for r in csv.DictReader(open(man / "oi_test_tuning_boxes.csv", newline="", encoding="utf-8")) if r["label"] == "cat"
             and imgs[r["image_id"]]["role"] == "cat_positive" and not any(int(r[k]) for k in ("is_occluded", "is_group_of", "is_depiction"))
             and 0.15 <= (float(r["xmax"]) - float(r["xmin"])) * (float(r["ymax"]) - float(r["ymin"])) <= 0.5]
    import cv2

    eng = AnalysisEngine(reference_settings(settings)).load()
    chosen = None
    for b in sorted(boxes, key=lambda r: r["image_id"])[:40]:
        im = cv2.imread(str(data / "openimages_test" / "images" / f"{b['image_id']}.jpg"))
        if im is None:
            continue
        h, w = im.shape[:2]
        crop = cv2.cvtColor(im[int(float(b["ymin"]) * h): int(float(b["ymax"]) * h), int(float(b["xmin"]) * w): int(float(b["xmax"]) * w)], cv2.COLOR_BGR2RGB)
        bg_row = bgs[0]
        bg = cv2.cvtColor(cv2.imread(str(data / "openimages_test" / "images" / f"{bg_row['image_id']}.jpg")), cv2.COLOR_BGR2RGB)
        path = [(0.3, 0.55)] * 40 + E8._move((0.3, 0.55), (0.7, 0.55), 0.1) + [(0.7, 0.55)] * 40
        seq = E8.Sequence("device-check", "still_walk_still", 0, bg, [E8.Actor(crop, "cat", path, [True] * len(path), 0.45)], len(path))
        mp4 = dc / "clip.mp4"
        E8.render(seq, mp4)
        r = eng.analyze(mp4, modalities=("audio", "video"))
        if len(r.observations.animal_tracks) == 1 and r.observations.animal_tracks[0].species == "cat":
            chosen = (b["image_id"], bg_row["image_id"])
            break
    if chosen is None:
        return {"status": "no suitable clip"}
    webm = dc / "clip.webm"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(dc / "clip.mp4"), "-c:v", "libvpx-vp9", "-crf", "24", "-b:v", "0",
                    "-deadline", "realtime", "-cpu-used", "8", str(webm)], check=True)
    expected = []
    for f, mime in (("device_check/clip.webm", 'video/webm; codecs="vp9"'), ("device_check/clip.mp4", 'video/mp4; codecs="avc1.64001f"')):
        r = eng.analyze(out / f, modalities=("audio", "video"))
        tr = r.observations.animal_tracks
        expected.append({"file": f, "mime": mime, "tracks": len(tr), "species": [t.species for t in tr],
                         "segment_kinds": sorted({b.kind for t in tr for b in t.behavior_segments}), "decision": r.decision.state,
                         "sound_labels": [s.label for s in r.observations.sound_events]})
    (dc / "expected.json").write_text(json.dumps(expected, indent=1))

    def attribution(image_id: str) -> dict:
        row = imgs.get(image_id) or {}
        return {"image_id": image_id, "author": row.get("author"), "license": row.get("license"), "source": row.get("original_landing_url")}

    bg_meta = next((r for r in csv.DictReader(open(man / "e8_backgrounds.csv", newline="", encoding="utf-8")) if r["image_id"] == chosen[1]), {})
    (dc / "ATTRIBUTION.json").write_text(json.dumps({"clip": "semi-synthetic: real Open Images cat crop moved along a scripted path on a real Open Images background",
                                                     "animal": attribution(chosen[0]),
                                                     "background": {"image_id": chosen[1], "author": bg_meta.get("author"),
                                                                    "license": bg_meta.get("license") or "CC BY 2.0 (Open Images)",
                                                                    "source": bg_meta.get("original_landing_url") or bg_meta.get("download_url")}}, indent=1))
    # Re-register files in the manifest (content hash changes the bundle version).
    for rel, kind in (("device_check/reference_outputs.json", "device_check"), ("device_check/expected.json", "device_check"),
                      ("device_check/ATTRIBUTION.json", "device_check"), ("device_check/clip.webm", "device_check"), ("device_check/clip.mp4", "device_check")):
        p = out / rel
        m["files"] = [f for f in m["files"] if f["path"] != rel] + [{"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size, "kind": kind, "resource_id": None}]
    m["files"].sort(key=lambda f: f["path"])
    m["total_bytes"] = sum(f["bytes"] for f in m["files"])
    m["bundle_version"] = hashlib.sha256("".join(f["sha256"] for f in m["files"]).encode()).hexdigest()[:16]
    (out / "manifest.json").write_text(json.dumps(m, indent=1))
    return {"status": "ok", "animal_image": chosen[0], "background_image": chosen[1], "expected": expected}
