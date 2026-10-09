"""Engine-level parity: Python reference engine vs the on-device (browser) engine.

``write_reference`` analyses a fixed media set with the Python engine
configured like the on-device bundle (same detector, crop verifier, Kalman
tracker, thresholds; no keypoints) and writes ``cases.json`` plus one result
per case. The browser test (web/tests/runtime/engine.spec.ts) runs the
TypeScript engine on the same files and writes its results next to them.
``compare`` scores agreement. Media decoding differs by design (FFmpeg vs the
browser), so agreement is measured, not assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Settings, load_settings
from ..context.zones import ZoneConfig
from ..utils import utc_now, write_json


def reference_settings(base: Settings) -> Settings:
    """The Python engine configured like the bundle: same model variants, no keypoints."""
    s = load_settings(overrides={"pose": {"provider": "none"}})
    s.paths = base.paths
    s.release_profile = base.release_profile
    s.vision.detector = base.vision.detector
    s.vision.verifier.classifier = base.vision.verifier.classifier
    s.posture.backbone, s.posture.model_file = base.posture.backbone, base.posture.model_file
    return s


def build_cases(settings: Settings) -> list[dict]:
    from ..eval import e8_behavior_sequences as E8
    from ..eval import scenarios

    data = settings.paths.resolve("data_root")
    cases: list[dict] = []
    for sc in scenarios.build(settings, data.parent / "examples"):
        cases.append({"name": sc.scenario_id, "media": str(sc.media), "zones": sc.zones.model_dump(mode="json") if sc.zones else None,
                      "modalities": list(sc.modalities), "source": "scenario (synthetic pairing of real public media)"})
    man = settings.paths.resolve("manifests_root")
    patches, backgrounds, _ = E8.load_assets(data, man)
    test_dir = data.parent / "e8" / "test"
    for script in E8.SCRIPTS:
        for seed in E8.TEST_SEEDS[E8.SCRIPTS.index(script)::len(E8.SCRIPTS)][:2]:
            mp4 = test_dir / f"{script}-{seed}.mp4"
            if not mp4.exists():
                continue
            seq = E8.build_sequence(script, seed, patches, backgrounds)
            cases.append({"name": f"e8-{script}-{seed}", "media": str(mp4), "zones": seq.zones.model_dump(mode="json") if seq.zones else None,
                          "modalities": ["video"], "source": "E8 semi-synthetic sequence"})
    big = data.parent / "e2e" / "big_cat_clip.mp4"
    if big.exists():
        cases.append({"name": "e2e-big-cat-clip", "media": str(big), "zones": None, "modalities": ["audio", "video"], "source": "E2E fixture"})
    return cases


def to_webm(src: Path, dst: Path) -> Path:
    """VP9/Opus copy: open-source Chromium builds (Playwright) lack H.264/AAC decoders."""
    import subprocess

    if src.suffix.lower() in (".wav", ".jpg", ".jpeg", ".png"):
        return src
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-c:v", "libvpx-vp9", "-crf", "24", "-b:v", "0", "-row-mt", "1",
                        "-deadline", "realtime", "-cpu-used", "8", "-c:a", "libopus", "-b:a", "128k", str(dst)], check=True)
    return dst


def write_reference(settings: Settings, out: Path) -> dict:
    from ..pipeline import AnalysisEngine

    s = reference_settings(settings)
    eng = AnalysisEngine(s).load()
    cases = build_cases(s)
    for c in cases:  # both engines analyse the identical (WebM) file
        if not c["name"].startswith("s11"):
            c["media"] = str(to_webm(Path(c["media"]), out / "media" / f"{c['name']}.webm"))
    (out / "reference").mkdir(parents=True, exist_ok=True)
    from ..ingest.media import MediaError

    for c in cases:
        zones = ZoneConfig.model_validate(c["zones"]) if c["zones"] else None
        try:
            r = eng.analyze(c["media"], zones=zones, modalities=tuple(c["modalities"]), analysis_id=c["name"])
            (out / "reference" / f"{c['name']}.json").write_text(r.model_dump_json())
        except MediaError as exc:  # expected for deliberately corrupt inputs
            (out / "reference" / f"{c['name']}.json").write_text(json.dumps({"error": exc.code}))
    write_json(out / "cases.json", {"cases": cases, "created_at": utc_now().isoformat(timespec="seconds"),
                                    "components": {k: v.model_dump() for k, v in eng.status.items()}})
    return {"cases": len(cases)}


def _tracks(r: dict) -> list[dict]:
    return r["observations"]["animal_tracks"]


def _match_intervals(a: list[tuple[str, float, float]], b: list[tuple[str, float, float]], tol: float) -> tuple[int, int, int, list[float]]:
    """Greedy match by label with start/end within tol seconds. Returns (matched, only_a, only_b, abs errors)."""
    used = set()
    hits, errs = 0, []
    for la, s0, e0 in a:
        best, bj = None, None
        for j, (lb, s1, e1) in enumerate(b):
            if j in used or la != lb:
                continue
            d = max(abs(s0 - s1), abs(e0 - e1))
            if d <= tol and (best is None or d < best):
                best, bj = d, j
        if bj is not None:
            used.add(bj)
            hits += 1
            errs.append(best)
    return hits, len(a) - hits, len(b) - hits, errs


def compare(out: Path) -> dict:
    cases = json.loads((out / "cases.json").read_text())["cases"]
    rows = []
    agg = {"tracks": [0, 0, 0], "sounds": [0, 0, 0], "segments": [0, 0, 0], "hypotheses": [0, 0, 0]}
    errs: dict[str, list[float]] = {k: [] for k in agg}
    decisions = 0
    compared = 0
    posture = {"ref_frames": 0, "web_frames": 0, "matched": 0, "same_label": 0, "both_reported": 0, "both_reported_same": 0, "gate_flips": 0, "abs_score_diff": []}
    for c in cases:
        ref_p, web_p = out / "reference" / f"{c['name']}.json", out / "web" / f"{c['name']}.json"
        if not (ref_p.exists() and web_p.exists()):
            continue
        ref, web = json.loads(ref_p.read_text()), json.loads(web_p.read_text())
        if "error" in ref:
            rows.append({"name": c["name"], "reference_error": ref["error"], "web_rejected_too": "error" in web})
            continue
        if "error" in web:
            rows.append({"name": c["name"], "web_error": web["error"]})
            continue
        compared += 1
        tol_t = 1.0 / 2.0 + 0.05  # one sampling interval at 2 fps
        tr = _match_intervals([(t["species"], t["first_seen_s"], t["last_seen_s"]) for t in _tracks(ref)],
                              [(t["species"], t["first_seen_s"], t["last_seen_s"]) for t in _tracks(web)], 2 * tol_t)
        snd = _match_intervals([(e["label"], e["t_start_s"], e["t_end_s"]) for e in ref["observations"]["sound_events"] + ref["observations"]["context_sounds"]],
                               [(e["label"], e["t_start_s"], e["t_end_s"]) for e in web["observations"]["sound_events"] + web["observations"]["context_sounds"]], 0.5)
        seg = _match_intervals([(b["kind"], b["t_start_s"], b["t_end_s"]) for t in _tracks(ref) for b in t["behavior_segments"]],
                               [(b["kind"], b["t_start_s"], b["t_end_s"]) for t in _tracks(web) for b in t["behavior_segments"]], 2 * tol_t)
        hyp = _match_intervals([(f"{h['name']}:{h['status']}", 0.0, 0.0) for h in ref["hypotheses"]],
                               [(f"{h['name']}:{h['status']}", 0.0, 0.0) for h in web["hypotheses"]], 0.1)
        for k, v in (("tracks", tr), ("sounds", snd), ("segments", seg), ("hypotheses", hyp)):
            for i in range(3):
                agg[k][i] += v[i]
            errs[k] += v[3]
        pr = {(t["species"], round(p["t_s"], 2)): p for t in _tracks(ref) for p in t.get("posture", [])}
        pw = {(t["species"], round(p["t_s"], 2)): p for t in _tracks(web) for p in t.get("posture", [])}
        posture["ref_frames"] += len(pr)
        posture["web_frames"] += len(pw)
        for k in pr.keys() & pw.keys():
            posture["matched"] += 1
            posture["same_label"] += int(pr[k]["label"] == pw[k]["label"])
            rep = (pr[k]["label"] != "uncertain", pw[k]["label"] != "uncertain")
            if all(rep):
                posture["both_reported"] += 1
                posture["both_reported_same"] += int(pr[k]["label"] == pw[k]["label"])
            posture["gate_flips"] += int(rep[0] != rep[1])
            posture["abs_score_diff"].append(abs(pr[k]["score"] - pw[k]["score"]))
        same_decision = ref["decision"]["state"] == web["decision"]["state"]
        decisions += int(same_decision)
        rows.append({"name": c["name"], "decision_ref": ref["decision"]["state"], "decision_web": web["decision"]["state"],
                     "tracks": tr[:3], "sounds": snd[:3], "segments": seg[:3], "hypotheses": hyp[:3],
                     "web_timings_ms": web.get("timings_ms"), "ref_timings_ms": {k: v for k, v in ref["timings_ms"].items() if not k.startswith("model_load")}})
    summary = {
        "cases_compared": compared, "decision_agreement": decisions / max(compared, 1),
        "agreement": {k: {"matched": v[0], "only_reference": v[1], "only_web": v[2],
                          "f1": 2 * v[0] / max(2 * v[0] + v[1] + v[2], 1),
                          "median_boundary_error_s": float(np.median(errs[k])) if errs[k] else None} for k, v in agg.items()},
        "posture": {"reference_frames": posture["ref_frames"], "web_frames": posture["web_frames"], "frames_matched_by_species_and_time": posture["matched"],
                    "label_agreement": posture["same_label"] / posture["matched"] if posture["matched"] else None,
                    "both_reported": posture["both_reported"],
                    "label_agreement_when_both_reported": posture["both_reported_same"] / posture["both_reported"] if posture["both_reported"] else None,
                    "gate_flips_one_side_uncertain": posture["gate_flips"],
                    "median_abs_score_diff": float(np.median(posture["abs_score_diff"])) if posture["abs_score_diff"] else None},
        "rows": rows,
        "tolerances": {"tracks_s": 2 * (1.0 / 2.0 + 0.05), "sounds_s": 0.5, "segments_s": 2 * (1.0 / 2.0 + 0.05)},
        "note": "Desktop headless Chromium (LiteRT.js WASM) vs Python LiteRT; media decoded by the browser vs FFmpeg.",
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    return summary
