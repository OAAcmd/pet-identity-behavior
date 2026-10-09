"""E5: fusion behavior on synthetic-pairing scenarios (missing modality,
contradictory evidence, background/context conflict, multiple pets, abstention).

These are integration tests of *behavior*, run with real models on real public
media that were paired synthetically. Multimodal accuracy remains
``not_evaluated``: no genuine paired household audio/video/context/outcome test
set exists.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from ..config import Settings
from ..ingest.media import MediaError
from ..pipeline import AnalysisEngine
from ..schemas.analysis import AnalysisResult
from ..utils import utc_now, write_json
from . import scenarios as SC

METHODS = ["rule_baseline", "weighted_average", "product_of_experts"]
FORBIDDEN = [r"\bI want\b", r"\bI am\b", r"\bdiagnos", r"\bdisease\b", r"\bpercent\b", r"%", r"\bprobability of\b", r"\btranslat", r"\bfeels?\b"]


def invariants(r: AnalysisResult) -> dict[str, bool]:
    ev_ids = {e.evidence_id for e in r.observations.evidence}
    hyps = r.hypotheses
    exp = [h for h in hyps if h.status == "experimental_hypothesis"]
    text = " ".join([h.explanation for h in hyps] + [s.text for s in r.suggestions])
    cited = set(re.findall(r"\[(ev-[^\]]+)\]", " ".join(h.explanation for h in hyps)))
    snd = {e.evidence_id: e for e in r.observations.sound_events}
    ok_audio_attr = True
    for h in exp:
        for i in h.supporting_evidence_ids:
            if i in snd and not (h.track_id and snd[i].attribution.status == "single_visible_candidate"
                                 and h.track_id in snd[i].attribution.candidate_track_ids):
                ok_audio_attr = False
    return {
        "no_calibrated_confidence": all(h.calibrated_confidence is None and h.calibration_status == "uncalibrated" for h in hyps),
        "experimental_needs_two_roots": all(len(set(h.independent_provenance_roots)) >= 2 for h in exp),
        "no_context_only_experimental": all(set(h.independent_provenance_roots) != {"context:zones"} for h in exp),
        "evidence_ids_resolve": all(i in ev_ids for h in hyps for i in h.supporting_evidence_ids + h.conflicting_evidence_ids),
        "explanations_cite_existing_evidence": cited <= ev_ids,
        "audio_used_only_with_single_candidate": ok_audio_attr,
        "fusion_marked_unvalidated": r.fusion.multimodal_accuracy == "not_evaluated" and r.fusion.validated is False,
        "no_forbidden_phrasing": not any(re.search(p, text, re.IGNORECASE) for p in FORBIDDEN),
        "relaxed_never_from_audio_absence": all(not any(e.kind == "audio_absence" for e in r.observations.evidence if e.evidence_id in h.supporting_evidence_ids)
                                                for h in hyps if h.name == "relaxed"),
    }


def expectations(sc: SC.Scenario, r: AnalysisResult | None, err: str | None, method: str) -> dict[str, bool]:
    e = sc.expect
    out: dict[str, bool] = {}
    if "error_code" in e:
        out["error_code"] = err == e["error_code"]
        return out
    assert r is not None
    exp_names = {h.name for h in r.hypotheses if h.status == "experimental_hypothesis"}
    snd = r.observations.sound_events
    if "food_experimental_rule_baseline" in e and method == "rule_baseline":
        out["food_experimental_rule_baseline"] = ("food" in exp_names) == e["food_experimental_rule_baseline"]
    if "food_experimental" in e:
        out["food_not_experimental"] = "food" not in exp_names
    if "attribution" in e:
        pet_snd = [s for s in snd if s.label in ("cat_vocalization", "dog_vocalization")]
        out["attribution"] = bool(pet_snd) and all(s.attribution.status == e["attribution"] for s in pet_snd)
    if e.get("no_experimental_with_audio"):
        audio_ids = {s.evidence_id for s in snd}
        out["no_experimental_with_audio"] = not any(h.status == "experimental_hypothesis" and set(h.supporting_evidence_ids) & audio_ids for h in r.hypotheses)
    if "min_tracks" in e:
        out["min_tracks"] = len(r.observations.animal_tracks) >= int(e["min_tracks"])
    if "decision_state" in e:
        out["decision_state"] = r.decision.state == e["decision_state"]
    if "reason" in e:
        out["reason_code"] = e["reason"] in r.decision.reason_codes
    if "relaxed_experimental" in e:
        out["relaxed_not_experimental"] = "relaxed" not in exp_names
    if e.get("relaxed_conflict"):
        out["relaxed_conflict_recorded"] = any(h.name == "relaxed" and "conflicting_evidence" in h.reason_codes for h in r.hypotheses)
    if "video_state" in e:
        out["video_state"] = r.modalities.video_state == e["video_state"]
    if "audio_state" in e:
        out["audio_state"] = r.modalities.audio_state == e["audio_state"]
    if "any_experimental" in e:
        out["no_experimental"] = not exp_names
    if e.get("camera_motion_flag"):
        out["camera_motion_flag"] = any(q.code == "camera_motion" for q in r.quality_flags)
        out["compensated_movement_low"] = all((t.movement.moving_fraction or 0.0) <= 0.2 for t in r.observations.animal_tracks)
    if e.get("movement_high"):
        out["movement_high_observed"] = any(ev.kind == "movement" and ev.summary.startswith("high") for ev in r.observations.evidence)
    if "quality_flag" in e:
        out["quality_flag"] = any(q.code == e["quality_flag"] for q in r.quality_flags)
    if e.get("relaxed_uses_no_audio"):
        audio_ids = {s.evidence_id for s in snd}
        out["relaxed_uses_no_audio"] = all(not (set(h.supporting_evidence_ids) & audio_ids) for h in r.hypotheses if h.name == "relaxed")
    return out


def run(settings: Settings, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    media_dir = settings.paths.resolve("data_root").parent / "examples"
    scs = SC.build(settings, media_dir)
    rows = []
    per_method_pass: dict[str, list[bool]] = {m: [] for m in METHODS}
    examples_dir = out_dir / "outputs"
    for m in METHODS:
        s2 = settings.model_copy(deep=True)
        s2.fusion.method = m
        eng = AnalysisEngine(s2).load()
        for sc in scs:
            err, r = None, None
            try:
                r = eng.analyze(sc.media, zones=sc.zones, modalities=sc.modalities, analysis_id=f"e5-{sc.scenario_id}-{m}")
            except MediaError as exc:
                err = exc.code
            inv = invariants(r) if r else {}
            exp = expectations(sc, r, err, m)
            passed = all(inv.values()) and all(exp.values())
            per_method_pass[m].append(passed)
            rows.append({
                "scenario_id": sc.scenario_id, "method": m, "synthetic_pairing": sc.synthetic, "description": sc.description,
                "error_code": err, "decision_state": r.decision.state if r else None,
                "decision_reasons": r.decision.reason_codes if r else None,
                "hypotheses": [{"name": h.name, "status": h.status, "track_id": h.track_id, "score": h.score,
                                "roots": h.independent_provenance_roots, "reasons": h.reason_codes} for h in r.hypotheses] if r else [],
                "tracks": len(r.observations.animal_tracks) if r else 0,
                "sound_events": [(s.label, round(s.t_start_s, 2), round(s.t_end_s, 2), s.attribution.status) for s in r.observations.sound_events] if r else [],
                "quality_flags": sorted({q.code for q in r.quality_flags}) if r else [],
                "invariants": inv, "expectations": exp, "passed": passed, "sources": sc.sources,
            })
            if r is not None and m == "rule_baseline":
                write_json(examples_dir / f"{sc.scenario_id}.json", r.model_dump(mode="json"))
    # Diagnostic: the camera-pan scenario with compensation disabled shows what it removes.
    s3 = settings.model_copy(deep=True)
    s3.vision.camera_motion_compensation = False
    pan = next(x for x in scs if x.scenario_id == "s09_camera_pan_cat")
    r_unc = AnalysisEngine(s3).load().analyze(pan.media, analysis_id="e5-s09-uncompensated")
    r_cmp = next(x for x in rows if x["scenario_id"] == "s09_camera_pan_cat" and x["method"] == "rule_baseline")
    compensation_diag = {
        "scenario_id": "s09_camera_pan_cat",
        "uncompensated_tracks": [t.movement.model_dump() for t in r_unc.observations.animal_tracks],
        "compensated_tracks_rows": r_cmp["tracks"],
    }
    summary = {
        "experiment": "E5",
        "camera_compensation_diagnostic": compensation_diag,
        "title": "Fusion/abstention behavior on synthetic-pairing scenarios",
        "status_note": "Behavioral integration checks only. Multimodal accuracy: not_evaluated (no genuine paired test set).",
        "synthetic_pairing": True,
        "scenarios": len(scs),
        "methods": METHODS,
        "pass_counts": {m: f"{sum(v)}/{len(v)}" for m, v in per_method_pass.items()},
        "multimodal_accuracy": "not_evaluated",
        "rows": rows,
        "runtime_s": time.perf_counter() - t0,
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    write_json(out_dir / "scenario_results.json", summary)
    return summary
