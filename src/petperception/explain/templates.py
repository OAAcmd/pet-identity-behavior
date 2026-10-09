"""Deterministic explanation and suggestion templates.

Every explanation is assembled only from evidence items that exist in the
result, cited by ID. There are no medical statements, no first-person pet
quotes, and no "translation" phrasing. Optional entertaining phrasing is
disabled in Phase 1 (there is no code path for it).
"""

from __future__ import annotations

from ..schemas.analysis import EvidenceItem, Hypothesis, Suggestion

REASON_TEXT = {
    "fusion_not_validated": "the fusion rules have not been validated on paired household data",
    "single_provenance_root": "all supporting evidence comes from a single source",
    "context_only_evidence": "only scene context (an owner-marked zone) supports it",
    "stale_evidence": "the supporting observations were too far apart in time",
    "missing_modality_audio": "no usable audio was available",
    "audio_silent": "the audio track is silent (silence is not evidence either way)",
    "audio_model_unavailable": "the audio-event model is unavailable",
    "missing_modality_video": "no usable video was available",
    "zones_not_configured": "no scene zones were configured",
    "pose_unavailable": "no verified animal pose model is available",
    "person_detection_disabled": "person detection is disabled",
    "uncertain_attribution_multiple_animals": "several animals were visible, so the sound could not be attributed",
    "no_visible_animal_for_attribution": "no animal was visible when the sound occurred (it may be off-screen or from a TV/speaker)",
    "conflicting_evidence": "co-occurring evidence points in a different direction",
    "low_support": "the experimental score was below the abstention threshold",
    "audio_attribution_unverified": "attributing the sound to this animal is unverified (one visible candidate)",
    "low_reliability_evidence": "some evidence has low reliability (quality limits)",
    "silence_not_used_as_evidence": "silence is never used as evidence of relaxation",
    "purr_does_not_rule_out_discomfort": "purring can also occur with discomfort",
}

STATUS_TEXT = {
    "experimental_hypothesis": "Experimental hypothesis",
    "insufficient_evidence": "Insufficient evidence",
    "abstained": "Abstained",
}


def explain_hypothesis(h: Hypothesis, evidence: dict[str, EvidenceItem]) -> str:
    sup = [f"{evidence[i].summary} [{i}]" for i in h.supporting_evidence_ids if i in evidence]
    con = [f"{evidence[i].summary} [{i}]" for i in h.conflicting_evidence_ids if i in evidence]
    subject = f" for {h.track_id}" if h.track_id else ""
    parts = [f"{STATUS_TEXT[h.status]} '{h.name}' ({h.semantic_type}){subject}."]
    if sup:
        parts.append("Supporting observations: " + "; ".join(sup) + ".")
    if con:
        parts.append("Conflicting observations: " + "; ".join(con) + ".")
    reasons = [REASON_TEXT[r] for r in h.reason_codes if r in REASON_TEXT]
    if reasons:
        parts.append("Notes: " + "; ".join(reasons) + ".")
    if h.score is not None:
        parts.append(f"Score {h.score:.2f} is an uncalibrated {h.score_kind} value, not a probability.")
    return " ".join(parts)


def suggestions_for(hyps: list[Hypothesis], quality_codes: set[str], n_tracks: int, any_pet_sound: bool) -> list[Suggestion]:
    out: list[Suggestion] = []

    def add(text: str, ids: list[str], kind: str = "observation_followup") -> None:
        out.append(Suggestion(suggestion_id=f"sug-{len(out) + 1}", text=text, based_on_evidence_ids=ids, kind=kind))

    for h in hyps:
        if h.status != "experimental_hypothesis":
            continue
        ids = h.supporting_evidence_ids
        if h.name == "food":
            add("The animal vocalized while in the area you marked for feeding. If it is close to the usual feeding time, "
                "you could check whether food and water are available. This is an unvalidated heuristic, not a determination of hunger.", ids)
        elif h.name == "outside":
            add("The animal vocalized while in the area you marked as a door. If your routine allows, you could check whether "
                "it wants to go out. This is an unvalidated heuristic.", ids)
        elif h.name == "attention":
            add("The animal vocalized while a person was visible. This may or may not be a request for attention; "
                "the system cannot tell.", ids)
        elif h.name in ("warning", "tense"):
            add("Hiss- or growl-like sounds were observed together with a visible animal. Giving animals space and not reaching "
                "toward them is a common precaution. If such signals are new or frequent, consider asking a veterinarian or a "
                "qualified behavior professional.", ids, "safety_general")
        elif h.name == "alert":
            add("Barking or door sounds co-occurred with movement. Checking what happened nearby may help interpret it.", ids)
        elif h.name == "play":
            add("Sustained movement in the play area was observed. Movement alone cannot distinguish play from other activity.", ids)
        elif h.name == "relaxed":
            add("Low movement in the rest area was observed. This does not confirm comfort or wellbeing.", ids)
    if n_tracks > 1 and any_pet_sound:
        add("Several animals were visible, so sounds were not assigned to a specific animal.", [], "data_quality")
    if "low_light" in quality_codes:
        add("Lighting was low in part of the video; visual observations there are less reliable.", [], "data_quality")
    if "bandwidth_limited_audio" in quality_codes:
        add("The audio has a low sample rate, so high-frequency sound content is missing and sound tagging is less reliable.", [], "data_quality")
    if "camera_motion" in quality_codes:
        add("The camera moved during recording; apparent animal movement is less reliable.", [], "data_quality")
    return out
