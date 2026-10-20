"""Household validation: annotation format and scoring of app results against it.

The collection and annotation procedure is in docs/household/. This module
defines the files the annotators produce (strict schemas: unknown fields are
rejected) and scores app results (docs/analysis_result.schema.json, phone or
server engine) against them with the metrics and acceptance thresholds fixed
in docs/EVALUATION_PLAN_REAL_PAIRED.md. Confidence intervals resample
households.

Layout of a dataset directory:

    <dir>/recordings/<recording_id>.json     RecordingManifest (owner side, written at recording time)
    <dir>/annotations/<recording_id>.<annotator_id>.json   Annotation (one per annotator, blind to app output)
    <dir>/adjudicated/<recording_id>.json     Annotation (final labels after adjudication)
    <dir>/results/<recording_id>.json         the app's analysis result for that recording

Two layers are kept apart on purpose: what an annotator can see and hear
(``observed``: presence, activities, sounds, posture, interactions,
furniture relations) and what the situation suggests about the animal's
motivation (``interpretation``), which may always be "cannot_tell".
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

SCHEMA = "petperception.household_annotation/1"
Species = Literal["cat", "dog"]
Situation = Literal["before_feeding", "waiting_at_door", "play", "greeting", "visitor_or_doorbell", "rest", "conflict_or_tension", "grooming_or_handling",
                    "alone_at_home", "ordinary", "pet_free_control", "other"]
ActivityKind = Literal["stationary", "moving", "active_bout", "pacing", "entered_view", "left_view", "eating", "drinking", "grooming_self", "playing",
                       "sleeping", "scratching", "eliminating", "other"]
VocalType = Literal["meow", "purr", "hiss", "growl", "chirp_trill", "caterwaul", "bark", "howl", "whine_whimper", "yip", "other_pet_sound", "unclear"]
PostureLabel = Literal["lying", "sitting", "standing", "other", "not_visible"]
FurnitureObject = Literal["bed", "couch", "chair", "table", "other"]
FurnitureRelation = Literal["on", "in_front_of", "beside", "under", "inside"]
Hypothesis = Literal["food", "outside", "attention", "play", "alert", "tense", "warning", "relaxed"]

# Acceptance thresholds (docs/EVALUATION_PLAN_REAL_PAIRED.md, fixed before data exist).
THRESHOLDS = {"presence_recall": 0.85, "pet_free_false_alarm_rate_max": 0.05, "segment_f1": 0.70, "vocal_event_f1": 0.80, "posture_agreement": 0.80,
              "person_interaction_f1": 0.70, "on_furniture_f1": 0.70, "false_on_rate_max": 0.20, "hypothesis_precision": 0.70}
SEGMENT_TOLERANCE_S = 1.0
VOCAL_TOLERANCE_S = 0.5
FRAME_STEP_S = 0.5  # the app samples 2 frames per second
# App segment kinds that correspond to annotated activities.
APP_KIND = {"stationary": "stationary", "moving": "moving", "active_bout": "active_bout", "pacing": "pacing_candidate",
            "entered_view": "entered_view", "left_view": "left_view"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Pet(_Strict):
    pet_id: str
    species: Species
    breed_reported_by_owner: str | None = None
    age_years: float | None = None


class Device(_Strict):
    model: str
    os: str
    browser: str


class OwnerContext(_Strict):
    situation: Situation
    what_happened_before: str = ""
    what_the_pet_did_next: str = ""
    note: str = ""


class Consent(_Strict):
    research_use: bool
    annotation_by_project_staff: bool
    people_in_recording_consented_or_blurred: bool
    aggregate_publication: bool


class RecordingManifest(_Strict):
    schema_: Literal["petperception.household_recording/1"] = Field(alias="schema")
    recording_id: str
    household_id: str
    pets_in_household: list[Pet]
    device: Device
    recorded_at: str
    duration_s: float
    room: str = ""
    lighting: Literal["daylight", "artificial", "low_light", "mixed"] = "artificial"
    owner_context: OwnerContext
    consent: Consent


class Interval(_Strict):
    t_start_s: float
    t_end_s: float


class Presence(Interval):
    pet_id: str
    species: Species


class Activity(Interval):
    pet_id: str
    kind: ActivityKind


class Vocalization(Interval):
    pet_id: str | None = None  # None: cannot tell which pet
    species: Species | None = None
    type: VocalType


class PostureFrame(_Strict):
    pet_id: str
    t_s: float
    label: PostureLabel


class PersonInteraction(Interval):
    pet_id: str
    kind: Literal["holding", "petting", "playing_with", "feeding", "other"]


class FurnitureInterval(Interval):
    pet_id: str
    object: FurnitureObject
    relation: FurnitureRelation


class Observed(_Strict):
    presence: list[Presence] = []
    activities: list[Activity] = []
    vocalizations: list[Vocalization] = []
    posture: list[PostureFrame] = []
    person_interactions: list[PersonInteraction] = []
    furniture: list[FurnitureInterval] = []


class Interpretation(_Strict):
    pet_id: str
    hypothesis: Hypothesis
    label: Literal["yes", "no", "cannot_tell"]
    evidence: str = ""


class Annotation(_Strict):
    schema_: Literal["petperception.household_annotation/1"] = Field(alias="schema")
    recording_id: str
    annotator_id: str
    blind_to_app_output: bool
    observed: Observed
    interpretation: list[Interpretation] = []
    quality: list[Literal["low_light", "pet_mostly_occluded", "heavy_camera_motion", "audio_noisy", "too_short"]] = []


def json_schemas() -> dict:
    return {"recording": RecordingManifest.model_json_schema(by_alias=True), "annotation": Annotation.model_json_schema(by_alias=True)}


# ---------------------------------------------------------------- scoring helpers

def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def match_intervals(ref: list[tuple[str, float, float]], hyp: list[tuple[str, float, float]], tol: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching by label with start and end within tol (or IoU >= 0.5). Returns (tp, fn, fp)."""
    used: set[int] = set()
    tp = 0
    for lab, s, e in ref:
        best, bj = None, None
        for j, (lb, s2, e2) in enumerate(hyp):
            if j in used or lb != lab:
                continue
            iou = _overlap((s, e), (s2, e2)) / max(max(e, e2) - min(s, s2), 1e-9)
            close = abs(s - s2) <= tol and abs(e - e2) <= tol
            if close or iou >= 0.5:
                d = abs(s - s2) + abs(e - e2)
                if best is None or d < best:
                    best, bj = d, j
        if bj is not None:
            used.add(bj)
            tp += 1
    return tp, len(ref) - tp, len(hyp) - tp


def _f1(tp: int, fn: int, fp: int) -> float | None:
    return None if tp + fn + fp == 0 else 2 * tp / max(2 * tp + fn + fp, 1)


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if not a or len(a) != len(b):
        return None
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / len(a)
    pe = sum((a.count(lab) / len(a)) * (b.count(lab) / len(b)) for lab in labels)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def _app_tracks(res: dict) -> list[dict]:
    return res.get("observations", {}).get("animal_tracks", [])


def score_recording(man: RecordingManifest, ann: Annotation, res: dict) -> dict:
    """Counts for one recording; aggregated per household by `evaluate`."""
    out: dict = {"recording_id": man.recording_id, "household_id": man.household_id, "pet_free": man.owner_context.situation == "pet_free_control"}
    tracks = _app_tracks(res)
    # Presence: sampled frames inside annotated presence; a frame counts as found when an app track of that species covers it.
    hit = tot = 0
    for p in ann.observed.presence:
        t = p.t_start_s
        while t <= p.t_end_s + 1e-9:
            tot += 1
            hit += any(tr["species"] == p.species and tr["first_seen_s"] - FRAME_STEP_S <= t <= tr["last_seen_s"] + FRAME_STEP_S for tr in tracks)
            t += FRAME_STEP_S
    out["presence"] = {"frames": tot, "found": hit}
    out["any_pet_reported"] = bool(tracks)
    # Activity / view segments (kinds the app reports), any track.
    segs = [s for tr in tracks for s in tr.get("behavior_segments", [])]
    out["segments"] = {}
    for kind, app_kind in APP_KIND.items():
        ref = [(kind, a.t_start_s, a.t_end_s) for a in ann.observed.activities if a.kind == kind]
        hyp = [(kind, s["t_start_s"], s["t_end_s"]) for s in segs if s["kind"] == app_kind]
        out["segments"][kind] = match_intervals(ref, hyp, SEGMENT_TOLERANCE_S)
    # Vocal events by species.
    sounds = res.get("observations", {}).get("sound_events", [])
    out["vocal"] = {}
    for sp in ("cat", "dog"):
        ref = [(sp, v.t_start_s, v.t_end_s) for v in ann.observed.vocalizations if v.species == sp]
        hyp = [(sp, s["t_start_s"], s["t_end_s"]) for s in sounds if s.get("label") == f"{sp}_vocalization"]
        out["vocal"][sp] = match_intervals(ref, hyp, VOCAL_TOLERANCE_S)
    # Posture agreement on frames where the app reports a label (nearest app frame within 0.25 s, same species).
    agree = n = 0
    for pf in ann.observed.posture:
        if pf.label in ("other", "not_visible"):
            continue
        cands = [(abs(f["t_s"] - pf.t_s), f["label"]) for tr in tracks for f in tr.get("posture", []) if f["label"] != "uncertain"]
        if cands and min(cands)[0] <= 0.25:
            n += 1
            agree += min(cands)[1] == pf.label
    out["posture"] = {"compared": n, "agree": agree}
    # Person interaction and on-furniture intervals.
    out["person_interaction"] = match_intervals([("pi", a.t_start_s, a.t_end_s) for a in ann.observed.person_interactions],
                                                [("pi", s["t_start_s"], s["t_end_s"]) for s in segs if s["kind"] == "person_interaction_candidate"], SEGMENT_TOLERANCE_S)
    on_ref = [(f.object, f.t_start_s, f.t_end_s) for f in ann.observed.furniture if f.relation == "on" and f.object != "other"]
    on_hyp = [(s.get("related_object"), s["t_start_s"], s["t_end_s"]) for s in segs if s["kind"] == "on_furniture_candidate"]
    out["on_furniture"] = match_intervals(on_ref, on_hyp, SEGMENT_TOLERANCE_S)
    near = [f for f in ann.observed.furniture if f.relation in ("in_front_of", "beside")]
    false_on = sum(any(_overlap((f.t_start_s, f.t_end_s), (s["t_start_s"], s["t_end_s"])) > 0 and s.get("related_object") == f.object for s in segs
                       if s["kind"] == "on_furniture_candidate") for f in near)
    out["near_furniture"] = {"intervals": len(near), "reported_as_on": false_on}
    # Hypotheses: a non-abstained app hypothesis is correct when the adjudicated interpretation is "yes".
    truth = {i.hypothesis: i.label for i in ann.interpretation}
    hyps = [h for h in res.get("hypotheses", []) if h.get("status") not in ("abstained", "not_supported", "insufficient_evidence")]
    out["hypotheses"] = [{"name": h.get("name"), "truth": truth.get(h.get("name"), "not_annotated")} for h in hyps]
    # Intent block (result schema 1.5.0): only supported hypotheses count as claims; tentative and context-only do not.
    it = res.get("intent") or None
    if it is not None:
        claims = [h for h in it.get("hypotheses", []) if h.get("status") in ("supported", "supported_corroborated")]
        out["intent"] = {"status": it.get("status"), "claims": [{"name": h.get("name"), "truth": truth.get(h.get("name"), "not_annotated")} for h in claims]}
    return out


def _bootstrap(households: list[str], stat, n: int = 1000, seed: int = 0) -> list[float] | None:
    uniq = sorted(set(households))
    if len(uniq) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        v = stat(list(pick))
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def evaluate(dataset: Path) -> dict:
    """Score every recording that has a manifest, an adjudicated annotation and an app result."""
    rows, missing = [], []
    for mp in sorted((dataset / "recordings").glob("*.json")):
        man = RecordingManifest.model_validate_json(mp.read_text())
        ap, rp = dataset / "adjudicated" / mp.name, dataset / "results" / mp.name
        if not (ap.exists() and rp.exists()):
            missing.append(man.recording_id)
            continue
        rows.append(score_recording(man, Annotation.model_validate_json(ap.read_text()), json.loads(rp.read_text())))
    by_h: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_h[r["household_id"]].append(r)
    hh = list(by_h)

    def agg(fn):
        def stat(sel: list[str]):
            return fn([r for h in sel for r in by_h[h]])
        return stat

    def presence(rs):
        f = sum(r["presence"]["frames"] for r in rs)
        return None if f == 0 else sum(r["presence"]["found"] for r in rs) / f

    def false_alarm(rs):
        pf = [r for r in rs if r["pet_free"]]
        return None if not pf else sum(r["any_pet_reported"] for r in pf) / len(pf)

    def f1_of(key, sub=None):
        def f(rs):
            c = np.array([(r[key][sub] if sub else r[key]) for r in rs]).sum(axis=0) if rs else np.zeros(3)
            return _f1(*map(int, c))
        return f

    def posture(rs):
        n = sum(r["posture"]["compared"] for r in rs)
        return None if n == 0 else sum(r["posture"]["agree"] for r in rs) / n

    def false_on(rs):
        n = sum(r["near_furniture"]["intervals"] for r in rs)
        return None if n == 0 else sum(r["near_furniture"]["reported_as_on"] for r in rs) / n

    metrics: dict = {}

    def add(name, fn, threshold, higher_is_better=True):
        v = agg(fn)(hh)
        ci = _bootstrap(hh, agg(fn))
        status = "not_evaluated" if v is None else ("met" if (v >= threshold if higher_is_better else v <= threshold) else "not_met")
        metrics[name] = {"value": v, "ci95_households": ci, "threshold": threshold, "status": status}

    add("presence_recall", presence, THRESHOLDS["presence_recall"])
    add("pet_free_false_alarm_rate", false_alarm, THRESHOLDS["pet_free_false_alarm_rate_max"], higher_is_better=False)
    for kind in APP_KIND:
        add(f"segment_f1:{kind}", f1_of("segments", kind), THRESHOLDS["segment_f1"])
    for sp in ("cat", "dog"):
        add(f"vocal_event_f1:{sp}", f1_of("vocal", sp), THRESHOLDS["vocal_event_f1"])
    add("posture_agreement", posture, THRESHOLDS["posture_agreement"])
    add("person_interaction_f1", f1_of("person_interaction"), THRESHOLDS["person_interaction_f1"])
    add("on_furniture_f1", f1_of("on_furniture"), THRESHOLDS["on_furniture_f1"])
    add("false_on_rate_near_furniture", false_on, THRESHOLDS["false_on_rate_max"], higher_is_better=False)
    hyp = defaultdict(lambda: [0, 0, 0])  # yes, no, cannot_tell / not annotated
    for r in rows:
        for h in r["hypotheses"]:
            hyp[h["name"]][0 if h["truth"] == "yes" else 1 if h["truth"] == "no" else 2] += 1
    metrics["hypothesis_precision"] = {name: {"non_abstained": sum(c), "precision": c[0] / (c[0] + c[1]) if c[0] + c[1] else None,
                                              "cannot_tell_or_unannotated": c[2], "threshold": THRESHOLDS["hypothesis_precision"]} for name, c in sorted(hyp.items())}
    with_intent = [r for r in rows if "intent" in r]
    if with_intent:
        ih = defaultdict(lambda: [0, 0, 0])
        for r in with_intent:
            for h in r["intent"]["claims"]:
                ih[h["name"]][0 if h["truth"] == "yes" else 1 if h["truth"] == "no" else 2] += 1
        metrics["intent_claim_precision"] = {name: {"claims": sum(c), "precision": c[0] / (c[0] + c[1]) if c[0] + c[1] else None,
                                                    "cannot_tell_or_unannotated": c[2], "threshold": THRESHOLDS["hypothesis_precision"]}
                                             for name, c in sorted(ih.items())}
        metrics["intent_coverage"] = {"recordings": len(with_intent),
                                      "with_a_supported_hypothesis": sum(bool(r["intent"]["claims"]) for r in with_intent),
                                      "status_counts": dict(Counter(r["intent"]["status"] for r in with_intent))}
    return {"schema": "petperception.household_evaluation/1", "recordings_scored": len(rows), "households": len(hh), "recordings_missing_labels_or_results": missing,
            "metrics": metrics}


def agreement(dataset: Path) -> dict:
    """Inter-annotator agreement on recordings with two annotations (posture kappa, interpretation kappa, interval agreement)."""
    pairs: dict[str, list[Annotation]] = defaultdict(list)
    for p in sorted((dataset / "annotations").glob("*.json")):
        a = Annotation.model_validate_json(p.read_text())
        pairs[a.recording_id].append(a)
    post_a, post_b, int_a, int_b = [], [], [], []
    seg = np.zeros(3, int)
    for anns in pairs.values():
        if len(anns) < 2:
            continue
        a, b = anns[0], anns[1]
        pb = {(f.pet_id, round(f.t_s * 2) / 2): f.label for f in b.observed.posture}
        for f in a.observed.posture:
            k = (f.pet_id, round(f.t_s * 2) / 2)
            if k in pb:
                post_a.append(f.label)
                post_b.append(pb[k])
        ib = {(i.pet_id, i.hypothesis): i.label for i in b.interpretation}
        for i in a.interpretation:
            if (i.pet_id, i.hypothesis) in ib:
                int_a.append(i.label)
                int_b.append(ib[(i.pet_id, i.hypothesis)])
        seg += np.array(match_intervals([(x.kind, x.t_start_s, x.t_end_s) for x in a.observed.activities],
                                        [(x.kind, x.t_start_s, x.t_end_s) for x in b.observed.activities], SEGMENT_TOLERANCE_S))
    return {"recordings_double_annotated": sum(1 for v in pairs.values() if len(v) >= 2), "posture_kappa": cohen_kappa(post_a, post_b),
            "interpretation_kappa": cohen_kappa(int_a, int_b), "activity_interval_f1_between_annotators": _f1(*map(int, seg))}
