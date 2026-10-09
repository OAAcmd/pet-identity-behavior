"""Fusion and abstention behavior on synthetic evidence (no models involved)."""

import math

import pytest

from petperception.config import FusionConfig
from petperception.fusion.engine import Ev, FusionContext, FusionEngine
from petperception.schemas.analysis import EvidenceItem, Provenance

pytestmark = pytest.mark.synthetic


def ev(eid, root, tags, t0, t1, track=None, attributed=None, kind="sound_event", rel="high"):
    item = EvidenceItem(evidence_id=eid, kind=kind, t_start_s=t0, t_end_s=t1, summary=eid, track_id=track, reliability=rel,
                        provenance=Provenance(component="test", root=root, method="synthetic"))
    return Ev(item, root, frozenset(tags), 1.0, attributed_track=attributed)


def ctx(**kw):
    base = dict(audio_available=True, video_available=True, zones_configured=True, person_detection_enabled=False, n_tracks=1)
    base.update(kw)
    return FusionContext(**base)


def by_name(hyps, name, track="track-1"):
    return [h for h in hyps if h.name == name and h.track_id == track]


def test_context_alone_never_yields_food():
    e = [ev("t", "vision:detector", {"animal_visible"}, 0, 8, track="track-1", kind="animal_track"),
         ev("z", "context:zones", {"zone:food_area"}, 0, 8, track="track-1", kind="zone_occupancy")]
    h = by_name(FusionEngine(FusionConfig()).evaluate(e, ctx()), "food")[0]
    assert h.status == "insufficient_evidence"
    assert "context_only_evidence" in h.reason_codes and h.calibrated_confidence is None


def test_cooccurring_vocalization_and_zone_is_experimental_only():
    e = [ev("s", "audio:yamnet", {"vocalization"}, 2, 4, attributed="track-1"),
         ev("z", "context:zones", {"zone:food_area"}, 0, 8, track="track-1", kind="zone_occupancy")]
    h = by_name(FusionEngine(FusionConfig()).evaluate(e, ctx()), "food")[0]
    assert h.status == "experimental_hypothesis"
    assert set(h.independent_provenance_roots) == {"audio:yamnet", "context:zones"}
    assert h.calibration_status == "uncalibrated" and h.calibrated_confidence is None
    assert "fusion_not_validated" in h.reason_codes and "audio_attribution_unverified" in h.reason_codes


def test_stale_evidence_is_not_combined():
    e = [ev("s", "audio:yamnet", {"vocalization"}, 0, 1, attributed="track-1"),
         ev("z", "context:zones", {"zone:food_area"}, 7, 9, track="track-1", kind="zone_occupancy")]
    h = by_name(FusionEngine(FusionConfig(co_occurrence_window_s=3.0, max_evidence_age_s=10.0)).evaluate(e, ctx()), "food")[0]
    assert h.status == "insufficient_evidence" and "stale_evidence" in h.reason_codes


def test_shared_provenance_counts_once():
    # Tag + embedding classifier from the same YAMNet root are one confirmation.
    e = [ev("s1", "audio:yamnet", {"hiss"}, 1, 2, attributed="track-1"),
         ev("s2", "audio:yamnet", {"growl"}, 1, 2, attributed="track-1")]
    hs = FusionEngine(FusionConfig()).evaluate(e, ctx())
    assert all(h.status != "experimental_hypothesis" for h in hs)
    assert any("single_provenance_root" in h.reason_codes for h in hs)


def test_conflict_forces_abstention_for_relaxed():
    e = [ev("m", "vision:detector", {"movement:low"}, 0, 8, track="track-1", kind="movement"),
         ev("z", "context:zones", {"zone:rest_area"}, 0, 8, track="track-1", kind="zone_occupancy"),
         ev("g", "audio:yamnet", {"growl"}, 3, 4, attributed="track-1")]
    h = by_name(FusionEngine(FusionConfig()).evaluate(e, ctx()), "relaxed")[0]
    assert h.status == "abstained" and "conflicting_evidence" in h.reason_codes and "g" in h.conflicting_evidence_ids


def test_purr_never_rules_out_discomfort_and_silence_is_not_evidence():
    e = [ev("p", "audio:yamnet", {"purr"}, 0, 5, attributed="track-1"),
         ev("m", "vision:detector", {"movement:low"}, 0, 8, track="track-1", kind="movement")]
    h = by_name(FusionEngine(FusionConfig()).evaluate(e, ctx()), "relaxed")[0]
    assert "purr_does_not_rule_out_discomfort" in h.reason_codes and "silence_not_used_as_evidence" in h.reason_codes
    # Silent audio produces no evidence at all, so it cannot support 'relaxed'.
    only_vision = [ev("m", "vision:detector", {"movement:low"}, 0, 8, track="track-1", kind="movement")]
    h2 = by_name(FusionEngine(FusionConfig()).evaluate(only_vision, ctx(audio_available=False, audio_state="silent")), "relaxed")[0]
    assert h2.status == "insufficient_evidence"


def test_multiple_animals_unattributed_sound():
    e = [ev("s", "audio:yamnet", {"vocalization", "bark"}, 2, 4),
         ev("t1", "vision:detector", {"animal_visible"}, 0, 8, track="track-1", kind="animal_track"),
         ev("t2", "vision:detector", {"animal_visible"}, 0, 8, track="track-2", kind="animal_track")]
    hs = FusionEngine(FusionConfig()).evaluate(e, ctx(n_tracks=2))
    assert hs and all(h.track_id is None for h in hs)
    assert all("uncertain_attribution_multiple_animals" in h.reason_codes for h in hs)
    assert all(h.status != "experimental_hypothesis" for h in hs)


def test_visible_animal_alone_opens_no_candidate():
    e = [ev("t", "vision:detector", {"animal_visible"}, 0, 8, track="track-1", kind="animal_track")]
    assert FusionEngine(FusionConfig()).evaluate(e, ctx()) == []


@pytest.mark.parametrize("method", ["rule_baseline", "weighted_average", "product_of_experts"])
def test_fusion_methods_are_bounded_and_finite(method):
    cfg = FusionConfig(method=method, weights={"audio": 50.0, "vision": 50.0, "context": 50.0, "pose": 1.0}, temperature=0.01)
    e = [ev("s", "audio:yamnet", {"vocalization"}, 2, 4, attributed="track-1"),
         ev("z", "context:zones", {"zone:food_area"}, 0, 8, track="track-1", kind="zone_occupancy")]
    for h in FusionEngine(cfg).evaluate(e, ctx()):
        assert h.score is not None and math.isfinite(h.score) and 0.0 <= h.score <= 1.0


def test_weighted_average_shrinks_when_experts_missing():
    cfg = FusionConfig(method="weighted_average")
    full = [ev("s", "audio:yamnet", {"vocalization"}, 2, 4, attributed="track-1"),
            ev("z", "context:zones", {"zone:food_area"}, 0, 8, track="track-1", kind="zone_occupancy")]
    h_full = by_name(FusionEngine(cfg).evaluate(full, ctx()), "food")[0]
    h_one = by_name(FusionEngine(cfg).evaluate(full[1:], ctx()), "food")[0]
    assert h_one.score < h_full.score


def test_low_reliability_lowers_rule_score():
    good = [ev("s", "audio:yamnet", {"vocalization"}, 2, 4, attributed="track-1"),
            ev("z", "context:zones", {"zone:food_area"}, 0, 8, track="track-1", kind="zone_occupancy")]
    bad = [good[0], ev("z", "context:zones", {"zone:food_area"}, 0, 8, track="track-1", kind="zone_occupancy", rel="low")]
    s_good = by_name(FusionEngine(FusionConfig()).evaluate(good, ctx()), "food")[0].score
    h_bad = by_name(FusionEngine(FusionConfig()).evaluate(bad, ctx()), "food")[0]
    assert h_bad.score < s_good and "low_reliability_evidence" in h_bad.reason_codes
