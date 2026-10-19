"""Subtype validation gate: only subtypes that passed E21 become tags or appear in text (synthetic inputs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from petperception.audio.events import load_subtype_validation, shown_subtypes, vocal_conflict_tags, vocal_tags

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.synthetic


def _tags(label, sub, shown):
    return sorted(vocal_tags(label, sub, shown))


def test_unvalidated_subtypes_are_not_tags_but_keep_events_non_request():
    shown = {"meow", "bark"}
    assert _tags("cat_vocalization", {"purr": 0.8}, shown) == []  # purring is not a request, and purr is not validated
    assert _tags("cat_vocalization", {"hiss": 0.6, "meow": 0.5}, shown) == ["vocalization"]
    assert _tags("dog_vocalization", {"bark": 0.5, "growl": 0.4}, shown) == ["bark", "vocalization"]
    assert _tags("growl", {"growl": 0.7}, shown) == []


def test_missing_validation_file_validates_nothing(tmp_path):
    assert load_subtype_validation(tmp_path / "absent.json") == {}
    assert load_subtype_validation(None) == {}
    assert shown_subtypes({}) == set()


def test_committed_artifact_matches_e21_and_the_gate():
    art = json.loads((REPO / "artifacts" / "audio_subtype_validation.json").read_text())
    e21 = json.loads((REPO / "results" / "e21" / "metrics.json").read_text())["results"]["vocalization_type"]
    gate = art["gate"]
    for k, v in art["subtypes"].items():
        if v["status"] in ("passed", "failed"):
            r = e21[v["target"]]["all"]
            assert r["positives"] >= gate["min_positives"]
            ok = r["f1"] >= gate["f1"] and r["f1_ci95_by_uploader"]["ci_low"] >= gate["ci_low"]
            assert (v["status"] == "passed") == ok, k
    assert shown_subtypes(load_subtype_validation(REPO / "artifacts" / "audio_subtype_validation.json")) == {"meow", "bark"}


def test_conflict_only_tags_are_the_unvalidated_confident_subtypes():
    shown = {"meow", "bark"}
    assert vocal_conflict_tags("cat_vocalization", {"purr": 0.8, "meow": 0.9, "hiss": 0.2}, shown) == {"purr"}
    assert vocal_conflict_tags("growl", {"growl": 0.7}, shown) == {"growl"}
    assert vocal_conflict_tags("dog_vocalization", {"bark": 0.9}, shown) == set()


def test_unvalidated_growl_blocks_relaxed_but_opens_nothing():
    """A possible growl must still make 'relaxed' abstain (cautious direction) without opening 'tense' or 'warning'."""
    from petperception.config import load_settings
    from petperception.fusion.engine import Ev, FusionContext, FusionEngine
    from petperception.schemas.analysis import EvidenceItem, Provenance

    def item(eid, kind, t0, t1, root):
        return EvidenceItem(evidence_id=eid, kind=kind, t_start_s=t0, t_end_s=t1, summary=eid, track_id="track-1" if kind != "sound_event" else None,
                            reliability="high", provenance=Provenance(component="x", root=root, method="x"))

    pool = [
        Ev(item("trk", "animal_track", 0, 8, "vision:detector"), "vision:detector", frozenset({"animal_visible", "species:cat"})),
        Ev(item("mv", "movement", 0, 8, "vision:detector"), "vision:detector", frozenset({"movement:low"})),
        Ev(item("zone", "zone_occupancy", 0, 8, "context:zones"), "context:zones", frozenset({"zone:rest_area"})),
        Ev(item("snd", "sound_event", 5, 6, "audio:yamnet"), "audio:yamnet", frozenset(), attributed_track="track-1", conflict_tags=frozenset({"growl"})),
    ]
    ctx = FusionContext(audio_available=True, video_available=True, zones_configured=True, person_detection_enabled=False, n_tracks=1)
    hyps: dict = {}
    for h in FusionEngine(load_settings(None).fusion).evaluate(pool, ctx):
        hyps.setdefault(h.name, []).append(h)
    relaxed = [h for h in hyps["relaxed"] if h.track_id == "track-1"]
    assert relaxed and all(h.status == "abstained" and "snd" in h.conflicting_evidence_ids for h in relaxed)
    assert all(h.status not in ("experimental_hypothesis", "supported_observation") for n in ("tense", "warning") for h in hyps.get(n, []))



def test_per_tag_context_thresholds_override_the_shared_one():
    import numpy as np

    from petperception.audio.events import CONTEXT_TAG_MIDS, extract_events
    from petperception.audio.yamnet import WindowScores

    class _Backend:  # just what extract_events reads
        mid_to_index = {CONTEXT_TAG_MIDS["knock"]: 0, CONTEXT_TAG_MIDS["door"]: 1, "/m/0ghcn6": 2}
        labels = ["knock", "door", "growl"]

    scores = np.zeros((4, 3), np.float32)
    scores[1, 0] = 0.4  # knock at 0.4
    scores[2, 1] = 0.4  # door at 0.4
    t = np.arange(4) * 0.48
    ws = WindowScores(t, t + 0.975, scores)
    labels = lambda ev: sorted(e.label for e in ev)  # noqa: E731
    shared = extract_events(ws, _Backend(), {}, {"growl": 0.5}, 0.3, 0.5)
    per_tag = extract_events(ws, _Backend(), {}, {"growl": 0.5}, 0.3, 0.5, {"knock": 0.59})
    assert labels(shared) == ["context:door", "context:knock"]
    assert labels(per_tag) == ["context:door"]  # knock now needs 0.59; door keeps the shared 0.3


def test_committed_context_thresholds_were_adopted_by_e21b():
    from petperception.audio.events import load_context_thresholds

    th = load_context_thresholds(REPO / "artifacts" / "audio_context_thresholds.json")
    e21b = json.loads((REPO / "results" / "e21b" / "metrics.json").read_text())["variants"]["cc0_ccby"]
    for tag, v in th.items():
        r = e21b["results"][f"context:{tag}"]
        assert r["adopt"] and e21b["selected"]["context"][tag] == v
