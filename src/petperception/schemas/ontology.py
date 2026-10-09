"""Research hypothesis ontology (future targets, not trained classes).

The eight research hypotheses mix needs, activities, states and a
communicative signal. They are evaluated independently (multi-label) and may
coexist, e.g. ``food`` (need) with ``alert`` (state). ``relaxed`` is in soft
tension with ``tense``/``warning``; tension is expressed as conflict, never as
hard mutual exclusion. ``unknown`` and ``insufficient_evidence`` are decision
outcomes rather than hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass

ONTOLOGY_VERSION = "ontology-v1"


@dataclass(frozen=True)
class HypothesisDef:
    name: str
    semantic_type: str  # need | activity | state | communicative_signal
    description: str
    soft_conflicts: tuple[str, ...]
    observable_evidence: tuple[str, ...]
    never_from: tuple[str, ...]


HYPOTHESES: dict[str, HypothesisDef] = {
    "food": HypothesisDef("food", "need", "Possible food-seeking / feeding-routine anticipation.", (),
                          ("vocalization co-occurring with dwell in an owner-marked food_area",),
                          ("zone dwell alone", "vocalization alone", "COCO 'bowl' detections (not validated)")),
    "outside": HypothesisDef("outside", "need", "Possible wish to go outside.", (),
                             ("vocalization co-occurring with dwell in an owner-marked exit_door zone",),
                             ("zone dwell alone", "vocalization alone")),
    "attention": HypothesisDef("attention", "need", "Possible attention/social solicitation.", (),
                               ("vocalization while a person is visible (requires 'person' context detection)",),
                               ("vocalization alone",)),
    "play": HypothesisDef("play", "activity", "Possible play activity.", ("tense", "warning"),
                          ("sustained high movement in an owner-marked play_area", "verified play-bow pose candidate (unavailable in Phase 1)"),
                          ("movement alone",)),
    "alert": HypothesisDef("alert", "state", "Possible alerting/vigilance.", ("relaxed",),
                           ("bark/howl co-occurring with visible movement", "doorbell/knock sound co-occurring with visible movement"),
                           ("sound alone",)),
    "tense": HypothesisDef("tense", "state", "Possible tension/defensive state.", ("relaxed", "play"),
                           ("hiss/growl-like sound with a single visible candidate animal that is mostly stationary",),
                           ("silence", "movement alone")),
    "warning": HypothesisDef("warning", "communicative_signal", "Possible warning signal (hiss/growl-like vocalization).", ("relaxed", "play"),
                             ("hiss/growl-like sound with a single visible candidate animal",),
                             ("sound with no visible animal (may be off-screen or from media)",)),
    "relaxed": HypothesisDef("relaxed", "state", "Possible relaxed state.", ("tense", "warning", "alert"),
                             ("sustained low movement in an owner-marked rest_area", "purr-like sound (never rules out discomfort)"),
                             ("silence", "absence of other signals")),
}

DECISION_OUTCOMES = ("unknown", "insufficient_evidence")
