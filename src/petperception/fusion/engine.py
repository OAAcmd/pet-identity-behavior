"""Experimental hypothesis fusion with explicit abstention.

Status: EXPERIMENTAL. There is no paired household audio/video/context/outcome
dataset, so no weight, temperature or threshold below has been fitted, and no
output score is a probability. The engine defaults to observations and
abstention; a hypothesis is only marked ``experimental_hypothesis`` when
independent, co-occurring, attributable evidence satisfies its requirements.

Key rules
* Evidence is grouped by *primary provenance root* (``audio:yamnet``,
  ``vision:detector``, ``context:zones``, ``pose:<provider>``). Items sharing a
  root count once: a sound tag, its embedding and a classifier on that
  embedding are one confirmation, not three.
* Supporting items from different roots must co-occur within
  ``co_occurrence_window_s``; older pairs are reported as stale.
* Audio evidence attaches to an animal track only when exactly one visible
  candidate exists; with several animals attribution stays unknown.
* Context alone (e.g. standing near the food bowl) never yields a hypothesis;
  silence never supports ``relaxed``; a purr never rules out discomfort.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product

from ..config import FusionConfig
from ..schemas.analysis import EvidenceItem, Hypothesis, Scorer
from ..schemas.ontology import HYPOTHESES, ONTOLOGY_VERSION

RULES_REVISION = "rules-v1"

RELIABILITY_FACTOR = {"high": 1.0, "medium": 0.7, "low": 0.4}


@dataclass
class Ev:
    """Internal evidence record used by fusion."""

    item: EvidenceItem
    root: str
    tags: frozenset[str]
    strength: float = 1.0  # [0, 1] how strongly the item expresses its tags
    attributed_track: str | None = None  # for audio: track it may be attached to
    # Tags that may only block a hypothesis (unvalidated sound subtypes, E21): never triggers or support.
    conflict_tags: frozenset[str] = frozenset()

    @property
    def id(self) -> str:
        return self.item.evidence_id

    @property
    def t0(self) -> float:
        return self.item.t_start_s

    @property
    def t1(self) -> float:
        return self.item.t_end_s


@dataclass(frozen=True)
class Rule:
    # Each requirement group is a set of alternative tags; ``min_groups`` of the
    # groups must be satisfied by co-occurring evidence from distinct roots.
    groups: tuple[frozenset[str], ...]
    min_groups: int
    tag_scores: dict[str, float]  # heuristic expert opinion when the tag is observed (0.5 = neutral)
    conflicts: frozenset[str] = frozenset()
    # Animal-originated, discriminative tags that can open a candidate. Enabling
    # tags such as "animal_visible" or scene sounds never open one on their own.
    triggers: frozenset[str] = frozenset()


REQUEST_TAGS = frozenset({"vocalization"})
RULES: dict[str, Rule] = {
    "food": Rule((REQUEST_TAGS, frozenset({"zone:food_area"})), 2, {"vocalization": 0.65, "zone:food_area": 0.65},
                 triggers=frozenset({"vocalization", "zone:food_area"})),
    "outside": Rule((REQUEST_TAGS, frozenset({"zone:exit_door"})), 2, {"vocalization": 0.65, "zone:exit_door": 0.65},
                    triggers=frozenset({"vocalization", "zone:exit_door"})),
    "attention": Rule((REQUEST_TAGS, frozenset({"person_visible"})), 2, {"vocalization": 0.6, "person_visible": 0.6},
                      triggers=frozenset({"vocalization"})),
    "play": Rule((frozenset({"movement:high"}), frozenset({"zone:play_area"}), frozenset({"pose:play_bow"})), 2,
                 {"movement:high": 0.6, "zone:play_area": 0.6, "pose:play_bow": 0.7}, frozenset({"hiss", "growl"}),
                 triggers=frozenset({"movement:high", "zone:play_area", "pose:play_bow"})),
    "alert": Rule((frozenset({"bark", "howl", "doorbell", "knock"}), frozenset({"movement:high"})), 2,
                  {"bark": 0.6, "howl": 0.55, "doorbell": 0.55, "knock": 0.55, "movement:high": 0.55},
                  triggers=frozenset({"bark", "howl"})),
    "tense": Rule((frozenset({"hiss", "growl"}), frozenset({"movement:low"})), 2, {"hiss": 0.65, "growl": 0.6, "movement:low": 0.55},
                  triggers=frozenset({"hiss", "growl"})),
    "warning": Rule((frozenset({"hiss", "growl"}), frozenset({"animal_visible"})), 2, {"hiss": 0.7, "growl": 0.65, "animal_visible": 0.55},
                    triggers=frozenset({"hiss", "growl"})),
    "relaxed": Rule((frozenset({"movement:low"}), frozenset({"zone:rest_area"}), frozenset({"purr"})), 2,
                    {"movement:low": 0.6, "zone:rest_area": 0.6, "purr": 0.55}, frozenset({"hiss", "growl", "movement:high", "bark"}),
                    triggers=frozenset({"movement:low", "zone:rest_area", "purr"})),
}

def _gap(a: Ev, b: Ev) -> float:
    return max(0.0, max(a.t0, b.t0) - min(a.t1, b.t1))


def _logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class FusionContext:
    audio_available: bool
    video_available: bool
    zones_configured: bool
    person_detection_enabled: bool
    n_tracks: int
    audio_state: str = "present"
    gates: dict[str, float] = field(default_factory=dict)  # root -> gate in [0, 1]
    global_reason_codes: list[str] = field(default_factory=list)


class FusionEngine:
    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.scorer = Scorer(name=f"{cfg.method}:{RULES_REVISION}", revision=f"{RULES_REVISION}+{ONTOLOGY_VERSION}", kind=cfg.method)

    # -- expert opinions -------------------------------------------------
    def _expert(self, ev: Ev, tag: str, rule: Rule) -> float:
        base = rule.tag_scores.get(tag, 0.5)
        q = RELIABILITY_FACTOR[ev.item.reliability] * max(0.0, min(1.0, ev.strength))
        return 0.5 + (base - 0.5) * q  # low quality shrinks toward neutral

    def _combine(self, opinions: dict[str, float], relevant_roots: set[str], ctx: FusionContext) -> tuple[float, str]:
        c = self.cfg
        if not opinions:
            return 0.5, "none"
        if c.method == "rule_baseline":
            return sum(opinions.values()) / len(opinions), "heuristic_support_0_1"
        weight = {r: c.weights.get(r.split(":")[0], 1.0) for r in relevant_roots | set(opinions)}
        if c.method == "weighted_average":
            avail = sum(weight[r] for r in opinions)
            wa = sum(weight[r] * s for r, s in opinions.items()) / avail
            coverage = avail / max(sum(weight[r] for r in relevant_roots | set(opinions)), 1e-9)
            return 0.5 + (wa - 0.5) * coverage, "fusion_experimental_score_0_1"
        # Context-gated product of experts for a binary hypothesis:
        # logit = prior + sum_r g_r * w_r * logit(clip(s_r)) / T  (log-space, stable)
        eps = c.smoothing_epsilon
        z = c.prior_logit
        for r, s in opinions.items():
            s = min(1 - eps, max(eps, s))
            z += ctx.gates.get(r, 1.0) * weight[r] * _logit(s) / c.temperature
        return _sigmoid(z), "fusion_experimental_score_0_1"

    # -- candidate search ------------------------------------------------
    def evaluate(self, evidence: list[Ev], ctx: FusionContext) -> list[Hypothesis]:
        tracks = {e.item.track_id for e in evidence if e.item.track_id} | {e.attributed_track for e in evidence if e.attributed_track}
        subjects = sorted(tracks | {None}, key=lambda s: (s is None, s or ""))
        out: list[Hypothesis] = []
        counter = 0
        for name, rule in RULES.items():
            for subject in subjects:
                pool = [e for e in evidence if self._belongs(e, subject)]
                relevant = [e for e in pool if e.tags & set(rule.tag_scores)]
                if not any(e.tags & rule.triggers for e in relevant):
                    continue
                counter += 1
                hyp = self._evaluate_one(f"hyp-{counter}", name, rule, subject, relevant, pool, ctx)
                if hyp is not None:
                    out.append(hyp)
        return self._deduplicate(out)

    @staticmethod
    def _belongs(e: Ev, subject: str | None) -> bool:
        if subject is None:
            # Subject-less candidates use only unattributed audio/context sounds.
            return e.item.track_id is None and e.attributed_track is None
        return e.item.track_id == subject or e.attributed_track == subject

    def _evaluate_one(self, hid: str, name: str, rule: Rule, subject: str | None, relevant: list[Ev],
                      pool: list[Ev], ctx: FusionContext) -> Hypothesis | None:
        c = self.cfg
        reasons: list[str] = ["fusion_not_validated"]
        # Which requirement groups can each item satisfy?
        group_items: list[list[Ev]] = [[e for e in relevant if e.tags & g] for g in rule.groups]
        satisfied_groups = [i for i, items in enumerate(group_items) if items]
        best: list[Ev] = []
        stale: list[Ev] = []
        # Search co-occurring combinations across satisfied groups (small sets).
        for k in range(len(satisfied_groups), 0, -1):
            found = None
            for combo_groups in _combinations(satisfied_groups, k):
                for combo in product(*[group_items[g] for g in combo_groups]):
                    roots = {e.root for e in combo}
                    if len(roots) < len(combo):
                        continue  # same root cannot satisfy two groups
                    if all(_gap(a, b) <= c.co_occurrence_window_s for a in combo for b in combo):
                        if found is None or sum(e.strength for e in combo) > sum(e.strength for e in found):
                            found = list(combo)
                if found:
                    break
            if found:
                best = found
                break
        if len(best) < rule.min_groups:
            # Report items that would have satisfied more groups but were too far apart in time.
            for a in relevant:
                for b in relevant:
                    if a.root != b.root and c.co_occurrence_window_s < _gap(a, b) <= c.max_evidence_age_s:
                        stale.append(a)
        support_ids = [e.id for e in (best or relevant)]
        roots_used = sorted({e.root for e in (best or relevant)})
        conflicts = [e for e in pool if (e.tags | e.conflict_tags) & rule.conflicts
                     and (not best or any(_gap(e, b) <= c.co_occurrence_window_s for b in best))]
        opinions: dict[str, float] = {}
        for e in best or relevant:
            for tag in e.tags & set(rule.tag_scores):
                opinions[e.root] = max(opinions.get(e.root, 0.5), self._expert(e, tag, rule))
        relevant_roots = {"audio:yamnet" if t in ("vocalization", "bark", "howl", "doorbell", "knock", "hiss", "growl", "purr") else
                          "context:zones" if t.startswith("zone:") else "pose:none" if t.startswith("pose:") else "vision:detector"
                          for t in rule.tag_scores}
        score, score_kind = self._combine(opinions, relevant_roots, ctx)

        status = "experimental_hypothesis"
        if len(best) < rule.min_groups:
            status = "insufficient_evidence"
            if len({e.root for e in relevant}) < 2:
                reasons.append("single_provenance_root")
            if {e.root for e in relevant} == {"context:zones"}:
                reasons.append("context_only_evidence")
            if stale:
                reasons.append("stale_evidence")
            missing_roots = relevant_roots - {e.root for e in relevant}
            if "audio:yamnet" in missing_roots and not ctx.audio_available:
                reasons.append("audio_silent" if ctx.audio_state == "silent" else
                               "audio_model_unavailable" if ctx.audio_state == "present" else "missing_modality_audio")
            if missing_roots & {"vision:detector", "context:zones"} and not ctx.video_available:
                reasons.append("missing_modality_video")
            if "context:zones" in missing_roots and ctx.video_available and not ctx.zones_configured:
                reasons.append("zones_not_configured")
            if any(t.startswith("pose:") for g in rule.groups for t in g):
                reasons.append("pose_unavailable")
            if name == "attention" and not ctx.person_detection_enabled:
                reasons.append("person_detection_disabled")
            if subject is None and ctx.n_tracks > 1:
                reasons.append("uncertain_attribution_multiple_animals")
            elif subject is None and ctx.video_available and ctx.n_tracks == 0:
                reasons.append("no_visible_animal_for_attribution")
        if conflicts:
            status = "abstained"
            reasons.append("conflicting_evidence")
        if status == "experimental_hypothesis" and score < c.abstain_below_score:
            status = "abstained"
            reasons.append("low_support")
        if any(e.root == "audio:yamnet" for e in best) and subject is not None:
            reasons.append("audio_attribution_unverified")
        for e in best or relevant:
            if e.item.reliability == "low":
                reasons.append("low_reliability_evidence")
                break
        if name == "relaxed":
            reasons.append("silence_not_used_as_evidence")
            if any("purr" in e.tags for e in best or relevant):
                reasons.append("purr_does_not_rule_out_discomfort")
        hdef = HYPOTHESES[name]
        t0 = min((e.t0 for e in best or relevant), default=None)
        t1 = max((e.t1 for e in best or relevant), default=None)
        return Hypothesis(
            hypothesis_id=hid, name=name, semantic_type=hdef.semantic_type, status=status, track_id=subject,
            t_start_s=t0, t_end_s=t1, supporting_evidence_ids=support_ids,
            conflicting_evidence_ids=[e.id for e in conflicts], independent_provenance_roots=roots_used,
            score=round(score, 4) if score_kind != "none" else None, score_kind=score_kind, scorer=self.scorer,
            calibration_status="uncalibrated", calibrated_confidence=None,
            reason_codes=list(dict.fromkeys(reasons)), explanation="",
        )

    @staticmethod
    def _deduplicate(hyps: list[Hypothesis]) -> list[Hypothesis]:
        # A subject-less candidate is redundant when the same evidence already
        # formed a track-level candidate.
        seen = {(h.name, frozenset(h.supporting_evidence_ids)) for h in hyps if h.track_id}
        return [h for h in hyps if h.track_id or (h.name, frozenset(h.supporting_evidence_ids)) not in seen]


def _combinations(items: list[int], k: int):
    from itertools import combinations

    return combinations(items, k)
