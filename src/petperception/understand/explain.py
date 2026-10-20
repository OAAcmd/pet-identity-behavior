"""Compose an answer: what the recording shows (with times), what the user reports, candidate explanations by kind, uncertainty,
observations that would tell the candidates apart, urgent signs, and sources.

The answer never states a cause as established from the recording. It keeps
requests and intentions apart from play, stress, compulsive and medical
explanations, and says plainly when a behaviour implies no request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import knowledge as K
from . import vocab as V
from .question import Query, cue_score
from .recognize import BehaviorReading, RecognizerConfig

ANSWER_VERSION = "behavior-answer-v1"
MAX_SEEN_EXPLAINED = 2  # behaviours seen but not asked about that also get explanations


@dataclass
class Answer:
    version: str
    species: str
    scene: str
    question: str
    behaviors_seen: list[dict]
    temporal_pattern: dict
    user_reported: list[dict]
    explained_behaviors: list[dict]
    uncertainty: list[str]
    observations_needed: list[str]
    urgent_signs: list[str]
    sources: list[dict]
    summary: str
    knowledge_version: str = K.KNOWLEDGE_VERSION
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt_bouts(r: BehaviorReading, limit: int = 4) -> str:
    if not r.bouts:
        return ""
    parts = [f"{b.t0:.1f}-{b.t1:.1f} s" for b in r.bouts[:limit]]
    more = f" and {len(r.bouts) - limit} more" if len(r.bouts) > limit else ""
    return ", ".join(parts) + more


def compose(q: Query, readings: list[BehaviorReading], pattern: dict, config: RecognizerConfig | None = None,
            quality: list[str] | None = None) -> Answer:
    cfg = config or RecognizerConfig()
    seen = [r for r in readings if r.score >= cfg.threshold(r.behavior) and (r.bouts or r.behavior == "vocalizing" or r.evidence.get("clip_level"))]
    seen_names = {r.behavior for r in seen}
    by_name = {r.behavior: r for r in readings}
    behaviors_seen = [{"behavior": r.behavior, "description": V.BEHAVIORS[r.behavior].description, "score": r.score,
                       "threshold": cfg.threshold(r.behavior), "bouts_s": [[round(b.t0, 2), round(b.t1, 2)] for b in r.bouts],
                       "evidence": r.evidence} for r in seen]
    reported = []
    for b in q.reported:
        r = by_name.get(b)
        if b in seen_names:
            status = "also seen in the recording"
        elif r is None:
            status = "cannot be checked in this recording (" + ("no sound" if V.BEHAVIORS[b].channel == "audio" else "the animal was not located") + ")"
        else:
            status = "not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)"
        reported.append({"behavior": b, "status": status, "recording_score": None if r is None else r.score})
    # which behaviours to explain: those the user asks about first, then those seen
    targets = list(dict.fromkeys(q.reported + [r.behavior for r in seen if r.behavior not in q.reported][:MAX_SEEN_EXPLAINED]))
    explained, sources_used, observations, urgent = [], {}, [], []
    for b in targets:
        e = K.entry(b, q.species)
        if e is None:
            continue
        cands = []
        for c in e.candidates:
            s, words = cue_score(q.text, c.context_cues)
            cands.append((s, c, words))
        cands.sort(key=lambda x: -x[0])
        groups: dict[str, list[dict]] = {}
        for s, c, words in cands:
            groups.setdefault(c.kind, []).append({"id": c.id, "explanation": c.text, "context_match": words, "context_score": round(s, 2),
                                                  "sources": list(c.sources)})
            for sid in c.sources:
                sources_used[sid] = K.SOURCES[sid]
        request_implied = any(c.kind == "request" for _, c, _ in cands)
        explained.append({"behavior": b, "basis": ("recording and user report" if b in seen_names and b in q.reported else
                                                   "recording" if b in seen_names else "user report only"),
                          "summary": e.summary, "temporal_notes": e.temporal_notes,
                          "candidates_by_kind": {k: groups[k] for k in K.KINDS if k in groups},
                          "kind_order_by_context": [c.kind for _, c, _ in cands],
                          "request_or_intention_possible": request_implied,
                          "no_request_note": None if request_implied else "No request or intention is implied by this behaviour itself."})
        observations += [o for o in e.distinguish if o not in observations]
        urgent += [u for u in e.urgent if u not in urgent]
    unc = ["The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, "
           "ordered by how well your description matches them."]
    if pattern.get("available") is False:
        unc.append("The animal could not be located reliably, so its movement over time was not measured.")
    elif pattern.get("animal_found_fraction", 1.0) < 0.5:
        unc.append(f"The animal was located in only {pattern['animal_found_fraction']:.0%} of sampled frames.")
    vote = pattern.get("detector_dog_vote")
    if vote is not None and ((q.species == "cat" and vote > 0.5) or (q.species == "dog" and vote < 0.5)):
        unc.append(f"The detector's species vote disagrees with the species you gave ({q.species}); your answer is used, and the detector only locates the animal.")
    unc.append("Behaviour recognition uses a general vision-language model (" + ("zero-shot" if cfg.method == "zero_shot" else
               "zero-shot, plus a head trained on a few hundred public clips for " + ", ".join(cfg.head_behaviors)) +
               "); scores are relative, not probabilities, and its measured accuracy is in results/e31.")
    if quality:
        unc += quality
    summary = _summary(q, seen, explained, pattern, by_name)
    return Answer(ANSWER_VERSION, q.species, q.scene, q.question, behaviors_seen, pattern, reported, explained, unc, observations[:8], urgent,
                  [{"id": k, **v} for k, v in sources_used.items()], summary)


def _summary(q: Query, seen: list[BehaviorReading], explained: list[dict], pattern: dict, by_name: dict) -> str:
    parts = []
    if seen:
        top = seen[:3]
        parts.append("In the recording: " + "; ".join(f"{V.BEHAVIORS[r.behavior].description} ({_fmt_bouts(r) or 'throughout'})" for r in top) + ".")
    else:
        parts.append("No specific behaviour was recognized with confidence in the recording.")
    if pattern.get("available"):
        bits = []
        if pattern.get("stays_in_place"):
            bits.append("the animal stays roughly in place while moving")
        if abs(pattern.get("turned_rad", 0.0)) >= 6.0:
            bits.append(f"it turns about {abs(pattern['turned_rad']) / 6.283:.1f} times in one direction")
        if pattern.get("repetition_strength", 0) >= 0.15 and pattern.get("repetition_period_s"):
            bits.append(f"its appearance repeats about every {pattern['repetition_period_s']:.1f} s")
        if bits:
            parts.append("Movement: " + ", ".join(bits) + ".")
    for ex in explained[:2]:
        kinds = [k for k in ex["kind_order_by_context"]]
        kinds = list(dict.fromkeys(kinds))
        parts.append(f"{V.BEHAVIORS[ex['behavior']].description.capitalize()} ({ex['basis']}) has several possible explanations: "
                     + "; ".join(K.KIND_TEXT[k] for k in kinds) + "." + ("" if ex["request_or_intention_possible"] else " No request is implied by it."))
    parts.append("The cause is not established from the recording; the questions listed would help tell the candidates apart.")
    return " ".join(parts)


def render_markdown(ans: dict, title: str = "") -> str:
    """A readable version of an answer (``Answer.to_dict()``)."""
    lines = ([f"### {title}", ""] if title else []) + [f"**Species (given):** {ans['species']}  ", f"**Scene:** {ans['scene'] or '-'}  ",
                                                         f"**Question:** {ans['question'] or '-'}", "", f"**Summary:** {ans['summary']}", "",
                                                         "**Seen in the recording:**"]
    lines += [f"- {b['behavior']} (score {b['score']:.2f}, threshold {b['threshold']:.2f}; bouts {b['bouts_s'][:4] or 'clip level'})"
              for b in ans["behaviors_seen"]] or ["- nothing above threshold"]
    lines += ["", "**What you reported:**"] + ([f"- {u['behavior']}: {u['status']}" for u in ans["user_reported"]] or ["- no behaviour recognized in the question"])
    for e in ans["explained_behaviors"]:
        lines += ["", f"**Possible explanations for {e['behavior']}** ({e['basis']}; kinds in order of fit to your description: "
                  f"{', '.join(dict.fromkeys(e['kind_order_by_context']))}):"]
        for kind, g in e["candidates_by_kind"].items():
            for c in g:
                lines.append(f"- [{kind}] {c['explanation']} (sources: {', '.join(c['sources']) or 'none'}; matched in your description: "
                             f"{', '.join(c['context_match']) or '-'})")
        lines.append(f"- request or intention possible: {'yes' if e['request_or_intention_possible'] else 'no'}"
                     + (f" - {e['no_request_note']}" if e["no_request_note"] else ""))
    lines += ["", "**Would help tell them apart:**"] + [f"- {o}" for o in ans["observations_needed"]]
    if ans["urgent_signs"]:
        lines += ["", "**See a veterinarian promptly if:**"] + [f"- {u}" for u in ans["urgent_signs"]]
    lines += ["", "**Uncertainty:**"] + [f"- {u}" for u in ans["uncertainty"]]
    lines += ["", "**Sources:**"] + [f"- [{x['id']}] {x['citation']} ({x['tier']}) {x['url']}" for x in ans["sources"]]
    return "\n".join(lines) + "\n"
