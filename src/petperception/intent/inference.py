"""Combined behaviour/intent inference over trained channel models, with explicit missing inputs and conflicts.

Channels and the trained models behind them (artifacts/intent/):

* situation (recorded context, cats and dogs): per-branch heads on audio,
  animal appearance/posture/keypoints/motion, environment (scene) and
  person interaction, combined by a trained late-fusion model that accepts
  any subset of branches (E28, Kinetics, synchronized audio and video);
* dog vocal activity / intent (dogs, audio): Barkopedia activities, of which
  "begging for food" and "seeking attention" are annotated intent, the rest
  behaviour or context (E27); optional environment context;
* dog bark context (dogs, audio): Yin & McCowan recording situations (E22);
* cat meow context (cats, audio): CatMeows recording situations (E2), below
  the evidence gate, so it can never support a hypothesis on its own.

How readings combine (taxonomy relations, intent/taxonomy.py):

* a hypothesis is *supported* only by a channel whose label is an intent
  annotation or a behaviour consistent with it, when that channel's top label
  passes the abstention threshold fitted on its training data;
* recorded contexts (situation, bark or meow context) *qualify* or
  *corroborate*; context alone yields at most a ``context_only`` entry, never
  a supported hypothesis (context is not a mental state);
* a *conflict* is reported, and no single interpretation given, when
  confident channels disagree: audio and video name different situations
  (E28 policy), the situation contradicts the detected species, or a
  supported hypothesis meets a confident reading of a soft-conflicting one
  (schemas/ontology.py);
* every missing or unusable input is listed with its reason; nothing is
  inferred from absence (silence never supports "relaxed").

There is no synchronized dataset with intent labels for all channels, so the
cross-channel intent layer itself is not validated; the channel models and
the situation fusion are (results/e27, results/e28). Scores are
temperature-scaled softmax values on the training distribution, not
probabilities of truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT
from ..schemas.ontology import HYPOTHESES
from . import heads as H
from . import taxonomy as T

ART = REPO_ROOT / "artifacts" / "intent"
INFERENCE_VERSION = "intent-inference-v1"
DOG_SOUND = ("Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)")
CAT_SOUND = ("Cat", "Purr", "Meow", "Hiss", "Caterwaul")
VOCAL_HEARD = 0.2
SITUATION_SPECIES = {"petting cat": "cat", "grooming cat": "cat", "grooming dog": "dog", "bathing dog": "dog", "training dog": "dog",
                     "walking the dog": "dog"}
DOG_ACTIVITY_HYP = {"begging for food": "food", "seeking attention": "attention", "playing with human": "play", "playing with other animals": "play",
                    "playing with toy": "play", "alerting to sounds": "alert", "rest": "relaxed", "taking shower": None}


@dataclass
class Reading:
    channel: str
    model: str
    label_space: str
    species: str
    scores: dict[str, float]
    top: str
    top_score: float
    threshold: float
    confident: bool
    inputs: list[str]
    evidence_gate_passed: bool = True
    note: str = ""


@dataclass
class HypothesisOut:
    name: str
    status: str  # supported | supported_corroborated | tentative | context_only | conflict
    support: float | None
    supporting: list[str] = field(default_factory=list)
    qualifying: list[str] = field(default_factory=list)
    contradicting: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class IntentResult:
    version: str
    species: str
    species_source: str
    inputs: dict[str, str]
    missing_inputs: dict[str, str]
    observations: list[str]
    readings: list[Reading]
    situation: dict | None
    qualifiers: list[str]
    hypotheses: list[HypothesisOut]
    conflicts: list[str]
    status: str  # interpretation | conflict | situation_only | insufficient_evidence
    summary: str
    caveats: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class IntentModels:
    """Trained channel models. ``variant`` "full" uses YAMNet embeddings and the research keypoint head;
    "phone_now" uses only what the phone bundle computes today (YAMNet class scores, no keypoints), so the
    embedding-only dog bark context head is unavailable there."""

    def __init__(self, art: Path = ART, variant: str = "full"):
        from ..utils import read_json

        self.variant = variant
        sd = art / "kinetics_situation_v1" / variant
        self.situation_heads = {p.stem.removeprefix("head_"): H.Head.from_json(p) for p in sorted(sd.glob("head_*.json"))}
        self.situation_fusion = H.LateFusion.from_json(sd / "late_fusion.json")
        self.situation_policy = read_json(sd / "policy.json")
        da = art / "dog_audio_activity_v1"
        a = "audio" if variant == "full" else "audio_scores"
        self.dog_audio = H.Head.from_json(da / f"head_{a}.json")
        self.dog_env = H.Head.from_json(da / "head_environment.json")
        self.dog_audio_env = H.LateFusion.from_json(da / f"late_fusion_{variant}.json")
        self.dog_bark_context = H.Head.from_json(art / "dog_bark_context_v1" / "head_audio.json") if variant == "full" else None
        self.cat_context = H.Head.from_json(art / "cat_audio_context_v1" / f"head_{a}.json")
        self.audio_branch = self.situation_policy["branches"][0]
        self.audio_input = a


def _reading(channel: str, model: str, space: str, species: str, head: H.Head, p: np.ndarray, inputs: list[str], gate: bool = True,
             note: str = "", threshold: float | None = None) -> Reading:
    k = int(np.argmax(p))
    thr = head.abstain_below if threshold is None else threshold
    return Reading(channel, model, space, species, {c: round(float(v), 4) for c, v in zip(head.classes, p)}, head.classes[k], round(float(p[k]), 4),
                   round(float(thr), 4), bool(p[k] >= thr), inputs, gate, note)


def infer(models: IntentModels, branches: dict[str, np.ndarray | None], cues: dict, yamnet_labels: list[str],
          environment: str | None = None) -> IntentResult:
    """branches/cues as produced by intent.kinetics.pool; environment: optional known location (E27 categories)."""
    missing = dict(cues.get("missing", {}))
    inputs = {b: ("present" if branches.get(b) is not None else "missing") for b in ("audio", "audio_scores", "animal", "animal_phone", "environment",
                                                                                  "interaction") if b in branches}
    obs: list[str] = []
    caveats = ["cross-channel intent combination not validated (no synchronized intent-labelled recordings)",
               "scores are temperature-scaled on training data, not probabilities of truth"]
    # species
    ym = dict(zip(yamnet_labels, map(float, cues["yamnet_max"]))) if "yamnet_max" in cues else {}
    dog_v = max((ym.get(n, 0.0) for n in DOG_SOUND), default=0.0)
    cat_v = max((ym.get(n, 0.0) for n in CAT_SOUND), default=0.0)
    if cues.get("dog_fraction") is not None:
        species = "dog" if cues["dog_fraction"] >= 0.5 else "cat"
        src = f"detector ({species} in {max(cues['dog_fraction'], 1 - cues['dog_fraction']):.0%} of {cues.get('pet_frames', 0)} animal frames)"
    elif max(dog_v, cat_v) >= VOCAL_HEARD:
        species, src = ("dog", "sound") if dog_v >= cat_v else ("cat", "sound")
    else:
        species, src = "unknown", "no animal seen or heard"
    if dog_v >= VOCAL_HEARD:
        obs.append(f"dog vocalization heard (YAMNet max {dog_v:.2f})")
    if cat_v >= VOCAL_HEARD:
        obs.append(f"cat vocalization heard (YAMNet max {cat_v:.2f})")
    heard = sorted(((n, v) for n, v in ym.items() if v >= VOCAL_HEARD), key=lambda t: -t[1])[:6]
    if heard:
        obs.append("sounds: " + ", ".join(f"{n} {v:.2f}" for n, v in heard))
    if cues.get("pet_frames"):
        obs.append(f"animal visible in {cues['pet_frames']}/{cues.get('frames', 0)} sampled frames")
    if cues.get("person_frames"):
        obs.append(f"person visible in {cues['person_frames']}/{cues.get('frames', 0)} frames; best person-animal interaction score "
                   f"{cues.get('interaction_max', 0.0):.2f}")
    if cues.get("posture_mean"):
        pm = cues["posture_mean"]
        obs.append("posture scores (lying/sitting/standing) " + "/".join(f"{v:.2f}" for v in pm))
    readings: list[Reading] = []
    conflicts: list[str] = []
    qualifiers: list[str] = []
    # ---- situation (trained late fusion over present branches)
    situation = None
    sb = {b: branches.get(b) for b in models.situation_fusion.branches}
    present = [b for b, v in sb.items() if v is not None]
    if present:
        bp = {b: (models.situation_heads[b].proba(v)[0][None, :] if v is not None else None) for b, v in sb.items()}
        p = models.situation_fusion.proba(bp)[0]
        classes = models.situation_fusion.classes
        r = _reading("situation", "kinetics_situation_v1", "kinetics_situation", species, models.situation_heads[present[0]], p, present,
                     threshold=models.situation_fusion.abstain_below)
        r.scores = {c: round(float(v), 4) for c, v in zip(classes, p)}
        r.top = classes[int(np.argmax(p))]
        readings.append(r)
        ab = models.audio_branch
        vis = [b for b in present if b != ab]
        sit_conflict = None
        if ab in present and vis:
            pa = bp[ab][0]
            pv = models.situation_fusion.proba({**dict.fromkeys(sb), **{b: bp[b] for b in vis}})[0]
            a_top, v_top = int(np.argmax(pa)), int(np.argmax(pv))
            if a_top != v_top and pa[a_top] >= models.situation_heads[ab].abstain_below and pv[v_top] >= models.situation_policy["visual_only_abstain_below"]:
                sit_conflict = f"audio suggests '{classes[a_top]}' ({pa[a_top]:.2f}) but video suggests '{classes[v_top]}' ({pv[v_top]:.2f})"
                conflicts.append("situation: " + sit_conflict)
        if species in ("cat", "dog") and SITUATION_SPECIES[r.top] != species and r.confident:
            conflicts.append(f"situation '{r.top}' implies a {SITUATION_SPECIES[r.top]} but the {src.split(' ')[0]} indicates a {species}")
        status = "conflict" if sit_conflict else ("interpretation" if r.confident else "uncertain")
        situation = {"label": r.top, "score": r.top_score, "status": status, "branches_used": present,
                     "qualifier": T.label("kinetics_pets", r.top).qualifier if status == "interpretation" else None}
        if situation["qualifier"]:
            qualifiers.append(situation["qualifier"])
    else:
        missing["situation"] = "no audio or video branch available"
    # ---- dog audio channels
    ai = getattr(models, "audio_input", "audio")
    if species in ("dog", "unknown") and branches.get(ai) is not None and dog_v >= VOCAL_HEARD:
        pa = models.dog_audio.proba(branches[ai])[0]
        if environment is not None:
            from ..eval.e27_dog_audio_intent import env_onehot

            pe = models.dog_env.proba(env_onehot([environment]))[0]
            pf = models.dog_audio_env.proba({models.dog_audio_env.branches[0]: pa[None, :], "environment": pe[None, :]})[0]
            readings.append(_reading("dog_vocal_activity", "dog_audio_activity_v1+environment", "barkopedia_activity", "dog", models.dog_audio, pf,
                                     ["audio", "environment(context)"], threshold=models.dog_audio_env.abstain_below,
                                     note=f"environment given as context: {environment}"))
        else:
            readings.append(_reading("dog_vocal_activity", "dog_audio_activity_v1", "barkopedia_activity", "dog", models.dog_audio, pa, ["audio"]))
        if models.dog_bark_context is not None and branches.get("audio") is not None:
            pb = models.dog_bark_context.proba(branches["audio"])[0]
            readings.append(_reading("dog_bark_context", "dog_bark_context_v1", "yin_bark_context", "dog", models.dog_bark_context, pb, ["audio"]))
        else:
            missing["dog_bark_context"] = "model needs the YAMNet embedding (not in this variant)"
    elif species in ("dog", "unknown"):
        missing["dog_vocal_activity"] = "audio missing" if branches.get(ai) is None else "no dog vocalization heard"
    if species in ("cat", "unknown") and branches.get(ai) is not None and cat_v >= VOCAL_HEARD:
        pc = models.cat_context.proba(branches[ai])[0]
        readings.append(_reading("cat_meow_context", "cat_audio_context_v1", "catmeows_context", "cat", models.cat_context, pc, ["audio"], gate=False,
                                 note="below the evidence gate (E2 leave-one-cat-out macro-F1 0.46): context hint only"))
    elif species in ("cat", "unknown"):
        missing["cat_meow_context"] = "audio missing" if branches.get(ai) is None else "no cat vocalization heard"
    # ---- hypotheses
    hyps = _hypotheses(readings, qualifiers, conflicts)
    if any(h.status == "conflict" for h in hyps) or (situation and situation["status"] == "conflict"):
        status = "conflict"
    elif any(h.status.startswith("supported") for h in hyps):
        status = "interpretation"
    elif situation and situation["status"] == "interpretation":
        status = "situation_only"
    else:
        status = "insufficient_evidence"
    if species == "unknown":
        caveats.append("species unknown: neither seen nor heard")
    return IntentResult(INFERENCE_VERSION, species, src, inputs, missing, obs, readings, situation, sorted(set(qualifiers)), hyps, conflicts, status,
                        _summary(status, hyps, situation, conflicts, missing), caveats)


def _hypotheses(readings: list[Reading], qualifiers: list[str], conflicts: list[str]) -> list[HypothesisOut]:
    out: dict[str, HypothesisOut] = {}
    direct = [r for r in readings if r.channel == "dog_vocal_activity"]
    context = [r for r in readings if r.channel in ("situation", "dog_bark_context", "cat_meow_context")]
    for r in direct:
        mass: dict[str, float] = {}
        for lab, v in r.scores.items():
            h = DOG_ACTIVITY_HYP[lab]
            if h:
                mass[h] = mass.get(h, 0.0) + v
        top_h = DOG_ACTIVITY_HYP[r.top]
        if r.top == "taking shower" and r.confident:
            qualifiers.append("bathing")
        for h, m in sorted(mass.items(), key=lambda t: -t[1]):
            rel = "annotated intent" if r.top in ("begging for food", "seeking attention") and h == top_h else "behaviour consistent with"
            if h == top_h and r.confident:
                st = "supported"
            elif h == top_h or m >= 0.3:
                st = "tentative"
            else:
                continue
            out[h] = HypothesisOut(h, st, round(m, 4), [f"{r.channel}: '{r.top}' {r.top_score:.2f} ({rel} '{h}')" if h == top_h else
                                                        f"{r.channel}: {m:.2f} of the score mass maps to '{h}'"])
    for r in context:
        if not r.evidence_gate_passed and not r.confident:
            continue
        ds = {"situation": "kinetics_pets", "dog_bark_context": "beans_dog_barks", "cat_meow_context": "catmeows"}[r.channel]
        if not r.confident:
            continue
        lab = T.label(ds, r.top)
        for h, rel in lab.hypotheses.items():
            if rel != "context_for":
                continue
            note = f"{r.channel}: context '{r.top}' {r.top_score:.2f} makes '{h}' plausible"
            if h in out:
                out[h].qualifying.append(note)
                if out[h].status == "supported" and r.channel == "situation":
                    out[h].status = "supported_corroborated"
            else:
                out[h] = HypothesisOut(h, "context_only", None, [], [note], [], "context alone is not evidence of what the animal wants")
    q = set(qualifiers)
    for h in out.values():
        if q & {"bathing", "handling"} and h.name in ("alert", "attention", "play") and h.status.startswith("supported"):
            h.qualifying.append("animal is being handled/bathed: vocal signals may be a reaction to the handling")
        for other in out.values():
            if other.name in HYPOTHESES[h.name].soft_conflicts and h.status.startswith("supported") and other.status.startswith("supported"):
                h.contradicting.append(f"'{other.name}' is also supported")
                h.status = "conflict"
                conflicts.append(f"hypotheses '{h.name}' and '{other.name}' both supported")
        if not h.explanation:
            h.explanation = {"supported": "the channel's top label passes its trained abstention threshold",
                             "supported_corroborated": "supported, and the synchronized situation reading is consistent",
                             "tentative": "evidence present but below the channel's abstention threshold",
                             "conflict": "confident readings disagree; no single interpretation"}.get(h.status, "")
    order = {"conflict": 0, "supported_corroborated": 1, "supported": 2, "tentative": 3, "context_only": 4}
    return sorted(out.values(), key=lambda h: (order[h.status], -(h.support or 0.0)))


def _summary(status: str, hyps: list[HypothesisOut], situation: dict | None, conflicts: list[str], missing: dict[str, str]) -> str:
    parts = []
    if status == "conflict":
        parts.append("Conflicting evidence: " + "; ".join(conflicts) + ". No single interpretation.")
    sup = [h for h in hyps if h.status.startswith("supported")]
    if sup:
        parts.append("Possible: " + ", ".join(f"{h.name} ({h.support:.2f})" for h in sup) + ".")
    tent = [h for h in hyps if h.status == "tentative"]
    if tent:
        parts.append("Weaker: " + ", ".join(h.name for h in tent) + ".")
    if situation:
        parts.append(f"Situation: {situation['label']} ({situation['score']:.2f}, {situation['status']}).")
    if missing:
        parts.append("Missing: " + "; ".join(f"{k} ({v})" for k, v in sorted(missing.items())) + ".")
    if status == "insufficient_evidence" and not sup:
        parts.append("Insufficient evidence for an interpretation.")
    return " ".join(parts)
