"""Label taxonomy: observable behaviours, recorded contexts and intent hypotheses are kept apart.

Three label layers occur in the data this project can use:

* ``observable_behavior``: what the animal visibly or audibly does (rests,
  plays, eats, alerts to a sound, vocalizes, is touched by a person, its
  posture, ear or tail configuration). It can be checked against the
  recording.
* ``recorded_context``: the situation the recording was made in, set by the
  protocol or seen in the scene (brushing session, isolation in an unfamiliar
  room, waiting for food, a stranger at the door, being bathed, walked,
  trained; the location). A context is NOT a mental state: a cat recorded
  while waiting for food is not thereby shown to want food at that moment.
* ``annotated_intent``: an annotator's judgement of what the animal is trying
  to obtain (Barkopedia "begging for food", "seeking attention"; a
  vision-language model proposed them from the video and people verified
  them). Still an inference by someone else, not ground truth.

The product's hypotheses (``schemas/ontology.py``: food, outside, attention,
play, alert, tense, warning, relaxed) are what the animal *may* want or be
doing. Each dataset label is linked to hypotheses by an explicit relation:

* ``annotated``: the label is an intent annotation for the hypothesis;
* ``consistent_with``: the behaviour is consistent with the hypothesis but
  does not establish it;
* ``context_for``: the situation makes the hypothesis plausible; it may
  qualify or strengthen evidence from the animal, never create a hypothesis
  on its own.

Context qualifiers (``QUALIFIERS``) are situations that change how evidence
reads (e.g. vocalizing while being bathed reads as a reaction to handling,
not as play), and are reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas.ontology import HYPOTHESES

BEHAVIOR, CONTEXT, INTENT = "observable_behavior", "recorded_context", "annotated_intent"
LAYERS = (BEHAVIOR, CONTEXT, INTENT)
RELATIONS = ("annotated", "consistent_with", "context_for")
TAXONOMY_VERSION = "intent-taxonomy-v1"

QUALIFIERS = {
    "handling": "a person is touching, brushing, grooming or bathing the animal",
    "bathing": "the animal is being bathed or showered",
    "training": "a training session with a person",
    "walk": "the animal is walked by a person (often on a leash, outdoors)",
    "isolation": "the animal is alone, separated from its owner",
    "disturbance": "an external disturbance such as a stranger at the door",
    "food_routine": "food is expected (a feeding routine is under way)",
    "location": "where the animal is (indoor, near a door or window, bathroom, vehicle, outdoors)",
}


@dataclass(frozen=True)
class LabelDef:
    raw: str
    layer: str
    concept: str
    hypotheses: dict[str, str] = field(default_factory=dict)  # hypothesis -> relation
    qualifier: str | None = None
    note: str = ""


@dataclass(frozen=True)
class DatasetDef:
    name: str
    species: tuple[str, ...]
    modalities: tuple[str, ...]
    synchronized_av: bool
    unit: str
    grouping: str
    label_source: str
    terms: str
    on_disk: str
    experiments: tuple[str, ...]
    labels: tuple[LabelDef, ...]


def _L(raw, layer, concept, hyps=None, qualifier=None, note=""):
    return LabelDef(raw, layer, concept, dict(hyps or {}), qualifier, note)


DATASETS: dict[str, DatasetDef] = {
    "catmeows": DatasetDef(
        "CatMeows (Ludovico et al. 2020, Zenodo 4008297)", ("cat",), ("audio",), False, "meow clip (8 kHz)", "cat id (21) and owner (12)",
        "recording protocol: the owner placed the cat in the situation", "CC BY 4.0", "var/data/catmeows (440 clips)", ("E2", "E27"),
        (_L("brushing", CONTEXT, "being_brushed", {}, "handling", "owner brushing the cat at home"),
         _L("isolation", CONTEXT, "isolation_unfamiliar_environment", {"attention": "context_for"}, "isolation",
            "cat alone in an unfamiliar room; contact-seeking is plausible, not shown"),
         _L("waiting_for_food", CONTEXT, "waiting_for_food", {"food": "context_for"}, "food_routine",
            "owner started the feeding routine; the label is the situation, not the cat's state"))),
    "beans_dog_barks": DatasetDef(
        "Yin & McCowan (2004) barks as packaged in BEANS", ("dog",), ("audio",), False, "bark clip", "dog id (10), breed (6)",
        "recording protocol (CSV terms; mapping to the paper's situations unconfirmed)", "no licence stated; research only",
        "var/data/beans_dog_barks/features_v1.npz (features only)", ("E22", "E27"),
        (_L("aggression", CONTEXT, "disturbance", {"alert": "context_for", "warning": "context_for"}, "disturbance",
            "paper situation: stranger ringing the doorbell (mapping unconfirmed)"),
         _L("contact", CONTEXT, "isolation_or_contact", {"attention": "context_for"}, "isolation",
            "paper situation: isolation from the owner (mapping unconfirmed)"),
         _L("play", CONTEXT, "play_session", {"play": "context_for"}, None, "owner playing with the dog"))),
    "barkopedia_act_env": DatasetDef(
        "ArlingtonCL2/Barkopedia_Dog_Act_Env @ 60a721a7", ("dog",), ("audio",), False, "bark clip from a video", "none published (pseudo-groups)",
        "a vision-language model labelled the source video; people verified", "MIT", "var/data/barkopedia_act_env (12,480 + 499 clips)",
        ("E26", "E27"),
        (_L("rest", BEHAVIOR, "resting", {"relaxed": "consistent_with"}),
         _L("alerting to sounds", BEHAVIOR, "alerting_to_sound", {"alert": "consistent_with"}),
         _L("seeking attention", INTENT, "seeking_attention", {"attention": "annotated"}),
         _L("begging for food", INTENT, "begging_for_food", {"food": "annotated"}),
         _L("playing with human", BEHAVIOR, "playing_with_person", {"play": "consistent_with"}, None, "partner: person"),
         _L("playing with other animals", BEHAVIOR, "playing_with_animal", {"play": "consistent_with"}, None, "partner: another animal"),
         _L("playing with toy", BEHAVIOR, "playing_with_toy", {"play": "consistent_with"}, None, "partner: toy"),
         _L("taking shower", CONTEXT, "being_bathed", {}, "bathing", "situation imposed by a person"))),
    "barkopedia_environment": DatasetDef(
        "Barkopedia_Dog_Act_Env environment column", ("dog",), ("audio",), False, "bark clip", "as above",
        "vision-language model + human verification", "MIT", "same files", ("E27",),
        tuple(_L(r, CONTEXT, "location:" + r.replace(" ", "_").replace("(", "").replace(")", ""), {}, "location") for r in
              ("indoor (general)", "near door", "near window", "near other animals", "bathroom", "vehicle interior", "on grass", "outdoor"))),
    "kinetics_pets": DatasetDef(
        "Kinetics-700-2020 pet classes (CVDF mirror)", ("cat", "dog"), ("audio", "video"), True, "10 s YouTube clip", "YouTube id",
        "annotators labelled the human action", "annotations CC BY 4.0; clips belong to their uploaders (research only)",
        "var/data/kinetics_pets (294 validation + 600 train clips)", ("E18", "E19", "E28"),
        (_L("petting cat", CONTEXT, "being_petted", {"attention": "context_for", "relaxed": "context_for"}, "handling", "person touches the cat"),
         _L("grooming cat", CONTEXT, "being_groomed", {}, "handling", "person brushes or grooms the cat"),
         _L("grooming dog", CONTEXT, "being_groomed", {}, "handling", "person brushes, trims or grooms the dog"),
         _L("bathing dog", CONTEXT, "being_bathed", {}, "bathing", "person bathes the dog"),
         _L("training dog", CONTEXT, "training_session", {"food": "context_for", "attention": "context_for"}, "training",
            "rewards and attention are usually part of training; not shown per clip"),
         _L("walking the dog", CONTEXT, "being_walked", {"outside": "context_for"}, "walk", "the dog is already outside"))),
    "mammalnet_dogs": DatasetDef(
        "MammalNet (dog videos)", ("dog",), ("video",), False, "behaviour segment", "YouTube id", "annotators labelled the behaviour",
        "research use (MammalNet terms)", "streamed; features cached by E11/E17", ("E11", "E17"),
        (_L("sleeps", BEHAVIOR, "sleeping", {"relaxed": "consistent_with"}),
         _L("eats_food", BEHAVIOR, "eating", {"food": "consistent_with"}, None, "consumption, i.e. the need is being met"),
         _L("drinks_water", BEHAVIOR, "drinking", {}),
         _L("fights", BEHAVIOR, "fighting", {"tense": "consistent_with", "warning": "consistent_with"}),
         _L("eliminates", BEHAVIOR, "eliminating", {"outside": "consistent_with"}, None, "only weakly related; often outdoors already"))),
    "openimages_relations": DatasetDef(
        "Open Images V7 relationships / Localized Narratives", ("cat", "dog"), ("image",), False, "image box pair", "image id",
        "crowd annotation; captions (weak)", "CC BY 4.0 (annotations); images per Flickr licence", "var/data/openimages*", ("E10", "E13", "E15"),
        (_L("holds / interacts with (person-pet)", BEHAVIOR, "touched_by_person", {"attention": "context_for"}, "handling"),
         _L("on (furniture)", BEHAVIOR, "on_furniture", {"relaxed": "context_for"}, "location"),
         _L("lying / sitting / standing (caption)", BEHAVIOR, "posture", {"relaxed": "consistent_with"}, None, "lying only"))),
    "fsd50k_vocal_types": DatasetDef(
        "FSD50K animal vocalization classes", ("cat", "dog"), ("audio",), False, "freesound clip", "uploader",
        "crowd annotation of sound class", "CC licences per clip (BY / BY-NC / CC0)", "var/data/fsd50k (subset)", ("E21", "E21b"),
        (_L("bark / bow-wow / yip", BEHAVIOR, "barking", {"alert": "consistent_with"}),
         _L("growling", BEHAVIOR, "growling", {"warning": "consistent_with", "tense": "consistent_with"}),
         _L("whimper (dog)", BEHAVIOR, "whimpering", {"attention": "consistent_with"}),
         _L("meow", BEHAVIOR, "meowing", {"attention": "consistent_with", "food": "consistent_with"}),
         _L("purr", BEHAVIOR, "purring", {"relaxed": "consistent_with"}, None, "never rules out discomfort"),
         _L("hiss", BEHAVIOR, "hissing", {"warning": "consistent_with", "tense": "consistent_with"}))),
    "pose_keypoints": DatasetDef(
        "AwA-Pose (cats, dogs) and Ultralytics dog-pose", ("cat", "dog"), ("image",), False, "keypoint set", "image",
        "manual keypoints", "research only", "var/data/awa_pose, var/data/dog_pose", ("E9", "E24"),
        (_L("ear base / tip keypoints", BEHAVIOR, "ear_configuration", {}, None, "geometry only; no ear-position-to-intent labels exist"),
         _L("tail start / end keypoints", BEHAVIOR, "tail_configuration", {}, None, "geometry only; no tail-position-to-intent labels exist"))),
}


def label(dataset: str, raw: str) -> LabelDef:
    for lab in DATASETS[dataset].labels:
        if lab.raw == raw:
            return lab
    raise KeyError(f"{dataset}: unknown label {raw!r}")


def evidence_map(dataset: str) -> dict[str, dict[str, str]]:
    """raw label -> {hypothesis: relation}."""
    return {lab.raw: dict(lab.hypotheses) for lab in DATASETS[dataset].labels}


def coverage() -> dict[str, dict[str, list[dict]]]:
    """hypothesis -> species -> sources that link a label to it (layer, relation, modality)."""
    out: dict[str, dict[str, list[dict]]] = {h: {"cat": [], "dog": []} for h in HYPOTHESES}
    for key, ds in DATASETS.items():
        for lab in ds.labels:
            for h, rel in lab.hypotheses.items():
                for sp in ds.species:
                    out[h][sp].append({"dataset": key, "label": lab.raw, "layer": lab.layer, "relation": rel, "modalities": list(ds.modalities),
                                       "synchronized_av": ds.synchronized_av})
    return out


MISSING = [
    {"need": "synchronized audio-video household recordings with an intent annotation per bout",
     "why": "no dataset pairs what a cat or dog does in all channels with what it wanted; the combined intent layer cannot be fitted or validated",
     "exact_examples": "per species >= 300 bouts from >= 40 animals in >= 30 households; per bout: onset/offset, animal id, household id, "
                       "owner-reported situation (feeding time, at the door, after a walk, alone), independent annotator intent label "
                       "(food, outside, attention, play, alert, tense/warning, relaxed, unknown) with confidence, second annotator on >= 20 %"},
    {"need": "cat intent annotations in any modality",
     "why": "cat labels available here are recording contexts only (CatMeows) or vocal types (FSD50K)",
     "exact_examples": ">= 150 cat vocal or visual bouts per hypothesis (food, attention, outside, play) from >= 30 cats, with cat ids"},
    {"need": "'outside' (wish to go out) examples",
     "why": "no dataset labels a door-directed request; only owner-configured exit-door zones exist",
     "exact_examples": "door-side bouts (scratching, vocalizing, waiting) with the owner's outcome (let out or not), >= 100 per species"},
    {"need": "facial action / ear / tail annotations linked to situations",
     "why": "keypoint models give geometry, but no data link ear or tail configuration to a context or intent; CatFLW/DogFLW (landmarks) "
            "need a Kaggle account and still carry no intent labels",
     "exact_examples": "frames with ear/tail/face landmarks plus the situation label (e.g. CatFACS/DogFACS-coded clips of the same bouts)"},
    {"need": "animal and household identifiers for Barkopedia clips",
     "why": "without ids, held-out evaluation can only use embedding pseudo-groups; household leakage cannot be excluded",
     "exact_examples": "per clip: source video id, dog id, uploader/household id"},
]
