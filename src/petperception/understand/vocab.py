"""Behaviour vocabulary: observable actions, how they are prompted to the vision-language encoder, and their temporal form.

Every behaviour here is something visible or audible in a recording. What a
behaviour may mean (a request, play, discomfort, a compulsive pattern) lives
in ``knowledge.py``, never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Behavior:
    name: str
    description: str
    species: tuple[str, ...]
    prompts: tuple[str, ...]  # "{a}" is replaced by "a cat" / "a dog"
    temporal: str  # bout | repetitive | posture | event
    channel: str = "visual"  # visual | audio
    aliases: tuple[str, ...] = field(default_factory=tuple)  # words users use for it (question parsing)


BEHAVIORS: dict[str, Behavior] = {b.name: b for b in [
    Behavior("tail_chasing", "turning in tight circles after its own tail", ("cat", "dog"),
             ("{a} chasing its own tail", "{a} spinning in circles", "{a} turning around after its tail", "{a} biting at its tail while turning"),
             "repetitive", aliases=("chasing its tail", "chases its tail", "chasing her tail", "chasing his tail", "turning in circles", "spinning",
                                    "spins", "circling", "goes in circles", "running in circles", "turns in circles", "keeps turning")),
    Behavior("tail_directed", "licking, biting or chewing its own tail", ("cat", "dog"),
             ("{a} biting its own tail", "{a} licking its tail", "{a} chewing its tail"), "bout",
             aliases=("biting its tail", "biting its own tail", "bites its tail", "chewing its tail", "licking its tail", "attacking its tail",
                      "biting her tail", "biting his tail")),
    Behavior("self_grooming", "licking or cleaning its own fur, paws or face", ("cat", "dog"),
             ("{a} grooming itself", "{a} licking its fur", "{a} licking its paw", "{a} cleaning itself"), "repetitive",
             aliases=("grooming", "licking itself", "licks itself", "cleaning itself", "washing itself", "licking its fur", "overgrooming",
                      "licking its paws", "licking its paw")),
    Behavior("scratching_self", "scratching its own body with a hind foot", ("cat", "dog"),
             ("{a} scratching itself with its hind leg", "{a} scratching its ear"), "repetitive", aliases=("scratching itself", "scratches", "itching")),
    Behavior("kneading", "pushing the front paws alternately into a soft surface", ("cat",),
             ("{a} kneading a blanket with its paws", "{a} kneading"), "repetitive", aliases=("kneading", "making biscuits", "pawing the blanket")),
    Behavior("playing_object", "batting, pouncing on or chasing a toy or object", ("cat", "dog"),
             ("{a} playing with a toy", "{a} pouncing on a toy", "{a} batting at a string", "{a} chasing a ball"), "bout",
             aliases=("playing with a toy", "plays with", "pouncing", "batting", "fetch", "playing with a ball", "chasing a toy", "chasing a laser")),
    Behavior("social_play", "play fighting or wrestling with another animal", ("cat", "dog"),
             ("two {s}s play fighting", "{a} wrestling with another {s}", "{s}s playing together"), "bout",
             aliases=("play fighting", "wrestling", "playing with the other", "playing together", "rough play")),
    Behavior("threat_display", "hissing, growling, ears flat or arched back toward something", ("cat", "dog"),
             ("{a} hissing", "{a} growling with bared teeth", "an angry {s} with flattened ears", "{a} with an arched back hissing"), "event",
             aliases=("hissing", "hisses", "growling", "growls", "snarling", "spitting", "arched back")),
    Behavior("eating", "eating food", ("cat", "dog"), ("{a} eating food from a bowl", "{a} eating"), "bout", aliases=("eating", "eats", "chewing food")),
    Behavior("drinking", "drinking water", ("cat", "dog"), ("{a} drinking water", "{a} lapping water"), "bout", aliases=("drinking", "drinks", "lapping")),
    Behavior("resting", "lying down, resting or sleeping", ("cat", "dog"),
             ("{a} sleeping", "{a} lying down resting", "{a} curled up asleep"), "posture", aliases=("sleeping", "resting", "lying down", "napping")),
    Behavior("locomotion", "walking or running across the scene", ("cat", "dog"),
             ("{a} walking", "{a} running across a room"), "bout", aliases=("walking", "running around", "zoomies", "pacing")),
    Behavior("being_petted", "being touched or stroked by a person", ("cat", "dog"),
             ("a person petting {a}", "a hand stroking {a}"), "bout", aliases=("petting", "when i pet", "stroking", "when touched", "being touched")),
    Behavior("vocalizing", "meowing, barking, howling or whining", ("cat", "dog"), (), "event", channel="audio",
             aliases=("meowing", "meows", "barking", "barks", "howling", "whining", "crying", "yowling", "vocalizing")),
]}

# neutral states compete with the behaviours in the zero-shot softmax, so "nothing in particular" is a possible answer
NEUTRAL_PROMPTS = ("a photo of {a}", "{a} sitting still", "{a} standing", "{a} looking at the camera")
SPECIES_WORD = {"cat": ("a cat", "cat"), "dog": ("a dog", "dog")}


def prompts_for(species: str) -> tuple[list[str], list[str]]:
    """-> (prompt texts, owner label per prompt: behaviour name or "neutral")."""
    a, s = SPECIES_WORD[species]
    texts, owners = [], []
    for b in BEHAVIORS.values():
        if b.channel != "visual" or species not in b.species:
            continue
        for p in b.prompts:
            texts.append(p.format(a=a, s=s))
            owners.append(b.name)
    for p in NEUTRAL_PROMPTS:
        texts.append(p.format(a=a, s=s))
        owners.append("neutral")
    return texts, owners


def visual_behaviors(species: str) -> list[str]:
    return [b.name for b in BEHAVIORS.values() if b.channel == "visual" and species in b.species]
