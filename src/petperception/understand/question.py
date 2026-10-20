"""Reading the user's question and scene description: reported behaviours and context words (transparent keyword matching)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import vocab as V

AGE_WORDS = {"kitten": "young", "puppy": "young", "young": "young", "months old": "young", "senior": "senior", "old cat": "senior",
             "old dog": "senior", "elderly": "senior", "adult": "adult", "years old": "adult"}
REPEAT_WORDS = ("keeps", "keep", "again and again", "repeatedly", "every day", "all the time", "constantly", "won't stop", "can't stop", "over and over",
                "always", "for hours")


@dataclass
class Query:
    species: str
    scene: str
    question: str
    reported: list[str] = field(default_factory=list)
    text: str = ""
    age: str | None = None
    repetition_reported: bool = False
    cues_found: list[str] = field(default_factory=list)


def parse(species: str, scene: str, question: str) -> Query:
    text = " ".join([scene or "", question or ""]).lower()
    text = re.sub(r"\s+", " ", text)
    reported = []
    for b in V.BEHAVIORS.values():
        if species in b.species and any(a in text for a in b.aliases):
            reported.append(b.name)
    # "turning in circles while biting its own tail" is tail chasing with tail-directed biting
    age = next((v for k, v in AGE_WORDS.items() if k in text), None)
    rep = any(w in text for w in REPEAT_WORDS)
    return Query(species, scene or "", question or "", reported, text, age, rep)


def cue_score(text: str, cues: dict[str, float]) -> tuple[float, list[str]]:
    hit = [(w, v) for w, v in cues.items() if w in text]
    return sum(v for _, v in hit), [w for w, _ in hit]
