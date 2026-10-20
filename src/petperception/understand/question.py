"""Reading the user's question and scene description: reported behaviours and context words (transparent keyword matching).

Matching is deliberately simple and inspectable: the text is lower-cased,
pronouns are normalized ("her tail" -> "its tail", "herself" -> "itself",
"own" dropped), words are reduced to a crude stem ("chases", "chasing" ->
"chas"), and a behaviour is reported when one of its aliases, normalized the
same way, occurs as a word sequence. No language model is involved, so what
the parser misses is easy to see (benchmark E31 measures it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import vocab as V

AGE_WORDS = {"kitten": "young", "puppy": "young", "young": "young", "months old": "young", "month-old": "young", "senior": "senior",
             "old cat": "senior", "old dog": "senior", "elderly": "senior", "adult": "adult", "years old": "adult", "year-old": "adult"}
REPEAT_WORDS = ("keeps", "keep", "again and again", "repeatedly", "every day", "all the time", "constantly", "won't stop", "can't stop", "over and over",
                "always", "for hours", "times a day", "for minutes")
_PRONOUN = {"his": "its", "her": "its", "their": "its", "himself": "itself", "herself": "itself", "themselves": "itself"}
_SUFFIXES = ("ing", "es", "ed", "s", "e")


def _stem(w: str) -> str:
    for s in _SUFFIXES:
        if len(w) > len(s) + 2 and w.endswith(s):
            return w[: -len(s)]
    return w


def normalize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", text.lower().replace("-", " "))
    out = []
    for w in words:
        if w == "own":
            continue
        out.append(_stem(_PRONOUN.get(w, w)))
    return out


def _contains(seq: list[str], sub: list[str]) -> bool:
    n = len(sub)
    return n > 0 and any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))


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
    words = normalize(text)
    reported = []
    for b in V.BEHAVIORS.values():
        if species in b.species and any(_contains(words, normalize(a)) for a in b.aliases):
            reported.append(b.name)
    age = next((v for k, v in AGE_WORDS.items() if k in text), None)
    rep = any(w in text for w in REPEAT_WORDS)
    return Query(species, scene or "", question or "", reported, text, age, rep)


def cue_score(text: str, cues: dict[str, float]) -> tuple[float, list[str]]:
    """Context cue words (substring match on the raw lower-cased text, so a cue like "wrestl" matches its inflections)."""
    hit = [(w, v) for w, v in cues.items() if w in text]
    return sum(v for _, v in hit), [w for w, _ in hit]
