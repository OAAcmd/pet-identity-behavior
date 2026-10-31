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

import numpy as np

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
    context_behaviors: list[str] = field(default_factory=list)


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


CONTEXT_MARKERS = ("after", "before", "until")  # "after eating she grooms": eating is context, grooming is the question


def _clauses(text: str) -> list[str]:
    parts = re.split(r"[.,;!?]| and | while | but | then | or ", text.lower())
    return [p.strip() for p in parts if len(p.strip().split()) >= 2]


_SUBJECT = {"she", "he", "it", "they", "my", "our", "the", "his", "her", "its", "i", "we"}


def split_context(text: str) -> tuple[str, list[str]]:
    """Separate phrases introduced by after / before / until (up to 4 words, stopping at a new subject) from the rest."""
    words = re.findall(r"[a-z0-9']+|[.,;!?]", text.lower())
    main, ctx, k = [], [], 0
    while k < len(words):
        w = words[k]
        if w in CONTEXT_MARKERS:
            phrase = []
            k += 1
            while k < len(words) and len(phrase) < 4 and words[k] not in _SUBJECT and words[k] not in ".,;!?":
                phrase.append(words[k])
                k += 1
            ctx.append(" ".join(phrase))
            continue
        main.append(w)
        k += 1
    return " ".join(main), ctx


def context_only(text: str, behavior: str, species: str) -> bool:
    """True when the behaviour's aliases occur only inside after / before / until phrases."""
    main, ctx = split_context(text)
    al = [normalize(a) for a in V.BEHAVIORS[behavior].aliases]
    in_ctx = any(_contains(normalize(c), a) for c in ctx for a in al)
    in_main = any(_contains(normalize(main), a) for a in al)
    return in_ctx and not in_main


class SemanticMatcher:
    """Sentence-embedding fallback for paraphrases (all-MiniLM-L6-v2, frozen). Each clause of the question adds its best-matching
    behaviour when the similarity is at least ``threshold`` and beats the runner-up by ``margin``; both were chosen on development
    questions (benchmarks/parser_dev.json and the E31 questions; E32 development record): 0.5 and 0.05."""

    MODEL_ID = "minilm_l6_v2"
    MODEL_REVISION = "sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

    def __init__(self, model_dir=None, threshold: float = 0.5, margin: float = 0.05):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if model_dir is None:
            from ..config import load_settings
            from ..registry import model_dir as _md

            model_dir = _md(load_settings(), self.MODEL_ID)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModel.from_pretrained(str(model_dir)).eval()
        self.threshold, self.margin = threshold, margin
        self._bank: dict[str, tuple[np.ndarray, list[str]]] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            x = self.tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
            h = self.model(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            return self.torch.nn.functional.normalize(e, dim=-1).numpy()

    def bank(self, species: str) -> tuple[np.ndarray, list[str]]:
        if species not in self._bank:
            a, sw = V.SPECIES_WORD[species]
            texts, owners = [], []
            for b in V.BEHAVIORS.values():
                if species not in b.species:
                    continue
                phrases = [f"my {sw} is {V.describe(b.name, species)}"] + [f"my {sw} {al}" for al in b.aliases] + [p.format(a=a, s=sw) for p in b.prompts]
                texts += phrases
                owners += [b.name] * len(phrases)
            self._bank[species] = (self.embed(texts), owners)
        return self._bank[species]

    def clause_matches(self, species: str, text: str) -> list[tuple[str, float, str]]:
        """Per clause, the best behaviour when its similarity >= threshold and beats the runner-up by >= margin."""
        clauses = _clauses(text)
        if not clauses:
            return []
        E, owners = self.bank(species)
        sim = self.embed(clauses) @ E.T
        names = list(dict.fromkeys(owners))
        cols = {n: [j for j, o in enumerate(owners) if o == n] for n in names}
        out = []
        for ci, clause in enumerate(clauses):
            ranked = sorted(((float(sim[ci, cols[n]].max()), n) for n in names), reverse=True)
            (s1, b1), (s2, _) = ranked[0], ranked[1]
            if s1 >= self.threshold and s1 - s2 >= self.margin:
                out.append((b1, round(s1, 3), clause))
        return out

    def scores(self, species: str, text: str) -> dict[str, float]:
        clauses = _clauses(text)
        if not clauses:
            return {}
        E, owners = self.bank(species)
        sim = self.embed(clauses) @ E.T  # [clauses, phrases]
        best: dict[str, float] = {}
        for j, o in enumerate(owners):
            best[o] = max(best.get(o, -1.0), float(sim[:, j].max()))
        return best


def parse_v2(species: str, scene: str, question: str, matcher: SemanticMatcher | None = None) -> Query:
    """Keyword parse + semantic paraphrase matching on the question; behaviours mentioned only as context are moved to ``context_behaviors``."""
    q = parse(species, scene, question)
    qtext = (question or "").lower()
    found = list(q.reported)
    if matcher is not None:
        for b, _, _ in matcher.clause_matches(species, question or ""):
            if b not in found:
                found.append(b)
    ctx = [b for b in found if context_only(qtext, b, species)]
    q.reported = [b for b in found if b not in ctx]
    q.cues_found = q.cues_found + [f"context:{b}" for b in ctx]
    q.context_behaviors = ctx
    return q


def cue_score(text: str, cues: dict[str, float]) -> tuple[float, list[str]]:
    """Context cue words (substring match on the raw lower-cased text, so a cue like "wrestl" matches its inflections)."""
    hit = [(w, v) for w, v in cues.items() if w in text]
    return sum(v for _, v in hit), [w for w, _ in hit]
