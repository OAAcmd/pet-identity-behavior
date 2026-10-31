"""E16: does pet detection hold up across breeds? Cats and dogs (Oxford-IIIT Pet).

Data: Oxford-IIIT Pet (data/oxford_pets.py): 37 breeds, 12 cat and 25 dog,
about 200 images each, with species and breed per image and a head box for
the trainval half. The photos are mostly close-up pet portraits, not
household video, so absolute rates are optimistic for the app; the point is
the *variation across breeds* (morphology: coat, face shape, colour, size),
which no other acquired source measures for cats.

Pipeline (what the phone computes per frame): the product detector at the
product threshold for cats and dogs, then the product verifier. Outcome per
image, in this order:
* found: an accepted detection of the correct species that covers at least
  80 % of the annotated head box (images without a head box: any accepted
  detection of the correct species);
* wrong_species: no correct-species acceptance, but an accepted detection of
  the other species;
* rejected_by_verifier: a correct-species detection existed but the verifier
  rejected it;
* missed: no correct-species detection at the threshold.
Rates per breed with Wilson 95 % intervals; breed heterogeneity per species
by a permutation test on the between-breed standard deviation of the found
rate (breed labels permuted across images, same group sizes). Nothing is
trained, so all images are evaluation images.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..config import Settings, load_settings
from ..data import oxford_pets as OP
from ..utils import utc_now, write_json
from ..vision.detector import detect_at_operating_point

N_PERMUTATIONS = 2000
HEAD_COVERAGE = 0.8


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def _coverage(head: list[float], box) -> float:
    ix = max(0.0, min(head[2], box[2]) - max(head[0], box[0]))
    iy = max(0.0, min(head[3], box[3]) - max(head[1], box[1]))
    area = max((head[2] - head[0]) * (head[3] - head[1]), 1e-9)
    return ix * iy / area


def collect(settings: Settings, recs: list[dict], img_dir: Path, cache: Path) -> list[dict]:
    import cv2

    from ..pipeline import AnalysisEngine

    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            done[r["image"]] = r
    s = load_settings(overrides={"audio": {"enabled": False}, "posture": {"enabled": False}, "pose": {"provider": "none"},
                                 "interaction": {"enabled": False}, "object_relation": {"enabled": False}})
    eng = AnalysisEngine(s).load()
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "a", encoding="utf-8") as fh:
        for r in recs:
            if r["image"] in done:
                continue
            im = cv2.imread(str(img_dir / r["image"]))
            if im is None:
                out = {**r, "outcome": "unreadable"}
            else:
                rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                dets = detect_at_operating_point(eng.detector, rgb, ["cat", "dog"], s.vision, 10)
                ver = []
                for d in dets:
                    v = eng.verifier.verify(rgb, d) if eng.verifier is not None else None
                    ver.append(True if v is None else bool(v[1]))

                def relevant(d) -> bool:
                    return r["head_box"] is None or _coverage(r["head_box"], d.box) >= HEAD_COVERAGE

                correct = [(d, a) for d, a in zip(dets, ver) if d.label == r["species"] and relevant(d)]
                other = [(d, a) for d, a in zip(dets, ver) if d.label != r["species"] and relevant(d)]
                if any(a for _, a in correct):
                    outcome = "found"
                elif any(a for _, a in other):
                    outcome = "wrong_species"
                elif correct:
                    outcome = "rejected_by_verifier"
                else:
                    outcome = "missed"
                best = max((d.score for d, _ in correct), default=None)
                out = {**r, "outcome": outcome, "best_correct_score": best, "n_detections": len(dets)}
            done[r["image"]] = out
            fh.write(json.dumps(out) + "\n")
    return [done[r["image"]] for r in recs]


def _heterogeneity(rows: list[dict], rng: np.random.Generator) -> dict:
    breeds = np.array([r["breed"] for r in rows])
    found = np.array([r["outcome"] == "found" for r in rows], dtype=float)

    def sd(labels: np.ndarray) -> float:
        return float(np.std([found[labels == b].mean() for b in np.unique(labels)]))

    obs = sd(breeds)
    null = np.array([sd(rng.permutation(breeds)) for _ in range(N_PERMUTATIONS)])
    return {"between_breed_sd_of_found_rate": obs, "null_sd_mean": float(null.mean()), "permutation_p": float((1 + (null >= obs).sum()) / (1 + len(null)))}


def run(settings: Settings, out_dir: Path) -> dict:
    data = settings.paths.resolve("data_root")
    recs = OP.records(data)
    img_dir = OP.root(data) / "x" / "oxford-iiit-pet" / "images"
    rows = collect(settings, recs, img_dir, data.parent / "cache" / "e16" / "outcomes.jsonl")
    rows = [r for r in rows if r["outcome"] != "unreadable"]
    rng = np.random.default_rng(20261017)
    summary: dict = {"experiment": "E16", "title": "Pet detection and verification across breeds (Oxford-IIIT Pet)", "protocol": __doc__.strip().splitlines(),
                     "images": len(rows), "unreadable": len(recs) - len(rows), "species": {}}
    for sp in ("cat", "dog"):
        rs = [r for r in rows if r["species"] == sp]
        oc = Counter(r["outcome"] for r in rs)
        by_breed = defaultdict(list)
        for r in rs:
            by_breed[r["breed"]].append(r)
        breeds = {}
        for b, br in sorted(by_breed.items()):
            c = Counter(r["outcome"] for r in br)
            n = len(br)
            breeds[b] = {"images": n, **{k: c.get(k, 0) for k in ("found", "wrong_species", "rejected_by_verifier", "missed")},
                         "found_rate": c.get("found", 0) / n, "found_rate_ci95": wilson(c.get("found", 0), n),
                         "with_head_box": sum(1 for r in br if r["head_box"] is not None)}
        rates = {b: v["found_rate"] for b, v in breeds.items()}
        summary["species"][sp] = {
            "images": len(rs), "breeds": len(breeds), "outcomes": dict(oc), "found_rate": oc.get("found", 0) / max(len(rs), 1),
            "found_rate_ci95": wilson(oc.get("found", 0), len(rs)), "lowest_breeds": sorted(rates, key=rates.get)[:4],
            "highest_breeds": sorted(rates, key=rates.get, reverse=True)[:4], "heterogeneity": _heterogeneity(rs, rng), "by_breed": breeds}
    summary["terms"] = "Oxford-IIIT Pet: CC BY-SA 4.0; image copyright with the original owners. Evaluation only; no image is stored in the repository."
    summary["created_at"] = utc_now().isoformat(timespec="seconds")
    write_json(out_dir / "metrics.json", summary)
    return summary
