"""E18: the full engine on real pet video with sound (Kinetics-700 pet actions).

Data: 10-second Kinetics-700-2020 validation clips of six human actions that
involve a cat or a dog (data/kinetics_pets.py): petting cat, grooming cat,
grooming dog, bathing dog, training dog, walking the dog. They are real,
synchronized audio and video, but YouTube footage chosen for the human
action, not household recordings, and the labels are clip-level actions.

Engine: the Python reference engine with the components of the shipped
phone bundle (release profile personal_noncommercial), audio and video.

What the labels support (and only that):
* pet presence: every clip shows the named species, so the share of clips
  with a track of that species is a clip-level recall; clips whose only
  tracks are the other species are counted as species confusion;
* person contact: petting, grooming and bathing imply a person touching the
  animal during the clip, so the share of those clips with a
  person_interaction_candidate segment is a clip-level recall; training and
  walking do not imply contact (often a leash or distance), so their rate is
  reported alongside but is not a false-positive rate;
* everything else (sounds, posture, on-furniture, interpretations) has no
  label here and is reported descriptively, never as accuracy.
Intervals are Wilson 95 % intervals over clips.

Variants (same clips, same engine otherwise; pose off):
* reference: MediaPipe EfficientDet-Lite0 float32 + crop verifier, the
  configuration E4-E16 measured (first run, cached as results.jsonl);
* no_verifier: the same detector without the crop verifier, to see how many
  misses the verifier causes;
* phone_build: the TF Hub detector with raw-output decoding + its verifier,
  the detector of the published phone test build (docs/ASSET_TERMS.md).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import Settings, load_settings
from ..data import kinetics_pets as KP
from ..utils import utc_now, write_json
from .e16_breed_robustness import wilson

VARIANTS = {
    "reference": {"vision": {"detector": "efficientdet_lite0_mediapipe_v1"}},
    "no_verifier": {"vision": {"detector": "efficientdet_lite0_mediapipe_v1", "verifier": {"enabled": False}}},
    "phone_build": {"vision": {"detector": "efficientdet_lite0_tfhub_rawdecode_v1"}},
}


def analyse(settings: Settings, rows: list[dict], cache: Path, overrides: dict | None = None) -> list[dict]:
    from ..ingest.media import MediaError
    from ..pipeline import AnalysisEngine

    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            done[r["clip"]] = r
    if all(r["clip"] in done for r in rows):
        return [done[r["clip"]] for r in rows]
    # research profile: the components equal the personal_noncommercial bundle except the detector licence check
    s = load_settings(overrides={"release_profile": "research", "pose": {"provider": "none"}, **(overrides or {})})
    eng = AnalysisEngine(s).load()
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "a", encoding="utf-8") as fh:
        for r in rows:
            if r["clip"] in done:
                continue
            try:
                res = json.loads(eng.analyze(r["path"], zones=None, modalities=("audio", "video"), analysis_id=f"e18-{r['clip']}").model_dump_json())
            except MediaError as exc:
                out = {**{k: r[k] for k in ("class", "species", "person_contact_implied", "clip")}, "error": exc.code}
            else:
                tracks = res["observations"]["animal_tracks"]
                segs = [b for t in tracks for b in t["behavior_segments"]]
                out = {**{k: r[k] for k in ("class", "species", "person_contact_implied", "clip")},
                       "duration_s": res["media"].get("duration_s"), "track_species": [t["species"] for t in tracks],
                       "segment_kinds": sorted({b["kind"] for b in segs}),
                       "interaction_s": float(sum(b["t_end_s"] - b["t_start_s"] for b in segs if b["kind"] == "person_interaction_candidate")),
                       "sounds": [e["label"] for e in res["observations"]["sound_events"]],
                       "decision": res["decision"]["state"],
                       "hypotheses": [h["name"] for h in res.get("hypotheses", []) if h.get("status") not in ("abstained", "not_supported", "insufficient_evidence")],
                       "quality_flags": [q["code"] for q in res.get("quality_flags", [])], "timings_ms": res.get("timings_ms", {})}
            done[r["clip"]] = out
            fh.write(json.dumps(out) + "\n")
    return [done[r["clip"]] for r in rows]


def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": wilson(k, n)}


def run(settings: Settings, out_dir: Path) -> dict:
    data = settings.paths.resolve("data_root")
    clips = KP.clips(data)
    cache = data.parent / "cache" / "e18"
    variants = {name: summarize(analyse(settings, clips, cache / ("results.jsonl" if name == "reference" else f"results_{name}.jsonl"), ov))
                for name, ov in VARIANTS.items()}
    summary = {"experiment": "E18", "title": "Full engine on real pet video with sound (Kinetics-700 pet actions; research only)",
               "protocol": __doc__.strip().splitlines(), **variants["reference"],
               "variants": {name: {"detector": VARIANTS[name]["vision"]["detector"],
                                   "verifier": VARIANTS[name]["vision"].get("verifier", {}).get("enabled", True),
                                   "by_species": {sp: {k: b[k] for k in ("pet_present", "only_other_species", "no_pet_track", "person_interaction_segment",
                                                                         "own_species_vocalization_detected", "analysis_ms_median")}
                                                  for sp, b in v["by_species"].items()},
                                   "person_interaction": v["person_interaction"]} for name, v in variants.items()}}
    summary["terms"] = "Kinetics-700-2020 via the CVDF mirror; annotations CC BY 4.0, clips are YouTube uploads (research use; not committed)."
    summary["created_at"] = utc_now().isoformat(timespec="seconds")
    write_json(out_dir / "metrics.json", summary)
    return summary


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    summary: dict = {"clips": len(rows), "decode_errors": dict(Counter(r["error"] for r in rows if "error" in r)), "by_class": {}, "by_species": {}}
    for key, sel in [("class", lambda r, v: r["class"] == v), ("species", lambda r, v: r["species"] == v)]:
        values = sorted({r[key] for r in ok})
        for v in values:
            rs = [r for r in ok if sel(r, v)]
            sp = rs[0]["species"]
            other = "dog" if sp == "cat" else "cat"
            block = {"clips": len(rs),
                     "pet_present": _rate(sum(sp in r["track_species"] for r in rs), len(rs)),
                     "only_other_species": _rate(sum(bool(r["track_species"]) and sp not in r["track_species"] and other in r["track_species"] for r in rs), len(rs)),
                     "no_pet_track": _rate(sum(not r["track_species"] for r in rs), len(rs)),
                     "person_interaction_segment": _rate(sum("person_interaction_candidate" in r["segment_kinds"] for r in rs), len(rs)),
                     "own_species_vocalization_detected": _rate(sum(f"{sp}_vocalization" in r["sounds"] for r in rs), len(rs)),
                     "segment_kinds": dict(Counter(k for r in rs for k in r["segment_kinds"])),
                     "decisions": dict(Counter(r["decision"] for r in rs)),
                     "clips_with_a_non_abstained_interpretation": _rate(sum(bool(r["hypotheses"]) for r in rs), len(rs)),
                     "interpretations": dict(Counter(h for r in rs for h in r["hypotheses"])),
                     "quality_flags": dict(Counter(q for r in rs for q in r["quality_flags"])),
                     "analysis_ms_median": float(np.median([sum(r["timings_ms"].values()) for r in rs if r["timings_ms"]])) if rs else None}
            summary[f"by_{key}"][v] = block
    contact = [r for r in ok if r["person_contact_implied"]]
    no_contact = [r for r in ok if not r["person_contact_implied"]]
    tracked = [r for r in contact if r["species"] in r["track_species"]]
    hit = lambda rs: sum("person_interaction_candidate" in r["segment_kinds"] for r in rs)  # noqa: E731
    summary["person_interaction"] = {
        "contact_classes_recall": _rate(hit(contact), len(contact)),
        "contact_classes_recall_given_pet_tracked": _rate(hit(tracked), len(tracked)),
        "non_contact_classes_rate": _rate(hit(no_contact), len(no_contact)),
        "note": "non-contact classes may still show contact; their rate is not a false-positive rate"}
    return summary
