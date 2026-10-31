"""Capability inventory: what the product can actually do per behaviour, computed from the code and the result files.

``python -m petperception.inventory`` writes ``docs/capability_inventory.json`` and prints the Markdown tables used in
docs/CAPABILITY_MATRIX.md. Counting rules: species detection is not a behaviour; aliases are not behaviours; a
knowledge-base entry is an explanation capability, never evidence of recognition from video; a behaviour counts as
"evaluated" only with a held-out or cross-validated measurement on real recordings (the metric is shown, whether or
not it met its bar).
"""

from __future__ import annotations

import json
from collections import Counter

from .config import REPO_ROOT

# Observations of the analysis engine (Python and phone engines; parity-tested in desktop Chromium, no physical phone).
# Mapped to the explanation vocabulary where they are the same behaviour, so they are not counted twice.
ENGINE_OBSERVATIONS = {
    "moving": {"same_as": "locomotion", "species": ["cat", "dog"], "status": "experimental",
               "evidence": "rule on the track; E8 semi-synthetic only; produced on real video in E18 (no labels)"},
    "stationary": {"same_as": None, "species": ["cat", "dog"], "status": "experimental", "evidence": "rule on the track; E8 semi-synthetic only"},
    "active_bout": {"same_as": None, "species": ["cat", "dog"], "status": "experimental", "evidence": "rule on motion; E8 semi-synthetic only"},
    "pacing_candidate": {"same_as": None, "species": ["cat", "dog"], "status": "experimental", "evidence": "rule; E8 semi-synthetic only"},
    "startle_candidate": {"same_as": None, "species": ["cat", "dog"], "status": "experimental", "evidence": "audio-motion rule; no real evaluation"},
    "vocal_bout": {"same_as": "vocalizing", "species": ["cat", "dog"], "status": "evaluated",
                   "evidence": "E21 FSD50K eval: cat vocalization F1 0.688; meow and bark subtypes passed the gate (results/e21)"},
    "person_interaction_candidate": {"same_as": None, "species": ["cat", "dog"], "status": "evaluated",
                                     "evidence": "E13 dog pairs AP 0.907 (photos); E18 real video contact clips: cat 35/56, dog 36/50 tracked clips (results/e13, results/e18)"},
    "on_furniture_candidate": {"same_as": None, "species": ["cat", "dog"], "status": "evaluated",
                               "evidence": "E15 Open Images rule precision / recall 0.97 pooled, end-to-end recall 0.40 (results/e15)"},
    "posture_lying": {"same_as": None, "species": ["cat", "dog"], "status": "evaluated", "evidence": "E10 weak labels: cat F1 0.75, dog 0.61 (results/e10)"},
    "posture_sitting": {"same_as": None, "species": ["cat", "dog"], "status": "evaluated", "evidence": "E10 (as lying)"},
    "posture_standing": {"same_as": None, "species": ["cat", "dog"], "status": "evaluated", "evidence": "E10 (as lying; 3 cat examples only)"},
}
ZONE_EVENTS = ["approach_zone", "enter_zone", "dwell_zone", "leave_zone", "entered_view", "left_view"]


def _eval_evidence() -> dict[str, list[str]]:
    ev: dict[str, list[str]] = {}
    r = REPO_ROOT / "results"
    e31 = json.loads((r / "e31" / "metrics.json").read_text())["part_a"]["methods"]["R0_zero_shot"]["per_behavior_ap"]
    for b, ap in e31.items():
        ev.setdefault(b, []).append(f"E31 pool CV (human labels): AP {ap:.2f}")
    e32 = json.loads((r / "e32" / "metrics.json").read_text())["v2"]["recognition_extra"]["by_behavior"]
    for b, x in e32.items():
        ev.setdefault(b, []).append(f"E32 fresh test: {x['k']}/{x['n']} recognized")
    snd = json.loads((r / "e31" / "metrics.json").read_text())["part_a"]["sound_vocalizing"]
    ev.setdefault("vocalizing", []).append(f"E31 sound vs title labels: AP {snd['ap']:.2f} (base rate {snd['base_rate']:.2f}); E21 FSD50K cat vocalization F1 0.69")
    e36 = json.loads((r / "e36" / "metrics.json").read_text())["methods"]["B1_siglip_zero_shot"]
    ev.setdefault("tail_chasing", []).append(f"E36 held-out sources: AP {e36['ap']:.2f}, {e36['tp']}/{e36['positives']} found, {e36['fp']} FP (not reliable)")
    return ev


def build() -> dict:
    from .understand import knowledge as K
    from .understand import vocab as V
    from .understand.e31 import vocab as V1

    ev = _eval_evidence()
    rows = []
    for name, b in V.BEHAVIORS.items():
        kb = {sp: K.entry(name, sp) is not None for sp in b.species}
        evaluated = name in ev
        rows.append({"behavior": name, "description": b.description, "species": list(b.species), "aliases": len(b.aliases),
                     "understood_from_text": len(b.aliases) > 0,
                     "recognized_from": "sound (YAMNet)" if b.channel == "audio" else "video (zero-shot SigLIP prompts" + (", turning-in-place model" if name == "tail_chasing" else "") + ")",
                     "recognition_status": "evaluated" if evaluated else "experimental",
                     "explained_with_sources": kb, "in_default_system": name in V1.BEHAVIORS, "evidence": ev.get(name, ["no measurement on real recordings"]),
                     "where": "server / desktop (Python); not on the phone"})
    engine = [{"observation": k, **v, "where": "phone engine (browser, parity-tested on desktop Chromium) and Python engine"}
              for k, v in ENGINE_OBSERVATIONS.items()]
    vocab_by_sp = {sp: {r["behavior"] for r in rows if sp in r["species"]} for sp in ("cat", "dog")}
    eng_by_sp = {sp: {k for k, v in ENGINE_OBSERVATIONS.items() if sp in v["species"] and v["same_as"] is None} for sp in ("cat", "dog")}
    by_sp = {sp: vocab_by_sp[sp] | eng_by_sp[sp] for sp in ("cat", "dog")}
    shared = by_sp["cat"] & by_sp["dog"]
    status = Counter(r["recognition_status"] for r in rows) + Counter(v["status"] for k, v in ENGINE_OBSERVATIONS.items() if v["same_as"] is None)
    subtypes = json.loads((REPO_ROOT / "artifacts" / "audio_subtype_validation.json").read_text())
    subtypes = subtypes.get("subtypes", subtypes)
    ctx_tags = json.loads((REPO_ROOT / "artifacts" / "audio_context_thresholds.json").read_text())["tags"]
    totals = {
        "behaviours_cat": len(by_sp["cat"]), "behaviours_dog": len(by_sp["dog"]), "behaviours_shared": len(shared),
        "behaviours_deduplicated": len(by_sp["cat"] | by_sp["dog"]),
        "behaviours_cat_only": sorted(by_sp["cat"] - by_sp["dog"]), "behaviours_dog_only": sorted(by_sp["dog"] - by_sp["cat"]),
        "recognition_status": dict(status),
        "explanation_vocabulary": len(rows), "engine_observations_not_duplicated": len({k for k, v in ENGINE_OBSERVATIONS.items() if v["same_as"] is None}),
        "sound_events": {"vocalization_events": ["cat_vocalization", "dog_vocalization"],
                         "vocal_types_shown": sorted(k for k, v in subtypes.items() if (v.get("status") if isinstance(v, dict) else v) == "passed"),
                         "vocal_types_blocked": sorted(k for k, v in subtypes.items() if (v.get("status") if isinstance(v, dict) else v) != "passed"),
                         "context_sound_tags": list(ctx_tags) if isinstance(ctx_tags, list) else list(ctx_tags)},
        "contexts": {"zone_events": ZONE_EVENTS, "recording_context_server": ["person near / in view", "9 object kinds", "4 sound kinds", "8 places"],
                     "situations_kinetics": 6, "dog_bark_context": 3, "cat_meow_context": 3, "dog_activity_from_barks": 8},
        "explanation": {"knowledge_entries": len(K.ENTRIES), "entries_cat": sum(sp == "cat" for _, sp in K.ENTRIES),
                        "entries_dog": sum(sp == "dog" for _, sp in K.ENTRIES), "sources": len(K.SOURCES), "candidate_kinds": list(K.KINDS),
                        "engine_hypotheses": ["food", "outside", "attention", "play", "alert", "tense", "warning", "relaxed"],
                        "cloud_understanding": "implemented, not evaluated (no API key)"},
        "phone_validation": "no physical phone has run any of this; desktop Chromium parity only",
    }
    return {"generated_by": "python -m petperception.inventory", "totals": totals, "behaviours": rows, "engine_observations": engine}


def markdown(inv: dict) -> str:
    t = inv["totals"]
    lines = ["| Count | Value |", "| --- | --- |",
             f"| Distinct behaviours, cats | {t['behaviours_cat']} |", f"| Distinct behaviours, dogs | {t['behaviours_dog']} |",
             f"| Shared | {t['behaviours_shared']} |", f"| Deduplicated total | {t['behaviours_deduplicated']} |",
             f"| Cat only / dog only | {', '.join(t['behaviours_cat_only'])} / {', '.join(t['behaviours_dog_only'])} |",
             f"| With a measurement on real recordings / experimental only | {t['recognition_status'].get('evaluated', 0)} / {t['recognition_status'].get('experimental', 0)} |",
             f"| Sound events | 2 vocalization events; vocal types shown: {', '.join(t['sound_events']['vocal_types_shown'])}; "
             f"{len(t['sound_events']['vocal_types_blocked'])} types blocked by the gate; {len(t['sound_events']['context_sound_tags'])} context sounds |",
             f"| Contexts | {len(t['contexts']['zone_events'])} zone / view events; server recording context; 6 situations; bark (3) and meow (3) contexts; 8 dog activities from barks |",
             f"| Explanation | {t['explanation']['knowledge_entries']} knowledge entries (cat {t['explanation']['entries_cat']}, dog {t['explanation']['entries_dog']}), "
             f"{t['explanation']['sources']} sources, 5 candidate kinds; 8 engine hypotheses (experimental) |", "",
             "| Behaviour | Species | From text | Recognized from | Recognition | Sourced explanation | In default answer system | Evidence |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in inv["behaviours"]:
        lines.append(f"| {r['behavior']} | {'/'.join(r['species'])} | {'yes' if r['understood_from_text'] else 'no'} | {r['recognized_from']} | "
                     f"{r['recognition_status']} | {', '.join(sp for sp, v in r['explained_with_sources'].items() if v) or 'no'} | {'yes' if r['in_default_system'] else 'no (system v2 only)'} | {'; '.join(r['evidence'])} |")
    for e in inv["engine_observations"]:
        if e["same_as"] is None:
            lines.append(f"| {e['observation']} (engine) | {'/'.join(e['species'])} | no | video (phone and Python engines) | {e['status']} | no | engine | {e['evidence']} |")
    return "\n".join(lines)


def main() -> None:
    inv = build()
    (REPO_ROOT / "docs" / "capability_inventory.json").write_text(json.dumps(inv, indent=1) + "\n")
    print(markdown(inv))


if __name__ == "__main__":
    main()
