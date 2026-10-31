"""E35: does cloud-assisted understanding improve answers to real behaviour questions? Input strategies compared.

Held-out set: every frame-reviewed clip of the E36 scenario manifest (10 tail-chasing positives and 64 hard negatives:
ordinary turning or running, grooming, toy play, tail flicking, neurological head turning, ...). Each clip gets the
priority question with the owner's species and the clip's setting from the frame review as the scene:
"My {species} keeps turning in circles while chasing or biting its tail. Why does it do this?" - so negatives test
whether an answer wrongly confirms the owner's description.

Per clip and strategy (``understand/cloud.py``: full_frames, local_preprocessed, features_only):
* packet: images, upload bytes, estimated input tokens (images ceil(w/28) x ceil(h/28) + text / 4), estimated cost at
  the configured prices, preprocessing time;
* with an API key on the backend, the hosted model's answer: tail chasing reported as observed (with times overlapping
  the reference bouts on positives) or not; validation problems (missing fields, citations outside the knowledge base);
  latency; billed tokens and cost;
* the local system's answer on the same clip as the comparator (version-2 recognizer: tail chasing seen or not).
Scored: recognition (true / false positives against the frame review), evidential support (answers whose explanations
cite only knowledge-base ids and list distinguishing observations), failures, latency, upload size, cost per analysis.
A hosted model's explanation remains a hypothesis; the frame review is machine-assisted, not human ground truth.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import Settings
from ..utils import utc_now, write_json

QUESTION = "My {species} keeps turning in circles while chasing or biting its tail. Why does it do this?"


def _overlap(obs: list[dict], ref: list[dict]) -> bool:
    for o in obs:
        try:
            a, b = float(o.get("t_start_s")), float(o.get("t_end_s"))
        except (TypeError, ValueError):
            continue
        if any(min(b, r["t1"]) > max(a, r["t0"]) for r in ref):
            return True
    return False


def score_cloud(ans: dict | None, row: dict) -> dict:
    if ans is None:
        return {"usable": False}
    tc = [o for o in ans.get("observed", []) if str(o.get("behavior", "")).startswith("tail_chasing")]
    seen = bool(tc)
    return {"usable": True, "tail_chasing_observed": seen, "times_overlap_reference": _overlap(tc, row["bouts"]) if seen and row["bouts"] else None,
            "says_not_observed": any("tail" in str(x).lower() for x in ans.get("not_observed", [])),
            "explanations": len(ans.get("candidate_explanations", [])), "distinguishing": len(ans.get("would_distinguish", []))}


def run(settings: Settings, out_dir: Path, call_cloud: bool) -> dict:
    from ..understand import cloud as CL
    from ..understand import dataset as D
    from ..understand.run import _Z, Explainer
    from .e36_tail_chasing_scenario import SCENARIO

    sc = json.loads(SCENARIO.read_text())
    rows = [r for r in sc["clips"] if r["label"] in ("positive", "hard_negative")]
    ann = D.load_assistant()
    clips = {c.key: c for c in D.clips(settings.paths.resolve("data_root"))}
    root = settings.paths.resolve("data_root").parent / "understand" / "features_v2"
    ex = Explainer.load(settings.runtime.num_threads, version=2)
    cfg = settings.cloud
    per = []
    for n, r in enumerate(rows, 1):
        c = clips[r["clip"]]
        scene = (ann.get(r["clip"], {}).get("setting") or "").strip()
        q = QUESTION.format(species=r["species"])
        t0 = time.perf_counter()
        z = _Z(dict(np.load(root / f"{r['clip']}.npz", allow_pickle=False)))
        local = ex.answer(z, r["species"], scene, q).to_dict()
        yam = ex.extractor.yamnet.labels if ex.extractor.yamnet is not None and "a_scores" in z.files else None
        ev = {"z": z, "answer": local, "yam": yam}
        local_s = time.perf_counter() - t0
        item = {"clip": r["clip"], "label": r["label"], "species": r["species"], "scene": scene, "duration_s": float(z["t"][-1]) if len(z["t"]) else None,
                "local": {"tail_chasing_seen": any(b["behavior"] == "tail_chasing" for b in local["behaviors_seen"]), "seconds_from_stored_features": round(local_s, 2)},
                "strategies": {}}
        for strat in CL.STRATEGIES:
            p = CL.build_packet(c.path, r["species"], scene, q, strat, ev)
            st = p.stats(CL.system_prompt(r["species"]))
            st["estimated_cost_usd"] = round((st["image_tokens_est"] + st["text_tokens_est"]) * cfg.price_in_per_mtok / 1e6
                                             + 1000 * cfg.price_out_per_mtok / 1e6, 5)
            entry = {"packet": st}
            if call_cloud:
                out = CL.call(p, cfg)
                entry["cloud"] = {k: v for k, v in out.items() if k != "answer"} | {"score": score_cloud(out.get("answer"), r)}
                entry["answer"] = out.get("answer")
            item["strategies"][strat] = entry
        per.append(item)
        if n % 10 == 0:
            print(f"{n}/{len(rows)}", flush=True)
    res = {"experiment": "E35", "created_at": utc_now().isoformat(timespec="seconds"), "clips": len(per),
           "positives": sum(r["label"] == "positive" for r in rows), "hard_negatives": sum(r["label"] == "hard_negative" for r in rows),
           "cloud_called": call_cloud, "cloud_model": cfg.model if call_cloud else None,
           "prices_usd_per_mtok": {"input": cfg.price_in_per_mtok, "output": cfg.price_out_per_mtok, "output_tokens_assumed_for_estimate": 1000}}
    y = np.array([p["label"] == "positive" for p in per])
    loc = np.array([p["local"]["tail_chasing_seen"] for p in per])
    res["local_v2"] = {"tp": int(np.sum(loc & y)), "fp": int(np.sum(loc & ~y)), "fn": int(np.sum(~loc & y))}
    res["packets"] = {}
    for strat in CL.STRATEGIES:
        st = [p["strategies"][strat]["packet"] for p in per]
        res["packets"][strat] = {k: {"median": float(np.median([s[k] for s in st])), "max": float(np.max([s[k] for s in st]))}
                                 for k in ("images", "upload_bytes_base64", "image_tokens_est", "text_tokens_est", "estimated_cost_usd", "preprocess_s")}
        if call_cloud:
            sc_ = [p["strategies"][strat]["cloud"] for p in per]
            obs = np.array([bool(s["score"].get("tail_chasing_observed")) for s in sc_])
            res["packets"][strat]["cloud"] = {
                "ok": sum(s.get("status") == "ok" for s in sc_), "tp": int(np.sum(obs & y)), "fp": int(np.sum(obs & ~y)), "fn": int(np.sum(~obs & y)),
                "times_overlap_on_tp": sum(bool(s["score"].get("times_overlap_reference")) for s, yy in zip(sc_, y) if yy),
                "answers_with_validation_problems": sum(bool(s.get("problems")) for s in sc_),
                "median_latency_s": float(np.median([s.get("latency_s", np.nan) for s in sc_])),
                "mean_cost_usd": float(np.mean([s.get("cost_usd", 0.0) or 0.0 for s in sc_]))}
    write_json(out_dir / "metrics.json", res)
    write_json(out_dir / "per_clip.json", per)
    return res


def main() -> None:
    import argparse
    import os

    from ..config import load_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--call-cloud", action="store_true", help="send the packets (needs the API key in cloud.api_key_env)")
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    s = load_settings()
    if a.threads:
        s.runtime.num_threads = a.threads
    if a.call_cloud and not os.environ.get(s.cloud.api_key_env):
        raise SystemExit(f"--call-cloud needs {s.cloud.api_key_env} in the environment")
    r = run(s, s.paths.resolve("results_root") / "e35", a.call_cloud)
    print(json.dumps({k: r[k] for k in ("clips", "positives", "hard_negatives", "cloud_called", "local_v2", "packets")}, indent=1))


if __name__ == "__main__":
    main()
