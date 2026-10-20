"""Train the remaining audio channel heads as intent-layer artifacts (no new test data is scored here).

* cat audio context (CatMeows, CC BY 4.0): brushing / isolation /
  waiting_for_food, the recording situation of a meow. Held-out evaluation is
  E2 (leave-one-cat-out macro-F1 0.46 with YAMNet embeddings, below the
  evidence gate), so the head is used as weak context evidence only.
* dog bark context (Yin & McCowan via BEANS, research only): aggression
  (stranger at the door) / contact (isolation) / play. Held-out evaluation is
  E22 (leave-one-dog-out macro-F1 0.72 with YAMNet embeddings).

Each head: logistic regression on YAMNet embedding mean | max (and, for
CatMeows, class scores mean | max), C by grouped CV over animals, temperature
and abstention threshold from out-of-fold scores (intent/heads.py). The OOF
figures are recorded next to the artifact; they are cross-validation on the
training animals, not a test result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT, Settings, load_settings
from ..utils import utc_now, write_json
from . import heads as H

ART = REPO_ROOT / "artifacts" / "intent"


def train_catmeows(settings: Settings, out_dir: Path) -> dict:
    from ..audio.yamnet import YamnetBackend
    from ..data import catmeows as CM
    from ..eval.e2_cat_context import _load
    from ..ingest.media import AudioTrack
    from ..registry import model_dir

    root = CM.dataset_dir(settings.paths.resolve("data_root"))
    recs, _ = CM.scan(root, settings.paths.resolve("manifests_root") / "catmeows_files.csv")
    be = YamnetBackend(model_dir(settings, settings.audio.yamnet.model_id), settings.runtime.num_threads, with_embeddings=True)
    E, S = [], []
    for r in recs:
        x, sr = _load(r.path)
        ws = be.score_track(AudioTrack("present", x, sr, 0.0), hop_s=0.48)
        E.append(np.concatenate([ws.embeddings.mean(0), ws.embeddings.max(0)]))
        S.append(np.concatenate([ws.scores.mean(0), ws.scores.max(0)]))
    y = np.array([r.context for r in recs])
    g = np.array([r.cat_id for r in recs])
    classes = sorted(set(y.tolist()))
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for b, X in (("audio", np.array(E)), ("audio_scores", np.array(S))):
        head, inf, _ = H.fit_head(b, X, y, g, classes, metadata={
            "branch": b, "dataset": "CatMeows (Zenodo 4008297, CC BY 4.0)", "label_layer": "recorded_context",
            "label_to_qualifier": {"brushing": "handling", "isolation": "isolation", "waiting_for_food": "food_routine"},
            "label_to_hypothesis": {"isolation": {"attention": "context_for"}, "waiting_for_food": {"food": "context_for"}},
            "held_out_evaluation": "results/e2/metrics.json (leave-one-cat-out macro-F1 0.464 YAMNet embedding; below the 0.7 evidence gate)",
            "evidence_gate_passed": False, "species": "cat"})
        head.to_json(out_dir / f"head_{b}.json")
        info[b] = {k: v for k, v in inf.items() if k != "inner_macro_f1"} | {"inner_macro_f1": inf.get("inner_macro_f1")}
    return {"clips": len(y), "cats": len(set(g.tolist())), "classes": classes, "heads": info}


def train_beans(settings: Settings, out_dir: Path) -> dict:
    z = np.load(settings.paths.resolve("data_root") / "beans_dog_barks" / "features_v1.npz", allow_pickle=False)
    meta = [json.loads(s) for s in z["rows"]]
    y = np.array([m["context"] for m in meta])
    g = np.array([m["name"] for m in meta])
    classes = sorted(set(y.tolist()))
    head, inf, _ = H.fit_head("audio", z["E"], y, g, classes, metadata={
        "branch": "audio", "dataset": "Yin & McCowan (2004) barks via BEANS (no licence stated; research only)", "label_layer": "recorded_context",
        "label_to_qualifier": {"aggression": "disturbance", "contact": "isolation", "play": None},
        "label_to_hypothesis": {"aggression": {"alert": "context_for", "warning": "context_for"}, "contact": {"attention": "context_for"},
                                "play": {"play": "context_for"}},
        "held_out_evaluation": "results/e22/metrics.json (leave-one-dog-out macro-F1 0.722 YAMNet embedding)",
        "evidence_gate_passed": True, "species": "dog"})
    out_dir.mkdir(parents=True, exist_ok=True)
    head.to_json(out_dir / "head_audio.json")
    return {"clips": len(y), "dogs": len(set(g.tolist())), "classes": classes, "heads": {"audio": inf}}


def main() -> None:
    s = load_settings()
    out = {"created_at": utc_now().isoformat(timespec="seconds"),
           "cat_audio_context_v1": train_catmeows(s, ART / "cat_audio_context_v1"),
           "dog_bark_context_v1": train_beans(s, ART / "dog_bark_context_v1")}
    write_json(REPO_ROOT / "results" / "intent" / "train_heads.json", out)
    print(json.dumps({k: (v["heads"] if isinstance(v, dict) and "heads" in v else v) for k, v in out.items()}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
