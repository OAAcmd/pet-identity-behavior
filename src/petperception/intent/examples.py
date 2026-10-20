"""End-to-end intent examples on real held-out recordings (written to results/intent/examples/).

* Kinetics validation clips (never used for training): one per situation,
  with all channels, and the same clip with audio removed and with video
  removed;
* a conflict fixture: the video of one validation clip with the audio of a
  clip from a different situation. Such pairs are fixtures for the conflict
  logic only, never multimodal evidence;
* Barkopedia validation clips (audio only), with and without the location
  given as context.
Each file carries the dataset label and its layer next to the output, so a
reader can compare without mistaking a context label for an intent.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from ..config import REPO_ROOT, load_settings
from ..utils import write_json
from . import inference as I
from . import kinetics as K
from . import taxonomy as T

OUT = REPO_ROOT / "results" / "intent" / "examples"


def _labels(settings) -> list[str]:
    from ..audio.yamnet import YamnetBackend
    from ..registry import model_dir

    return YamnetBackend(model_dir(settings, settings.audio.yamnet.model_id), 1).labels


def _as_batch(br: dict) -> dict:
    return {b: (None if v is None else np.asarray(v, np.float64)[None, :]) for b, v in br.items()}


def _drop(br: dict, cues: dict, which: str) -> tuple[dict, dict]:
    br, cues = dict(br), {**cues, "missing": dict(cues.get("missing", {}))}
    if which == "audio":
        for b in ("audio", "audio_scores"):
            br[b] = None
            cues["missing"][b] = "audio removed for this example"
        cues.pop("yamnet_max", None)
    else:
        for b in ("animal", "animal_phone", "environment", "interaction"):
            br[b] = None
            cues["missing"][b] = "video removed for this example"
        for k in ("object_frames", "dog_fraction", "pet_frames", "person_frames", "interaction_max", "posture_mean", "frames"):
            cues.pop(k, None)
        cues["dog_fraction"] = None
    return br, cues


def _write(name: str, meta: dict, res: I.IntentResult) -> dict:
    d = {"example": name, **meta, "output": res.to_dict()}
    write_json(OUT / f"{name}.json", d)
    return {"example": name, **{k: meta[k] for k in ("source", "dataset_label", "label_layer")}, "status": res.status, "summary": res.summary}


def main() -> None:
    s = load_settings()
    data_root = s.paths.resolve("data_root")
    labels = _labels(s)
    models = I.IntentModels()
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    test = [c for c in K.list_clips(data_root) if c.split == "test"]
    rows, _ = K.load_pooled(test, data_root.parent)
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["clip"].label, []).append(r)
    picks = {lab: sorted(v, key=lambda r: r["clip"].key)[0] for lab, v in sorted(by_label.items())}
    for lab, r in picks.items():
        c = r["clip"]
        meta = {"source": f"Kinetics-700 validation clip {c.youtube_id} ({c.path.name})", "dataset_label": lab,
                "label_layer": T.label("kinetics_pets", lab).layer, "note": "held out: never used to train any head"}
        slug = lab.replace(" ", "_")
        index.append(_write(f"kinetics_{slug}_all_channels", meta, I.infer(models, _as_batch(r["branches"]), r["cues"], labels)))
        for which in ("audio", "video"):
            br, cues = _drop(r["branches"], r["cues"], which)
            index.append(_write(f"kinetics_{slug}_{which}_removed", meta | {"note": f"{which} removed to show missing-input handling"},
                                I.infer(models, _as_batch(br), cues, labels)))
    # conflict fixtures: video of one clip + audio of a clip from another situation (unrelated recordings)
    for vl, al in (("walking the dog", "bathing dog"), ("grooming dog", "petting cat")):
        v, a = picks[vl], picks[al]
        br = dict(v["branches"])
        br["audio"], br["audio_scores"] = a["branches"]["audio"], a["branches"]["audio_scores"]
        cues = {**v["cues"], "yamnet_max": a["cues"].get("yamnet_max")}
        meta = {"source": f"FIXTURE: video {v['clip'].path.name} + audio {a['clip'].path.name} (unrelated recordings)",
                "dataset_label": f"{vl} (video) / {al} (audio)", "label_layer": "recorded_context",
                "note": "unrelated pair: tests the conflict logic only; not multimodal evidence"}
        index.append(_write(f"fixture_swap_{vl.replace(' ', '_')}_video_{al.replace(' ', '_')}_audio", meta,
                            I.infer(models, _as_batch(br), cues, labels)))
    # Barkopedia validation, audio only
    from ..eval import e26_bark_activity as E26

    r = E26.root(data_root)
    va = dict(np.load(r / "features_validation.npz", allow_pickle=False))
    sc = np.load(r / "intent_scores_validation.npz", allow_pickle=False)
    pos = {k: i for i, k in enumerate(sc["id"].tolist())}
    env = {row["audio_id"]: row["environement"] for row in csv.DictReader(open(r / "valid_labels.csv", newline="", encoding="utf-8"))}
    from ..eval.e27_dog_audio_intent import ENV

    chosen = {}
    for i, y in enumerate(va["y"].tolist()):
        if y not in chosen:
            chosen[y] = i
    for y, i in sorted(chosen.items()):
        cid = va["id"][i]
        S = sc["S"][pos[cid]]
        br = {"audio": va["E"][i][None, :].astype(np.float64), "audio_scores": S[None, :].astype(np.float64), "animal": None, "animal_phone": None,
              "environment": None, "interaction": None}
        cues = {"missing": {b: "audio-only recording" for b in ("animal", "animal_phone", "environment", "interaction")},
                "yamnet_max": S[len(S) // 2:], "dog_fraction": None}
        lab = T.label("barkopedia_act_env", y)
        meta = {"source": f"Barkopedia validation clip {cid}", "dataset_label": y, "label_layer": lab.layer,
                "dataset_environment": env[cid], "note": "held out: provider validation split"}
        slug = y.replace(" ", "_")
        index.append(_write(f"barkopedia_{slug}_audio_only", meta, I.infer(models, br, cues, labels)))
        e = ENV.get(E26.norm(env[cid]), "other")
        index.append(_write(f"barkopedia_{slug}_audio_with_location", meta | {"note": f"location given as context: {e}"},
                            I.infer(models, br, cues, labels, environment=env[cid])))
    write_json(OUT / "index.json", {"examples": index})
    print(json.dumps(index, indent=1)[:4000])


if __name__ == "__main__":
    main()
