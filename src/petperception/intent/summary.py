"""End-to-end status of the combined inference on every held-out clip (descriptive; written to results/intent/).

Runs intent.inference.infer (the product policy, both variants) on all 294
Kinetics validation clips and all 500 Barkopedia validation clips and counts
the output statuses. For Kinetics it also reports how often the situation
label is right when the policy gives an interpretation. The policy includes
two rules added after E28 (both make it more conservative): no
interpretation from audio alone, and a conflict when the situation
contradicts the detected species. These figures therefore describe the
shipped policy on data that informed it; they are not a fresh test.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from ..config import REPO_ROOT, load_settings
from ..utils import utc_now, write_json
from . import inference as I
from . import kinetics as K


def main() -> None:
    s = load_settings()
    data_root = s.paths.resolve("data_root")
    from ..audio.yamnet import YamnetBackend
    from ..registry import model_dir

    labels = YamnetBackend(model_dir(s, s.audio.yamnet.model_id), 1).labels
    test = [c for c in K.list_clips(data_root) if c.split == "test"]
    rows, _ = K.load_pooled(test, data_root.parent)
    out: dict = {"created_at": utc_now().isoformat(timespec="seconds"), "protocol": __doc__.strip().splitlines()}
    for variant in ("full", "phone_now"):
        models = I.IntentModels(variant=variant)
        st, sit_ok, sit_n, hyp = Counter(), 0, 0, Counter()
        conf_kinds = Counter()
        for r in rows:
            br = {b: (None if v is None else np.asarray(v, np.float64)[None, :]) for b, v in r["branches"].items()}
            if variant == "phone_now":
                br["audio"] = None  # the phone computes class scores only
            res = I.infer(models, br, r["cues"], labels)
            st[res.status] += 1
            for c in res.conflicts:
                conf_kinds["species" if "implies a" in c else "audio_vs_video" if c.startswith("situation:") else "hypotheses"] += 1
            if res.situation and res.situation["status"] == "interpretation":
                sit_n += 1
                sit_ok += res.situation["label"] == r["clip"].label
            for h in res.hypotheses:
                hyp[f"{h.name}:{h.status}"] += 1
        out[f"kinetics_validation:{variant}"] = {"clips": len(rows), "status": dict(st), "conflict_kinds": dict(conf_kinds),
                                                  "situation_interpretations": sit_n,
                                                  "situation_accuracy_when_interpreted": sit_ok / sit_n if sit_n else None,
                                                  "hypothesis_statuses": dict(hyp)}
    # Barkopedia validation: audio only
    from ..eval import e26_bark_activity as E26

    r = E26.root(data_root)
    va = dict(np.load(r / "features_validation.npz", allow_pickle=False))
    sc = np.load(r / "intent_scores_validation.npz", allow_pickle=False)
    pos = {k: i for i, k in enumerate(sc["id"].tolist())}
    for variant in ("full", "phone_now"):
        models = I.IntentModels(variant=variant)
        st, hyp, top_ok, top_n = Counter(), Counter(), 0, 0
        from ..eval.e27_dog_audio_intent import HYPOTHESIS

        base = Counter(HYPOTHESIS[v] for v in va["y"].tolist())

        for i, (cid, y) in enumerate(zip(va["id"].tolist(), va["y"].tolist())):
            S = sc["S"][pos[cid]]
            br = {"audio": None if variant == "phone_now" else va["E"][i][None, :].astype(np.float64), "audio_scores": S[None, :].astype(np.float64),
                  "animal": None, "animal_phone": None, "environment": None, "interaction": None}
            cues = {"missing": {}, "yamnet_max": S[len(S) // 2:], "dog_fraction": None}
            res = I.infer(models, br, cues, labels)
            st[res.status] += 1
            for h in res.hypotheses:
                hyp[f"{h.name}:{h.status}"] += 1
            tent = [h for h in res.hypotheses if h.status in ("tentative", "supported", "supported_corroborated")]
            if tent:
                top_n += 1
                top_ok += tent[0].name == HYPOTHESIS[y]
        out[f"barkopedia_validation:{variant}"] = {"clips": len(va["y"]), "status": dict(st), "hypothesis_statuses": dict(hyp),
                                                    "clips_with_a_vocal_hypothesis": top_n,
                                                    "first_vocal_hypothesis_matches_label": top_ok / top_n if top_n else None,
                                                    "base_rate_most_common_hypothesis": max(base.values()) / sum(base.values())}
    write_json(REPO_ROOT / "results" / "intent" / "end_to_end_status.json", out)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
