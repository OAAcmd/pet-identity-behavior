"""Feature-level parity fixture for the phone intent module (web/src/engine/intent.ts).

Writes web/tests/unit/fixtures/intent_parity.json: the bundle's intent assets,
per-frame / per-window inputs for a few cases (real held-out Kinetics clips,
truncated to keep the file small, plus edge cases: audio absent, no animal
detected, silent audio) and the Python reference outputs (pooled branches,
channel readings, hypotheses, status). The TypeScript unit test recomputes
everything from the same inputs and must match.
"""

from __future__ import annotations

import base64
import json

import numpy as np

from ..config import REPO_ROOT, load_settings
from . import inference as I
from . import kinetics as K

OUT = REPO_ROOT / "web" / "tests" / "unit" / "fixtures" / "intent_parity.json"
T_FRAMES, W_WINDOWS = 3, 3


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()


class _Z:
    def __init__(self, d: dict):
        self._d = d
        self.files = list(d)

    def __getitem__(self, k):
        return self._d[k]


def _case(name: str, z: dict, models, labels) -> dict:
    br, cues = K.pool(_Z(z))
    res = I.infer(models, {b: (None if v is None else v[None, :].astype(np.float64)) for b, v in br.items()}, cues, labels)
    frames = None
    if str(z.get("video_state", "absent")) == "present":
        frames = []
        for i in range(len(z["v_t"])):
            present = z["v_pet"][i, 0] > 0
            frames.append({"t": float(z["v_t"][i]), "whole_b64": _b64(z["v_whole"][i].astype(np.float32)),
                           "objects_b64": _b64(z["v_objects"][i].astype(np.float32)), "pet": [float(v) for v in z["v_pet"][i]],
                           "box": [float(v) for v in z["v_box"][i]], "crop_b64": _b64(z["v_crop"][i].astype(np.float32)) if present else None,
                           "person": [float(v) for v in z["v_person"][i]], "motion": float(z["v_motion"][i])})
    audio = None
    if "a_scores" in z:
        audio = {"scores_b64": [_b64(r.astype(np.float32)) for r in z["a_scores"]], "rmsDb": [float(v) for v in z["a_rms_db"]]}
    return {"name": name, "frames": frames, "audio": audio, "videoState": str(z.get("video_state", "absent")),
            "audioState": "present" if "a_scores" in z else "absent",
            "expected": {"branches": {b: (None if v is None else _b64(v)) for b, v in br.items() if b in ("audio_scores", "animal_phone", "environment",
                                                                                                         "interaction")},
                         "species": res.species, "status": res.status, "missing_inputs": res.missing_inputs,
                         "situation": res.situation, "readings": [{"channel": r.channel, "top": r.top, "top_score": r.top_score, "scores": r.scores,
                                                                   "confident": r.confident} for r in res.readings],
                         "hypotheses": [{"name": h.name, "status": h.status, "support": h.support} for h in res.hypotheses],
                         "conflicts": res.conflicts, "qualifiers": res.qualifiers}}


def _truncate(z, t: int, w: int) -> dict:
    d = {k: z[k] for k in z.files}
    for k in [k for k in d if k.startswith("v_") and d[k].ndim >= 1 and k != "video_state"]:
        d[k] = d[k][:t]
    d.pop("a_emb", None)  # the phone computes YAMNet class scores only
    for k in ("a_t", "a_scores", "a_rms_db"):
        if k in d:
            d[k] = d[k][:w]
    # float16 storage -> the float32 values both sides read
    for k in ("v_whole", "v_crop", "v_objects", "a_scores"):
        if k in d:
            d[k] = d[k].astype(np.float32)
    return d


def main() -> None:
    from ..ondevice.bundle import _intent_assets

    s = load_settings()
    data_root = s.paths.resolve("data_root")
    from ..audio.yamnet import YamnetBackend
    from ..registry import model_dir

    labels = YamnetBackend(model_dir(s, s.audio.yamnet.model_id), 1).labels
    models = I.IntentModels(variant="phone_now")
    assets = _intent_assets(lambda rid: True)
    test = [c for c in K.list_clips(data_root) if c.split == "test"]
    d = K.feature_dir(data_root.parent)
    picks = {}
    for c in test:
        if c.label not in picks and (d / f"{c.key}.npz").exists():
            with np.load(d / f"{c.key}.npz") as z:
                if str(z["audio_state"]) == "present" and (z["v_pet"][:T_FRAMES, 0] > 0).sum() >= 2:
                    picks[c.label] = (c, _truncate(z, T_FRAMES, W_WINDOWS))
        if len(picks) == 4:
            break
    cases = []
    for lab, (c, z) in picks.items():
        cases.append(_case(f"kinetics_{lab.replace(' ', '_')}_{c.youtube_id}", z, models, labels))
    c0, z0 = next(iter(picks.values()))
    za = {k: v for k, v in z0.items() if not k.startswith("a_")}
    za["audio_state"] = np.array("absent")
    cases.append(_case("audio_absent", za, models, labels))
    zn = dict(z0)
    zn["v_pet"] = np.zeros_like(z0["v_pet"])
    cases.append(_case("no_animal_detected", zn, models, labels))
    zs = dict(z0)
    zs["a_rms_db"] = np.full_like(z0["a_rms_db"], -80.0)
    zs["audio_state"] = np.array("silent")
    cases.append(_case("silent_audio", zs, models, labels))
    # synthetic edits of real inputs, to reach the remaining code paths
    bark, meow = labels.index("Bark"), labels.index("Meow")
    for name, (lab, edit) in {"dog_bark_heard": ("grooming dog", ("score", bark)), "cat_meow_heard": ("petting cat", ("score", meow)),
                              "species_flipped": ("petting cat", ("species", None))}.items():
        if lab not in picks:
            continue
        z = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in picks[lab][1].items()}
        if edit[0] == "score":
            z["a_scores"][:, edit[1]] = 0.9
        else:
            pet = z["v_pet"][:, 0] > 0
            z["v_pet"][pet, 3] = 1.0 - z["v_pet"][pet, 3]
        cases.append(_case(name, z, models, labels))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"note": "generated by python -m petperception.intent.parity; inputs truncated to "
                                       f"{T_FRAMES} frames / {W_WINDOWS} windows", "assets": assets, "yamnet_labels": labels, "cases": cases}))
    print(OUT, OUT.stat().st_size, [c["name"] for c in cases])


if __name__ == "__main__":
    main()
