"""Kinetics-700 pet clips as synchronized audio-visual data for situation (recorded context) recognition.

Splits (fixed before any model was fitted):
* train: the 600 Kinetics *train*-split clips already on disk (train_sample,
  clips 1-50 of each class, and train_holdout, clips 51-100; both fetched for
  E19/E19b, which tuned only the detector operating point);
* test: the 294 Kinetics *validation*-split clips (E18). No test clip shares a
  YouTube id with a training clip (checked at fetch time and again here).
* never read: train_final (clips 101-160), sealed for E23.

Unit of grouping: the YouTube id (one id = one upload; the uploader is not
published, so near-duplicate scenes across ids are audited by embedding
similarity instead).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data import kinetics_pets as KP

SPLIT_DIRS = {"train": ("train_sample", "train_holdout"), "test": ("clips",)}
SEALED = (KP.FINAL_SUBDIR,)


@dataclass(frozen=True)
class Clip:
    key: str
    path: Path
    split: str
    source_dir: str
    label: str
    species: str
    person_contact: bool
    youtube_id: str


def youtube_id(stem: str) -> str:
    # Kinetics file names: <11-char id>_<start 6 digits>_<end 6 digits>
    parts = stem.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else stem


def list_clips(data_root: Path) -> list[Clip]:
    root = KP.root(data_root)
    out: list[Clip] = []
    for split, dirs in SPLIT_DIRS.items():
        for d in dirs:
            if d in SEALED:
                raise RuntimeError("sealed final set must not be listed")
            for label, (_, species, contact) in KP.CLASSES.items():
                for p in sorted((root / d / label).glob("*.mp4")):
                    key = f"{d}__{label.replace(' ', '_')}__{p.stem}"
                    out.append(Clip(key, p, split, d, label, species, contact, youtube_id(p.stem)))
    tr = {c.youtube_id for c in out if c.split == "train"}
    te = {c.youtube_id for c in out if c.split == "test"}
    if tr & te:
        raise RuntimeError(f"YouTube ids shared by train and test: {sorted(tr & te)[:5]}")
    return out


def feature_dir(var_root: Path, variant: str = "clean") -> Path:
    return var_root / "intent" / "kinetics_av" / variant


BRANCHES = ("audio", "animal", "environment", "interaction")
FPS = 2.0


def pool(z) -> tuple[dict[str, np.ndarray | None], dict]:
    """Clip-level branch vectors from per-frame/per-window features, plus interpretable cues.

    Branches (None = input missing, with the reason in the cue dict):
    * audio: YAMNet embedding mean | max (2048); audio_scores: YAMNet class scores mean | max (1042, phone-ready today)
    * animal: crop embedding mean | max over frames with an animal, posture scores, keypoint (x, y, visibility) means,
      box size / shape / speed statistics, detector and verifier scores, share of frames with an animal (2644)
    * animal_phone: the animal branch without posture and keypoints (2569): what the phone bundle computes today
      (the keypoint head is research-only; posture is not ported to the web engine)
    * environment: whole-frame embedding mean | COCO object best scores (1357)
    * interaction: person presence, interaction-model scores, person-animal IoU, global frame motion (8)
    """
    b: dict[str, np.ndarray | None] = dict.fromkeys(BRANCHES + ("audio_scores", "animal_phone"))
    cues: dict = {"missing": {}}
    a_state = str(z["audio_state"]) if "audio_state" in z.files else "absent"
    if a_state == "present":
        s = z["a_scores"].astype(np.float32)
        b["audio_scores"] = np.concatenate([s.mean(0), s.max(0)])
        cues["yamnet_max"] = s.max(0)
        if "a_emb" in z.files:
            e = z["a_emb"].astype(np.float32)
            b["audio"] = np.concatenate([e.mean(0), e.max(0)])
        else:
            cues["missing"]["audio"] = "YAMNet embedding not computed (phone_now variant)"
    else:
        cues["missing"]["audio"] = f"audio {a_state}"
    v_state = str(z["video_state"]) if "video_state" in z.files else "absent"
    if v_state != "present":
        for k in ("animal", "animal_phone", "environment", "interaction"):
            cues["missing"][k] = f"video {v_state}"
        return b, cues
    pet = z["v_pet"][:, 0] > 0
    T = len(pet)
    obj = z["v_objects"].astype(np.float32)
    b["environment"] = np.concatenate([z["v_whole"].astype(np.float32).mean(0), obj.max(0)])
    cues["object_frames"] = (obj > 0).sum(0)
    per, mot = z["v_person"], z["v_motion"]
    b["interaction"] = np.array([(per[:, 0] > 0).mean(), per[:, 0].mean(), per[:, 2].max(), per[pet, 2].mean() if pet.any() else 0.0,
                                 per[:, 3].max(), per[pet, 3].mean() if pet.any() else 0.0, float(mot[1:].mean()) if T > 1 else 0.0,
                                 float(np.percentile(mot[1:], 90)) if T > 1 else 0.0], np.float32)
    cues["pet_frames"] = int(pet.sum())
    cues["frames"] = T
    cues["person_frames"] = int((per[:, 0] > 0).sum())
    cues["interaction_max"] = float(per[:, 2].max()) if T else 0.0
    if not pet.any():
        cues["missing"]["animal"] = cues["missing"]["animal_phone"] = "no cat or dog detected in any sampled frame"
        cues["dog_fraction"] = None
        return b, cues
    crop = z["v_crop"][pet].astype(np.float32)
    box, pv = z["v_box"][pet], z["v_pet"][pet]
    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    ctr = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], 1)
    t = z["v_t"][pet]
    sp = np.linalg.norm(np.diff(ctr, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3) if len(t) > 1 else np.zeros(1)
    ver = pv[:, 2][pv[:, 2] >= 0]
    stats = np.array([float((w * h).mean()), float((w * h).std()), float(np.log(np.maximum(w, 1e-4) / np.maximum(h, 1e-4)).mean()),
                      float(np.median(sp)), float(np.percentile(sp, 90)), float(pv[:, 1].mean()), float(ver.mean()) if ver.size else 0.5,
                      float(pv[:, 3].mean())], np.float32)
    kp = z["v_kp"][pet].reshape(len(crop), -1).mean(0)
    b["animal"] = np.concatenate([crop.mean(0), crop.max(0), z["v_posture"][pet].mean(0), kp, stats, [pet.mean()]]).astype(np.float32)
    b["animal_phone"] = np.concatenate([crop.mean(0), crop.max(0), stats, [pet.mean()]]).astype(np.float32)
    cues["dog_fraction"] = float(pv[:, 3].mean())
    cues["posture_mean"] = z["v_posture"][pet].mean(0).tolist()
    return b, cues


def load_pooled(clips: list[Clip], var_root: Path, variant: str = "clean") -> tuple[list[dict], list[str]]:
    """-> rows (clip, branches, cues) for clips whose cache file exists, and the keys that are missing."""
    d = feature_dir(var_root, variant)
    rows, missing = [], []
    for c in clips:
        p = d / f"{c.key}.npz"
        if not p.exists():
            missing.append(c.key)
            continue
        with np.load(p, allow_pickle=False) as z:
            br, cues = pool(z)
        rows.append({"clip": c, "branches": br, "cues": cues})
    return rows, missing


VARIANTS = ("clean", "audio_snr10", "audio_snr0", "video_poor")


def main(argv: list[str] | None = None) -> None:
    import argparse

    from ..config import load_settings
    from . import av_features as AF
    from . import robustness as RB

    ap = argparse.ArgumentParser(description="Extract synchronized audio-visual features for the Kinetics pet clips (resumable).")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    ap.add_argument("--variant", choices=VARIANTS, default="clean",
                    help="degraded copies for the robustness conditions (audio-only or video-only features)")
    a = ap.parse_args(argv)
    s = load_settings()
    data_root = s.paths.resolve("data_root")
    clips = [c for c in list_clips(data_root) if a.split in ("all", c.split)]
    comp = AF.load_components(a.threads, with_audio=a.variant != "video_poor")
    out = feature_dir(data_root.parent, a.variant)
    if a.variant == "clean":
        AF.extract_many(comp, [(c.key, c.path) for c in clips], out)
        return
    import time

    out.mkdir(parents=True, exist_ok=True)
    pool = RB.noise_pool(data_root / "esc50") if a.variant.startswith("audio") else None
    t0 = time.perf_counter()
    todo = [c for c in clips if not (out / f"{c.key}.npz").exists()]
    for n, c in enumerate(todo, 1):
        if a.variant.startswith("audio_snr"):
            f = AF.extract_clip(comp, c.path, video=False, audio_fn=RB.audio_noise_fn(c.key, float(a.variant.removeprefix("audio_snr")), pool))
        else:
            f = AF.extract_clip(comp, c.path, audio=False, frame_fn=RB.poor_visual_fn(c.key))
        np.savez(out / f"{c.key}.tmp.npz", **f)
        (out / f"{c.key}.tmp.npz").replace(out / f"{c.key}.npz")
        if n % 25 == 0 or n == len(todo):
            print(f"{a.variant}: {n}/{len(todo)} clips, {(time.perf_counter() - t0) / n:.1f} s/clip", flush=True)


if __name__ == "__main__":
    main()
