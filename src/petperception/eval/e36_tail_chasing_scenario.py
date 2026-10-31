"""E36: the priority scenario - "my cat keeps circling while chasing or biting its tail; why?" - on every usable real video.

Protocol fixed before any E36 score was computed (this docstring and the scenario manifest were committed first).

Data (``dataset`` writes ``benchmarks/tail_chasing_scenario.json``)
* All clips with features: the E31 collection (Wikimedia Commons, paper supplements, Internet Archive) and the E32 clips
  (paper supplements, Mixkit, Internet Archive). Research-permitted terms per clip in the manifests.
* Positive: the assistant frame review saw tail chasing / circling after the tail, with coarse bouts (machine-assisted
  temporal labels, about +-1 s; the uploader's or authors' label agrees in every case).
* Excluded: a human label says tail chasing but the frame review could not verify it, or the file duplicates another.
* Hard negative: frame-reviewed clip without tail chasing (ordinary turning or running, grooming, toy play, tail
  flicking, a kitten biting another cat's tail, neurological head turning, ...). Negative: every other clip.
* Groups: uploader or paper (all clips of one source together, so one animal never sits on both sides of a split).
  Duplicates: identical file hashes, and pairs of clips whose mean crop embeddings have cosine >= 0.97.
Acquisition beyond these sources was attempted and is logged in the manifest (``acquisition_log``).

Evidence per 2-s window (hop 0.5 s, 6 fps track):
* track dynamics (``understand/temporal.py``: staying in place, box-shape oscillation, motion flips, missed detections);
* pose (reused E24 AwA-Pose cat+dog keypoint head on EfficientNet-Lite0, both already in this repository): nose to
  tail-end and nose to tail-start distance relative to body length, body-axis rotation, keypoint visibility;
* SigLIP zero-shot circling / neutral probabilities (E31 prompts);
* X-CLIP (microsoft/xclip-base-patch32, Kinetics-400 video-text model, pinned) zero-shot probability of
  "chasing its tail" against ordinary pet actions on 8-frame windows (2 s).

Methods (clip score = mean of the top quarter of window scores):
* pretrained, untrained here: B1 SigLIP zero-shot; B2 X-CLIP zero-shot;
* adapted on this data (logistic regression, C = 0.1, balanced classes; window labels from the bouts: inside = 1,
  outside every bout = 0, partial overlap ignored; negatives' windows 0): A1 track + SigLIP (E32 features);
  P phone route = track + pose only (detector and backbone already in the phone bundle); F = track + pose + SigLIP +
  X-CLIP.
Evaluation: leave-one-group-out (every group held out once; adapted models trained on all other groups); the clip
threshold of every method is the best-F1 threshold on the training groups' clips, applied to the held-out group.
Reported: clip AP; precision, recall, false positives (all negatives and hard negatives apart) with Wilson intervals;
temporal agreement of predicted bouts with the frame-review bouts on detected positives; usable-answer coverage
(positive clips where tail chasing is reported with a bout overlapping a reference bout; negatives without a false
tail-chasing claim). A method counts as reliable only if, with at least 10 positive clips, out-of-fold recall and
precision are both at least 0.8; otherwise results are indicative.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT, Settings
from ..utils import utc_now, write_json

SCENARIO = REPO_ROOT / "benchmarks" / "tail_chasing_scenario.json"
UNVERIFIED_POSITIVE = {"commons_7f743aa53c22": "uploader title 'Outside234 5' (spinning dog); frame review could not verify tail chasing",
                       "commons_edba93e378de": "label parsed from a description about the Crab Nebula; not a real dog clip"}
DUPLICATE_OF = {"commons_d8737cf1d9c3": "paper_tiira2012_s2"}
ACQUISITION_LOG = [
    {"source": "Wikimedia Commons search (30 terms in 9 languages, 2026-10-31)", "result": "159 video hits, 149 not yet in the collection, none showing tail chasing (space probes, troops, spiders, Wikipedia campaign videos)"},
    {"source": "Internet Archive advanced search (19 phrases)", "result": "40 items; the tail-chasing ones are already in the collection (3 cats, 1 dog 'spinning' that is fetching); the rest unrelated or cartoons"},
    {"source": "Flickr (CC-licensed video search)", "result": "CC BY / BY-SA tail-chasing videos listed, but the video host live.staticflickr.com answers 403/429 through this environment's egress proxy"},
    {"source": "Pexels, Pixabay, Vimeo, Videvo", "result": "HTTP 403 from this environment"},
    {"source": "YouTube (Creative Commons filter)", "result": "search reachable; downloads refused ('confirm you are not a bot')"},
    {"source": "Mixkit, Coverr (free stock)", "result": "reachable; no tail-chasing clip in their pet catalogues"},
    {"source": "paper supplements", "result": "already collected: Tiira 2012 (3 dogs), Vet Res Commun 2026 (1 cat), PLoS ONE 2020 sensory-impaired dog"},
]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_dataset(settings: Settings) -> dict:
    from ..understand import dataset as D

    data_root = settings.paths.resolve("data_root")
    fdir = data_root.parent / "understand" / "features_v2"
    ann = D.load_assistant()
    rows, embs, shas = [], {}, {}
    for c in D.clips(data_root):
        f = fdir / f"{c.key}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        a = ann.get(c.key)
        bouts = [{"t0": b["t0"], "t1": b["t1"]} for b in (a or {}).get("bouts", []) if b["behavior"] == "tail_chasing"]
        if c.key in DUPLICATE_OF:
            label, why = "excluded", f"duplicate of {DUPLICATE_OF[c.key]}"
        elif c.key in UNVERIFIED_POSITIVE:
            label, why = "excluded", UNVERIFIED_POSITIVE[c.key]
        elif bouts:
            label, why = "positive", "frame review: tail chasing with bouts"
        elif a is not None:
            label, why = "hard_negative", "frame-reviewed: " + (", ".join(a.get("behaviors", [])) or "no listed behaviour")
        else:
            label, why = "negative", "no tail-chasing label from source or review"
        shas[c.key] = _sha(c.path)
        if "emb" in z.files and len(z["emb"]):
            e = z["emb"].astype(np.float32).mean(0)
            embs[c.key] = e / np.linalg.norm(e)
        rows.append({"clip": c.key, "species": c.species, "source": c.source, "group": c.group, "set": c.set, "licence": c.licence,
                     "title": c.title[:120], "label": label, "why": why, "bouts": bouts,
                     "temporal_label_uncertainty_s": 1.0 if bouts else None, "label_basis": "assistant frame review (machine-assisted)" if a else "source labels"})
    # duplicates by hash and by near-identical content
    by_sha = Counter(shas.values())
    keys = list(embs)
    E = np.stack([embs[k] for k in keys]) if keys else np.zeros((0, 768))
    near = []
    sim = E @ E.T if len(E) else np.zeros((0, 0))
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if sim[i, j] >= 0.97:
                near.append([keys[i], keys[j], round(float(sim[i, j]), 3)])
    for r in rows:
        r["sha256"] = shas[r["clip"]]
        if by_sha[shas[r["clip"]]] > 1 and r["label"] != "excluded":
            r["duplicate_hash"] = True
    count = Counter(r["label"] for r in rows)
    pos = [r for r in rows if r["label"] == "positive"]
    out = {"version": "tail-chasing-scenario-v1", "created_at": utc_now().isoformat(timespec="seconds"), "counts": dict(count),
           "positives": {"clips": len(pos), "groups": len({r["group"] for r in pos}), "cats": sum(r["species"] == "cat" for r in pos),
                         "dogs": sum(r["species"] == "dog" for r in pos), "cat_groups": len({r["group"] for r in pos if r["species"] == "cat"})},
           "near_duplicate_pairs": near, "acquisition_log": ACQUISITION_LOG, "clips": rows}
    write_json(SCENARIO, out)
    return {k: out[k] for k in ("counts", "positives", "near_duplicate_pairs")}


# ------------------------------------------------------------------------------------------------ cached evidence
POSE_KEYS = ("nose", "tail_start", "tail_end", "withers")


def pose_cache(settings: Settings, keys: list[str]) -> int:
    """AwA-Pose cat+dog head on the stored track at 2 fps -> var/understand/pose/<clip>.npz (resumable)."""
    from ..registry import check_model
    from ..understand import dataset as D
    from ..understand.features import decode
    from ..vision.keypoints import DogKeypointModel

    data_root = settings.paths.resolve("data_root")
    out = data_root.parent / "understand" / "pose"
    out.mkdir(parents=True, exist_ok=True)
    bb = Path(next(iter(check_model(settings, "efficientnet_lite0_mediapipe_v1").files.values())))
    model = DogKeypointModel.load(bb, REPO_ROOT / "artifacts" / "keypoints" / "catdog_keypoints_head_awa_v1.npz", settings.runtime.num_threads)
    names = json.loads((REPO_ROOT / "artifacts" / "keypoints" / "catdog_keypoints_head_awa_v1.json").read_text())["keypoints"]
    sel = [names.index(k) for k in POSE_KEYS]
    clips = {c.key: c for c in D.clips(data_root)}
    n = 0
    for k in keys:
        p = out / f"{k}.npz"
        if p.exists():
            continue
        z = np.load(data_root.parent / "understand" / "features_v2" / f"{k}.npz", allow_pickle=False)
        frames, t = decode(clips[k].path, fps=6.0)
        box = z["box"] if "box" in z.files else np.zeros((0, 4))
        idx = [i for i in range(0, min(len(frames), len(box)), 3) if not np.isnan(box[i, 0])]
        xy = np.full((len(idx), len(sel), 2), np.nan, np.float32)
        vis = np.zeros((len(idx), len(sel)), np.float32)
        for j, i in enumerate(idx):
            kp, _, v = model.predict(frames[i], box[i])
            xy[j], vis[j] = kp[sel], v[sel]
        np.savez(p.with_suffix(".tmp.npz"), t=np.array([t[i] for i in idx], np.float32), xy=xy, vis=vis, box=box[idx] if idx else np.zeros((0, 4)))
        p.with_suffix(".tmp.npz").replace(p)
        n += 1
        if n % 25 == 0:
            print(f"pose {n}", flush=True)
    return n


XCLIP_PROMPTS = {"tail": ["a {s} chasing its own tail", "a {s} spinning in circles after its tail", "a {s} biting its tail while turning"],
                 "other": ["a {s} walking", "a {s} running", "a {s} sitting still", "a {s} lying down", "a {s} grooming itself",
                           "a {s} playing with a toy", "a {s} eating", "a {s} being petted", "a {s} playing with another animal",
                           "a {s} jumping", "a {s} looking around"]}


def xclip_cache(settings: Settings, keys: list[str], hop_s: float = 1.0) -> int:
    """X-CLIP zero-shot on 8-frame windows (2 s at 4 fps), hop 1 s -> var/understand/xclip/<clip>.npz (resumable)."""
    import torch
    from transformers import XCLIPModel, XCLIPProcessor

    from ..registry import model_dir
    from ..understand import dataset as D
    from ..understand.features import decode

    data_root = settings.paths.resolve("data_root")
    out = data_root.parent / "understand" / "xclip"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(settings.runtime.num_threads)
    md = model_dir(settings, "xclip_base_patch32")
    model, proc = XCLIPModel.from_pretrained(str(md)).eval(), XCLIPProcessor.from_pretrained(str(md))
    clips = {c.key: c for c in D.clips(data_root)}
    n = 0
    for k in keys:
        p = out / f"{k}.npz"
        if p.exists():
            continue
        c = clips[k]
        s = "cat" if c.species == "cat" else "dog"
        texts = [x.format(s=s) for x in XCLIP_PROMPTS["tail"]] + [x.format(s=s) for x in XCLIP_PROMPTS["other"]]
        frames, t = decode(c.path, fps=4.0, max_side=320)
        starts, probs = [], []
        step = int(round(hop_s * 4))
        with torch.no_grad():
            for st in range(0, max(1, len(frames) - 7), step):
                win = frames[st:st + 8]
                if len(win) < 8:
                    break
                x = proc(text=texts, videos=list(win), return_tensors="pt", padding=True)
                lp = model(**x).logits_per_video[0].softmax(-1).numpy()
                probs.append(lp)
                starts.append(t[st])
        np.savez(p.with_suffix(".tmp.npz"), t=np.array(starts, np.float32), p=np.array(probs, np.float32).reshape(len(starts), len(texts)),
                 n_tail=np.array(len(XCLIP_PROMPTS["tail"])))
        p.with_suffix(".tmp.npz").replace(p)
        n += 1
        if n % 10 == 0:
            print(f"xclip {n}", flush=True)
    return n


def main() -> None:
    import argparse

    from ..config import load_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["dataset", "pose", "xclip", "evaluate"])
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    s = load_settings()
    if a.threads:
        s.runtime.num_threads = a.threads
    if a.step == "dataset":
        print(json.dumps(build_dataset(s), indent=1))
        return
    keys = [r["clip"] for r in json.loads(SCENARIO.read_text())["clips"] if r["label"] != "excluded"]
    if a.step == "pose":
        print(pose_cache(s, keys))
    elif a.step == "xclip":
        # positives and hard negatives first, so a partial cache already covers the informative clips
        sc = {r["clip"]: r["label"] for r in json.loads(SCENARIO.read_text())["clips"]}
        print(xclip_cache(s, sorted(keys, key=lambda k: {"positive": 0, "hard_negative": 1}.get(sc[k], 2))))
    else:
        from .e36_evaluate import evaluate

        r = evaluate(s, s.paths.resolve("results_root") / "e36")
        print(json.dumps(r["summary"], indent=1))


if __name__ == "__main__":
    main()
