"""E33: a phone-sized student of the behaviour encoder (SigLIP base -> MobileNetV3) for on-device zero-shot behaviour recognition.

Why: behaviour recognition in the explanation system (E31, E32) runs SigLIP
base (812 MB) on the animal crops, which no phone budget allows. A student
that reproduces SigLIP's image embedding for cat and dog crops keeps the
zero-shot design (text prompts embedded once by SigLIP, offline) with a
model of a few megabytes.

Pre-registered (this text and the training script were committed before any
student score on evaluation clips was computed):

Data
* Training: square crops (scale 1.2, 224 px, as ``understand.features.square_crop``)
  of cat and dog boxes of the G1 Open Images training images (box area at
  least 0.04 of the image; no group-of or depiction boxes; CC BY 2.0 images)
  and of the 4,538 CoP3D training frames of G1b (boxes from the masks;
  CC BY-NC 4.0). Validation (model selection only): the same from G1's Open
  Images development images and 400 CoP3D development frames (seeded
  sample). Crops are stored as JPEG (quality 95) and both teacher and student
  see the decoded JPEG.
* Teacher targets: SigLIP base image embeddings of the crops (frozen).
* Evaluation: the E31 pool (278 clips with located animals: Wikimedia
  Commons, paper supplements, Internet Archive), crops recut from the videos
  with the stored version-1 boxes at the stored embedding times (the crops
  the teacher embedded in E31). No training image comes from these videos.
Student (``training/student/train_student.py``, TensorFlow 2.15)
* Keras MobileNetV3-Large (ImageNet weights, built-in preprocessing) +
  Dense(768) + L2 normalization. Loss = (1 - cosine to the teacher
  embedding) + KL(teacher || student) over the softmax of SigLIP logits
  (scale, bias) against the E31 prompts of both species. Adam 3e-4 with
  cosine decay, batch 32, 6 epochs, no augmentation; a checkpoint after
  every epoch (resumable); model selection: lowest validation loss.
* Exports: TFLite float32 (evaluated), float16 and int8 (dynamic range).
Evaluation
* R0 zero-shot macro AP over the 9 behaviours eligible in E31, with the
  frozen E31 prompts and clip score (``understand/e31``): teacher = the
  stored SigLIP embeddings (must reproduce E31's 0.416), student = the
  float32 TFLite export on the recut crops; paired difference with a
  cluster bootstrap over uploader / paper groups (1,000 resamples, seed 0).
* Criterion: the student becomes the phone behaviour-encoder candidate if it
  retains at least 80 % of the teacher's macro AP above chance:
  AP_student >= chance + 0.8 (AP_teacher - chance).
* Reported regardless: per-behaviour AP, macro AP of the float16 and int8
  exports, file sizes, desktop CPU latency per crop (Python LiteRT), and,
  descriptive, agreement of the per-clip most likely behaviour on the 37 E32
  test clips. Nothing here is a phone result.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from ..config import REPO_ROOT, Settings
from ..utils import utc_now, write_json

AREA_MIN = 0.04
COP3D_VAL = 400
SEED = 20261031
RETAIN = 0.8


def work_dir(settings: Settings) -> Path:
    return settings.paths.resolve("data_root").parent / "student"


def _crop(rgb: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    from ..understand.features import square_crop

    return square_crop(rgb, np.array(box, np.float32))


def build(settings: Settings) -> dict:
    """Write the training and validation crops (resumable: existing files are kept) and ``index.json``."""
    g1 = settings.paths.resolve("data_root").parent / "training" / "g1"
    out = work_dir(settings) / "crops"
    imgs = {r["image_id"]: r for r in csv.DictReader(gzip.open(g1 / "images.csv.gz", "rt"))}
    items = {"train": [], "val": []}
    for r in csv.DictReader(gzip.open(g1 / "boxes.csv.gz", "rt")):
        if r["label"] not in ("cat", "dog") or r["is_depiction"] == "1" or r["is_group_of"] == "1":
            continue
        box = tuple(float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
        if (box[2] - box[0]) * (box[3] - box[1]) < AREA_MIN:
            continue
        split = {"train": "train", "development": "val"}.get(imgs.get(r["image_id"], {}).get("split", ""))
        if split:
            items[split].append({"src": f"openimages:{r['image_id']}", "file": str(g1 / "images" / f"{r['image_id']}.jpg"), "box": box,
                                 "species": r["label"]})
    cop = list(csv.DictReader(open(g1 / "cop3d_frames.csv")))
    frames_dir = g1 / "images"  # the G1b copies of the CoP3D frames
    rng = np.random.default_rng(SEED)
    dev = [r for r in cop if r["split"] == "development"]
    dev = [dev[i] for i in sorted(rng.choice(len(dev), min(COP3D_VAL, len(dev)), replace=False))]
    for split, rows in (("train", [r for r in cop if r["split"] == "train"]), ("val", dev)):
        for r in rows:
            items[split].append({"src": f"cop3d:{r['file_name']}", "file": str(frames_dir / r["file_name"]),
                                 "box": tuple(float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")), "species": r["species"]})
    index = {"created_at": utc_now().isoformat(timespec="seconds"), "splits": {}}
    for split, rows in items.items():
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        kept = []
        for i, it in enumerate(rows):
            name = f"{i:06d}.jpg"
            p = d / name
            if not p.exists():
                img = cv2.imread(it["file"])
                if img is None:
                    continue
                crop = _crop(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), it["box"])
                cv2.imwrite(str(p.with_suffix(".tmp.jpg")), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                p.with_suffix(".tmp.jpg").replace(p)
            kept.append({**it, "crop": f"{split}/{name}"})
        index["splits"][split] = kept
    index["counts"] = {s: len(v) for s, v in index["splits"].items()}
    write_json(out.parent / "index.json", index)
    return index["counts"]


def e31_text(settings: Settings, encoder) -> dict:
    """SigLIP text embeddings of the frozen E31 prompts of both species (the student's KL targets and the zero-shot bank)."""
    from ..understand.e31 import vocab as V1

    texts, owners, species = [], [], []
    for sp in ("cat", "dog"):
        t, o = V1.prompts_for(sp)
        texts += t
        owners += o
        species += [sp] * len(t)
    T = encoder.texts(texts)
    out = work_dir(settings) / "text_e31.npz"
    np.savez(out, T=T.astype(np.float32), scale=np.float32(encoder.scale), bias=np.float32(encoder.bias), texts=np.array(texts),
             owners=np.array(owners), species=np.array(species))
    return {"prompts": len(texts)}


def teacher(settings: Settings, chunk: int = 512) -> dict:
    """SigLIP embeddings of every crop, in resumable chunks (``teacher/<split>_<k>.npy``)."""
    from ..understand.features import SiglipEncoder

    wd = work_dir(settings)
    index = json.loads((wd / "index.json").read_text())
    enc = SiglipEncoder(threads=settings.runtime.num_threads)
    if not (wd / "text_e31.npz").exists():
        e31_text(settings, enc)
    tdir = wd / "teacher"
    tdir.mkdir(parents=True, exist_ok=True)
    done = {}
    for split, rows in index["splits"].items():
        for k in range(0, len(rows), chunk):
            p = tdir / f"{split}_{k // chunk:04d}.npy"
            if p.exists():
                continue
            imgs = [cv2.cvtColor(cv2.imread(str(wd / "crops" / r["crop"])), cv2.COLOR_BGR2RGB) for r in rows[k:k + chunk]]
            e = enc.images(imgs, batch=32).astype(np.float16)
            np.save(p.with_suffix(".tmp.npy"), e)
            p.with_suffix(".tmp.npy").replace(p)
            print(f"{split} {k + len(imgs)}/{len(rows)}", flush=True)
        done[split] = len(rows)
    return done


# ----------------------------------------------------------------------------------------------------------------- evaluation


class TFLiteEncoder:
    """The student export as an image encoder with the SigLIP text bank's scale and bias (``ZeroShot`` interface)."""

    def __init__(self, path: Path, text_npz: Path, threads: int = 2):
        from ai_edge_litert.interpreter import Interpreter

        self.interp = Interpreter(model_path=str(path), num_threads=threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        t = np.load(text_npz)
        self.scale, self.bias = float(t["scale"]), float(t["bias"])
        self.bank = {str(x): v for x, v in zip(t["texts"], t["T"])}

    def images(self, imgs: list[np.ndarray], batch: int = 1) -> np.ndarray:
        out = []
        for im in imgs:
            self.interp.set_tensor(self.inp["index"], im[None].astype(self.inp["dtype"]))
            self.interp.invoke()
            out.append(self.interp.get_tensor(self.out["index"])[0])
        e = np.array(out, np.float32).reshape(len(out), -1)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)

    def texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.bank[t] for t in texts])


class _TextBank:
    """SigLIP text bank with SigLIP's scale and bias, for scoring stored teacher embeddings."""

    def __init__(self, text_npz: Path):
        t = np.load(text_npz)
        self.scale, self.bias = float(t["scale"]), float(t["bias"])
        self.bank = {str(x): v for x, v in zip(t["texts"], t["T"])}

    def texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.bank[x] for x in texts])


def recut(path: Path, z) -> list[np.ndarray]:
    """The crops the teacher embedded in E31: frames at ``MOTION_FPS``, every third with a box, cut with the stored box."""
    from ..understand.features import EMB_FPS, MOTION_FPS, decode, square_crop

    frames, _ = decode(path)
    box = z["box"]
    step = int(round(MOTION_FPS / EMB_FPS))
    idx = [i for i in range(0, min(len(frames), len(box)), step) if not np.isnan(box[i, 0])]
    return [square_crop(frames[i], box[i]) for i in idx]


def pool(settings: Settings):
    from ..understand import dataset as D
    from ..understand.e31 import vocab as V1
    from . import e31_behavior_questions as E31

    data_root = settings.paths.resolve("data_root")
    fdir = data_root.parent / "understand" / "features"
    out = []
    for c in D.clips(data_root):
        if c.set != "dev" or (any(d in c.title for d in E31.DUPLICATES) and c.source == "commons"):
            continue
        f = fdir / f"{c.key}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        if "emb" not in z.files or len(z["emb"]) < E31.MIN_CROPS:
            continue
        labels = set()
        for layer in ("source", "expert"):
            for lab in c.labels.get(layer, []):
                if lab in E31.UNMAPPED:
                    continue
                lab = E31.SOURCE_MAP.get(lab, lab)
                if lab in V1.BEHAVIORS and c.species in V1.BEHAVIORS[lab].species:
                    labels.add(lab)
        out.append((c, z, labels))
    return out


def macro_scores(rows, embs, bank_encoder, eligible) -> tuple[np.ndarray, list[str]]:
    from ..understand.e31 import recognize as R1

    zsm = R1.ZeroShot(bank_encoder)
    S = np.zeros((len(rows), len(eligible)))
    for i, ((c, _, _), e) in enumerate(zip(rows, embs)):
        sc = R1.zero_shot_scores(e, c.species, zsm)
        S[i] = [sc.get(b, 0.0) for b in eligible]
    return S, eligible


class _Z(dict):
    @property
    def files(self):
        return list(self.keys())


def evaluate(settings: Settings, models: dict[str, Path], out_dir: Path) -> dict:
    from . import e31_behavior_questions as E31

    wd = work_dir(settings)
    e31 = json.loads((REPO_ROOT / "results" / "e31" / "metrics.json").read_text())
    eligible = e31["part_a"]["eligible"]
    rows = pool(settings)
    groups = np.array([c.group for c, _, _ in rows])
    Y = {b: np.array([b in lab for _, _, lab in rows], int) for b in eligible}
    bank = _TextBank(wd / "text_e31.npz")
    teacher_emb = [_Z(emb=z["emb"].astype(np.float32)) for _, z, _ in rows]
    St, _ = macro_scores(rows, teacher_emb, bank, eligible)
    encs = {name: TFLiteEncoder(p, wd / "text_e31.npz", settings.runtime.num_threads) for name, p in models.items()}
    student_emb = {name: [] for name in encs}
    check = []
    for i, (c, z, _) in enumerate(rows):
        crops = recut(c.path, z)
        if len(crops) != len(z["emb"]):
            check.append({"clip": c.key, "recut": len(crops), "stored": int(len(z["emb"]))})
        for name, enc in encs.items():
            student_emb[name].append(_Z(emb=enc.images(crops)))
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)} clips", flush=True)

    def macro(S, idx=None):
        idx = np.arange(len(rows)) if idx is None else idx
        v = [E31.ap(Y[b][idx], S[idx, j]) for j, b in enumerate(eligible)]
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    chance = float(np.mean([Y[b].mean() for b in eligible]))
    res = {"experiment": "E33", "created_at": utc_now().isoformat(timespec="seconds"), "clips": len(rows), "groups": int(len(set(groups))),
           "eligible": eligible, "chance_macro_ap": chance, "recut_count_mismatches": check,
           "teacher": {"macro_ap": macro(St), "macro_ap_ci95": E31.cluster_boot(groups, lambda idx: macro(St, idx)),
                       "per_behavior_ap": {b: E31.ap(Y[b], St[:, j]) for j, b in enumerate(eligible)}}, "students": {}}
    for name in encs:
        Ss, _ = macro_scores(rows, student_emb[name], encs[name], eligible)
        d = macro(Ss) - macro(St)
        top_agree = float(np.mean(Ss.argmax(1) == St.argmax(1)))
        res["students"][name] = {"model": str(models[name]), "bytes": Path(models[name]).stat().st_size, "macro_ap": macro(Ss),
                                 "macro_ap_ci95": E31.cluster_boot(groups, lambda idx, Ss=Ss: macro(Ss, idx)),
                                 "difference_to_teacher": d, "difference_ci95": E31.cluster_boot(groups, lambda idx, Ss=Ss: macro(Ss, idx) - macro(St, idx)),
                                 "per_behavior_ap": {b: E31.ap(Y[b], Ss[:, j]) for j, b in enumerate(eligible)},
                                 "clip_top_behavior_agreement": top_agree}
    # secondary, descriptive: the 37 E32 test clips (never used for any choice here)
    from ..understand import dataset as D
    from ..understand.e31 import recognize as R1

    bench = json.loads((REPO_ROOT / "benchmarks" / "behavior_questions_v2.json").read_text())["questions"]
    cl = {c.key: c for c in D.clips(settings.paths.resolve("data_root"))}
    fdir = settings.paths.resolve("data_root").parent / "understand" / "features"
    agree = {name: [] for name in encs}
    zt = R1.ZeroShot(bank)
    for it in bench:
        z = np.load(fdir / f"{it['clip']}.npz", allow_pickle=False)
        if "emb" not in z.files or len(z["emb"]) < E31.MIN_CROPS:
            continue
        st = R1.zero_shot_scores(_Z(emb=z["emb"].astype(np.float32)), it["species"], zt)
        crops = recut(cl[it["clip"]].path, z)
        for name, enc in encs.items():
            ss = R1.zero_shot_scores(_Z(emb=enc.images(crops)), it["species"], R1.ZeroShot(enc))
            agree[name].append(max(st, key=st.get) == max(ss, key=ss.get))
    res["e32_clips_top_behavior_agreement"] = {name: {"agree": int(sum(v)), "n": len(v)} for name, v in agree.items()}
    need = chance + RETAIN * (res["teacher"]["macro_ap"] - chance)
    prim = res["students"].get("float32")
    res["criterion"] = {"rule": "AP_student >= chance + 0.8 (AP_teacher - chance), float32 export", "required": need,
                        "met": bool(prim and prim["macro_ap"] >= need)}
    write_json(out_dir / "metrics.json", res)
    return res


def latency(models: dict[str, Path], threads=(1, 2), n: int = 50) -> dict:
    import time

    from ai_edge_litert.interpreter import Interpreter

    x = (np.random.default_rng(0).random((1, 224, 224, 3)) * 255).astype(np.float32)
    out = {}
    for name, p in models.items():
        for th in threads:
            it = Interpreter(model_path=str(p), num_threads=th)
            it.allocate_tensors()
            i = it.get_input_details()[0]
            for _ in range(5):
                it.set_tensor(i["index"], x.astype(i["dtype"]))
                it.invoke()
            t0 = time.perf_counter()
            for _ in range(n):
                it.set_tensor(i["index"], x.astype(i["dtype"]))
                it.invoke()
            out[f"{name}_threads{th}_ms"] = round((time.perf_counter() - t0) / n * 1000, 2)
    return out


def main() -> None:
    import argparse

    from ..config import load_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "teacher", "evaluate"])
    ap.add_argument("--export-dir", default="models/student/e33_mnv3l")
    ap.add_argument("--threads", type=int, help="CPU threads (default: configuration)")
    a = ap.parse_args()
    s = load_settings()
    if a.threads:
        s.runtime.num_threads = a.threads
    if a.step == "build":
        print(build(s))
    elif a.step == "teacher":
        print(teacher(s))
    else:
        d = REPO_ROOT / a.export_dir
        models = {k: d / f"student_{k}.tflite" for k in ("float32", "float16", "int8") if (d / f"student_{k}.tflite").exists()}
        out = s.paths.resolve("results_root") / "e33"
        r = evaluate(s, models, out)
        r["latency_desktop_cpu"] = latency(models)
        write_json(out / "metrics.json", r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("recut_count_mismatches",)}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
