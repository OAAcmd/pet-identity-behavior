"""E26: dog vocal context from barks: the dog's activity (Barkopedia Activity and Environment, MIT).

Data. ArlingtonCL2/Barkopedia_Dog_Act_Env at revision 60a721a7 (Hugging Face;
the archives match the published LFS SHA-256). 12,480 training clips and 499
validation clips, each labelled with the activity the dog was engaged in:
rest, alerting to sounds, seeking attention, playing with a human, another
animal or a toy, begging for food, taking a shower. A vision-language model
inferred the labels from the clips' videos, and people then verified them
(dataset card). Label spelling is normalised (case). The labels describe the
situation, not the dog's state.

Protocol, fixed before the validation clips were scored:
* Features: YAMNet embeddings (mean and max over windows; the variant the
  phone could run once the bundle's YAMNet exposes its embedding) and MFCC
  statistics (E1's feature set).
* No dog or video ids are published. Near-duplicate clips (cosine >= 0.95
  of mean YAMNet embeddings) are linked into pseudo-groups, so selection
  cross-validates without splitting near-duplicates.
* Selection: logistic regression with C chosen by 5-fold cross-validation
  grouped by pseudo-group, on the training clips only.
* Test, once: the provider's validation set. 95 % CIs bootstrap validation
  pseudo-groups.
* Leakage audit: each validation clip's highest cosine similarity to any
  training clip. Results are repeated on the clips whose maximum is below
  0.95.
* Gate (as for vocalization types, docs/TRAINING_PROGRAM.md G4): an
  activity could serve as evidence only with F1 >= 0.7 and a lower bound
  >= 0.6. Nothing is shipped by this experiment.
"""

from __future__ import annotations

import csv
import hashlib
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import linear_model
from ..audio.features import FeatureConfig, extract
from ..config import Settings
from ..ingest.media import AudioTrack
from ..utils import utc_now, write_json
from . import metrics as M

REVISION = "60a721a7f9477634971f6b3df29dc2a5e56994d8"
ARCHIVES = {"train": ["split1.zip", "split2.zip"], "validation": ["valid_audios.zip"]}
LABELS_CSV = {"train": "train_label.csv", "validation": "valid_labels.csv"}
DUP = 0.95
GATE = {"f1": 0.7, "ci_low": 0.6}


def root(data_root: Path) -> Path:
    return data_root / "barkopedia_act_env"


def norm(label: str) -> str:
    return " ".join(label.strip().lower().split())


def load_split(data_root: Path, split: str) -> list[tuple[str, np.ndarray, int, str]]:
    r = root(data_root)
    labels = {row["audio_id"]: norm(row["activity"]) for row in csv.DictReader(open(r / LABELS_CSV[split], newline="", encoding="utf-8"))}
    out = []
    for a in ARCHIVES[split]:
        with zipfile.ZipFile(r / a) as z:
            for n in sorted(z.namelist()):
                stem = Path(n).stem
                if n.endswith(".wav") and "__MACOSX" not in n and stem in labels:
                    x, sr = sf.read(io.BytesIO(z.read(n)), dtype="float32", always_2d=False)
                    out.append((stem, x.mean(axis=1) if x.ndim > 1 else x, sr, labels[stem]))
    return out


def features(settings: Settings, clips: list, cache: Path) -> dict[str, np.ndarray]:
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}
    from ..audio.yamnet import YamnetBackend
    from ..registry import model_dir

    be = YamnetBackend(model_dir(settings, settings.audio.yamnet.model_id), settings.runtime.num_threads, with_embeddings=True)
    fc = FeatureConfig(sample_rate=16000)
    E, X, dur = [], [], []
    for _, x, sr, _ in clips:
        ws = be.score_track(AudioTrack("present", x, sr, 0.0), hop_s=settings.audio.yamnet.hop_s)
        E.append(np.concatenate([ws.embeddings.mean(0), ws.embeddings.max(0)]))
        X.append(extract(x, sr, fc))
        dur.append(len(x) / sr)
    out = {"E": np.array(E, np.float32), "X": np.array(X, np.float32), "duration_s": np.array(dur), "id": np.array([c[0] for c in clips]),
           "y": np.array([c[3] for c in clips])}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def pseudo_groups(E: np.ndarray, thr: float = DUP) -> np.ndarray:
    """Connected components of the cosine >= thr graph on mean embeddings (union-find)."""
    m = E[:, : E.shape[1] // 2]
    m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    parent = list(range(len(m)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(0, len(m), 1024):
        sim = m[s:s + 1024] @ m.T
        for i, j in zip(*np.where(sim >= thr)):
            a, b = find(s + i), find(j)
            if a != b:
                parent[a] = b
    return np.array([f"g{find(i)}" for i in range(len(m))])


def max_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = A[:, : A.shape[1] // 2]
    b = B[:, : B.shape[1] // 2]
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return np.max(a @ b.T, axis=1)


def run(settings: Settings, out_dir: Path, n_boot: int = 1000, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    data_root = settings.paths.resolve("data_root")
    tr = features(settings, load_split(data_root, "train") if not (root(data_root) / "features_train.npz").exists() else [], root(data_root) / "features_train.npz")
    va = features(settings, load_split(data_root, "validation") if not (root(data_root) / "features_validation.npz").exists() else [],
                  root(data_root) / "features_validation.npz")
    labels = sorted(set(tr["y"].tolist()))
    g_tr, g_va = pseudo_groups(tr["E"]), pseudo_groups(va["E"])
    mc = max_cos(va["E"], tr["E"])
    res: dict = {"labels": labels, "methods": {}}
    for name, key in (("yamnet_embedding_logreg", "E"), ("mfcc_logreg", "X")):
        model, info = linear_model.fit(tr[key], tr["y"], g_tr, labels, seed=seed, inner_splits=5)
        pred = np.array(model.predict(va[key]))
        y = va["y"]
        rep = M.classification_report(y.tolist(), pred.tolist(), labels)
        rep["macro_f1_ci95_by_pseudo_group"] = M.group_bootstrap(lambda ix: M.macro_f1(y[ix].tolist(), pred[ix].tolist(), labels), g_va.tolist(), n_boot, seed)
        per_ci = {}
        for c in labels:
            def f1c(ix, c=c):
                t, p = y[ix] == c, pred[ix] == c
                return float(2 * np.sum(t & p) / max(1, 2 * np.sum(t & p) + np.sum(~t & p) + np.sum(t & ~p)))
            ci = M.group_bootstrap(f1c, g_va.tolist(), n_boot, seed)
            f = rep["per_class"][c]["f1"]
            per_ci[c] = {"f1": f, "ci95": [ci["ci_low"], ci["ci_high"]], "passes_gate": bool(f >= GATE["f1"] and ci["ci_low"] >= GATE["ci_low"])}
        rep["per_class_gate"] = per_ci
        far = mc < DUP
        rep["without_near_duplicates"] = {"clips": int(far.sum()), "macro_f1": M.macro_f1(y[far].tolist(), pred[far].tolist(), labels)}
        rep["selection"] = info
        res["methods"][name] = rep
    vals, cnts = np.unique(tr["y"], return_counts=True)
    res["majority_macro_f1"] = M.macro_f1(va["y"].tolist(), [vals[np.argmax(cnts)]] * len(va["y"]), labels)
    rng = np.random.default_rng(seed)
    res["stratified_random_macro_f1"] = float(np.mean([M.macro_f1(va["y"].tolist(), rng.choice(vals, len(va["y"]), p=cnts / cnts.sum()).tolist(), labels)
                                                        for _ in range(200)]))
    res["leakage_audit"] = {"validation_max_cosine_to_train": {"median": float(np.median(mc)), ">=0.95": int((mc >= 0.95).sum()), ">=0.98": int((mc >= 0.98).sum()),
                                                               ">=0.99": int((mc >= 0.99).sum()), "clips": len(mc)},
                            "train_pseudo_groups": len(set(g_tr.tolist())), "largest_train_group": int(max(np.unique(g_tr, return_counts=True)[1])),
                            "validation_pseudo_groups": len(set(g_va.tolist()))}
    res["data"] = {"dataset": f"ArlingtonCL2/Barkopedia_Dog_Act_Env @ {REVISION} (MIT)", "train_clips": len(tr["y"]), "validation_clips": len(va["y"]),
                   "train_label_counts": {k: int(v) for k, v in zip(vals, cnts)},
                   "validation_label_counts": {k: int(v) for k, v in zip(*np.unique(va["y"], return_counts=True))},
                   "median_duration_s": float(np.median(tr["duration_s"])),
                   "feature_digest": hashlib.sha256(tr["E"].tobytes()[:1 << 20]).hexdigest()[:16]}
    out = {"experiment": "E26", "status": "completed", "created_at": utc_now().isoformat(timespec="seconds"), "protocol": __doc__.strip().splitlines(),
           "gate": GATE, "results": res, "runtime_s": round(time.perf_counter() - t0, 1),
           "terms": "MIT (dataset card); labels VLM-inferred then human-verified; research experiment, nothing shipped"}
    write_json(out_dir / "metrics.json", out)
    return out
