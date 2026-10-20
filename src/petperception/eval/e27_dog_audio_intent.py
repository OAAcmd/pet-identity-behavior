"""E27: the dog audio channel for intent hypotheses (Barkopedia activities), with environment context, noise and coverage.

Labels (intent/taxonomy.py): of the eight Barkopedia activities, "begging for
food" and "seeking attention" are annotated intent (a vision-language model
proposed them from the video, people verified); rest, alerting to sounds and
the three kinds of playing are observable behaviour; "taking shower" is a
recorded context. Hypothesis-level view: food, attention, play (three play
labels pooled), alert, relaxed (rest), bathing (context qualifier).

Protocol (fixed before the validation clips were scored by E27; E26 had
scored the same clips with the embedding model only):
* Features: YAMNet embedding mean | max (E26 cache) and YAMNet class scores
  mean | max (phone-ready today); environment = the dataset's environment
  label merged to 8 categories (indoor, near door, near window, near other
  animals, bathroom, vehicle, outdoor, other), standing in for context the
  product may know (camera scene, owner-configured zone).
* Heads (intent/heads.py): logistic regression, C by 5-fold CV grouped by
  near-duplicate pseudo-group (E26), temperature and abstention threshold
  (OOF selective accuracy 0.8) from OOF scores. Late fusion of the audio and
  environment heads with availability flags and modality dropout.
* Rule baseline: intent/rules.py barkopedia_activity (committed before
  scoring).
* Test: the provider's validation clips; CIs bootstrap validation
  pseudo-groups. Conditions: clean; ESC-50 appliance noise at 10 and 0 dB
  SNR (audio re-embedded); audio missing (environment only); environment
  missing (audio only).
* External checks with mappings fixed in advance, on data never used for
  training: (a) BEANS / Yin & McCowan barks (10 dogs): play -> play,
  aggression (stranger at the door) -> alert, contact (isolation) ->
  attention; restricted 3-way decision; dog-level bootstrap. (b) Kinetics dog
  clips (all 595, train and validation; never used here): does the
  "taking shower" score separate "bathing dog" clips from the other three
  dog situations (ROC AUC, top-1 rate)?
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import Settings
from ..intent import heads as H
from ..intent import rules as R
from ..utils import utc_now, write_json
from . import e26_bark_activity as E26

HYPOTHESIS = {"begging for food": "food", "seeking attention": "attention", "playing with human": "play", "playing with other animals": "play",
              "playing with toy": "play", "alerting to sounds": "alert", "rest": "relaxed", "taking shower": "bathing"}
HYPS = ["alert", "attention", "bathing", "food", "play", "relaxed"]
ENV = {"indoor (general)": "indoor", "indoor": "indoor", "near door": "near_door", "near window": "near_window", "near other animals": "near_other_animals",
       "bathroom": "bathroom", "vehicle interior": "vehicle", "on grass": "outdoor", "outdoor": "outdoor", "outdoors": "outdoor", "beach": "outdoor",
       "outdoor garden": "outdoor", "playground": "outdoor", "near tree": "outdoor", "garage": "other"}
ENVS = ["bathroom", "indoor", "near_door", "near_other_animals", "near_window", "other", "outdoor", "vehicle"]
TARGET_ACC = 0.8
SNRS = (10.0, 0.0)
BEANS_MAP = {"play": "play", "aggression": "alert", "contact": "attention"}


def env_onehot(raw: list[str]) -> np.ndarray:
    out = np.zeros((len(raw), len(ENVS)), np.float32)
    for i, r in enumerate(raw):
        out[i, ENVS.index(ENV.get(E26.norm(r), "other"))] = 1.0
    return out


def environments(data_root: Path, split: str, ids: np.ndarray) -> np.ndarray:
    import csv

    r = E26.root(data_root)
    env = {row["audio_id"]: row["environement"] for row in csv.DictReader(open(r / E26.LABELS_CSV[split], newline="", encoding="utf-8"))}
    return np.array([env[i] for i in ids])


def score_features(settings: Settings, data_root: Path, split: str, cache: Path, audio_fn_for=None) -> dict[str, np.ndarray]:
    """YAMNet scores mean|max and embeddings mean|max per clip (optionally after a degradation)."""
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}
    from ..audio.yamnet import YamnetBackend
    from ..ingest.media import resample
    from ..registry import model_dir

    be = YamnetBackend(model_dir(settings, settings.audio.yamnet.model_id), settings.runtime.num_threads, with_embeddings=True)
    r = E26.root(data_root)
    ids, S, Em = [], [], []
    for a in E26.ARCHIVES[split]:
        with zipfile.ZipFile(r / a) as z:
            for n in sorted(z.namelist()):
                stem = Path(n).stem
                if not n.endswith(".wav") or "__MACOSX" in n:
                    continue
                x, sr = sf.read(io.BytesIO(z.read(n)), dtype="float32", always_2d=False)
                x = x.mean(axis=1) if x.ndim > 1 else x
                y = resample(np.clip(x, -1, 1), sr, 16000)
                if audio_fn_for is not None:
                    y = audio_fn_for(stem)(y)
                ws = be.score_waveform16k(y, 0.0, settings.audio.yamnet.hop_s)
                ids.append(stem)
                S.append(np.concatenate([ws.scores.mean(0), ws.scores.max(0)]))
                Em.append(np.concatenate([ws.embeddings.mean(0), ws.embeddings.max(0)]))
    out = {"id": np.array(ids), "S": np.array(S, np.float32), "E": np.array(Em, np.float32)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def _align(ids_ref: np.ndarray, d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pos = {k: i for i, k in enumerate(d["id"].tolist())}
    ix = np.array([pos[k] for k in ids_ref.tolist()])
    return {k: v[ix] for k, v in d.items()}


def _to_hyp(p: np.ndarray, classes: list[str]) -> np.ndarray:
    out = np.zeros((len(p), len(HYPS)))
    for j, c in enumerate(classes):
        out[:, HYPS.index(HYPOTHESIS[c])] += p[:, j]
    return out


def _report(y: np.ndarray, p: np.ndarray, classes: list[str], groups: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    from . import metrics as M

    pred = np.array([classes[i] for i in p.argmax(1)])
    rep = M.classification_report(y.tolist(), pred.tolist(), classes)
    ci = M.group_bootstrap(lambda ix: M.macro_f1(y[ix].tolist(), pred[ix].tolist(), classes), groups.tolist(), n_boot, seed)
    yi = np.array([classes.index(v) for v in y])
    return {"n": int(len(y)), "macro_f1": rep["macro_f1"], "macro_f1_ci95": [ci["ci_low"], ci["ci_high"]], "accuracy": rep["accuracy"],
            "per_class_f1": {c: rep["per_class"][c]["f1"] for c in classes}, "ece": H.ece(p, yi)}


def _selective(y: np.ndarray, p: np.ndarray, classes: list[str], tau: float) -> dict:
    cov = p.max(1) >= tau
    pred = np.array([classes[i] for i in p.argmax(1)])
    return {"threshold": tau, "coverage": float(cov.mean()), "selective_accuracy": float((pred[cov] == y[cov]).mean()) if cov.any() else None}


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run(settings: Settings, out_dir: Path, artifact_dir: Path | None = None, n_boot: int = 1000) -> dict:
    t0 = time.perf_counter()
    data_root = settings.paths.resolve("data_root")
    r = E26.root(data_root)
    tr = dict(np.load(r / "features_train.npz", allow_pickle=False))
    va = dict(np.load(r / "features_validation.npz", allow_pickle=False))
    tr.update({k: v for k, v in _align(tr["id"], score_features(settings, data_root, "train", r / "intent_scores_train.npz")).items() if k == "S"})
    va.update({k: v for k, v in _align(va["id"], score_features(settings, data_root, "validation", r / "intent_scores_validation.npz")).items() if k == "S"})
    classes = sorted(HYPOTHESIS)
    y_tr, y_va = tr["y"], va["y"]
    g_tr, g_va = E26.pseudo_groups(tr["E"]), E26.pseudo_groups(va["E"])
    env_tr, env_va = env_onehot(environments(data_root, "train", tr["id"]).tolist()), env_onehot(environments(data_root, "validation", va["id"]).tolist())
    feats_tr = {"audio": tr["E"], "audio_scores": tr["S"], "environment": env_tr}
    feats_va = {"audio": va["E"], "audio_scores": va["S"], "environment": env_va}
    heads, infos, oof = {}, {}, {}
    for b, X in feats_tr.items():
        heads[b], infos[b], oof[b] = H.fit_head(b, X, y_tr, g_tr, classes, target_accuracy=TARGET_ACC, metadata={"branch": b})
    fusions = {}
    for name, (a_b) in (("full", "audio"), ("phone_now", "audio_scores")):
        fusions[name], infos[f"late_fusion:{name}"] = H.LateFusion.fit({a_b: oof[a_b], "environment": oof["environment"]}, y_tr, g_tr, classes,
                                                                       target_accuracy=TARGET_ACC)

    res: dict = {"classes": classes, "hypotheses": HYPS, "label_to_hypothesis": HYPOTHESIS}
    P: dict[str, np.ndarray] = {}
    for b in feats_va:
        P[f"branch:{b}"] = heads[b].proba(feats_va[b])
    for name, a_b in (("full", "audio"), ("phone_now", "audio_scores")):
        f = fusions[name]
        P[f"{name}:audio+environment"] = f.proba({a_b: P[f"branch:{a_b}"], "environment": P["branch:environment"]})
        P[f"{name}:audio_only(fusion, environment missing)"] = f.proba({a_b: P[f"branch:{a_b}"], "environment": None})
        P[f"{name}:environment_only(fusion, audio missing)"] = f.proba({a_b: None, "environment": P["branch:environment"]})
    clean = {}
    for k, p in P.items():
        clean[k] = {"activity": _report(y_va, p, classes, g_va, n_boot), "hypothesis": _report(np.array([HYPOTHESIS[v] for v in y_va]), _to_hyp(p, classes),
                                                                                                HYPS, g_va, n_boot)}
    from ..audio.yamnet import YamnetBackend
    from ..registry import model_dir

    yl = YamnetBackend(model_dir(settings, settings.audio.yamnet.model_id), 1).labels
    rule_pred = np.array([R.barkopedia_activity(R.score_max_by_name(s, yl))[0] for s in va["S"]])
    rp = np.zeros((len(rule_pred), len(classes)))
    rp[np.arange(len(rule_pred)), [classes.index(c) for c in rule_pred]] = 1.0
    clean["rules"] = {"activity": _report(y_va, rp, classes, g_va, n_boot), "hypothesis": _report(np.array([HYPOTHESIS[v] for v in y_va]),
                                                                                                    _to_hyp(rp, classes), HYPS, g_va, n_boot)}
    vals, cnts = np.unique(y_tr, return_counts=True)
    maj = np.zeros((len(y_va), len(classes)))
    maj[:, classes.index(vals[np.argmax(cnts)])] = 1
    clean["majority"] = {"activity": _report(y_va, maj, classes, g_va, 200)}
    res["clean"] = clean
    # paired difference: audio+environment vs audio only (activity macro-F1), by pseudo-group bootstrap
    from . import metrics as M

    def paired(pa, pb):
        ya = y_va.tolist()
        a = [classes[i] for i in pa.argmax(1)]
        b = [classes[i] for i in pb.argmax(1)]
        ci = M.group_bootstrap(lambda ix: M.macro_f1([ya[i] for i in ix], [a[i] for i in ix], classes) - M.macro_f1([ya[i] for i in ix], [b[i] for i in ix], classes),
                               g_va.tolist(), n_boot, 0)
        return {"difference": M.macro_f1(ya, a, classes) - M.macro_f1(ya, b, classes), "ci95": [ci["ci_low"], ci["ci_high"]]}

    res["environment_contribution"] = {
        "full: audio+environment - audio": paired(P["full:audio+environment"], P["branch:audio"]),
        "phone_now: audio+environment - audio_scores": paired(P["phone_now:audio+environment"], P["branch:audio_scores"]),
        "full: audio+environment - environment only": paired(P["full:audio+environment"], P["branch:environment"]),
        "audio - rules": paired(P["branch:audio"], rp),
        "audio_scores - audio": paired(P["branch:audio_scores"], P["branch:audio"]),
    }
    # coverage / selective accuracy at thresholds chosen on training OOF scores
    res["selective"] = {
        "branch:audio": _selective(y_va, P["branch:audio"], classes, heads["audio"].abstain_below),
        "branch:audio_scores": _selective(y_va, P["branch:audio_scores"], classes, heads["audio_scores"].abstain_below),
        "full:audio+environment": _selective(y_va, P["full:audio+environment"], classes, fusions["full"].abstain_below),
        "phone_now:audio+environment": _selective(y_va, P["phone_now:audio+environment"], classes, fusions["phone_now"].abstain_below),
    }
    # noise
    from ..intent import robustness as RB

    pool = RB.noise_pool(data_root / "esc50")
    noisy = {}
    for snr in SNRS:
        d = _align(va["id"], score_features(settings, data_root, "validation", r / f"intent_scores_validation_snr{int(snr)}.npz",
                                            lambda key, snr=snr: RB.audio_noise_fn(key, snr, pool)))
        pa, ps = heads["audio"].proba(d["E"]), heads["audio_scores"].proba(d["S"])
        pf = fusions["full"].proba({"audio": pa, "environment": P["branch:environment"]})
        noisy[f"snr{int(snr)}"] = {
            "branch:audio": _report(y_va, pa, classes, g_va, 200), "branch:audio_scores": _report(y_va, ps, classes, g_va, 200),
            "full:audio+environment": _report(y_va, pf, classes, g_va, 200),
            "selective:branch:audio": _selective(y_va, pa, classes, heads["audio"].abstain_below),
            "selective:full:audio+environment": _selective(y_va, pf, classes, fusions["full"].abstain_below)}
    res["noisy"] = noisy
    # leakage audit (as E26)
    mc = E26.max_cos(va["E"], tr["E"])
    far = mc < E26.DUP
    res["leakage_audit"] = {"validation_max_cos_>=0.95": int((~far).sum()), "clips": int(len(mc)),
                            "branch:audio_without_near_duplicates": _report(y_va[far], P["branch:audio"][far], classes, g_va[far], 200),
                            "full:audio+environment_without_near_duplicates": _report(y_va[far], P["full:audio+environment"][far], classes, g_va[far], 200)}
    # external check (a): BEANS barks
    bz = np.load(data_root / "beans_dog_barks" / "features_v1.npz", allow_pickle=False)
    meta = [json.loads(s) for s in bz["rows"]]
    yb = np.array([BEANS_MAP[m["context"]] for m in meta])
    dogs = np.array([m["name"] for m in meta])
    ph = _to_hyp(heads["audio"].proba(bz["E"]), classes)
    sub = ["alert", "attention", "play"]
    pr = np.array([sub[i] for i in ph[:, [HYPS.index(h) for h in sub]].argmax(1)])
    rep = M.classification_report(yb.tolist(), pr.tolist(), sub)
    ci = M.group_bootstrap(lambda ix: M.macro_f1(yb[ix].tolist(), pr[ix].tolist(), sub), dogs.tolist(), n_boot, 0)
    res["external_beans"] = {"mapping": BEANS_MAP, "n": int(len(yb)), "dogs": int(len(set(dogs))), "macro_f1": rep["macro_f1"],
                             "macro_f1_ci95_by_dog": [ci["ci_low"], ci["ci_high"]], "per_class_f1": {c: rep["per_class"][c]["f1"] for c in sub},
                             "confusion": rep.get("confusion_matrix"), "chance_macro_f1_uniform": 1 / 3,
                             "caveat": "CSV terms -> paper situations unconfirmed; recording situations, not annotated intent"}
    # external check (b): Kinetics dog clips
    res["external_kinetics_bathing"] = _kinetics_bathing(settings, heads["audio"], classes)
    res["training"] = infos
    res["data"] = {"train_clips": int(len(y_tr)), "validation_clips": int(len(y_va)), "train_pseudo_groups": int(len(set(g_tr.tolist()))),
                   "validation_pseudo_groups": int(len(set(g_va.tolist()))),
                   "environment_counts_train": {e: int(v) for e, v in zip(ENVS, env_tr.sum(0))},
                   "environment_counts_validation": {e: int(v) for e, v in zip(ENVS, env_va.sum(0))}}
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for b, h in heads.items():
            h.metadata.update({"dataset": f"ArlingtonCL2/Barkopedia_Dog_Act_Env @ {E26.REVISION} (MIT)", "label_to_hypothesis": HYPOTHESIS,
                               "features": {"audio": "YAMNet embedding mean|max over 0.96 s windows, hop 0.48 s (2048)",
                                            "audio_scores": "YAMNet class scores mean|max (1042)",
                                            "environment": f"one-hot {ENVS}"}[b]})
            h.to_json(artifact_dir / f"head_{b}.json")
        for name, f in fusions.items():
            f.to_json(artifact_dir / f"late_fusion_{name}.json")
    out = {"experiment": "E27", "status": "completed", "created_at": utc_now().isoformat(timespec="seconds"), "protocol": __doc__.strip().splitlines(),
           "rules_sha256": R.source_sha256(), "results": res, "runtime_s": round(time.perf_counter() - t0, 1),
           "terms": "Barkopedia MIT; BEANS/Yin & McCowan research only; Kinetics research only; heads trained on MIT data"}
    write_json(out_dir / "metrics.json", out)
    return out


def _kinetics_bathing(settings: Settings, head: H.Head, classes: list[str]) -> dict:
    from ..intent import kinetics as K

    data_root = settings.paths.resolve("data_root")
    clips = [c for c in K.list_clips(data_root) if c.species == "dog"]
    d = K.feature_dir(data_root.parent)
    X, lab = [], []
    for c in clips:
        p = d / f"{c.key}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as z:
            if str(z["audio_state"]) != "present":
                continue
            e = z["a_emb"].astype(np.float32)
            X.append(np.concatenate([e.mean(0), e.max(0)]))
            lab.append(c.label)
    if not X:
        return {"status": "not run (no Kinetics features)"}
    p = head.proba(np.array(X))
    lab = np.array(lab)
    j = classes.index("taking shower")
    pos, neg = p[lab == "bathing dog", j], p[lab != "bathing dog", j]
    top = np.array([classes[i] for i in p.argmax(1)])
    by = {c: {"n": int((lab == c).sum()), "top1_taking_shower": float((top[lab == c] == "taking shower").mean()),
              "top1_distribution": {k: int(v) for k, v in zip(*np.unique(top[lab == c], return_counts=True))}} for c in sorted(set(lab.tolist()))}
    return {"clips": int(len(lab)), "roc_auc_bathing_vs_other_dog_situations": _auc(pos, neg), "by_label": by,
            "note": "Kinetics clips never used to train this head; audio of YouTube uploads (music overdubs occur)"}
