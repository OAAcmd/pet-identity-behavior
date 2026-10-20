"""E28: situation (recorded context) from synchronized audio and video: single channels, combined, rules.

Data: Kinetics-700-2020 pet clips (intent/kinetics.py). Six labels describe
what a person does with the animal: petting cat, grooming cat, grooming dog,
bathing dog, training dog, walking the dog. They are recorded contexts, not
the animal's intent (intent/taxonomy.py); the product uses them only as
context qualifiers. Train: 600 Kinetics train-split clips; test: 294
validation clips, scored once by this protocol; no YouTube id in both.

Protocol (fixed before the test clips were scored):
* Branches (clip-level pooling, intent/kinetics.py): audio (YAMNet embedding),
  audio_scores (YAMNet class scores; what the phone computes today), animal
  (crop appearance, posture, keypoints, box motion), environment (whole-frame
  embedding and COCO objects), interaction (person presence and person-animal
  interaction geometry, frame motion). A branch is missing for a clip when its
  input is absent (no audio stream, silent audio, no detected animal).
* Heads: multinomial logistic regression per branch, C by 5-fold
  cross-validation grouped by YouTube id on the training clips, temperature
  and abstention threshold (OOF selective accuracy 0.8) from OOF scores.
* Combined: (a) late fusion: logistic regression on the branches' OOF
  log-probabilities with availability flags, trained with modality dropout;
  (b) early fusion: concatenated standardized features with flags, trained
  on five availability patterns (all present; audio, video, animal or
  environment missing). Visual-only and audio-only are the combined models with the
  other channel absent. Variant "phone_now" uses audio_scores instead of the
  embedding and animal_phone (no posture, no research-only keypoints): the
  features the phone bundle computes today.
* Rule baseline: intent/rules.py (committed before scoring; SHA-256 stored).
* Test conditions: clean; missing (audio absent, video absent, animal branch
  absent, environment absent); noisy (ESC-50 appliance noise at 10 and 0 dB
  SNR; poor visual conditions); conflicting (audio from a different-label
  test clip, same species and cross species).
* Conflict flag: the audio head and the visual-only combined model each pass
  their own abstention threshold and name different labels.
* Output policy: interpretation when the combined top score passes its
  threshold and no conflict is flagged; otherwise uncertain / conflict.
  Coverage = share of clips with an interpretation.
* Metrics: macro-F1 and accuracy with 95 % bootstrap CIs over clips (one clip
  per YouTube id); paired bootstrap for differences.
* Leakage audit: each test clip's maximum cosine similarity to any training
  clip (whole-frame and audio embeddings); results repeated without clips
  >= 0.95.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import Settings
from ..data import kinetics_pets as KP
from ..intent import heads as H
from ..intent import kinetics as K
from ..intent import rules as R
from ..utils import utc_now, write_json

BRANCH_SETS = {"full": ("audio", "animal", "environment", "interaction"),
               "phone_now": ("audio_scores", "animal_phone", "environment", "interaction")}
VISUAL = ("animal", "animal_phone", "environment", "interaction")
TARGET_ACC = 0.8
N_BOOT = 1000


def _matrix(rows: list[dict], b: str, dim: int | None = None) -> np.ndarray:
    d = dim or next(len(r["branches"][b]) for r in rows if r["branches"][b] is not None)
    return np.stack([r["branches"][b] if r["branches"][b] is not None else np.full(d, np.nan, np.float32) for r in rows]).astype(np.float64)


def _metrics(y: np.ndarray, pred: np.ndarray, classes: list[str], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    from . import metrics as M

    rep = M.classification_report(y.tolist(), pred.tolist(), classes)
    ci = M.group_bootstrap(lambda ix: M.macro_f1(y[ix].tolist(), pred[ix].tolist(), classes), [str(i) for i in range(len(y))], n_boot, seed)
    return {"n": int(len(y)), "macro_f1": rep["macro_f1"], "macro_f1_ci95": [ci["ci_low"], ci["ci_high"]], "accuracy": rep["accuracy"],
            "per_class_f1": {c: rep["per_class"][c]["f1"] for c in classes}, "confusion": rep.get("confusion_matrix")}


def _paired(y: np.ndarray, a: np.ndarray, b: np.ndarray, classes: list[str], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    from . import metrics as M

    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        d.append(M.macro_f1(y[ix].tolist(), a[ix].tolist(), classes) - M.macro_f1(y[ix].tolist(), b[ix].tolist(), classes))
    point = M.macro_f1(y.tolist(), a.tolist(), classes) - M.macro_f1(y.tolist(), b.tolist(), classes)
    return {"difference": point, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def _pred(p: np.ndarray, classes: list[str], missing_label: str = "(no output)") -> np.ndarray:
    out = np.array([classes[i] for i in np.nan_to_num(p, nan=-1).argmax(1)], dtype=object)
    out[np.isnan(p).any(axis=1)] = missing_label
    return out


AUDIO_CUES = ("yamnet_max",)
VIDEO_CUES = ("object_frames", "dog_fraction", "pet_frames", "person_frames", "interaction_max", "posture_mean", "frames")


def _with_branches(rows: list[dict], variant_rows: dict[str, dict], replace: tuple[str, ...]) -> list[dict]:
    """Rows whose `replace` branches (and the matching interpretable cues) come from a degraded variant."""
    audio = any(b.startswith("audio") for b in replace)
    out = []
    for r in rows:
        v = variant_rows.get(r["clip"].key)
        br, cues = dict(r["branches"]), {**r["cues"], "missing": dict(r["cues"].get("missing", {}))}
        if v is not None:
            for b in replace:
                br[b] = v["branches"].get(b)
                cues["missing"].pop(b, None)
                if b in v["cues"].get("missing", {}):
                    cues["missing"][b] = v["cues"]["missing"][b]
            for k in (AUDIO_CUES if audio else VIDEO_CUES):
                cues.pop(k, None)
                if k in v["cues"]:
                    cues[k] = v["cues"][k]
        out.append({**r, "branches": br, "cues": cues})
    return out


class Model:
    """Trained heads + fusion for one branch set."""

    def __init__(self, branches: tuple[str, ...], classes: list[str]):
        self.branches, self.classes = branches, classes
        self.heads: dict[str, H.Head] = {}
        self.info: dict = {}

    def fit(self, train: list[dict]) -> None:
        y = np.array([r["clip"].label for r in train])
        g = np.array([r["clip"].youtube_id for r in train])
        oof: dict[str, np.ndarray] = {}
        self.dims = {}
        for b in self.branches:
            X = _matrix(train, b)
            self.dims[b] = X.shape[1]
            ok = ~np.isnan(X).any(axis=1)
            head, info, p = H.fit_head(b, X[ok], y[ok], g[ok], self.classes, target_accuracy=TARGET_ACC, metadata={"branch": b})
            self.heads[b] = head
            self.info[b] = {**info, "train_rows": int(ok.sum())}
            full = np.full((len(y), len(self.classes)), np.nan)
            full[ok] = p
            oof[b] = full
        self.late, self.info["late_fusion"] = H.LateFusion.fit(oof, y, g, self.classes, target_accuracy=TARGET_ACC)
        feats = {b: _matrix(train, b, self.dims[b]) for b in self.branches}
        # realistic availability patterns: all present, audio missing, video missing, animal not detected, environment missing
        self.early, self.info["early_fusion"] = H.EarlyFusion.fit(feats, y, g, self.classes,
                                                                  patterns=[(1, 1, 1, 1), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 1, 1), (1, 1, 0, 1)])
        visual = {b: oof[b] for b in self.branches if b in VISUAL}
        pv = self.late.proba({**dict.fromkeys(self.branches), **visual})
        yi = np.array([self.classes.index(v) for v in y])
        self.visual_tau = H.selective_threshold(pv, yi, TARGET_ACC)
        self.info["visual_only_abstain_below"] = self.visual_tau

    def branch_proba(self, rows: list[dict], drop: tuple[str, ...] = ()) -> dict[str, np.ndarray | None]:
        out: dict[str, np.ndarray | None] = {}
        for b in self.branches:
            if b in drop:
                out[b] = None
                continue
            X = _matrix(rows, b, self.dims[b])
            ok = ~np.isnan(X).any(axis=1)
            p = np.full((len(rows), len(self.classes)), np.nan)
            if ok.any():
                p[ok] = self.heads[b].proba(X[ok])
            out[b] = p
        return out

    def late_proba(self, rows: list[dict], drop: tuple[str, ...] = ()) -> np.ndarray:
        return self.late.proba(self.branch_proba(rows, drop))

    def early_proba(self, rows: list[dict], drop: tuple[str, ...] = ()) -> np.ndarray:
        return self.early.proba({b: (None if b in drop else _matrix(rows, b, self.dims[b])) for b in self.branches})

    def decide(self, rows: list[dict], drop: tuple[str, ...] = ()) -> dict:
        """Output policy: interpretation / uncertain / conflict per clip, with the combined label."""
        bp = self.branch_proba(rows, drop)
        p = self.late.proba(bp)
        audio_b = self.branches[0]
        pa = bp.get(audio_b)
        pv = self.late.proba({**{b: None for b in self.branches}, **{b: bp[b] for b in self.branches if b in VISUAL}}) \
            if any(bp.get(b) is not None for b in VISUAL) else None
        status, label = [], []
        n_conf = 0
        for i in range(len(rows)):
            top = int(np.argmax(p[i]))
            conflict = False
            if pa is not None and pv is not None and not np.isnan(pa[i]).any() and any(
                    bp.get(b) is not None and not np.isnan(bp[b][i]).any() for b in VISUAL):
                a_top, v_top = int(np.nanargmax(pa[i])), int(np.argmax(pv[i]))
                conflict = (a_top != v_top and pa[i, a_top] >= self.heads[audio_b].abstain_below and pv[i, v_top] >= self.visual_tau)
            n_conf += conflict
            if conflict:
                status.append("conflict")
            elif p[i, top] >= self.late.abstain_below:
                status.append("interpretation")
            else:
                status.append("uncertain")
            label.append(self.classes[top])
        return {"proba": p, "status": np.array(status), "label": np.array(label, dtype=object), "conflicts": n_conf}


def _policy_metrics(y: np.ndarray, dec: dict) -> dict:
    st, lab = dec["status"], dec["label"]
    cov = st == "interpretation"
    return {"coverage": float(cov.mean()), "selective_accuracy": float((lab[cov] == y[cov]).mean()) if cov.any() else None,
            "conflict_rate": float((st == "conflict").mean()), "uncertain_rate": float((st == "uncertain").mean()),
            "accuracy_all_clips_top_label": float((lab == y).mean())}


def _max_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return (A @ B.T).max(axis=1)


def _rule_preds(rows: list[dict], yamnet_labels: list[str], coco: list[str], drop: tuple[str, ...] = ()) -> np.ndarray:
    out = []
    for r in rows:
        c = r["cues"]
        sm = None if ("audio" in drop or "yamnet_max" not in c) else dict(zip(yamnet_labels, map(float, c["yamnet_max"])))
        video = not ({"animal", "environment", "interaction"} <= set(drop)) and "object_frames" in c
        of = dict(zip(coco, map(int, c["object_frames"]))) if video else None
        dog_frac = c.get("dog_fraction") if video else None
        out.append(R.kinetics_situation(sm, of, dog_frac, c.get("pet_frames", 0) if video else 0)[0])
    return np.array(out, dtype=object)


def swap_audio(rows: list[dict], same_species: bool, seed: int = 0) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed)
    out, src = [], []
    for r in rows:
        cand = [q for q in rows if q["clip"].label != r["clip"].label and (q["clip"].species == r["clip"].species) == same_species
                and q["branches"]["audio"] is not None]
        q = cand[int(rng.integers(0, len(cand)))]
        br = dict(r["branches"])
        br["audio"], br["audio_scores"] = q["branches"]["audio"], q["branches"]["audio_scores"]
        out.append({**r, "branches": br})
        src.append(q["clip"].label)
    return out, np.array(src, dtype=object)


def run(settings: Settings, out_dir: Path, artifact_dir: Path | None = None) -> dict:
    t0 = time.perf_counter()
    data_root = settings.paths.resolve("data_root")
    var_root = data_root.parent
    clips = K.list_clips(data_root)
    rows, missing = K.load_pooled(clips, var_root)
    if missing:
        raise RuntimeError(f"{len(missing)} clips lack features; run: python -m petperception.intent.kinetics")
    classes = sorted(KP.CLASSES)
    train = [r for r in rows if r["clip"].split == "train"]
    test = [r for r in rows if r["clip"].split == "test"]
    y = np.array([r["clip"].label for r in test], dtype=object)
    from ..audio.yamnet import YamnetBackend  # labels only
    from ..registry import model_dir

    yl = YamnetBackend(model_dir(settings, settings.audio.yamnet.model_id), 1).labels
    from ..intent import av_features as AF

    coco = [lab for lab in AF.load_components(1, with_audio=False).object_classes]

    res: dict = {"classes": classes, "train_clips": len(train), "test_clips": len(test),
                 "missing_branches_test": {b: int(sum(r["branches"][b] is None for r in test)) for b in K.BRANCHES},
                 "missing_branches_train": {b: int(sum(r["branches"][b] is None for r in train)) for b in K.BRANCHES},
                 "rules_commit_note": "intent/rules.py committed in 48e0c82 before any E28 scoring"}
    models = {}
    for name, branches in BRANCH_SETS.items():
        m = Model(branches, classes)
        m.fit(train)
        models[name] = m
    m = models["full"]
    clean: dict = {}
    for b in ("audio", "audio_scores", "animal", "animal_phone", "environment", "interaction"):
        mm = models["full"] if b not in ("audio_scores", "animal_phone") else models["phone_now"]
        p = mm.branch_proba(test)[b]
        pr = _pred(p, classes)
        ok = pr != "(no output)"
        clean[f"branch:{b}"] = {**_metrics(y, pr, classes), "coverage_branch_present": float(ok.mean()),
                                "macro_f1_on_present": _metrics(y[ok], pr[ok], classes, n_boot=200)["macro_f1"]}
    preds: dict[str, np.ndarray] = {}
    for name, mm in models.items():
        audio_b, an = mm.branches[0], mm.branches[1]
        for cond, drop in (("combined", ()), ("visual_only", (audio_b,)), ("audio_only", VISUAL),
                           ("no_environment", ("environment",)), ("audio+environment", (an, "interaction")),
                           ("animal+interaction", (audio_b, "environment")), ("audio+animal+interaction", ("environment",))):
            key = f"{name}:late:{cond}"
            preds[key] = _pred(mm.late_proba(test, drop), classes)
            clean[key] = _metrics(y, preds[key], classes)
        preds[f"{name}:early:combined"] = _pred(mm.early_proba(test), classes)
        clean[f"{name}:early:combined"] = _metrics(y, preds[f"{name}:early:combined"], classes)
        preds[f"{name}:early:visual_only"] = _pred(mm.early_proba(test, (audio_b,)), classes)
        clean[f"{name}:early:visual_only"] = _metrics(y, preds[f"{name}:early:visual_only"], classes)
    preds["rules"] = _rule_preds(test, yl, coco)
    clean["rules"] = _metrics(y, preds["rules"], classes)
    clean["rules_video_only"] = _metrics(y, _rule_preds(test, yl, coco, ("audio",)), classes)
    clean["rules_audio_only"] = _metrics(y, _rule_preds(test, yl, coco, ("animal", "environment", "interaction")), classes)
    vals, cnts = np.unique([r["clip"].label for r in train], return_counts=True)
    clean["majority"] = _metrics(y, np.array([vals[np.argmax(cnts)]] * len(y), dtype=object), classes, n_boot=200)
    res["clean"] = clean
    best_single = max(("branch:audio", "branch:animal", "branch:environment", "branch:interaction"), key=lambda k: clean[k]["macro_f1"])
    res["paired_differences"] = {
        "full:late:combined - best single branch (" + best_single + ")": _paired(y, preds["full:late:combined"],
                                                                                   _pred(m.branch_proba(test)[best_single.split(":")[1]], classes), classes),
        "full:late:combined - full:late:visual_only": _paired(y, preds["full:late:combined"], preds["full:late:visual_only"], classes),
        "full:late:combined - full:late:audio_only": _paired(y, preds["full:late:combined"], preds["full:late:audio_only"], classes),
        "full:late:combined - full:late:no_environment": _paired(y, preds["full:late:combined"], preds["full:late:no_environment"], classes),
        "full:late:combined - rules": _paired(y, preds["full:late:combined"], preds["rules"], classes),
        "full:late:combined - full:early:combined": _paired(y, preds["full:late:combined"], preds["full:early:combined"], classes),
        "phone_now:late:combined - full:late:combined": _paired(y, preds["phone_now:late:combined"], preds["full:late:combined"], classes),
    }
    # output policy on clean, missing and conflicting inputs
    policy = {}
    for name, mm in models.items():
        audio_b = mm.branches[0]
        for cond, drop in (("clean", ()), ("audio_missing", (audio_b,)), ("video_missing", VISUAL), ("animal_missing", (mm.branches[1],)),
                           ("environment_missing", ("environment",))):
            policy[f"{name}:{cond}"] = _policy_metrics(y, mm.decide(test, drop))
        for same in (True, False):
            sw, src = swap_audio(test, same)
            dec = mm.decide(sw)
            pm = _policy_metrics(y, dec)
            pm["follows_foreign_audio"] = float((dec["label"] == src).mean())
            pm["top_label_matches_video"] = float((dec["label"] == y).mean())
            policy[f"{name}:audio_swapped_{'same' if same else 'cross'}_species"] = pm
    res["output_policy"] = policy
    # noisy inputs
    noisy = {}
    for variant, replace in (("audio_snr10", ("audio", "audio_scores")), ("audio_snr0", ("audio", "audio_scores")),
                             ("video_poor", ("animal", "animal_phone", "environment", "interaction"))):
        vrows, vmiss = K.load_pooled([r["clip"] for r in test], var_root, variant)
        if vmiss:
            noisy[variant] = {"status": "not run", "missing_clips": len(vmiss)}
            continue
        rows_v = _with_branches(test, {r["clip"].key: r for r in vrows}, replace)
        out = {}
        for name, mm in models.items():
            out[f"{name}:late:combined"] = _metrics(y, _pred(mm.late_proba(rows_v), classes), classes, n_boot=200)
            out[f"{name}:late:visual_only"] = _metrics(y, _pred(mm.late_proba(rows_v, (mm.branches[0],)), classes), classes, n_boot=200)
            out[f"{name}:late:audio_only"] = _metrics(y, _pred(mm.late_proba(rows_v, VISUAL), classes), classes, n_boot=200)
            out[f"{name}:policy"] = _policy_metrics(y, mm.decide(rows_v))
        out["rules"] = _metrics(y, _rule_preds(rows_v, yl, coco), classes, n_boot=200)
        out["missing_branches"] = {b: int(sum(r["branches"][b] is None for r in rows_v)) for b in K.BRANCHES}
        noisy[variant] = out
    res["noisy"] = noisy
    # leakage audit
    env_tr, env_te = _matrix(train, "environment")[:, :1280], _matrix(test, "environment")[:, :1280]
    au_tr, au_te = _matrix(train, "audio")[:, :1024], _matrix(test, "audio")[:, :1024]
    mc_env = _max_cos(np.nan_to_num(env_te), np.nan_to_num(env_tr))
    mc_au = _max_cos(np.nan_to_num(au_te), np.nan_to_num(au_tr))
    far = (mc_env < 0.95) & (mc_au < 0.95)
    res["leakage_audit"] = {"environment_max_cos": {"median": float(np.median(mc_env)), ">=0.95": int((mc_env >= 0.95).sum())},
                            "audio_max_cos": {"median": float(np.median(mc_au)), ">=0.95": int((mc_au >= 0.95).sum())},
                            "clips_kept": int(far.sum()),
                            "full:late:combined_without_near_duplicates": _metrics(y[far], preds["full:late:combined"][far], classes, n_boot=200)}
    res["training"] = {name: mm.info for name, mm in models.items()}
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for name, mm in models.items():
            d = artifact_dir / name
            d.mkdir(exist_ok=True)
            for b, h in mm.heads.items():
                h.to_json(d / f"head_{b}.json")
            mm.late.to_json(d / "late_fusion.json")
            write_json(d / "policy.json", {"visual_only_abstain_below": mm.visual_tau, "combined_abstain_below": mm.late.abstain_below,
                                           "audio_abstain_below": mm.heads[mm.branches[0]].abstain_below, "classes": classes,
                                           "branches": list(mm.branches), "feature_pooling": "petperception.intent.kinetics.pool"})
    out = {"experiment": "E28", "status": "completed", "created_at": utc_now().isoformat(timespec="seconds"), "protocol": __doc__.strip().splitlines(),
           "rules_sha256": R.source_sha256(), "results": res, "runtime_s": round(time.perf_counter() - t0, 1),
           "terms": "Kinetics annotations CC BY 4.0; clips are YouTube uploads (research only); no clip or frame is committed"}
    write_json(out_dir / "metrics.json", out)
    return out
