"""E10: coarse posture (lying / sitting / standing) from WEAK caption labels.

Labels: parsed from Localized Narratives captions (see data/posture_weak.py).
Boxes: the product detector + verifier (exactly one accepted detection of the
captioned species and none of the other species), as at analysis time.
Features: pooled EfficientNet-Lite0 embedding of the square crop, box shape,
species indicator. Model: multinomial logistic regression.

Splits (Open Images): train -> fit; test -> development (regularisation and
the abstention gate); validation -> final evaluation, used once.

Every number is agreement with caption-derived labels. Caption noise was not
measured: a caption can be wrong, vague ("resting" while sitting) or refer to
a different animal than the one detected.
Baselines: majority class and a box-shape-only model, to show what the
embedding adds.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from ..config import Settings, load_settings
from ..utils import sha256_file, utc_now, write_json
from ..vision import keypoints as KP
from ..vision.posture import POSTURES, posture_features
from . import metrics as M

C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
GATE_TARGET = 0.80  # development agreement required for reported (non-uncertain) labels


def boxes(settings: Settings, rows: list[dict], img_dir: Path) -> tuple[list[dict], dict]:
    """One accepted detection per image, chosen by the product detector + verifier."""
    from ..pipeline import AnalysisEngine

    eng = AnalysisEngine(load_settings(overrides={"audio": {"enabled": False}})).load()
    out, drop = [], {"missing_image": 0, "no_single_accepted_detection": 0}
    for r in rows:
        im = cv2.imread(str(img_dir / f"{r['image_id']}.jpg"))
        if im is None:
            drop["missing_image"] += 1
            continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        dets = eng.detector.detect(rgb, ["cat", "dog"], 0.35, 0.5, 10)
        own = [d for d in dets if d.label == r["species"]]
        other = [d for d in dets if d.label != r["species"]]
        acc = [d for d in own if (eng.verifier.verify(rgb, d) or (0.0, True))[1]] if eng.verifier else own
        if len(acc) != 1 or other:
            drop["no_single_accepted_detection"] += 1
            continue
        out.append({"image_id": r["image_id"], "split": r["split"], "species": r["species"], "posture": r["posture"],
                    "box": [float(v) for v in acc[0].box]})
    return out, drop


def featurize(backbone_path: Path, items: list[dict], img_dir: Path, num_threads: int) -> list[dict]:
    rb = KP.Backbone(backbone_path, num_threads)
    out = []
    for it in items:
        rgb = cv2.cvtColor(cv2.imread(str(img_dir / f"{it['image_id']}.jpg")), cv2.COLOR_BGR2RGB)
        box = np.array(it["box"])
        f = rb.features(KP.square_crop(rgb, box)[0])
        out.append({**it, "x": posture_features(f["pooled"], box, rgb.shape[:2], it["species"])})
    return out


def _samples(settings: Settings, rows: list[dict], img_dir: Path) -> tuple[list[dict], dict]:
    items, drop = boxes(settings, rows, img_dir)
    return featurize(_backbone_path(settings), items, img_dir, settings.runtime.num_threads), drop


def _backbone_path(settings: Settings) -> Path:
    from ..registry import check_model

    r = check_model(settings, settings.pose.backbone)
    if not r.ready:
        raise RuntimeError(r.reason)
    return Path(next(iter(r.files.values())))


def _fit(X: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000, class_weight="balanced")).fit(X, y)


def _report(y: np.ndarray, p: np.ndarray, species: list[str], gate: float | None) -> dict:
    pred = p.argmax(1)
    labels = list(POSTURES)
    yl = [labels[i] for i in y]
    pl = [labels[i] for i in pred]
    out = {"n": int(len(y)), "macro_f1": M.macro_f1(yl, pl, labels), "accuracy": float(np.mean(pred == y)),
           "confusion_rows_true_cols_pred": M.confusion(yl, pl, labels), "per_class": M.per_class(yl, pl, labels),
           "macro_f1_ci95": M.group_bootstrap(lambda ix: M.macro_f1([yl[i] for i in ix], [pl[i] for i in ix], labels), list(range(len(y))), 1000, 0)}
    for sp in ("cat", "dog"):
        ix = [i for i, s in enumerate(species) if s == sp]
        if ix:
            out[f"{sp}_macro_f1"] = M.macro_f1([yl[i] for i in ix], [pl[i] for i in ix], labels)
            out[f"{sp}_n"] = len(ix)
    if gate is not None:
        rep = p.max(1) >= gate
        out["gated"] = {"gate": gate, "coverage": float(rep.mean()), "agreement_on_reported": float(np.mean(pred[rep] == y[rep])) if rep.any() else None}
    return out


def fit_select(Xtr: np.ndarray, ytr: np.ndarray, Xdv: np.ndarray, ydv: np.ndarray):
    """Regularisation by development macro-F1, then the abstention gate on development."""
    dev_scores = {}
    for C in C_GRID:
        m = _fit(Xtr, ytr, C)
        dev_scores[C] = M.macro_f1([POSTURES[i] for i in ydv], [POSTURES[i] for i in m.predict(Xdv)], list(POSTURES))
    C = max(dev_scores, key=dev_scores.get)
    model = _fit(Xtr, ytr, C)
    pdv = model.predict_proba(Xdv)
    gate = 1.0
    for g in np.arange(0.34, 0.95, 0.02):
        rep = pdv.max(1) >= g
        if rep.sum() >= 20 and np.mean(pdv.argmax(1)[rep] == ydv[rep]) >= GATE_TARGET:
            gate = float(round(g, 2))
            break
    return model, C, gate, dev_scores


def load_rows(settings: Settings) -> tuple[list[dict], Path]:
    data = settings.paths.resolve("data_root")
    man = settings.paths.resolve("manifests_root")
    img_dir = data / "localized_narratives" / "images"
    pinned = {r["image_id"]: r["sha256"] for r in csv.DictReader(open(man / "posture_weak_sha256.csv", newline=""))}
    rows = [r for r in csv.DictReader(open(man / "posture_weak_images.csv", newline="")) if r["image_id"] in pinned]
    bad = [r["image_id"] for r in rows if (img_dir / f"{r['image_id']}.jpg").exists() and sha256_file(img_dir / f"{r['image_id']}.jpg") != pinned[r["image_id"]]]
    if bad:
        raise RuntimeError(f"{len(bad)} posture images fail their pinned SHA-256")
    return rows, img_dir


def run(settings: Settings, out_dir: Path, artifact_dir: Path | None = None) -> dict:
    rows, img_dir = load_rows(settings)
    samples, drops = _samples(settings, rows, img_dir)
    by = {sp: [s for s in samples if s["split"] == sp] for sp in ("train", "test", "validation")}
    lab = {p: i for i, p in enumerate(POSTURES)}

    def xy(lst: list[dict], cols: slice | None = None):
        X = np.stack([s["x"] for s in lst])
        return (X[:, cols] if cols is not None else X), np.array([lab[s["posture"]] for s in lst]), [s["species"] for s in lst]

    Xtr, ytr, _ = xy(by["train"])
    Xdv, ydv, sdv = xy(by["test"])
    Xte, yte, ste = xy(by["validation"])
    model, C, gate, dev_scores = fit_select(Xtr, ytr, Xdv, ydv)
    pdv = model.predict_proba(Xdv)
    pte = model.predict_proba(Xte)
    shape_cols = slice(Xtr.shape[1] - 4, Xtr.shape[1])
    shape_model = _fit(Xtr[:, shape_cols], ytr, 1.0)
    majority = np.bincount(ytr).argmax()
    maj_p = np.zeros_like(pte)
    maj_p[:, majority] = 1.0
    summary = {
        "experiment": "E10",
        "title": "Coarse posture from weak caption labels (cats and dogs)",
        "protocol": __doc__.strip().splitlines(),
        "samples": {k: len(v) for k, v in by.items()}, "dropped": drops,
        "label_counts": {k: {f"{s}/{p}": sum(1 for x in v if x["species"] == s and x["posture"] == p) for s in ("cat", "dog") for p in POSTURES}
                         for k, v in by.items()},
        "selected_C_on_dev": C, "dev_macro_f1_by_C": {str(k): v for k, v in dev_scores.items()},
        "gate_rule": f"smallest top-score gate with >= {GATE_TARGET:.0%} development agreement on reported labels",
        "dev": _report(ydv, pdv, sdv, gate),
        "final_validation": _report(yte, pte, ste, gate),
        "baselines_on_validation": {"majority_class": _report(yte, maj_p, ste, None)["macro_f1"],
                                    "box_shape_only": _report(yte, shape_model.predict_proba(Xte[:, shape_cols]), ste, None)["macro_f1"]},
        "label_source_caveat": "Agreement with caption-derived weak labels; caption noise unmeasured.",
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        sc, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
        p = artifact_dir / "posture_v1.npz"
        np.savez(p, mean=sc.mean_, scale=sc.scale_, coef=lr.coef_, intercept=lr.intercept_, gate=np.array(gate), version=np.array("posture_v1"))
        summary["artifact_sha256"] = sha256_file(p)
    write_json(out_dir / "metrics.json", summary)
    return summary
