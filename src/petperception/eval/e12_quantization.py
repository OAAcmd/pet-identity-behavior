"""E12: quantized model builds for phones, re-evaluated against float32.

Variants (MediaPipe builds of the same networks, pinned in manifests/resources.json):
* detector: EfficientDet-Lite0 float32 (shipped), float16 weights, int8;
* crop classifier / feature backbone: EfficientNet-Lite0 float32 (shipped), int8.
Quantized inputs are the quantized encoding of the float input defined by the
model metadata; quantized outputs and intermediate tensors are dequantized
(vision/tflite_io.py).

Evaluations, each re-running the protocol of the experiment it replaces:
1. Detection and false-positive rejection, E7 protocol (fit and every
   selection on the Open Images test-split tuning subset; final evaluation once
   on the E4 validation subset) for each (detector, classifier) pair. The
   verifier is refitted per pair, as it would be for a quantized release.
   The shipped float-trained verifier is also applied unchanged ("no refit")
   to show the drift caused by swapping models alone.
2. Posture, E10 protocol: boxes come from the float product engine (identical
   for all variants), features from each backbone. posture_v1 is applied
   unchanged to int8 features, and refitted on int8 features (C and gate on
   development, final on validation).
3. Agreement on identical detections (same float detector, float vs int8
   classifier): verifier score differences and accept/reject flips at tau;
   posture gated-label flips. This is the uncertainty / rejection behaviour.
4. Model size and single-image CPU latency (Python LiteRT on this VM).

Replacement rule, fixed before the results were seen: a quantized variant may
replace float32 in the phone bundle only if, on the final evaluation with its
refitted verifier and per class, image-level recall drops by at most 0.02 and
specificity on verified negatives by at most 0.03 (point estimates), and
posture final macro-F1 drops by at most 0.03. Confidence intervals are
reported; with 47 verified-negative cat images they are wide, so a pass under
this rule is not evidence of equivalence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import Settings
from ..data import openimages as OI
from ..registry import check_model
from ..utils import read_json, utc_now, write_json
from ..vision import verifier as V
from ..vision.posture import POSTURES, PostureModel
from . import e4_visual as E4
from . import e7_fp_rejection as E7
from . import e10_posture as E10

DETECTORS = {"float32": "efficientdet_lite0_mediapipe_v1", "float16": "efficientdet_lite0_mediapipe_f16_v1", "int8": "efficientdet_lite0_mediapipe_int8_v1"}
CLASSIFIERS = {"float32": "efficientnet_lite0_mediapipe_v1", "int8": "efficientnet_lite0_mediapipe_int8_v1"}
PAIRS = [("float32", "float32"), ("float16", "float32"), ("int8", "float32"), ("float32", "int8"), ("int8", "int8")]
SHIPPED_VERIFIER = "artifacts/verifier/verifier_efficientdet_lite0_mediapipe_v1.json"
SHIPPED_POSTURE = "artifacts/posture/posture_v1.npz"
MAX_RECALL_DROP, MAX_SPEC_DROP, MAX_POSTURE_DROP = 0.02, 0.03, 0.03


def _path(settings: Settings, mid: str) -> Path:
    r = check_model(settings, mid)
    if not r.ready:
        raise RuntimeError(f"{mid}: {r.reason}")
    return Path(next(iter(r.files.values())))


def _digest(score: dict) -> dict:
    out = {}
    for c in E7.CLASSES:
        k = score["classes"][c]
        il = k["image_level"]
        out[c] = {"ap50": k["ap50"], "box_precision": k["precision_at_op"], "box_recall": k["recall_at_op"],
                  "image_recall": il["recall"], "image_recall_ci95": [il["recall_ci95"]["ci_low"], il["recall_ci95"]["ci_high"]],
                  "specificity_on_verified_negatives": il["specificity_on_verified_negatives"],
                  "specificity_ci95": [il["specificity_ci95"]["ci_low"], il["specificity_ci95"]["ci_high"]]}
    return out


def _shipped_verifier(settings: Settings) -> dict[str, V.ClassVerifier]:
    from ..config import REPO_ROOT

    p = read_json(REPO_ROOT / SHIPPED_VERIFIER)
    return {c: V.ClassVerifier(v["feature_set"], np.array(v["mean"]), np.array(v["scale"]), np.array(v["coef"]), float(v["intercept"]), float(v["tau"]))
            for c, v in p["classes"].items()}


def _rows(settings: Settings, det: str, clf: str, which: str) -> list[dict]:
    suffix = "" if clf == E7.CLASSIFIER else f"__{clf}"
    return json.loads((settings.paths.resolve("data_root").parent / "cache" / "e7" / f"{det}{suffix}_{which}.json").read_text())


def detection(settings: Settings, out_dir: Path) -> dict:
    e4_imgs, e4_boxes = OI.read_subset(settings.paths.resolve("manifests_root"))
    ids = [i["image_id"] for i in e4_imgs]
    shipped = _shipped_verifier(settings)
    res: dict = {}
    for dq, cq in PAIRS:
        det, clf = DETECTORS[dq], CLASSIFIERS[cq]
        name = f"detector_{dq}__classifier_{cq}"
        summ = E7.run(settings, out_dir, detectors=[det], classifier=clf, filename=f"e7_{name}.json")["results"][det]
        e4 = summ["e4"]
        rows = _rows(settings, det, clf, "e4")
        v = np.array([shipped[r["label"]].score(V.features(r, shipped[r["label"]].feature_set)) for r in rows]) if rows else np.zeros(0)
        keep = np.array([r["score"] >= E7.OP for r in rows]) & np.array([v[k] >= shipped[r["label"]].tau for k, r in enumerate(rows)])
        no_refit = E4.score_predictions(E7._preds(rows, keep, ids), e4_imgs, e4_boxes, E7.OP, extra_ci=True)
        res[name] = {"detector": det, "classifier": clf, "e4_detections": summ["e4_detections"],
                     "detector_only@0.35": _digest(e4["detector_only@0.35"]),
                     "verifier_crop_refit": _digest(e4["verifier[crop]"]), "verifier_crop_refit_tau": e4["verifier[crop]"]["tau"],
                     "verifier_crop_shipped_no_refit": _digest(no_refit),
                     "tuning_selected_feature_set": summ["selected"]}
    return res


def agreement(settings: Settings) -> dict:
    """Same float detector, float vs int8 classifier: identical detections, compare verifier behaviour."""
    shipped = _shipped_verifier(settings)
    det = DETECTORS["float32"]
    a, b = _rows(settings, det, CLASSIFIERS["float32"], "e4"), _rows(settings, det, CLASSIFIERS["int8"], "e4")
    if [(r["image_id"], r["label"], r["box"]) for r in a] != [(r["image_id"], r["label"], r["box"]) for r in b]:
        raise RuntimeError("detections differ between classifier variants; agreement is undefined")
    out = {}
    for c in E7.CLASSES:
        ix = [k for k, r in enumerate(a) if r["label"] == c and r["score"] >= E7.OP]
        va = np.array([shipped[c].score(V.features(a[k], shipped[c].feature_set)) for k in ix])
        vb = np.array([shipped[c].score(V.features(b[k], shipped[c].feature_set)) for k in ix])
        acc_a, acc_b = va >= shipped[c].tau, vb >= shipped[c].tau
        out[c] = {"detections_at_op": len(ix), "mean_abs_score_diff": float(np.mean(np.abs(va - vb))) if ix else None,
                  "p95_abs_score_diff": float(np.percentile(np.abs(va - vb), 95)) if ix else None,
                  "accept_reject_flips": int((acc_a != acc_b).sum()), "rejected_float": int((~acc_a).sum()), "rejected_int8": int((~acc_b).sum())}
    return out


def posture(settings: Settings, cache: Path) -> dict:
    from ..config import REPO_ROOT

    rows, img_dir = E10.load_rows(settings)
    bpath = cache / "posture_boxes.json"
    if bpath.exists():
        items, drops = json.loads(bpath.read_text()).values()
    else:
        items, drops = E10.boxes(settings, rows, img_dir)
        bpath.parent.mkdir(parents=True, exist_ok=True)
        bpath.write_text(json.dumps({"items": items, "drops": drops}))
    shipped = PostureModel.load(REPO_ROOT / SHIPPED_POSTURE)
    lab = {p: i for i, p in enumerate(POSTURES)}
    feats = {q: E10.featurize(_path(settings, mid), items, img_dir, settings.runtime.num_threads) for q, mid in CLASSIFIERS.items()}

    def split(q: str, sp: str):
        lst = [s for s in feats[q] if s["split"] == sp]
        return np.stack([s["x"] for s in lst]), np.array([lab[s["posture"]] for s in lst]), [s["species"] for s in lst]

    out: dict = {"samples": {sp: sum(1 for s in items if s["split"] == sp) for sp in ("train", "test", "validation")}, "dropped": drops}
    probs = {}
    for q in CLASSIFIERS:
        Xte, yte, ste = split(q, "validation")
        p = np.stack([shipped.scores(x) for x in Xte])
        probs[q] = p
        out[f"shipped_posture_v1_on_{q}_features"] = E10._report(yte, p, ste, shipped.gate)
    Xtr, ytr, _ = split("int8", "train")
    Xdv, ydv, _ = split("int8", "test")
    Xte, yte, ste = split("int8", "validation")
    model, C, gate, dev = E10.fit_select(Xtr, ytr, Xdv, ydv)
    out["refit_on_int8_features"] = {"C": C, "gate": gate, "dev_macro_f1_by_C": {str(k): v for k, v in dev.items()},
                                     "final_validation": E10._report(yte, model.predict_proba(Xte), ste, gate)}
    pf, pi = probs["float32"], probs["int8"]
    rep_f, rep_i = pf.max(1) >= shipped.gate, pi.max(1) >= shipped.gate
    out["agreement_float_vs_int8_shipped_model"] = {
        "validation_animals": int(len(pf)), "argmax_agreement": float(np.mean(pf.argmax(1) == pi.argmax(1))),
        "median_abs_top_score_diff": float(np.median(np.abs(pf.max(1) - pi.max(1)))),
        "gate_flips": int((rep_f != rep_i).sum()), "both_reported_same_label": int(((pf.argmax(1) == pi.argmax(1)) & rep_f & rep_i).sum()),
        "both_reported": int((rep_f & rep_i).sum())}
    return out


def latency(settings: Settings, n: int = 30) -> dict:
    from ai_edge_litert.interpreter import Interpreter

    out = {}
    for mid in [*DETECTORS.values(), *CLASSIFIERS.values()]:
        p = _path(settings, mid)
        row = {"bytes": p.stat().st_size}
        for th in (1, 4):
            it = Interpreter(model_path=str(p), num_threads=th)
            it.allocate_tensors()
            d = it.get_input_details()[0]
            x = np.zeros(d["shape"], d["dtype"])
            it.set_tensor(d["index"], x)
            it.invoke()
            ts = []
            for _ in range(n):
                t = time.perf_counter()
                it.set_tensor(d["index"], x)
                it.invoke()
                ts.append(1000 * (time.perf_counter() - t))
            row[f"median_ms_{th}_thread"] = float(np.median(ts))
        out[mid] = row
    return out


def _decide(det: dict, post: dict) -> dict:
    base = det["detector_float32__classifier_float32"]["verifier_crop_refit"]
    f1_base = post["shipped_posture_v1_on_float32_features"]["macro_f1"]
    out = {}
    for name, r in det.items():
        if name == "detector_float32__classifier_float32":
            continue
        checks = {}
        for c in E7.CLASSES:
            q = r["verifier_crop_refit"][c]
            checks[f"{c}_recall_drop"] = base[c]["image_recall"] - q["image_recall"]
            checks[f"{c}_specificity_drop"] = base[c]["specificity_on_verified_negatives"] - q["specificity_on_verified_negatives"]
        ok = all(checks[f"{c}_recall_drop"] <= MAX_RECALL_DROP and checks[f"{c}_specificity_drop"] <= MAX_SPEC_DROP for c in E7.CLASSES)
        if "classifier_int8" in name:
            checks["posture_macro_f1_drop_refit"] = f1_base - post["refit_on_int8_features"]["final_validation"]["macro_f1"]
            checks["posture_macro_f1_drop_no_refit"] = f1_base - post["shipped_posture_v1_on_int8_features"]["macro_f1"]
            ok = ok and min(checks["posture_macro_f1_drop_refit"], checks["posture_macro_f1_drop_no_refit"]) <= MAX_POSTURE_DROP
        out[name] = {"drops_vs_float32": checks, "meets_replacement_rule": ok}
    return out


def run(settings: Settings, out_dir: Path) -> dict:
    cache = settings.paths.resolve("data_root").parent / "cache" / "e12"
    t0 = time.perf_counter()
    det = detection(settings, out_dir)
    agr = agreement(settings)
    post = posture(settings, cache)
    lat = latency(settings)
    summary = {
        "experiment": "E12", "title": "Quantized (int8 / float16) detector and backbone builds re-evaluated against float32",
        "protocol": __doc__.strip().splitlines(),
        "detection_and_rejection_final_e4": det,
        "verifier_agreement_float_vs_int8_classifier": agr,
        "posture": post,
        "size_and_cpu_latency_python_litert": lat,
        "replacement_rule": {"max_recall_drop": MAX_RECALL_DROP, "max_specificity_drop": MAX_SPEC_DROP, "max_posture_macro_f1_drop": MAX_POSTURE_DROP},
        "decision": _decide(det, post),
        "not_measured_here": ["browser (LiteRT.js) compatibility and latency of the quantized builds: web/tests/runtime",
                              "household footage", "phones"],
        "runtime_s": round(time.perf_counter() - t0, 1),
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    write_json(out_dir / "metrics.json", summary)
    return summary
