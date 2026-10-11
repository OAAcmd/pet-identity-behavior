"""E22: dog bark context (Yin & McCowan 2004, as packaged in BEANS), grouped by dog.

The dog counterpart of E2. Labels are the recording contexts in the package's
annotations.csv: aggression, contact and play. The paper's situations were
disturbance (a stranger ringing the doorbell), isolation from the owner, and
play. The mapping of the CSV terms to those situations is not confirmed, so
results are reported under the CSV terms. They describe the situation the
dog was recorded in, not a verified emotion.

Data handling (disk budget):
* the archive (901,568,218 bytes) is never stored. Its MD5 is checked against
  the publisher's object metadata in one streaming pass; the URL pins the
  object generation, so the bytes cannot change between that pass and the
  reads;
* annotations.csv and each audio member are then read by HTTP Range requests
  (`RemoteZip`, CRC-checked) and turned into features in memory. Only the
  features, labels and per-member SHA-256 values are cached
  (var/data/beans_dog_barks/features_v1.npz).

Protocol (as E2):
* outer: leave-one-dog-out (10 dogs). The stricter leave-one-breed-out
  variant (6 breeds) is also reported;
* inner: grouped CV by dog on the training dogs only, selecting C;
  standardisation fitted inside each training partition;
* features: MFCC statistics at 16 kHz, and YAMNet embeddings (mean and max
  over frames), the variant the phone can run;
* baselines: majority and stratified random; dog-level bootstrap 95% CIs;
  a random split as a leakage diagnostic only.

Terms: no data licence is stated; research evaluation only. Cite Yin, S. &
McCowan, B. (2004), Animal Behaviour 68, 343-355. No model is shipped.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.model_selection import StratifiedKFold

from ..audio import linear_model
from ..audio.features import FeatureConfig, extract
from ..config import Settings
from ..data import mirror as MR
from ..utils import utc_now, write_json
from . import metrics as M

DATASET_KEY = "beans_dog_barks/dog_barks.zip"
GENERATION = "1728757838735376"
FEATURES = FeatureConfig(sample_rate=16000, version="mfcc-stats-v1")
CACHE_VERSION = "features_v1"


def _source_url(settings: Settings) -> tuple[str, dict]:
    cat = {cf.key: cf for cf in MR.load_catalog(settings.paths.resolve("manifests_root") / "mirror_files.json")}
    cf = cat[DATASET_KEY]
    url = MR.url_for(cf, MR.mirror_urls())
    if url is None:  # the publisher's bucket is reachable from here; pin the object generation
        url = f"{cf.official_url}?generation={GENERATION}"
    return url, {"size": cf.size, **cf.checksums}


def _members_by_basename(z: MR.RemoteZip) -> dict[str, str]:
    out = {}
    for n in z.names():
        if not n.endswith("/"):
            out.setdefault(Path(n).name, n)
            out.setdefault(Path(n).stem, n)
    return out


def build_features(settings: Settings, cache: Path, yamnet) -> dict:
    url, expect = _source_url(settings)
    t0 = time.perf_counter()
    dig = MR.stream_digests(url, ("md5", "sha256"))
    if dig["bytes"] != expect["size"] or dig["md5"] != expect["md5"]:
        raise MR.MirrorError(f"{MR.redact(url)}: size/md5 {dig['bytes']}/{dig['md5']} do not match the catalogue {expect}")
    t_verify = time.perf_counter() - t0
    z = MR.RemoteZip([url])
    ann_name = next(n for n in z.names() if Path(n).name == "annotations.csv")
    rows = list(csv.DictReader(io.StringIO(z.read(ann_name).decode("utf-8"))))
    by_base = _members_by_basename(z)
    X, E, keep, shas, durs, srs, missing = [], [], [], [], [], [], []
    from ..ingest.media import AudioTrack

    for r in rows:
        member = by_base.get(Path(r["filename"]).name) or by_base.get(Path(r["filename"]).stem)
        if member is None:
            missing.append(r["filename"])
            continue
        data = z.read(member)
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        X.append(extract(x, sr, FEATURES))
        if yamnet is not None:
            ws = yamnet.score_track(AudioTrack("present", x, sr, 0.0), hop_s=0.48)
            E.append(np.concatenate([ws.embeddings.mean(axis=0), ws.embeddings.max(axis=0)]))
        keep.append(r)
        shas.append(hashlib.sha256(data).hexdigest())
        durs.append(len(x) / sr)
        srs.append(sr)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=np.stack(X), E=np.stack(E) if E else np.zeros((len(X), 0)),
                        rows=np.array([json.dumps(r) for r in keep]), sha256=np.array(shas), duration_s=np.array(durs), sample_rate=np.array(srs))
    prov = {"source": MR.redact(url), "verified": {"bytes": dig["bytes"], "md5": dig["md5"], "sha256": dig["sha256"]},
            "verify_seconds": round(t_verify, 1), "archive_stored": False, "annotation_member": ann_name,
            "rows_in_annotations": len(rows), "members_read": len(keep), "rows_without_member": missing,
            "feature_seconds": round(time.perf_counter() - t0 - t_verify, 1), "created_at": utc_now().isoformat(timespec="seconds")}
    cache.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2))
    return prov


def _grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, labels: list[str], seed: int) -> tuple[np.ndarray, list[dict]]:
    pred = np.empty(len(y), dtype=object)
    folds = []
    for g in sorted(set(groups.tolist())):
        te, tr = groups == g, groups != g
        lacking = sorted(set(labels) - set(y[tr].tolist()))
        if lacking:
            folds.append({"held_out": g, "status": "excluded", "reason": f"training data lacks {lacking}"})
            pred[te] = "excluded"
            continue
        model, info = linear_model.fit(X[tr], y[tr], groups[tr], labels, seed=seed, inner_splits=5)
        pred[te] = model.predict(X[te])
        present = sorted(set(y[te].tolist()))
        folds.append({"held_out": g, "status": "ok", "n_test": int(te.sum()), "classes_present": present,
                      "macro_f1_present_classes": M.macro_f1(y[te].tolist(), pred[te].tolist(), present), "selected_C": info["selected_C"]})
    return pred, folds


def evaluate(X: np.ndarray, E: np.ndarray, rows: list[dict], seed: int = 0, n_boot: int = 1000) -> dict:
    y = np.array([r["context"] for r in rows])
    dogs = np.array([r["name"] for r in rows])
    breeds = np.array([r["breed"] for r in rows])
    labels = sorted(set(y.tolist()))
    variants = [("lodo_mfcc_logreg", X, dogs), ("lobo_mfcc_logreg", X, breeds)]
    if E.shape[1]:
        variants += [("lodo_yamnet_embedding_logreg", E, dogs), ("lobo_yamnet_embedding_logreg", E, breeds)]
    res: dict = {"labels": labels, "methods": {}}
    for name, F, groups in variants:
        pred, folds = _grouped_cv(F, y, groups, labels, seed)
        keep = pred != "excluded"
        rep = M.classification_report(y[keep].tolist(), pred[keep].tolist(), labels)
        rep["folds"] = folds
        rep["excluded_predictions"] = int((~keep).sum())
        rep["macro_f1_ci95_by_dog"] = M.group_bootstrap(lambda ix: M.macro_f1(y[keep][ix].tolist(), pred[keep][ix].tolist(), labels),
                                                        dogs[keep].tolist(), n_boot, seed)
        res["methods"][name] = rep
    vals, cnts = np.unique(y, return_counts=True)
    res["methods"]["majority"] = M.classification_report(y.tolist(), [vals[np.argmax(cnts)]] * len(y), labels)
    rng = np.random.default_rng(seed)
    strat = [M.macro_f1(y.tolist(), rng.choice(vals, size=len(y), p=cnts / cnts.sum()).tolist(), labels) for _ in range(200)]
    res["methods"]["stratified_random"] = {"macro_f1_mean": float(np.mean(strat)), "macro_f1_sd": float(np.std(strat))}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    leak = np.empty(len(y), dtype=object)
    for tr, te in skf.split(X, y):
        m, _ = linear_model.fit(X[tr], y[tr], dogs[tr], labels, seed=seed, inner_splits=5)
        leak[te] = m.predict(X[te])
    res["leakage_diagnostic_random_split"] = {"macro_f1": M.macro_f1(y.tolist(), leak.tolist(), labels),
                                              "note": "NOT a valid estimate: the same dogs appear in training and test"}
    res["dataset_observed"] = {
        "recordings": len(rows), "dogs": len(set(dogs.tolist())), "breeds": len(set(breeds.tolist())),
        "label_counts": {l: int(np.sum(y == l)) for l in labels},
        "dog_by_context": {d: {l: int(np.sum((dogs == d) & (y == l))) for l in labels} for d in sorted(set(dogs.tolist()))},
        "breed_of_dog": {d: sorted(set(breeds[dogs == d].tolist())) for d in sorted(set(dogs.tolist()))},
    }
    return res


def run(settings: Settings, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = settings.paths.resolve("data_root") / "beans_dog_barks" / f"{CACHE_VERSION}.npz"
    from ..audio.yamnet import YamnetBackend
    from ..registry import check_model, model_dir

    yamnet = YamnetBackend(model_dir(settings, settings.audio.yamnet.model_id), settings.runtime.num_threads, with_embeddings=True) \
        if check_model(settings, settings.audio.yamnet.model_id).ready else None
    if not cache.exists():
        build_features(settings, cache, yamnet)
    d = np.load(cache)
    rows = [json.loads(s) for s in d["rows"]]
    res = evaluate(d["X"], d["E"], rows)
    res["data_source"] = json.loads(cache.with_suffix(".provenance.json").read_text())
    res["audio"] = {"duration_s_median": float(np.median(d["duration_s"])), "duration_s_total": float(np.sum(d["duration_s"])),
                    "sample_rates_hz": {str(k): int(v) for k, v in zip(*np.unique(d["sample_rate"], return_counts=True))},
                    "member_sha256_digest": hashlib.sha256("".join(sorted(d["sha256"].tolist())).encode()).hexdigest()}
    res["feature_config"] = FEATURES.to_dict()
    res["terms"] = "no data licence stated; research evaluation only; cite Yin & McCowan 2004; nothing is shipped"
    res["label_semantics"] = "recording context from annotations.csv (aggression / contact / play); mapping to the paper's disturbance / isolation / play is unconfirmed"
    write_json(out_dir / "metrics.json", {"experiment": "E22", "status": "completed", "results": res,
                                          "runtime_s": round(time.perf_counter() - t0, 1), "created_at": utc_now().isoformat(timespec="seconds")})
    return res
