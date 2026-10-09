"""E9: dog keypoints (ears, tail, limbs) on the Ultralytics dog-pose data.

Data: Ultralytics dog-pose (Stanford Dogs images, 24 keypoints; documented
as Research-Only, archive ships an AGPL-3.0 licence text). Splits:
* fit: dog-pose ``train`` minus a development hold-out;
* development (model selection, gates): 10% of ``train`` images, seed 20261011;
* final evaluation: dog-pose ``val`` (1,703 images), used once.
Breed (the Stanford Dogs synset prefix of the file name) is recorded so
confidence intervals resample breeds, not images.

Protocol: top-down with the annotated box (as in the dataset) and, separately,
end-to-end with the configured detector's box (IoU >= 0.5 with the annotation).
Metrics: PCK@{0.05, 0.1, 0.2} of max(box w, h) over labelled points; per-point
PCK; visibility ranking AUC; precision/recall of the gated "visible" output;
derived features (ear-tip elevation sign, tail elevation) compared with the
same features computed from the annotation.

What this does not show: accuracy on household video frames, motion blur,
night/IR footage, puppies, or cats (no cat keypoint data was available).
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..config import Settings
from ..registry import check_model
from ..utils import sha256_file, utc_now, write_json
from ..vision import keypoints as KP
from . import metrics as M

SEED = 20261011
DEV_FRACTION = 0.10
HIDDEN = 128
EPOCHS = 30
BATCH = 128
SIGMA_CELLS = 0.75
K = len(KP.DOG_KEYPOINTS)


def read_split(root: Path, split: str) -> list[dict]:
    """One record per annotated dog instance."""
    out = []
    for lab in sorted((root / "labels" / split).glob("*.txt")):
        img = root / "images" / split / f"{lab.stem}.jpg"
        if not img.exists():
            continue
        for n, line in enumerate(lab.read_text().splitlines()):
            v = [float(x) for x in line.split()]
            if len(v) != 5 + 3 * K:
                raise ValueError(f"{lab}: unexpected label length {len(v)}")
            cx, cy, w, h = v[1:5]
            kp = np.array(v[5:]).reshape(K, 3)
            out.append({"id": f"{lab.stem}#{n}", "image": str(img), "breed": lab.stem.split("_")[0],
                        "box": np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]), "kp": kp})
    return out


def _crop_targets(rec: dict, hw: tuple[int, int], flip: bool) -> tuple[np.ndarray, np.ndarray]:
    """Keypoints in crop units [K, 2] and a labelled mask (inside the crop)."""
    h, w = hw
    b = rec["box"]
    x1, y1, x2, y2 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
    side = max(x2 - x1, y2 - y1, 8.0) * KP.CROP_SCALE
    x0, y0 = (x1 + x2) / 2 - side / 2, (y1 + y2) / 2 - side / 2
    xy = np.stack([(rec["kp"][:, 0] * w - x0) / side, (rec["kp"][:, 1] * h - y0) / side], axis=-1)
    lab = (rec["kp"][:, 2] > 0) & np.all((xy >= 0) & (xy < 1), axis=-1)
    if flip:
        xy[:, 0] = 1.0 - xy[:, 0]
        for a, c in KP.FLIP_PAIRS:
            xy[[a, c]] = xy[[c, a]]
            lab[[a, c]] = lab[[c, a]]
    return xy, lab


def extract(backbone: KP.Backbone, recs: list[dict], out: Path, flip: bool) -> dict:
    """Write float16 feature memmaps and targets for recs (and their mirror images if flip)."""
    n = len(recs) * (2 if flip else 1)
    out.mkdir(parents=True, exist_ok=True)
    maps = {k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=np.float16, shape=(n, g, g, c))
            for k, (g, c) in KP.FEATURE_TENSORS.items()}
    xy = np.zeros((n, K, 2), np.float32)
    lab = np.zeros((n, K), bool)
    j = 0
    for rec in recs:
        rgb = cv2.cvtColor(cv2.imread(rec["image"]), cv2.COLOR_BGR2RGB)
        crop, _ = KP.square_crop(rgb, rec["box"])
        for f in ([False, True] if flip else [False]):
            feats = backbone.features(crop[:, ::-1].copy() if f else crop)
            for k in maps:
                maps[k][j] = feats[k]
            xy[j], lab[j] = _crop_targets(rec, rgb.shape[:2], f)
            j += 1
    for m in maps.values():
        m.flush()
    np.save(out / "xy.npy", xy)
    np.save(out / "lab.npy", lab)
    return {"instances": n}


def _load(d: Path) -> dict[str, np.ndarray]:
    return {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in [*KP.FEATURE_TENSORS, "xy", "lab"]}


def _assemble_batch(data: dict[str, np.ndarray], ix: np.ndarray) -> np.ndarray:
    f14 = np.asarray(data["f14"][ix], np.float32)
    f7 = np.asarray(data["f7"][ix], np.float32)
    f28 = np.asarray(data["f28"][ix], np.float32)
    n = len(ix)
    f7u = np.repeat(np.repeat(f7, 2, axis=1), 2, axis=2)
    f28s = f28.reshape(n, KP.GRID, 2, KP.GRID, 2, -1).transpose(0, 1, 3, 2, 4, 5).reshape(n, KP.GRID, KP.GRID, -1)
    return np.concatenate([f14, f7u, f28s], axis=-1)


def _channel_stats(data: dict[str, np.ndarray], n_sample: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    ix = np.sort(rng.choice(len(data["xy"]), size=min(n_sample, len(data["xy"])), replace=False))
    x = _assemble_batch(data, ix).reshape(-1, 592)
    return x.mean(0).astype(np.float32), (x.std(0) + 1e-3).astype(np.float32)


def train(train_dir: Path, dev_dir: Path, log: list) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp
    import optax

    tr, dv = _load(train_dir), _load(dev_dir)
    mean, std = _channel_stats(tr)
    key = jax.random.PRNGKey(SEED)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    c_in = 592

    def init(k, shape, fan_in):
        return jax.random.normal(k, shape) * math.sqrt(2.0 / fan_in)

    params = {"W1": init(k1, (c_in, HIDDEN), c_in), "b1": jnp.zeros(HIDDEN),
              "Wc": init(k2, (3, 3, HIDDEN, HIDDEN), 9 * HIDDEN) * 0.5, "bc": jnp.zeros(HIDDEN),
              "W2": init(k3, (HIDDEN, 3 * K), HIDDEN) * 0.1, "b2": jnp.zeros(3 * K),
              "Wv": init(k4, (HIDDEN, K), HIDDEN) * 0.1, "bv": jnp.zeros(K)}
    g = KP.GRID
    gy, gx = jnp.meshgrid(jnp.arange(g), jnp.arange(g), indexing="ij")

    def forward(p, x):
        z = (x - mean) / std
        h = jax.nn.relu(z @ p["W1"] + p["b1"])
        conv = jax.lax.conv_general_dilated(h, p["Wc"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
        h2 = jax.nn.relu(conv + p["bc"]) + h
        out = h2 @ p["W2"] + p["b2"]
        heat = out[..., :K]
        offs = out[..., K:].reshape(x.shape[0], g, g, K, 2)
        vis = h2.mean(axis=(1, 2)) @ p["Wv"] + p["bv"]
        return heat, offs, vis

    def loss_fn(p, x, xy, lab):
        heat, offs, vis = forward(p, x)
        n = x.shape[0]
        tx, ty = xy[..., 0] * g, xy[..., 1] * g  # [N, K] in cells
        d2 = (gx[None, :, :, None] + 0.5 - tx[:, None, None, :]) ** 2 + (gy[None, :, :, None] + 0.5 - ty[:, None, None, :]) ** 2
        tgt = jnp.exp(-d2 / (2 * SIGMA_CELLS ** 2))
        tgt = tgt / tgt.sum(axis=(1, 2), keepdims=True)
        logp = jax.nn.log_softmax(heat.reshape(n, g * g, K), axis=1).reshape(n, g, g, K)
        ce = -(tgt * logp).sum(axis=(1, 2))  # [N, K]
        cx = jnp.clip(jnp.floor(tx), 0, g - 1).astype(jnp.int32)
        cy = jnp.clip(jnp.floor(ty), 0, g - 1).astype(jnp.int32)
        ni, ki = jnp.meshgrid(jnp.arange(n), jnp.arange(K), indexing="ij")
        o = offs[ni, cy, cx, ki]
        off_t = jnp.stack([tx - cx - 0.5, ty - cy - 0.5], axis=-1)
        l1 = jnp.abs(o - off_t).sum(-1)
        m = lab.astype(jnp.float32)
        denom = jnp.maximum(m.sum(), 1.0)
        bce = optax.sigmoid_binary_cross_entropy(vis, m).mean()
        return (ce * m).sum() / denom + 1.0 * (l1 * m).sum() / denom + 0.5 * bce

    n_tr = len(tr["xy"])
    steps = EPOCHS * math.ceil(n_tr / BATCH)
    opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(optax.cosine_decay_schedule(2e-3, steps), weight_decay=1e-4))
    state = opt.init(params)

    @jax.jit
    def step(p, s, x, xy, lab):
        lv, gr = jax.value_and_grad(loss_fn)(p, x, xy, lab)
        up, s = opt.update(gr, s, p)
        return optax.apply_updates(p, up), s, lv

    fwd = jax.jit(forward)
    best, best_p = -1.0, None
    rng = np.random.default_rng(SEED)
    dev_xy, dev_lab = np.asarray(dv["xy"]), np.asarray(dv["lab"])
    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        perm = rng.permutation(n_tr)
        losses = []
        for b in range(0, n_tr, BATCH):
            ix = np.sort(perm[b:b + BATCH])
            x = jnp.asarray(_assemble_batch(tr, ix))
            params, state, lv = step(params, state, x, jnp.asarray(tr["xy"][ix]), jnp.asarray(tr["lab"][ix]))
            losses.append(float(lv))
        preds = []
        for b in range(0, len(dev_xy), 256):
            ix = np.arange(b, min(b + 256, len(dev_xy)))
            h, o, v = fwd(params, jnp.asarray(_assemble_batch(dv, ix)))
            preds.append(KP.decode(np.asarray(h), np.asarray(o), np.asarray(v))[0])
        pxy = np.concatenate(preds)
        err = np.linalg.norm(pxy - dev_xy, axis=-1) * KP.CROP_SCALE  # crop units -> box max-side units
        pck = float((err[dev_lab] <= 0.1).mean())
        log.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "dev_pck@0.1": pck, "seconds": round(time.perf_counter() - t0, 1)})
        if pck > best:
            best, best_p = pck, jax.tree_util.tree_map(np.asarray, params)
    assert best_p is not None
    return {**{k: np.asarray(v, np.float32) for k, v in best_p.items()}, "mean": mean, "std": std}


def predict_dir(params: dict[str, np.ndarray], d: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _load(d)
    xy, pk, vi = [], [], []
    for b in range(0, len(data["xy"]), 256):
        ix = np.arange(b, min(b + 256, len(data["xy"])))
        h, o, v = KP.head_forward(params, _assemble_batch(data, ix))
        a, p, s = KP.decode(h, o, v)
        xy.append(a)
        pk.append(p)
        vi.append(s)
    return np.concatenate(xy), np.concatenate(pk), np.concatenate(vi)


def _derived(xy: np.ndarray, ok: np.ndarray) -> dict[str, np.ndarray]:
    """Per-instance derived features (same definitions as vision/pose.py), NaN where not computable."""
    from ..vision.pose import KeypointSet, compute_features

    class _P:
        name, species, available, unavailable_reason = "eval", ("dog",), True, None
        keypoint_vocabulary = tuple(KP.DOG_KEYPOINTS)

    out = {"ear_tip_elevation": np.full(len(xy), np.nan), "tail_elevation": np.full(len(xy), np.nan),
           "play_bow_candidate": np.full(len(xy), np.nan)}
    for i in range(len(xy)):
        ks = KeypointSet(0.0, list(KP.DOG_KEYPOINTS), xy[i], ok[i].astype(float))
        for f in compute_features(_P(), [ks], 0.5):
            if f.name in out and f.available and f.value is not None:
                out[f.name][i] = float(f.value)
    return out


def evaluate(params: dict[str, np.ndarray], d: Path, recs: list[dict], gate_vis: float, gate_peak: float) -> dict:
    data = _load(d)
    gt, lab = np.asarray(data["xy"]), np.asarray(data["lab"])
    pxy, peak, vis = predict_dir(params, d)
    err = np.linalg.norm(pxy - gt, axis=-1) * KP.CROP_SCALE
    groups = [r["breed"] for r in recs]

    def pck_at(thr: float):
        return lambda ix: float((err[ix][lab[ix]] <= thr).mean()) if lab[ix].any() else float("nan")

    res = {"instances": len(gt), "labelled_points": int(lab.sum()), "breeds": len(set(groups))}
    for thr in (0.05, 0.1, 0.2):
        f = pck_at(thr)
        res[f"pck@{thr}"] = f(np.arange(len(gt)))
        res[f"pck@{thr}_ci95"] = M.group_bootstrap(f, groups, 1000, 0)
    res["per_keypoint_pck@0.1"] = {name: (float((err[lab[:, k], k] <= 0.1).mean()) if lab[:, k].any() else None, int(lab[:, k].sum()))
                                   for k, name in enumerate(KP.DOG_KEYPOINTS)}
    res["visibility_average_precision"] = M.average_precision(vis.ravel(), lab.ravel())
    rep = (vis >= gate_vis) & (peak >= gate_peak)
    res["gated_output"] = {"gate_visible": gate_vis, "gate_peak": gate_peak,
                           "reported_points": int(rep.sum()),
                           "reported_that_are_labelled": float(lab[rep].mean()) if rep.any() else None,
                           "reported_and_within_0.1": float(((err <= 0.1) & lab)[rep].mean()) if rep.any() else None,
                           "labelled_points_reported": float(rep[lab].mean())}
    dg, dp = _derived(gt, lab), _derived(pxy, rep)
    der = {}
    for name in dg:
        both = ~np.isnan(dg[name]) & ~np.isnan(dp[name])
        if both.sum() >= 10:
            from scipy.stats import spearmanr

            g_, p_ = dg[name][both], dp[name][both]
            entry = {"n": int(both.sum()), "available_rate_where_gt_available": float(both.sum() / max((~np.isnan(dg[name])).sum(), 1))}
            if name == "play_bow_candidate":
                entry |= {"gt_positive": int(g_.sum()), "pred_positive": int(p_.sum()), "agreement": float(np.mean(g_ == p_)),
                          "recall": float(p_[g_ == 1].mean()) if (g_ == 1).any() else None,
                          "precision": float(g_[p_ == 1].mean()) if (p_ == 1).any() else None}
            else:
                entry |= {"sign_agreement": float(np.mean(np.sign(g_) == np.sign(p_))), "spearman": float(spearmanr(g_, p_).statistic),
                          "mae": float(np.mean(np.abs(g_ - p_)))}
            der[name] = entry
    res["derived_features"] = der
    return res


def end_to_end(settings: Settings, params: dict[str, np.ndarray], backbone: KP.Backbone, recs: list[dict]) -> dict:
    """Detector box -> keypoints, on images with exactly one annotated dog."""
    from ..vision.detector import load_detector

    r = check_model(settings, settings.vision.detector)
    if not r.ready:
        return {"status": "unavailable", "reason": r.reason}
    det = load_detector(Path(next(iter(r.files.values()))), settings.runtime.num_threads)
    by_img: dict[str, list[dict]] = defaultdict(list)
    for rec in recs:
        by_img[rec["image"]].append(rec)
    errs, labs, found = [], [], 0
    single = [v[0] for v in by_img.values() if len(v) == 1]
    for rec in single:
        rgb = cv2.cvtColor(cv2.imread(rec["image"]), cv2.COLOR_BGR2RGB)
        dets = [d for d in det.detect(rgb, ["dog"], settings.vision.score_threshold, 0.5, 5)]
        ious = [M.iou_xyxy(d.box[None], rec["box"][None])[0, 0] for d in dets]
        if not dets or max(ious) < 0.5:
            continue
        found += 1
        d = dets[int(np.argmax(ious))]
        crop, (x0, y0, side) = KP.square_crop(rgb, d.box)
        h, o, v = KP.head_forward(params, KP.assemble(backbone.features(crop))[None])
        xy = KP.decode(h, o, v)[0][0]
        hh, ww = rgb.shape[:2]
        img_xy = np.stack([(x0 + xy[:, 0] * side) / ww, (y0 + xy[:, 1] * side) / hh], -1)
        bw, bh = (rec["box"][2] - rec["box"][0]) * ww, (rec["box"][3] - rec["box"][1]) * hh
        e = np.linalg.norm((img_xy - rec["kp"][:, :2]) * np.array([ww, hh]), axis=-1) / max(bw, bh)
        errs.append(e)
        labs.append(rec["kp"][:, 2] > 0)
    if not errs:
        return {"status": "no_matches"}
    E, L = np.array(errs), np.array(labs)
    return {"images_with_one_dog": len(single), "detector_found_iou50": found, "detector": settings.vision.detector,
            "pck@0.1": float((E[L] <= 0.1).mean()), "pck@0.2": float((E[L] <= 0.2).mean())}


def _choose_gates(params: dict[str, np.ndarray], dev_dir: Path) -> tuple[float, float, dict]:
    """Gate the 'visible' output on the development split: keep points that are labelled and
    within 0.2 of the box size at >= 90% precision with the largest coverage."""
    data = _load(dev_dir)
    gt, lab = np.asarray(data["xy"]), np.asarray(data["lab"])
    pxy, peak, vis = predict_dir(params, dev_dir)
    good = lab & (np.linalg.norm(pxy - gt, axis=-1) * KP.CROP_SCALE <= 0.2)
    best = (0.5, 0.2, 0.0, 0.0)
    for gv in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for gp in (0.05, 0.1, 0.2, 0.3, 0.4):
            rep = (vis >= gv) & (peak >= gp)
            if rep.sum() == 0:
                continue
            prec, cov = float(good[rep].mean()), float(rep[lab].mean())
            if prec >= 0.9 and cov > best[3]:
                best = (gv, gp, prec, cov)
    return best[0], best[1], {"dev_precision": best[2], "dev_coverage_of_labelled": best[3], "rule": "precision >= 0.90, max coverage"}


def run(settings: Settings, out_dir: Path, artifact_dir: Path | None = None) -> dict:
    root = settings.paths.resolve("data_root") / "dog_pose" / "extracted"
    if not (root / "labels" / "train").exists():
        raise FileNotFoundError("dog-pose data missing; run: uv run petperception fetch-data dog-pose")
    rb = check_model(settings, "efficientnet_lite0_mediapipe_v1")
    if not rb.ready:
        raise RuntimeError(rb.reason)
    backbone = KP.Backbone(Path(next(iter(rb.files.values()))), settings.runtime.num_threads)
    train_all, val = read_split(root, "train"), read_split(root, "val")
    images = sorted({r["image"] for r in train_all})
    random.Random(SEED).shuffle(images)
    dev_imgs = set(images[: int(len(images) * DEV_FRACTION)])
    fit = [r for r in train_all if r["image"] not in dev_imgs]
    dev = [r for r in train_all if r["image"] in dev_imgs]
    cache = settings.paths.resolve("data_root").parent / "cache" / "e9"
    t0 = time.perf_counter()
    for name, recs, flip in (("fit", fit, True), ("dev", dev, False), ("val", val, False)):
        if not (cache / name / "lab.npy").exists():
            extract(backbone, recs, cache / name, flip)
    extract_s = time.perf_counter() - t0
    log: list = []
    t0 = time.perf_counter()
    params = train(cache / "fit", cache / "dev", log)
    train_s = time.perf_counter() - t0
    gv, gp, gate_info = _choose_gates(params, cache / "dev")
    dev_res = evaluate(params, cache / "dev", dev, gv, gp)
    val_res = evaluate(params, cache / "val", val, gv, gp)
    e2e = end_to_end(settings, params, backbone, val)
    head_sha = None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        p = artifact_dir / "dog_keypoints_head_v1.npz"
        np.savez(p, **params, version=np.array("dog_keypoints_head_v1"))
        head_sha = sha256_file(p)
        write_json(artifact_dir / "dog_keypoints_head_v1.json", {
            "file": p.name, "sha256": head_sha, "backbone": "efficientnet_lite0_mediapipe_v1", "keypoints": KP.DOG_KEYPOINTS,
            "gates": {"min_visible": gv, "min_peak": gp, **gate_info}, "trained_on": "Ultralytics dog-pose train split minus dev hold-out",
            "terms": "Research-Only (dog-pose data; Stanford Dogs / ImageNet terms). Do not use in commercial releases.",
            "created_at": utc_now().isoformat(timespec="seconds")})
    summary = {
        "experiment": "E9",
        "title": "Dog keypoints (ears, tail, limbs): frozen EfficientNet-Lite0 features + trained head, Ultralytics dog-pose",
        "protocol": __doc__.strip().splitlines(),
        "splits": {"fit_instances": len(fit), "fit_instances_with_flips": 2 * len(fit), "dev_instances": len(dev), "val_instances": len(val),
                   "dev_images": len(dev_imgs), "seed": SEED},
        "training": {"epochs": EPOCHS, "batch": BATCH, "hidden": HIDDEN, "log": log, "extract_seconds": round(extract_s, 1),
                     "train_seconds": round(train_s, 1), "selected_epoch_by_dev_pck@0.1": max(log, key=lambda r: r["dev_pck@0.1"])["epoch"]},
        "gates": {"min_visible": gv, "min_peak": gp, **gate_info},
        "dev": dev_res,
        "final_val": val_res,
        "end_to_end_val_with_detector": e2e,
        "head_sha256": head_sha,
        "terms": "Research-Only data; the trained head inherits that restriction.",
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    write_json(out_dir / "metrics.json", summary)
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=1))
    return summary
