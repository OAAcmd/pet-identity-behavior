"""E17 (training job G3): temporal model over per-frame features, dog behaviour in real video.

Question: does modelling time beat the segment-level pooling of E11 on the
same segments? Data, classes and splits are E11's (MammalNet dog-breed
segments; sleeps, eats_food, drinks_water, fights, eliminates; MammalNet's
own training / validation / testing subsets, video-level). Research profile
only: MammalNet publishes no dataset licence and the videos are YouTube
uploads (streamed, never stored).

Per frame (2 fps, what the phone computes): product detector -> largest dog
box -> EfficientNet-Lite0 pooled embedding of the square crop (zeros and a
flag when no dog is found), pooled embedding of the whole frame, box
position and size.

Models (JAX, trained on CPU):
* temporal: per-frame projection (64) -> two 1-D convolutions over time
  (kernel 3, dilation 1 and 2) -> masked attention pooling + masked max
  pooling -> 5 classes;
* pooled ablation: the same network without the convolutions (projection ->
  pooling), isolating what temporal modelling adds;
* reference: E11's logistic regression on pooled features (its own run).
Selection (weight decay, number of epochs) by 5-fold cross-validation
grouped by video on training + validation; then 3 seeds are trained on
training + validation and the test split is scored once per seed.
"""

from __future__ import annotations

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..config import Settings, load_settings
from ..utils import utc_now, write_json
from . import e11_dog_behavior_video as E11
from . import metrics as M

T_MAX = 60  # 30 s at 2 fps
WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2)
MAX_EPOCHS = 80
SEEDS = (0, 1, 2)
HIDDEN = 64


def frame_features(eng, backbone, frames: list[np.ndarray]) -> dict | None:
    import cv2

    from ..vision.keypoints import square_crop

    if not frames:
        return None
    crop = np.zeros((len(frames), 1280), np.float16)
    whole = np.zeros((len(frames), 1280), np.float16)
    box = np.zeros((len(frames), 5), np.float32)
    for i, rgb in enumerate(frames):
        whole[i] = backbone.features(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA))["pooled"]
        dets = eng.detector.detect(rgb, ["dog"], 0.35, 0.5, 5)
        if dets:
            d = max(dets, key=lambda x: x.area)
            crop[i] = backbone.features(square_crop(rgb, d.box)[0])["pooled"]
            b = d.box
            box[i] = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1], 1.0]
    return {"crop": crop, "whole": whole, "box": box}


def build_sequences(settings: Settings, segs: list[dict], cache: Path, workers: int = 3) -> list[dict]:
    from ..pipeline import AnalysisEngine
    from ..registry import check_model
    from ..vision.keypoints import Backbone

    cache.mkdir(parents=True, exist_ok=True)
    key = lambda s: cache / f"{s['video']}_{s['t0']:.2f}.npz"  # noqa: E731
    todo = [s for s in segs if not key(s).exists()]
    if todo:
        eng = AnalysisEngine(load_settings(overrides={"audio": {"enabled": False}, "posture": {"enabled": False}, "pose": {"provider": "none"}})).load()
        rb = check_model(settings, "efficientnet_lite0_mediapipe_v1")
        backbone = Backbone(Path(next(iter(rb.files.values()))), settings.runtime.num_threads)

        def store(s: dict, frames: list[np.ndarray]) -> None:
            f = frame_features(eng, backbone, frames)
            np.savez(key(s), **(f if f is not None else {"missing": np.array(1)}))

        with ThreadPoolExecutor(max_workers=workers) as ex:  # bounded prefetch, as in E11
            pending: deque = deque()
            for s in todo:
                pending.append((s, ex.submit(E11.stream_frames, s["video"], s["t0"], s["t1"])))
                if len(pending) >= 2 * workers:
                    s0, fut = pending.popleft()
                    store(s0, fut.result())
            while pending:
                s0, fut = pending.popleft()
                store(s0, fut.result())
    out = []
    for s in segs:
        with np.load(key(s)) as z:
            if "missing" in z.files:
                continue
            out.append({**s, "x": np.concatenate([z["crop"].astype(np.float32), z["whole"].astype(np.float32), z["box"]], axis=1)})
    return out


def _pad(rows: list[dict], mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((len(rows), T_MAX, mean.shape[0]), np.float32)
    mask = np.zeros((len(rows), T_MAX), np.float32)
    for i, r in enumerate(rows):
        x = r["x"][:T_MAX]
        X[i, : len(x)] = (x - mean) / std
        mask[i, : len(x)] = 1.0
    return X, mask


def _init(rng, d_in: int, temporal: bool):
    import jax

    ks = jax.random.split(rng, 6)
    p = {"w_in": jax.random.normal(ks[0], (d_in, HIDDEN)) * np.sqrt(2.0 / d_in), "b_in": np.zeros(HIDDEN, np.float32),
         "q": jax.random.normal(ks[1], (HIDDEN,)) * 0.1, "w_out": jax.random.normal(ks[2], (2 * HIDDEN, len(E11.LABELS))) * np.sqrt(1.0 / HIDDEN),
         "b_out": np.zeros(len(E11.LABELS), np.float32)}
    if temporal:
        p["c1"] = jax.random.normal(ks[3], (3, HIDDEN, HIDDEN)) * np.sqrt(2.0 / (3 * HIDDEN))
        p["c2"] = jax.random.normal(ks[4], (3, HIDDEN, HIDDEN)) * np.sqrt(2.0 / (3 * HIDDEN))
    return p


def _forward(p, X, mask, key=None, dropout: float = 0.0):
    import jax
    import jax.numpy as jnp

    h = jax.nn.relu(X @ p["w_in"] + p["b_in"])
    if key is not None and dropout > 0:
        h = h * jax.random.bernoulli(key, 1 - dropout, h.shape) / (1 - dropout)
    m = mask[..., None]
    if "c1" in p:
        for w, dil in ((p["c1"], 1), (p["c2"], 2)):
            hp = jnp.pad(h * m, ((0, 0), (dil, dil), (0, 0)))
            T = h.shape[1]
            conv = hp[:, 0:T] @ w[0] + hp[:, dil:dil + T] @ w[1] + hp[:, 2 * dil:2 * dil + T] @ w[2]
            h = jax.nn.relu(conv) + h  # residual
    a = jnp.where(mask > 0, h @ p["q"], -1e9)
    att = jax.nn.softmax(a, axis=1)[..., None]
    pooled = jnp.concatenate([(att * h).sum(1), jnp.where(m > 0, h, -1e9).max(1)], axis=1)
    return pooled @ p["w_out"] + p["b_out"]


def _train(rows: list[dict], wd: float, epochs: int, seed: int, temporal: bool, eval_rows: list[dict] | None = None):
    """Returns (predict function, per-epoch macro-F1 on eval_rows or None)."""
    import jax
    import jax.numpy as jnp
    import optax

    allx = np.concatenate([r["x"] for r in rows])
    mean, std = allx.mean(0), allx.std(0) + 1e-6
    X, mask = _pad(rows, mean, std)
    y = np.array([E11.LABELS.index(r["label"]) for r in rows])
    w_cls = jnp.asarray(len(y) / (len(E11.LABELS) * np.maximum(np.bincount(y, minlength=len(E11.LABELS)), 1)), jnp.float32)
    params = _init(jax.random.PRNGKey(seed), X.shape[2], temporal)
    opt = optax.adamw(1e-3, weight_decay=wd)
    state = opt.init(params)

    def loss(p, xb, mb, yb, k):
        logits = _forward(p, xb, mb, k, dropout=0.3)
        ll = optax.softmax_cross_entropy_with_integer_labels(logits, yb)
        return (ll * w_cls[yb]).sum() / w_cls[yb].sum()

    @jax.jit
    def step(p, s, xb, mb, yb, k):
        g = jax.grad(loss)(p, xb, mb, yb, k)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s

    fwd = jax.jit(lambda p, xb, mb: _forward(p, xb, mb))
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 1000)
    curve = []
    if eval_rows is not None:
        Xe, me = _pad(eval_rows, mean, std)
        ye = [r["label"] for r in eval_rows]
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for i in range(0, len(y), 32):
            b = order[i:i + 32]
            key, k = jax.random.split(key)
            params, state = step(params, state, X[b], mask[b], y[b], k)
        if eval_rows is not None:
            pred = [E11.LABELS[i] for i in np.asarray(fwd(params, Xe, me)).argmax(1)]
            curve.append(M.macro_f1(ye, pred, E11.LABELS))

    def predict(rs: list[dict]) -> list[str]:
        Xp, mp = _pad(rs, mean, std)
        return [E11.LABELS[i] for i in np.asarray(fwd(params, Xp, mp)).argmax(1)]

    return predict, curve


def run(settings: Settings, out_dir: Path) -> dict:
    from sklearn.model_selection import GroupKFold

    data = settings.paths.resolve("data_root")
    ann = data / "mammalnet" / "_download" / "x" / "annotation" / "detection_annotations.json"
    segs = E11.segments(ann)
    rows = build_sequences(settings, segs, data.parent / "cache" / "e17")
    dev = [r for r in rows if r["subset"] in ("training", "validation")]
    test = [r for r in rows if r["subset"] == "testing"]
    groups = [r["video"] for r in dev]
    summary: dict = {"experiment": "E17", "job": "G3", "title": "Temporal model vs pooling, dog behaviour in MammalNet video (research only)",
                     "protocol": __doc__.strip().splitlines(), "segments": {"development": len(dev), "test": len(test)},
                     "frames_per_segment_median": float(np.median([len(r["x"]) for r in rows])), "models": {}}
    for name, temporal in (("temporal_conv_attention", True), ("pooled_ablation", False)):
        curves = {wd: np.zeros(MAX_EPOCHS) for wd in WEIGHT_DECAYS}
        for tr, va in GroupKFold(n_splits=5).split(dev, groups=groups):
            for wd in WEIGHT_DECAYS:
                _, c = _train([dev[i] for i in tr], wd, MAX_EPOCHS, 0, temporal, eval_rows=[dev[i] for i in va])
                curves[wd] += np.array(c) / 5
        wd, ep = max(((w, int(np.argmax(c)) + 1) for w, c in curves.items()), key=lambda t: curves[t[0]][t[1] - 1])
        per_seed = []
        for seed in SEEDS:
            predict, _ = _train(dev, wd, ep, seed, temporal)
            pred = predict(test)
            y = [r["label"] for r in test]
            rep = M.classification_report(y, pred, E11.LABELS)
            rep["macro_f1_ci95_by_video"] = M.group_bootstrap(lambda ix: M.macro_f1([y[i] for i in ix], [pred[i] for i in ix], E11.LABELS),
                                                              [r["video"] for r in test], 1000, seed)
            per_seed.append({"seed": seed, "macro_f1": rep["macro_f1"], "ci95": rep["macro_f1_ci95_by_video"],
                             "per_class_f1": {k: v["f1"] for k, v in rep["per_class"].items()}})
        summary["models"][name] = {"selected": {"weight_decay": wd, "epochs": ep}, "cv_macro_f1_curves": {str(w): c.round(4).tolist() for w, c in curves.items()},
                                   "test_per_seed": per_seed, "test_macro_f1_mean": float(np.mean([s["macro_f1"] for s in per_seed])),
                                   "test_macro_f1_sd": float(np.std([s["macro_f1"] for s in per_seed]))}
    e11 = json.loads((settings.paths.resolve("results_root") / "e11" / "metrics.json").read_text())
    summary["reference_e11_pooled_logreg"] = e11.get("results", {}).get("full", {}).get("test", e11.get("results"))
    summary["release"] = "research profile only (MammalNet: no dataset licence; YouTube videos). Nothing is exported to the app."
    summary["created_at"] = utc_now().isoformat(timespec="seconds")
    write_json(out_dir / "metrics.json", summary)
    return summary
