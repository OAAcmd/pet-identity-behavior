"""E33: distil SigLIP base image embeddings of cat and dog crops into a MobileNetV3-Large student (CPU, resumable).

Inputs (written by ``python -m petperception.eval.e33_phone_student build`` and ``teacher``, under --work):
* ``index.json`` (crop files per split), ``crops/<split>/*.jpg`` (224 x 224),
* ``teacher/<split>_<k>.npy`` (SigLIP embeddings, chunks of 512 in index order),
* ``text_e31.npz`` (SigLIP text embeddings of the frozen E31 prompts of both species, logit scale and bias).

Model: Keras MobileNetV3-Large (ImageNet weights, built-in preprocessing, input 0-255) + Dense(768) + L2 normalization.
Loss: (1 - cosine to the teacher) + KL(teacher || student) over softmax(scale * e @ T^T + bias). Adam with cosine decay.

Resumable: a TF checkpoint (weights, optimizer state, step counter) every --ckpt-minutes, at every epoch end and on
SIGTERM/SIGINT, with ``state.json`` (epoch, step in epoch, history). The data order of each epoch is a seeded
permutation (seed * 1000 + epoch) and completed steps are skipped on resume. A resume refuses to continue if the
configuration, the data index or this script changed. After the last epoch the weights of the epoch with the lowest
validation loss are exported to TFLite float32, float16 and int8 (dynamic range) with a run record (hashes, history).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import signal
import time
from pathlib import Path

import numpy as np


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_split(work: Path, index: dict, split: str) -> tuple[list[str], np.ndarray]:
    rows = index["splits"][split]
    chunks = sorted((work / "teacher").glob(f"{split}_*.npy"))
    emb = np.concatenate([np.load(c) for c in chunks]).astype(np.float32) if chunks else np.zeros((0, 768), np.float32)
    if len(emb) != len(rows):
        raise SystemExit(f"teacher embeddings for {split}: {len(emb)} of {len(rows)} crops - run the teacher step first")
    return [str(work / "crops" / r["crop"]) for r in rows], emb


def build_model(tf, weights: str | None):
    base = tf.keras.applications.MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights=weights, pooling="avg",
                                                  include_preprocessing=True)
    x = tf.keras.layers.Dense(768, name="proj")(base.output)
    out = tf.keras.layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=-1), name="l2norm")(x)
    return tf.keras.Model(base.input, out, name="student_mnv3l")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ckpt-minutes", type=float, default=10.0)
    a = ap.parse_args()

    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    work, run, exp = Path(a.work), Path(a.run_dir), Path(a.export_dir)
    run.mkdir(parents=True, exist_ok=True)
    index = json.loads((work / "index.json").read_text())
    config = {k: getattr(a, k) for k in ("epochs", "batch_size", "learning_rate", "kl_weight", "seed")}
    config.update({"model": "MobileNetV3Large+Dense768+L2", "init": "imagenet"})
    ident = {"config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
             "index_sha256": sha256_file(work / "index.json"), "code_sha256": sha256_file(Path(__file__)),
             "text_sha256": sha256_file(work / "text_e31.npz")}
    state_p = run / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"epoch": 0, "step_in_epoch": 0, "history": [], **ident, "config": config}
    for k, v in ident.items():
        if state.get(k) != v:
            raise SystemExit(f"refusing to resume: {k} changed")

    tr_files, tr_emb = load_split(work, index, "train")
    va_files, va_emb = load_split(work, index, "val")
    t = np.load(work / "text_e31.npz")
    T = tf.constant(t["T"].astype(np.float32))
    scale, bias = float(t["scale"]), float(t["bias"])
    steps_per_epoch = len(tr_files) // a.batch_size
    total = steps_per_epoch * a.epochs

    model = build_model(tf, "imagenet")
    sched = tf.keras.optimizers.schedules.CosineDecay(a.learning_rate, total)
    opt = tf.keras.optimizers.Adam(sched)
    opt.build(model.trainable_variables)  # slots exist before a restore, so a checkpoint written before the first step keeps them
    ckpt = tf.train.Checkpoint(model=model, optimizer=opt)
    mgr = tf.train.CheckpointManager(ckpt, str(run / "ckpt"), max_to_keep=2)
    if mgr.latest_checkpoint:
        ckpt.restore(mgr.latest_checkpoint).assert_existing_objects_matched()
        print(json.dumps({"resumed": mgr.latest_checkpoint, "epoch": state["epoch"], "step_in_epoch": state["step_in_epoch"]}), flush=True)

    def load(path):
        return tf.cast(tf.io.decode_jpeg(tf.io.read_file(path), channels=3), tf.float32)

    def loss_fn(s, te):
        cos = tf.reduce_sum(s * te, axis=-1)
        ls = scale * tf.matmul(s, T, transpose_b=True) + bias
        lt = scale * tf.matmul(te, T, transpose_b=True) + bias
        pt = tf.nn.softmax(lt)
        kl = tf.reduce_sum(pt * (tf.nn.log_softmax(lt) - tf.nn.log_softmax(ls)), axis=-1)
        return tf.reduce_mean(1.0 - cos) + a.kl_weight * tf.reduce_mean(kl), tf.reduce_mean(cos), tf.reduce_mean(kl)

    @tf.function
    def train_step(x, te):
        with tf.GradientTape() as tape:
            s = model(x, training=True)
            loss, cos, kl = loss_fn(s, te)
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss, cos, kl

    @tf.function
    def eval_step(x, te):
        return loss_fn(model(x, training=False), te)

    stop = {"flag": False}

    def on_signal(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def save(reason: str):
        path = mgr.save()
        state_p.with_suffix(".tmp").write_text(json.dumps(state, indent=1))
        state_p.with_suffix(".tmp").replace(state_p)
        print(json.dumps({"checkpoint": path, "reason": reason, "epoch": state["epoch"], "step_in_epoch": state["step_in_epoch"]}), flush=True)

    va_ds = tf.data.Dataset.from_tensor_slices((va_files, va_emb)).map(lambda p, e: (load(p), e), num_parallel_calls=2).batch(a.batch_size)
    t_start = time.time()
    last_ckpt = time.time()
    while state["epoch"] < a.epochs:
        ep = state["epoch"]
        order = np.random.default_rng(a.seed * 1000 + ep).permutation(len(tr_files))[: steps_per_epoch * a.batch_size]
        files = [tr_files[i] for i in order]
        ds = tf.data.Dataset.from_tensor_slices((files, tr_emb[order])).skip(state["step_in_epoch"] * a.batch_size)
        ds = ds.map(lambda p, e: (load(p), e), num_parallel_calls=2).batch(a.batch_size, drop_remainder=True).prefetch(2)
        run_loss, n = 0.0, 0
        for x, te in ds:
            loss, cos, kl = train_step(x, te)
            state["step_in_epoch"] += 1
            run_loss += float(loss)
            n += 1
            if state["step_in_epoch"] % 20 == 0:
                print(json.dumps({"epoch": ep, "step": state["step_in_epoch"], "of": steps_per_epoch, "loss": round(run_loss / n, 4),
                                  "cos": round(float(cos), 4), "kl": round(float(kl), 4), "elapsed_min": round((time.time() - t_start) / 60, 1),
                                  "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}), flush=True)
            if stop["flag"]:
                save("signal")
                return
            if time.time() - last_ckpt > a.ckpt_minutes * 60:
                save("interval")
                last_ckpt = time.time()
        vl = [eval_step(x, te) for x, te in va_ds]
        val = {"loss": float(np.mean([v[0] for v in vl])), "cos": float(np.mean([v[1] for v in vl])), "kl": float(np.mean([v[2] for v in vl]))}
        model.save_weights(str(run / f"weights_epoch{ep}.h5"))
        state["history"].append({"epoch": ep, "train_loss": run_loss / max(n, 1), "val": val})
        state["epoch"], state["step_in_epoch"] = ep + 1, 0
        save("epoch_end")
        print(json.dumps({"epoch_done": ep, "val": val}), flush=True)

    best = min(state["history"], key=lambda h: h["val"]["loss"])
    model.load_weights(str(run / f"weights_epoch{best['epoch']}.h5"))
    exp.mkdir(parents=True, exist_ok=True)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    files = {"student_float32.tflite": conv.convert()}
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    files["student_float16.tflite"] = conv.convert()
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    files["student_int8.tflite"] = conv.convert()
    for name, data in files.items():
        (exp / name).write_bytes(data)
    model.save_weights(str(exp / "student_best.weights.h5"))
    record = {"schema": "petperception.student_run/1", "experiment": "E33", "selected_epoch": best["epoch"], "history": state["history"],
              "config": config, **ident, "train_crops": len(tr_files), "val_crops": len(va_files),
              "files": {n: {"bytes": (exp / n).stat().st_size, "sha256": sha256_file(exp / n)} for n in sorted(list(files) + ["student_best.weights.h5"])},
              "input": "uint8 RGB 224x224 as float32 0-255 (MobileNetV3 built-in preprocessing)", "output": "L2-normalized 768-d, SigLIP image space",
              "versions": {"tensorflow": tf.__version__, "python": platform.python_version()}, "train_minutes": round((time.time() - t_start) / 60, 1)}
    (exp / "run_record.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"exported": str(exp), "selected_epoch": best["epoch"]}), flush=True)


if __name__ == "__main__":
    main()
