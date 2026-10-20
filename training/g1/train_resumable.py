"""G1: resumable detector training on CPU (MediaPipe Model Maker RetinaNet, own loop).

Why an own loop: Model Maker's `create()` trains in one uninterrupted Keras
`fit` and saves weights only at the end. A training run here spans several
sessions, so this loop checkpoints everything needed to continue exactly
where it stopped:

* model weights and optimizer state (momentum slots, iteration counter that
  drives the warm-up + cosine learning-rate schedule) - TF checkpoint;
* training position: epoch and step within the epoch;
* random state: the per-epoch data-order seed (the shuffled order is
  re-created exactly and the completed steps are skipped), the TF seed used
  for augmentation, NumPy and Python RNG states;
* identifiers: SHA-256 of the run configuration, of the training and
  development label files, of this script, and the package versions. A
  resume refuses to continue if the configuration or data changed.

Checkpoints are written every --ckpt-minutes, at every epoch end and on
SIGTERM/SIGINT (CheckpointManager keeps the last 3). Each checkpoint has a
sidecar `state-<n>.json` and a `latest.json` pointer written atomically.
Exact replay of augmentation is not guaranteed: augmentation runs in a
parallel tf.data map, so the interleaving of random draws can differ
after a resume; data order and every other state are exact.

Training stages (--stages): a comma-separated list of `<epochs>:<trainable>`
with trainable in {all, head, decoder_head}, for example `2:decoder_head,8:all`
(staged fine-tuning) or `10:all` (full fine-tuning). Frozen layers run in
inference mode (BatchNorm statistics fixed).

Initialisation (--init): `coco` (default) starts backbone, decoder and the
head's convolutions from Model Maker's COCO-trained RetinaNet checkpoint (the
classifier layer is always re-initialised for the product classes);
`scratch` skips that checkpoint, so no COCO-derived weight is used. Both are
recorded in the configuration hash. `--init-from <checkpoint>` then loads all
model weights of an earlier G1 run (same model and classes): a warm start with
a new optimizer and schedule, recorded in the configuration hash with the
source weights' SHA-256.

Benchmark mode (--benchmark N): measures data loading alone, N training
steps (after warm-up), a checkpoint write and restore, and a validation
pass, then exits; results go to <run-dir>/benchmark.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def atomic_write(p: Path, text: str) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


def parse_stages(spec: str) -> list[tuple[int, str]]:
    out = []
    for part in spec.split(","):
        n, what = part.split(":")
        if what not in ("all", "head", "decoder_head"):
            raise ValueError(f"unknown trainable set {what}")
        out.append((int(n), what))
    return out


def stage_for_epoch(stages: list[tuple[int, str]], epoch: int) -> str:
    e = 0
    for n, what in stages:
        e += n
        if epoch < e:
            return what
    return stages[-1][1]


def _iou(b: np.ndarray, g: np.ndarray) -> np.ndarray:
    y0, x0 = np.maximum(b[0], g[:, 0]), np.maximum(b[1], g[:, 1])
    y1, x1 = np.minimum(b[2], g[:, 2]), np.minimum(b[3], g[:, 3])
    inter = np.clip(y1 - y0, 0, None) * np.clip(x1 - x0, 0, None)
    area = (b[2] - b[0]) * (b[3] - b[1]) + (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) - inter
    return inter / np.maximum(area, 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float, max_out: int) -> list[int]:
    order, keep = list(np.argsort(-scores)), []
    while order and len(keep) < max_out:
        i = order.pop(0)
        keep.append(i)
        if order:
            ious = _iou(boxes[i], boxes[order])
            order = [o for o, v in zip(order, ious) if v < thr]
    return keep


# Cache layout written by install_coco_writer(); a new name so caches of the stock writer are never reused.
CACHE_LAYOUT = "coco-v2-negatives-crowd"


def coco_examples(labels: dict, image_bytes, decode_hw):
    """Yield one tf.train.Example feature dict per COCO image, including images without boxes.

    Model Maker 0.2.1.4's COCO cache writer differs from what G1 needs in two ways:
    * it skips every image without a box, which silently removes the hard
      negatives (confusable animals, dolls) that G1 selects on purpose;
    * it does not write `is_crowd`, so its evaluation ground truth treats
      Open Images group-of boxes as ordinary boxes.
    This version keeps box-less images (empty box lists) and writes is_crowd.
    Training is unchanged for group-of boxes: Model Maker's preprocessor
    trains on them as ordinary boxes and uses is_crowd only in evaluation
    ground truth. Boxes outside the image or without area are dropped, as in
    the stock writer. `image_bytes(file_name)` returns the encoded JPEG and
    `decode_hw(bytes)` its (height, width).
    """
    by_image: dict[int, list] = {}
    for a in labels["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    for img in labels["images"]:
        enc = image_bytes(img["file_name"])
        h, w = decode_hw(enc)
        xmin, xmax, ymin, ymax, cls, crowd = [], [], [], [], [], []
        for a in by_image.get(img["id"], []):
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0 or x + bw > w or y + bh > h:
                continue
            xmin.append(x / w)
            xmax.append((x + bw) / w)
            ymin.append(y / h)
            ymax.append((y + bh) / h)
            cls.append(int(a["category_id"]))
            crowd.append(int(a.get("iscrowd", 0)))
        yield img, enc, h, w, {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "category_id": cls, "is_crowd": crowd}


def install_coco_writer() -> None:
    """Replace Model Maker's COCO cache writer with coco_examples() (see there)."""
    import tensorflow as tf
    from mediapipe_model_maker.python.vision.object_detector import dataset_util
    from official.vision.data import tfrecord_lib

    def get_example(self, data_dir: str):  # noqa: ARG001 - signature of CacheFilesWriter._get_example
        data_dir = os.path.abspath(data_dir)
        labels = json.loads(Path(data_dir, "labels.json").read_text())

        def image_bytes(name: str) -> bytes:
            return Path(data_dir, "images", name).read_bytes()

        def decode_hw(enc: bytes) -> tuple[int, int]:
            s = tf.io.decode_jpeg(enc, channels=3).shape
            return int(s[0]), int(s[1])

        for img, enc, h, w, d in coco_examples(labels, image_bytes, decode_hw):
            f = tfrecord_lib.image_info_to_feature_dict(h, w, img["file_name"], img["id"], enc, "jpg")
            for key, vals, kind in (("xmin", d["xmin"], "float_list"), ("xmax", d["xmax"], "float_list"), ("ymin", d["ymin"], "float_list"),
                                    ("ymax", d["ymax"], "float_list")):
                f[f"image/object/bbox/{key}"] = tfrecord_lib.convert_to_feature(vals, kind)
            f["image/object/class/label"] = tfrecord_lib.convert_to_feature(d["category_id"], "int64_list")
            f["image/object/is_crowd"] = tfrecord_lib.convert_to_feature(d["is_crowd"], "int64_list")
            yield tf.train.Example(features=tf.train.Features(feature=f))

    dataset_util.COCOCacheFilesWriter._get_example = get_example


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True, help="COCO view directory with train/ and development/")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default="MOBILENET_V2")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=0.3, help="Model Maker convention: scaled by batch/256")
    ap.add_argument("--stages", default="10:all")
    ap.add_argument("--init", choices=["coco", "scratch"], default="coco", help="COCO-pretrained RetinaNet weights or random initialisation")
    ap.add_argument("--init-from", default="", help="warm start: model weights (not optimizer or position) from a checkpoint prefix of an "
                                                       "earlier G1 run with the same model and classes, e.g. models/g1/<run>/ckpt/ckpt-27")
    ap.add_argument("--val-every", type=int, default=1, help="validate every K epochs (the last epoch is always validated)")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--shuffle-buffer", type=int, default=0,
                    help="seeded shuffle buffer in examples (0 = the whole training set). Model Maker yields decoded images (~230 KB at "
                         "320 px), so a full buffer over 50k images needs ~12 GB of RAM; the cache is in random image-id order already")
    ap.add_argument("--ckpt-minutes", type=float, default=10.0)
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop cleanly (with a checkpoint) after this wall time; 0 = no limit")
    ap.add_argument("--max-steps", type=int, default=0, help="stop cleanly after this many steps in this invocation (tests)")
    ap.add_argument("--benchmark", type=int, default=0, help="benchmark N training steps and exit")
    ap.add_argument("--threads", type=int, default=0, help="TF intra/inter op threads (0 = default)")
    ap.add_argument("--validate-only", action="store_true", help="restore the latest checkpoint of --run-dir, validate, print and exit")
    a = ap.parse_args()

    import tensorflow as tf

    if a.threads:
        tf.config.threading.set_intra_op_parallelism_threads(a.threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    from mediapipe_model_maker import object_detector
    from mediapipe_model_maker.python.vision.object_detector import model_options as model_opt
    from mediapipe_model_maker.python.vision.object_detector import model_spec as ms
    from mediapipe_model_maker.python.vision.object_detector import object_detector as od_mod

    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    view = Path(a.view)
    stages = parse_stages(a.stages)
    total_epochs = sum(n for n, _ in stages)
    config = {"view": str(view), "model": a.model, "batch_size": a.batch_size, "learning_rate": a.learning_rate, "stages": a.stages, "seed": a.seed,
              "cache_layout": CACHE_LAYOUT}
    if a.init != "coco":
        config["init"] = a.init
    if a.init_from:  # the source weights are part of the configuration
        src_state = Path(a.init_from).parent.parent / "state.json"
        src_state = src_state if src_state.exists() else Path(a.init_from + ".state.json")
        config["init_from"] = {"checkpoint": a.init_from, "weights_sha256": json.loads(src_state.read_text())["weights_sha256"] if src_state.exists() else None}
    if a.shuffle_buffer:  # recorded only when set, so earlier runs keep their configuration hash
        config["shuffle_buffer"] = a.shuffle_buffer
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    data_ids = {s: sha256_file(view / s / "labels.json") for s in ("train", "development")}
    code_sha = sha256_file(Path(__file__))

    latest_p = run / "latest.json"
    prev = json.loads(latest_p.read_text()) if latest_p.exists() else None
    if prev and not a.benchmark and not a.validate_only:
        if prev["config_sha256"] != config_sha or prev["data_sha256"] != data_ids:
            sys.exit(f"refusing to resume: configuration or data changed since {prev['checkpoint']}")

    t_load = time.time()
    install_coco_writer()
    cache = view / "_cache" / CACHE_LAYOUT
    train_data = object_detector.Dataset.from_coco_folder(str(view / "train"), cache_dir=str(cache / "train"))
    dev_data = object_detector.Dataset.from_coco_folder(str(view / "development"), cache_dir=str(cache / "development"))
    t_load = time.time() - t_load
    n_train = len(train_data)
    steps_per_epoch = n_train // a.batch_size

    hp = object_detector.HParams(export_dir=str(run / "export"), epochs=total_epochs, batch_size=a.batch_size, learning_rate=a.learning_rate)
    det = od_mod.ObjectDetector(model_spec=ms.SupportedModels.get(getattr(object_detector.SupportedModels, a.model)),
                                label_names=train_data.label_names, hparams=hp, model_options=model_opt.ObjectDetectorModelOptions())
    det._optimizer = det._create_optimizer(steps_per_epoch)  # warm-up + cosine schedule over all epochs
    if a.init == "scratch":
        from mediapipe_model_maker.python.vision.object_detector import model as model_lib

        def build_only(self, checkpoint_path, include_last_layer=False):  # noqa: ARG001 - replaces the COCO checkpoint load
            self._model(tf.zeros([1] + self._model_spec.input_image_shape), training=True)

        model_lib.ObjectDetectorModel.load_checkpoint = build_only
    det._create_model()  # COCO-pretrained backbone, decoder and head (classifier re-initialised)
    if a.init_from:
        tf.train.Checkpoint(model=det._model._model).restore(a.init_from).expect_partial().assert_existing_objects_matched()
        print(json.dumps({"warm_start_from": a.init_from}), flush=True)
    model, opt, pre = det._model, det._optimizer, det._preprocessor
    inner = model._model
    opt.build(inner.trainable_variables)  # slots for every variable, so stages can change the trainable set

    epoch_v = tf.Variable(0, dtype=tf.int64, trainable=False)
    step_v = tf.Variable(0, dtype=tf.int64, trainable=False)
    ckpt = tf.train.Checkpoint(model=inner, optimizer=opt, epoch=epoch_v, step_in_epoch=step_v)
    mgr = tf.train.CheckpointManager(ckpt, str(run / "ckpt"), max_to_keep=3)
    history: list[dict] = []
    if prev and not a.benchmark and not a.validate_only:
        ckpt.restore(prev["checkpoint"]).assert_existing_objects_matched()
        history = prev.get("history", [])
        np.random.set_state(tuple(prev["numpy_rng_state"][:1]) + (np.array(prev["numpy_rng_state"][1], dtype=np.uint32),)
                            + tuple(prev["numpy_rng_state"][2:]))
        random.setstate((prev["python_rng_state"][0], tuple(prev["python_rng_state"][1]), prev["python_rng_state"][2]))
        print(json.dumps({"resumed_from": prev["checkpoint"], "epoch": int(epoch_v.numpy()), "step_in_epoch": int(step_v.numpy())}), flush=True)
    else:
        np.random.seed(a.seed)
        random.seed(a.seed)

    def weights_sha() -> str:
        h = hashlib.sha256()
        for v in inner.weights:
            h.update(v.numpy().tobytes())
        return h.hexdigest()

    def save(reason: str) -> str:
        path = mgr.save()
        ns = np.random.get_state()
        st = {"checkpoint": path, "reason": reason, "epoch": int(epoch_v.numpy()), "step_in_epoch": int(step_v.numpy()),
              "optimizer_iterations": int(opt.iterations.numpy()), "steps_per_epoch": steps_per_epoch, "total_epochs": total_epochs,
              "config": config, "config_sha256": config_sha, "data_sha256": data_ids, "code_sha256": code_sha,
              "data_order": "per-epoch shuffle (buffer: config shuffle_buffer, default the whole set) seeded with seed*1000+epoch; completed steps are skipped on resume",
              "tf_augmentation_seed_rule": "tf.random.set_seed(seed*1000+epoch) before each epoch iterator",
              "numpy_rng_state": [ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]], "python_rng_state": [random.getstate()[0], list(random.getstate()[1]), random.getstate()[2]],
              "weights_sha256": weights_sha(), "history": history, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "versions": {"tensorflow": tf.__version__, "python": platform.python_version()}}
        atomic_write(Path(path + ".state.json"), json.dumps(st))
        atomic_write(latest_p, json.dumps(st))
        return path

    def trainable_for(what: str) -> list:
        inner.backbone.trainable = what == "all"
        inner.decoder.trainable = what in ("all", "decoder_head")
        inner.head.trainable = True
        return inner.trainable_variables

    def make_step(variables):
        @tf.function(reduce_retracing=True)
        def step(x, y):
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = model.compute_loss(x, y, y_pred)
            grads = tape.gradient(loss, variables)
            opt.apply_gradients(zip(grads, variables))
            return loss
        return step

    def epoch_dataset(epoch: int):
        seed = a.seed * 1000 + epoch
        tf.random.set_seed(seed)
        raw = train_data._dataset.shuffle(a.shuffle_buffer or n_train, seed=seed, reshuffle_each_iteration=False)
        ep = object_detector.Dataset(raw, train_data.label_names, n_train)
        return ep.gen_tf_dataset(batch_size=a.batch_size, is_training=True, shuffle=False, preprocess=pre, drop_remainder=True)

    dev_ds = dev_data.gen_tf_dataset(a.batch_size, is_training=False, preprocess=pre)
    label_names = list(train_data.label_names)

    def validate() -> dict:
        """One pass over the development split: losses and per-class AP50.

        Raw outputs (backbone, decoder, head in inference mode) are decoded with
        the model's box coder (dy, dx, dh, dw on the anchors), scored with a
        sigmoid, suppressed per class in NumPy (IoU 0.5, at most 100 per image)
        and matched to the ground truth at IoU 0.5; group-of (crowd) boxes are
        ignored, as in COCO. This is a training-time selection metric; the
        product evaluation runs on the exported model afterwards.
        """
        t = time.time()
        for m in model.metrics:
            m.reset_state()
        dets: dict[int, list] = {c: [] for c in range(1, len(label_names))}
        npos = {c: 0 for c in range(1, len(label_names))}
        crowd: dict[tuple, list] = {}
        gts: dict[tuple, list] = {}
        img = 0
        for x, y in dev_ds:
            feats = inner.backbone(x, training=False)
            feats = inner.decoder(feats, training=False)
            head_out = inner.head(feats, training=False)
            cls_out, box_out = head_out[0], head_out[1]
            model.compute_loss(x, y, {"cls_outputs": cls_out, "box_outputs": box_out})
            levels = sorted(cls_out)
            nb = int(x.shape[0])
            cls = np.concatenate([cls_out[k].numpy().reshape(nb, -1, len(label_names)) for k in levels], 1)
            box = np.concatenate([box_out[k].numpy().reshape(nb, -1, 4) for k in levels], 1)
            anc = np.concatenate([y["anchor_boxes"][k].numpy()[0].reshape(-1, 4) for k in levels], 0)
            ah, aw = anc[:, 2] - anc[:, 0], anc[:, 3] - anc[:, 1]
            ayc, axc = anc[:, 0] + 0.5 * ah, anc[:, 1] + 0.5 * aw
            gt = y["groundtruths"]
            for b in range(nb):
                scale = y["image_info"][b, 2].numpy()
                d = box[b]
                yc, xc = d[:, 0] * ah + ayc, d[:, 1] * aw + axc
                hh, ww = np.exp(np.minimum(d[:, 2], 4.135)) * ah, np.exp(np.minimum(d[:, 3], 4.135)) * aw
                bx = np.stack([yc - hh / 2, xc - ww / 2, yc + hh / 2, xc + ww / 2], 1) / np.array([scale[0], scale[1], scale[0], scale[1]])
                sc = 1 / (1 + np.exp(-cls[b]))
                n_gt = int(gt["num_detections"][b].numpy().reshape(-1)[0])
                gb, gc, gcr = gt["boxes"][b].numpy()[:n_gt], gt["classes"][b].numpy()[:n_gt], gt["is_crowds"][b].numpy()[:n_gt]
                for c in range(1, len(label_names)):
                    m_ = gc == c
                    gts[(img, c)] = [g for g, cr in zip(gb[m_], gcr[m_]) if not cr]
                    crowd[(img, c)] = [g for g, cr in zip(gb[m_], gcr[m_]) if cr]
                    npos[c] += len(gts[(img, c)])
                    s = sc[:, c]
                    idx = np.where(s >= 0.05)[0]
                    idx = idx[np.argsort(-s[idx])][:300]
                    for k in _nms(bx[idx], s[idx], 0.5, 100):
                        dets[c].append((float(s[idx[k]]), img, bx[idx[k]]))
                img += 1
        ap = {}
        for c in range(1, len(label_names)):
            if npos[c] == 0:
                continue
            used: dict[tuple, np.ndarray] = {}
            tp, fp = [], []
            for _, im, b in sorted(dets[c], key=lambda r: -r[0]):
                g = gts[(im, c)]
                if g:
                    ious = _iou(b, np.array(g))
                    u = used.setdefault((im, c), np.zeros(len(g), bool))
                    j = int(np.argmax(np.where(u, -1, ious)))
                    if ious[j] >= 0.5 and not u[j]:
                        u[j] = True
                        tp.append(1)
                        fp.append(0)
                        continue
                if crowd[(im, c)] and _iou(b, np.array(crowd[(im, c)])).max() >= 0.5:
                    continue
                tp.append(0)
                fp.append(1)
            tpc, fpc = np.cumsum(tp), np.cumsum(fp)
            rec = tpc / npos[c] if len(tpc) else np.array([0.0])
            prec = tpc / np.maximum(tpc + fpc, 1) if len(tpc) else np.array([0.0])
            mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[0], prec, [0]])
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            ii = np.where(mrec[1:] != mrec[:-1])[0]
            ap[label_names[c]] = {"ap50": float(np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii + 1])), "positives": npos[c]}
        out = {"dev_" + m.name: float(m.result().numpy()) for m in model.metrics}
        out["dev_ap50"] = ap
        out["dev_map50"] = float(np.mean([v["ap50"] for v in ap.values()])) if ap else None
        out["seconds"] = time.time() - t
        return out

    stop = {"flag": False}

    def on_signal(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    if a.validate_only:
        if prev is None:
            sys.exit("no checkpoint to validate")
        ckpt.restore(prev["checkpoint"]).assert_existing_objects_matched()
        restored_sha = weights_sha()
        v = validate()
        v["checkpoint"] = prev["checkpoint"]
        v["restore_weights_sha256"] = restored_sha
        v["restore_weights_identical"] = restored_sha == prev["weights_sha256"]
        atomic_write(run / f"validation-{prev['checkpoint'].rsplit('-', 1)[-1]}.json", json.dumps(v, indent=2))
        print(json.dumps(v, indent=2))
        return

    if a.benchmark:
        bench: dict = {"config": config, "cpu_count": os.cpu_count(), "train_images": n_train, "development_images": len(dev_data),
                       "view_images": {s: len(json.loads((view / s / "labels.json").read_text())["images"]) for s in ("train", "development")},
                       "dataset_load_and_cache_seconds": t_load}
        it = iter(epoch_dataset(0))
        t = time.time()
        for _ in range(min(a.benchmark, steps_per_epoch)):
            next(it)
        bench["data_only_images_per_second"] = min(a.benchmark, steps_per_epoch) * a.batch_size / (time.time() - t)
        for what in sorted({w for _, w in stages}, key=["head", "decoder_head", "all"].index):
            variables = trainable_for(what)
            step = make_step(variables)
            it = iter(epoch_dataset(1))
            t = time.time()
            step(*next(it))  # trace
            trace_s = time.time() - t
            times = []
            for _ in range(a.benchmark):
                x, y = next(it)
                t = time.time()
                step(x, y).numpy()
                times.append(time.time() - t)
            bench[f"train_{what}"] = {"trainable_variables": len(variables), "trainable_parameters": int(sum(np.prod(v.shape) for v in variables)),
                                      "trace_seconds": trace_s, "median_step_seconds": float(np.median(times)),
                                      "images_per_second": a.batch_size / float(np.median(times)), "peak_rss_mb_so_far": peak_rss_mb()}
        t = time.time()
        path = save("benchmark")
        bench["checkpoint_write_seconds"] = time.time() - t
        bench["checkpoint_bytes"] = sum(p.stat().st_size for p in Path(path).parent.glob(Path(path).name + "*") if p.is_file())
        before = weights_sha()
        for v in inner.weights:
            v.assign(tf.zeros_like(v))
        t = time.time()
        ckpt.restore(path).assert_existing_objects_matched()
        bench["checkpoint_restore_seconds"] = time.time() - t
        bench["restore_weights_identical"] = weights_sha() == before
        v = validate()
        bench["validation"] = v
        bench["validation_images_per_second"] = len(dev_data) / v["seconds"]
        bench["validation_note"] = "fast validation: raw outputs + NumPy decoding/NMS + AP50 per class"
        bench["peak_rss_mb"] = peak_rss_mb()
        bench["view_disk_bytes"] = sum(p.stat().st_size for p in (view / "_cache").rglob("*") if p.is_file())
        atomic_write(run / "benchmark.json", json.dumps(bench, indent=2))
        print(json.dumps(bench, indent=2))
        return

    t_start = time.time()
    last_ckpt = time.time()
    steps_run = 0
    current_what = None
    step = None
    while int(epoch_v.numpy()) < total_epochs and not stop["flag"]:
        epoch = int(epoch_v.numpy())
        what = stage_for_epoch(stages, epoch)
        if what != current_what:
            step = make_step(trainable_for(what))
            current_what = what
        it = iter(epoch_dataset(epoch))
        skip = int(step_v.numpy())
        for _ in range(skip):  # replay the order up to the resume point
            next(it)
        t_ep = time.time()
        losses = []
        while int(step_v.numpy()) < steps_per_epoch:
            loss = float(step(*next(it)).numpy())
            losses.append(loss)
            step_v.assign_add(1)
            steps_run += 1
            if not np.isfinite(loss):
                save("non_finite_loss")
                sys.exit("non-finite loss; checkpoint written for inspection")
            if steps_run % 20 == 0:
                print(json.dumps({"epoch": epoch, "step": int(step_v.numpy()), "of": steps_per_epoch, "loss": round(float(np.mean(losses[-20:])), 4),
                                  "optimizer_iterations": int(opt.iterations.numpy()),
                                  "elapsed_min": round((time.time() - t_start) / 60, 1), "peak_rss_mb": round(peak_rss_mb())}), flush=True)
            limit = (a.max_minutes and time.time() - t_start > a.max_minutes * 60) or (a.max_steps and steps_run >= a.max_steps)
            if stop["flag"] or limit:
                save("signal" if stop["flag"] else "time_or_step_limit")
                print(json.dumps({"stopped": True, "epoch": epoch, "step_in_epoch": int(step_v.numpy())}), flush=True)
                return
            if time.time() - last_ckpt > a.ckpt_minutes * 60:
                save("periodic")
                last_ckpt = time.time()
        rec = {"epoch": epoch, "stage": what, "train_loss_mean": float(np.mean(losses)) if losses else None, "epoch_minutes": (time.time() - t_ep) / 60,
               "images_per_second": len(losses) * a.batch_size / max(1e-9, time.time() - t_ep)}
        if (epoch + 1) % a.val_every == 0 or epoch + 1 == total_epochs:
            rec.update(validate())
        history.append(rec)
        print(json.dumps(rec), flush=True)
        epoch_v.assign_add(1)
        step_v.assign(0)
        save("epoch_end")
        last_ckpt = time.time()
    if int(epoch_v.numpy()) >= total_epochs:
        (run / "export").mkdir(parents=True, exist_ok=True)  # Model Maker's save_tflite does not create it
        det.export_model("model_float32.tflite")
        det.export_model("model_float16.tflite", quantization_config=__import__("mediapipe_model_maker").quantization.QuantizationConfig.for_float16())
        det.export_model("model_int8_ptq.tflite",
                         quantization_config=__import__("mediapipe_model_maker").quantization.QuantizationConfig.for_int8(representative_data=dev_data))
        summary = {"config": config, "history": history, "exported": {p.name: sha256_file(p) for p in sorted((run / "export").glob("*.tflite"))},
                   "peak_rss_mb": peak_rss_mb()}
        atomic_write(run / "final.json", json.dumps(summary, indent=2))
        print(json.dumps({"finished": True}), flush=True)


if __name__ == "__main__":
    main()
