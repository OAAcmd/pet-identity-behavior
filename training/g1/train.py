"""G1: train, evaluate and export the pet-and-household detector (MediaPipe Model Maker).

Runs in the G1 training environment (training/g1/requirements.txt; GPU
container: training/g1/Dockerfile). Inputs are the COCO folders written by
prepare_coco.py. Outputs in --export-dir:

* model_float32.tflite: raw box encodings and class scores, no in-graph NMS
  (decoded on the phone like the E14 detector); metadata.json holds anchors;
* model_int8_qat.tflite (unless --no-qat): quantization-aware-trained int8;
* metrics.json: versions, hyperparameters, data sizes, COCO metrics of the
  development split for float32 and int8, per-epoch history, wall time.

Selection uses the development split only (Flickr authors disjoint from
train). The product evaluation (E4 split, E7 rejection protocol, E18 real
video, browser parity) runs afterwards in the main environment on the
exported file, once.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="var/training/g1/coco")
    ap.add_argument("--export-dir", default="var/training/g1/export")
    ap.add_argument("--model", default="MOBILENET_MULTI_AVG_I384",
                    choices=["MOBILENET_V2", "MOBILENET_V2_I320", "MOBILENET_MULTI_AVG", "MOBILENET_MULTI_AVG_I384"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=0.3)
    ap.add_argument("--qat-epochs", type=int, default=10)
    ap.add_argument("--no-qat", action="store_true")
    ap.add_argument("--max-images", type=int, default=None, help="smoke test")
    a = ap.parse_args()

    import tensorflow as tf
    from mediapipe_model_maker import object_detector

    t0 = time.time()
    exp = Path(a.export_dir)
    exp.mkdir(parents=True, exist_ok=True)
    cache = Path(a.data) / "_cache"
    train = object_detector.Dataset.from_coco_folder(str(Path(a.data) / "train"), cache_dir=str(cache / "train"), max_num_images=a.max_images)
    dev = object_detector.Dataset.from_coco_folder(str(Path(a.data) / "development"), cache_dir=str(cache / "development"), max_num_images=a.max_images)
    hp = object_detector.HParams(export_dir=str(exp), epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.learning_rate)
    opts = object_detector.ObjectDetectorOptions(supported_model=getattr(object_detector.SupportedModels, a.model), hparams=hp)
    model = object_detector.ObjectDetector.create(train_data=train, validation_data=dev, options=opts)
    history = {k: [float(x) for x in v] for k, v in (getattr(model, "_history", None).history.items() if getattr(model, "_history", None) else [])}
    t_train = time.time() - t0
    loss, coco = model.evaluate(dev, batch_size=min(8, a.batch_size))
    model.export_model("model_float32.tflite")
    out = {"job": "G1", "model_spec": a.model, "epochs": a.epochs, "batch_size": a.batch_size, "learning_rate": a.learning_rate,
           "train_images": train.size, "development_images": dev.size, "labels": train.label_names,
           "float32": {"development_loss": [float(x) for x in (loss if isinstance(loss, (list, tuple)) else [loss])],
                       "development_coco": {k: float(v) for k, v in coco.items()}},
           "history": history, "train_seconds": t_train,
           "versions": {"tensorflow": tf.__version__, "python": platform.python_version(), "gpus": [d.name for d in tf.config.list_physical_devices("GPU")]}}
    if not a.no_qat:
        qat = object_detector.QATHParams(learning_rate=0.3, batch_size=a.batch_size, epochs=a.qat_epochs, decay_steps=6, decay_rate=0.96)
        model.quantization_aware_training(train, dev, qat_hparams=qat)
        qloss, qcoco = model.evaluate(dev, batch_size=min(8, a.batch_size))
        model.export_model("model_int8_qat.tflite")
        out["int8_qat"] = {"qat_epochs": a.qat_epochs, "development_coco": {k: float(v) for k, v in qcoco.items()}}
    out["total_seconds"] = time.time() - t0
    (exp / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("train_images", "development_images", "total_seconds")}))


if __name__ == "__main__":
    main()
