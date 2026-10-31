"""E33: export the behaviour-encoder student to TFLite with a fixed batch of 1 (float32, float16, int8 dynamic range).

LiteRT.js cannot allocate an input tensor whose batch dimension is dynamic (Keras exports have shape signature
[-1, 224, 224, 3]), so the exported graphs take exactly one 224 x 224 crop. Usage (TensorFlow environment):

    var/venv-g1/bin/python training/student/export_student.py --weights models/student/e33_mnv3l/student_best.weights.h5 \\
        --out models/student/e33_mnv3l [--compare-with <dir of earlier exports> --crops var/student/crops/val]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fixed_batch(tf, model):
    inp = tf.keras.Input((224, 224, 3), batch_size=1, name="crop")
    return tf.keras.Model(inp, model(inp), name="student_mnv3l_b1")


def export(tf, model, out: Path) -> dict[str, dict]:
    m1 = fixed_batch(tf, model)
    out.mkdir(parents=True, exist_ok=True)
    variants = {"student_float32.tflite": None, "student_float16.tflite": "float16", "student_int8.tflite": "int8"}
    for name, kind in variants.items():
        conv = tf.lite.TFLiteConverter.from_keras_model(m1)
        if kind:
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if kind == "float16":
            conv.target_spec.supported_types = [tf.float16]
        (out / name).write_bytes(conv.convert())
    return {n: {"bytes": (out / n).stat().st_size, "sha256": sha256_file(out / n)} for n in variants}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-with", help="directory with earlier exports to compare outputs against")
    ap.add_argument("--crops", help="directory of crops (jpg) for the comparison")
    a = ap.parse_args()
    import sys

    import tensorflow as tf

    sys.path.insert(0, str(Path(__file__).parent))
    from train_student import build_model

    model = build_model(tf, None)
    model.load_weights(a.weights)
    files = export(tf, model, Path(a.out))
    report = {"files": files}
    if a.compare_with and a.crops:
        import cv2

        imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) for p in sorted(Path(a.crops).glob("*.jpg"))[:64]]
        for name in files:
            outs = []
            for path in (Path(a.out) / name, Path(a.compare_with) / name):
                it = tf.lite.Interpreter(model_path=str(path))
                it.allocate_tensors()
                i, o = it.get_input_details()[0], it.get_output_details()[0]
                ys = []
                for im in imgs:
                    it.set_tensor(i["index"], im[None])
                    it.invoke()
                    ys.append(it.get_tensor(o["index"])[0].copy())
                outs.append(np.array(ys))
                report.setdefault("input_shape_signature", {})[f"{path.parent.name}/{name}"] = it.get_input_details()[0]["shape_signature"].tolist()
            report.setdefault("max_abs_diff_vs_earlier", {})[name] = float(np.abs(outs[0] - outs[1]).max())
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
