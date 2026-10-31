"""E33 behaviour-encoder student: the committed exports load in Python LiteRT, give unit-length 768-d SigLIP-space embeddings,
agree across precisions, and match the run record. Evaluation results are in results/e33."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from petperception.config import REPO_ROOT
from petperception.ondevice.parity import CASES, synthetic_image

D = REPO_ROOT / "models" / "student" / "e33_mnv3l"


def _run(path: Path, x: np.ndarray) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=str(path), num_threads=1)
    it.allocate_tensors()
    i = it.get_input_details()[0]
    assert list(i["shape_signature"]) == [1, 224, 224, 3]  # fixed batch: LiteRT.js cannot allocate a dynamic one
    it.set_tensor(i["index"], x)
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])[0]


@pytest.mark.skipif(not (D / "run_record.json").exists(), reason="student exports not present")
def test_student_exports_match_record_and_agree():
    rec = json.loads((D / "run_record.json").read_text())
    for name, meta in rec["files"].items():
        assert hashlib.sha256((D / name).read_bytes()).hexdigest() == meta["sha256"], name
    x = CASES["student_float32"]()[0]
    assert x.dtype == np.float32 and x.max() <= 255 and np.array_equal(x[0], synthetic_image(224).astype(np.float32))
    out = {k: _run(D / f"student_{k}.tflite", x) for k in ("float32", "float16", "int8")}
    for k, e in out.items():
        assert e.shape == (768,) and abs(float(np.linalg.norm(e)) - 1.0) < 1e-3, k
    assert float(out["float32"] @ out["float16"]) > 0.999 and float(out["float32"] @ out["int8"]) > 0.99
