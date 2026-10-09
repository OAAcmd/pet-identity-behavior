"""Deterministic model-level parity fixtures (reference: Python LiteRT).

Inputs are generated from integer-only formulas plus a xorshift32 stream so
the browser (web/src/engine/parity.ts) reproduces them bit-exactly without
shipping media. They exercise every operator of each model; they are not
meaningful images or sounds, so they test numerical agreement only.
End-to-end agreement on real media is tested separately (engine parity).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from ..utils import sha256_file


def xorshift32(seed: int, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.uint32)
    x = seed & 0xFFFFFFFF
    for i in range(n):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out[i] = x
    return out


def synthetic_audio(n: int = 15600, seed: int = 12345) -> np.ndarray:
    r = xorshift32(seed, n)
    k = np.arange(n, dtype=np.int64)
    tri = ((k * 7) % 200 - 100).astype(np.float64) / 400.0
    sq = np.where((k // 40) % 2 == 0, 0.1, -0.1)
    noise = ((r >> 16).astype(np.float64) / 65536.0 - 0.5) * 0.1
    return (tri + sq + noise).astype(np.float32)


def synthetic_image(size: int, seed: int = 777) -> np.ndarray:
    r = xorshift32(seed, size * size * 3).reshape(size, size, 3)
    y, x, c = np.meshgrid(np.arange(size), np.arange(size), np.arange(3), indexing="ij")
    v = (x * 3 + y * 5 + c * 70 + ((x // 16 + y // 16) % 2) * 60 + (r >> 24) % 32) % 256
    return v.astype(np.uint8)


def _run(model_bytes: bytes, inputs: list[np.ndarray]) -> list[np.ndarray]:
    from ai_edge_litert.interpreter import Interpreter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        it = Interpreter(model_content=model_bytes, num_threads=1)
    it.allocate_tensors()
    for d, x in zip(it.get_input_details(), inputs):
        it.set_tensor(d["index"], x.reshape(d["shape"]).astype(d["dtype"]))
    it.invoke()
    return [it.get_tensor(d["index"]) for d in it.get_output_details()]


CASES = {
    # name: (input builder, input dtype note)
    "yamnet": lambda: [synthetic_audio()],
    "detector_float": lambda: [((synthetic_image(320).astype(np.float32) - 127.5) / 127.5)[None]],
    "detector_uint8": lambda: [synthetic_image(320)[None]],
    "backbone": lambda: [((synthetic_image(224).astype(np.float32) - 127.0) / 128.0)[None]],
}


def export(models: dict[str, Path], out_dir: Path) -> dict:
    """models: case name -> .tflite path. Writes raw float32/uint8 arrays and a manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"schema": "petperception.parity/1", "cases": {}}
    for name, path in models.items():
        inputs = CASES[name]()
        outs = _run(path.read_bytes(), inputs)
        case = {"model_file": path.name, "model_sha256": sha256_file(path), "inputs": [], "outputs": []}
        for i, x in enumerate(inputs):
            f = out_dir / f"{name}_in{i}.bin"
            f.write_bytes(np.ascontiguousarray(x).tobytes())
            case["inputs"].append({"file": f.name, "dtype": str(x.dtype), "shape": list(x.shape)})
        for i, y in enumerate(outs):
            f = out_dir / f"{name}_out{i}.bin"
            y32 = np.ascontiguousarray(y.astype(np.float32))
            f.write_bytes(y32.tobytes())
            case["outputs"].append({"file": f.name, "shape": list(y.shape), "mean_abs": float(np.abs(y32).mean()), "max_abs": float(np.abs(y32).max())})
        manifest["cases"][name] = case
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
