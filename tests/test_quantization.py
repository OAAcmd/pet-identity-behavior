"""Quantized-model input/output handling (E12) and in-graph dequantized outputs."""

from pathlib import Path

import numpy as np
import pytest

from petperception.vision.tflite_io import dequantize, encode_image

IMG = np.array([[[0, 1, 2], [127, 128, 129], [200, 254, 255]]], dtype=np.uint8)


def test_float_input_is_metadata_normalization():
    x = encode_image(IMG, {"dtype": np.float32, "quantization": (0.0, 0)}, 127.5, 127.5)
    assert x.dtype == np.float32 and x.shape == (1, 1, 3, 3)
    assert np.array_equal(x[0], (IMG.astype(np.float32) - 127.5) / 127.5)


def test_quantized_input_encodes_the_float_input():
    # Same values are asserted in web/tests/unit/budgets.test.ts (encodeImage) for cross-runtime parity.
    u8 = encode_image(IMG, {"dtype": np.uint8, "quantization": (0.012566016986966133, 131)}, 127.0, 128.0)
    assert u8.ravel().tolist() == [52, 53, 53, 131, 132, 132, 176, 210, 211]
    i8 = encode_image(IMG, {"dtype": np.int8, "quantization": (0.0078125, -1)}, 127.5, 127.5)
    assert i8.ravel().tolist() == [-128, -128, -127, -2, 0, 1, 72, 126, 127]
    with pytest.raises(RuntimeError):
        encode_image(IMG, {"dtype": np.uint8, "quantization": (0.0, 0), "name": "x"}, 127.5, 127.5)


def test_dequantize():
    q = np.array([-128, 0, 127], dtype=np.int8)
    assert np.allclose(dequantize(q, (0.5, -128)), [0.0, 64.0, 127.5])
    f = np.array([1.5], dtype=np.float32)
    assert dequantize(f, (0.0, 0)) is f


INT8 = Path(__file__).resolve().parents[1] / "var/models/efficientnet_lite0_mediapipe_int8_v1/efficientnet_lite0.tflite"


@pytest.mark.skipif(not INT8.exists(), reason="int8 classifier not fetched (petperception fetch-models)")
def test_dequantized_graph_outputs_match_numpy():
    from ai_edge_litert.interpreter import Interpreter

    from petperception.ondevice.bundle import BACKBONE_OUTPUTS
    from petperception.ondevice.tflite_patch import expose_outputs

    src = INT8.read_bytes()
    it = Interpreter(model_content=expose_outputs(src, BACKBONE_OUTPUTS))
    it.allocate_tensors()
    assert [str(d["dtype"].__name__) for d in it.get_output_details()][1:] == ["float32"] * 4
    x = (np.arange(224 * 224 * 3) % 251).astype(np.uint8).reshape(1, 224, 224, 3)
    it.set_tensor(it.get_input_details()[0]["index"], x)
    it.invoke()
    ref = Interpreter(model_content=src, experimental_preserve_all_tensors=True)
    ref.allocate_tensors()
    ref.set_tensor(ref.get_input_details()[0]["index"], x)
    ref.invoke()
    q = {d["index"]: d["quantization"] for d in ref.get_tensor_details()}
    for k, i in enumerate(BACKBONE_OUTPUTS):
        got = it.get_tensor(it.get_output_details()[1 + k]["index"])
        assert np.array_equal(got, dequantize(ref.get_tensor(i), q[i]))
