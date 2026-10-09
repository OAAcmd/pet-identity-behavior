"""Expose intermediate tensors of a TFLite model as additional graph outputs.

On-device runtimes (LiteRT.js in the browser, LiteRT on Android/iOS) return
only graph outputs; ``experimental_preserve_all_tensors`` is a debugging aid
that is not available there. This rewrites subgraph 0's ``outputs`` vector so
the requested tensors are outputs too. The weights and operators are
untouched; the new vector is appended at the end of the flatbuffer (forward
uoffset), so no other offsets move. Models with SignatureDefs are rejected
(their output maps would also need rewriting).

TFLite schema field ids used: Model.subgraphs = 2, Model.signature_defs = 7,
SubGraph.outputs = 2.
"""

from __future__ import annotations

import struct

import numpy as np


def _u32(b: bytes | bytearray, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]


def _table_field(b: bytes | bytearray, table: int, field_id: int) -> int | None:
    """Absolute position of a table field, or None if absent."""
    vt = table - struct.unpack_from("<i", b, table)[0]
    vt_len = struct.unpack_from("<H", b, vt)[0]
    slot = 4 + 2 * field_id
    if slot >= vt_len:
        return None
    rel = struct.unpack_from("<H", b, vt + slot)[0]
    return table + rel if rel else None


def subgraph_outputs(model: bytes) -> list[int]:
    root = _u32(model, 0)
    sgs_field = _table_field(model, root, 2)
    sgs = sgs_field + _u32(model, sgs_field)
    sg0 = sgs + 4 + _u32(model, sgs + 4)
    out_field = _table_field(model, sg0, 2)
    vec = out_field + _u32(model, out_field)
    n = _u32(model, vec)
    return list(struct.unpack_from(f"<{n}i", model, vec + 4))


def add_outputs(model: bytes, tensor_indices: list[int]) -> bytes:
    b = bytearray(model)
    root = _u32(b, 0)
    sig = _table_field(b, root, 7)
    if sig is not None and _u32(b, sig + _u32(b, sig)) > 0:
        raise ValueError("model has SignatureDefs; refusing to patch")
    sgs_field = _table_field(b, root, 2)
    sgs = sgs_field + _u32(b, sgs_field)
    sg0 = sgs + 4 + _u32(b, sgs + 4)
    out_field = _table_field(b, sg0, 2)
    if out_field is None:
        raise ValueError("subgraph has no outputs field")
    current = subgraph_outputs(model)
    new = current + [i for i in tensor_indices if i not in current]
    while len(b) % 4:
        b.append(0)
    pos = len(b)
    b += struct.pack(f"<I{len(new)}i", len(new), *new)
    struct.pack_into("<I", b, out_field, pos - out_field)
    return bytes(b)


def expose_outputs(model: bytes, tensor_indices: list[int]) -> bytes:
    """Expose tensors as graph outputs; integer-quantized tensors are exposed through an
    appended DEQUANTIZE operator as float32 (LiteRT.js 2.5.3 rejects INT8 graph outputs).

    Float tensors use the byte-level ``add_outputs`` (weights, operators and offsets
    untouched). Quantized tensors need new operators, so the model is rebuilt with the
    TFLite schema object API: the original operators, tensors and weight buffers are
    kept unchanged; one DEQUANTIZE operator and one float32 tensor are appended per
    exposed tensor. Bytes after the flatbuffer (the metadata zip with label files) are
    not carried over; read labels and normalization from the source model.
    """
    from ai_edge_litert import schema_py_generated as S

    m = S.ModelT.InitFromPackedBuf(model, 0)
    if m.signatureDefs:
        raise ValueError("models with SignatureDefs are not supported")
    sg = m.subgraphs[0]
    quantized = [i for i in tensor_indices if sg.tensors[i].type != S.TensorType.FLOAT32]
    if not quantized:
        return add_outputs(model, tensor_indices)
    import flatbuffers

    op = next((k for k, c in enumerate(m.operatorCodes) if max(c.builtinCode, c.deprecatedBuiltinCode) == S.BuiltinOperator.DEQUANTIZE), None)
    if op is None:
        code = S.OperatorCodeT()
        code.builtinCode = code.deprecatedBuiltinCode = S.BuiltinOperator.DEQUANTIZE
        code.version = 2  # int8 inputs
        m.operatorCodes.append(code)
        op = len(m.operatorCodes) - 1
    outputs = list(sg.outputs)
    for i in tensor_indices:
        src = sg.tensors[i]
        if src.type == S.TensorType.FLOAT32:
            outputs.append(i)
            continue
        m.buffers.append(S.BufferT())
        t = S.TensorT()
        t.shape, t.type, t.buffer = list(src.shape), S.TensorType.FLOAT32, len(m.buffers) - 1
        t.name = (src.name or b"") + b"/dequantized" if isinstance(src.name, bytes) else f"{src.name}/dequantized"
        sg.tensors.append(t)
        o = S.OperatorT()
        o.opcodeIndex, o.inputs, o.outputs = op, [i], [len(sg.tensors) - 1]
        sg.operators.append(o)
        outputs.append(len(sg.tensors) - 1)
    sg.outputs = outputs
    b = flatbuffers.Builder(len(model) + 1024)
    b.Finish(m.Pack(b), file_identifier=b"TFL3")
    return bytes(b.Output())


def strip_detection_postprocess(model: bytes) -> tuple[bytes, np.ndarray, dict]:
    """Remove a final ``TFLite_Detection_PostProcess`` custom op so its raw float inputs
    (box encodings ``[1, N, 4]`` and class scores ``[1, N, C]``) become the graph outputs.

    LiteRT.js 2.5.3 crashes inside this custom op; decoding the raw tensors outside the
    graph (same anchor decoding as the MediaPipe builds) avoids it. Returns the new model
    bytes, the op's constant anchor table ``[N, 4]`` as stored (y_center, x_center, h, w)
    and the op's options (scales, number of classes, NMS settings).
    """
    from ai_edge_litert import schema_py_generated as S
    from flatbuffers import flexbuffers

    m = S.ModelT.InitFromPackedBuf(model, 0)
    if m.signatureDefs:
        raise ValueError("models with SignatureDefs are not supported")
    sg = m.subgraphs[0]
    idx = [i for i, o in enumerate(sg.operators) if m.operatorCodes[o.opcodeIndex].customCode == b"TFLite_Detection_PostProcess"]
    if len(idx) != 1 or idx[0] != len(sg.operators) - 1:
        raise ValueError("expected exactly one TFLite_Detection_PostProcess as the last operator")
    op = sg.operators[idx[0]]
    boxes_t, scores_t, anchors_t = (int(v) for v in op.inputs)
    for t in (boxes_t, scores_t, anchors_t):
        if sg.tensors[t].type != S.TensorType.FLOAT32:
            raise ValueError("post-processing inputs are not float32")
    anchors = np.frombuffer(bytes(m.buffers[sg.tensors[anchors_t].buffer].data), dtype="<f4").reshape(-1, 4).copy()
    options = dict(flexbuffers.Loads(bytes(op.customOptions)))
    sg.operators.pop(idx[0])
    sg.outputs = [boxes_t, scores_t]
    import flatbuffers

    b = flatbuffers.Builder(len(model) + 1024)
    b.Finish(m.Pack(b), file_identifier=b"TFL3")
    return bytes(b.Output()), anchors, options
