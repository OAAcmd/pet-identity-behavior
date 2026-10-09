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
