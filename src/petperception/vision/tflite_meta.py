"""Minimal readers for metadata embedded in TFLite model files.

* TFLite Model Metadata (file identifier ``M001``, schema
  tensorflow/tflite-support metadata_schema.fbs) - used for input
  NormalizationOptions.
* MediaPipe ObjectDetectorOptions (file identifier ``V001``, schema
  mediapipe/tasks/metadata/object_detector_metadata_schema.fbs) - fixed SSD
  anchors and tensor-decoding options.
* Associated files are stored as a ZIP appended to the flatbuffer.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import flatbuffers
import numpy as np
from flatbuffers import number_types as N
from flatbuffers.table import Table


def _root(buf: bytes, ident: bytes) -> Table:
    i = buf.find(ident)
    if i < 4:
        raise ValueError(f"flatbuffer with identifier {ident!r} not found")
    b = bytearray(buf[i - 4:])
    return Table(b, flatbuffers.encode.Get(N.UOffsetTFlags.packer_type, b, 0))


def _field_offset(t: Table, slot: int) -> int:
    return t.Offset(4 + 2 * slot)


def _sub_table(t: Table, slot: int) -> Table | None:
    o = _field_offset(t, slot)
    return Table(t.Bytes, t.Indirect(o + t.Pos)) if o else None


def _vec_tables(t: Table, slot: int) -> list[Table]:
    o = _field_offset(t, slot)
    if not o:
        return []
    n, start = t.VectorLen(o), t.Vector(o)
    return [Table(t.Bytes, t.Indirect(start + 4 * k)) for k in range(n)]


def _vec_floats(t: Table, slot: int) -> list[float]:
    o = _field_offset(t, slot)
    if not o:
        return []
    n, start = t.VectorLen(o), t.Vector(o)
    return [t.Get(N.Float32Flags, start + 4 * k) for k in range(n)]


def _scalar(t: Table, slot: int, flags, default):
    o = _field_offset(t, slot)
    return t.Get(flags, o + t.Pos) if o else default


def associated_files(model_bytes: bytes) -> dict[str, bytes]:
    try:
        z = zipfile.ZipFile(io.BytesIO(model_bytes))
    except zipfile.BadZipFile:
        return {}
    return {n: z.read(n) for n in z.namelist()}


def input_normalization(model_bytes: bytes) -> tuple[list[float], list[float]] | None:
    """(mean, std) of the first input tensor's NormalizationOptions, if present."""
    mm = _root(model_bytes, b"M001")
    for sg in _vec_tables(mm, 3):  # subgraph_metadata
        for tm in _vec_tables(sg, 2):  # input_tensor_metadata
            for pu in _vec_tables(tm, 4):  # process_units
                if _scalar(pu, 0, N.Uint8Flags, 0) == 1:  # NormalizationOptions
                    opts = _sub_table(pu, 1)
                    if opts is not None:
                        return _vec_floats(opts, 0), _vec_floats(opts, 1)
    return None


@dataclass
class DetectorDecoding:
    anchors: np.ndarray  # [N, 4] x_center, y_center, width, height (normalized)
    num_classes: int
    num_boxes: int
    num_coords: int
    x_scale: float
    y_scale: float
    w_scale: float
    h_scale: float
    apply_exponential_on_box_size: bool
    sigmoid_score: bool


def mediapipe_detector_options(model_bytes: bytes) -> DetectorDecoding:
    root = _root(model_bytes, b"V001")
    ssd = _sub_table(root, 1)
    fas = _sub_table(ssd, 0) if ssd else None
    if fas is None:
        raise ValueError("no fixed anchors in detector metadata")
    anchors = np.array([[_scalar(a, j, N.Float32Flags, 0.0) for j in range(4)] for a in _vec_tables(fas, 0)], np.float32)
    dec = _sub_table(root, 2)
    if dec is None:
        raise ValueError("no TensorsDecodingOptions in detector metadata")
    return DetectorDecoding(
        anchors=anchors,
        num_classes=_scalar(dec, 0, N.Int32Flags, 0),
        num_boxes=_scalar(dec, 1, N.Int32Flags, 0),
        num_coords=_scalar(dec, 2, N.Int32Flags, 0),
        x_scale=_scalar(dec, 6, N.Float32Flags, 0.0),
        y_scale=_scalar(dec, 7, N.Float32Flags, 0.0),
        w_scale=_scalar(dec, 8, N.Float32Flags, 0.0),
        h_scale=_scalar(dec, 9, N.Float32Flags, 0.0),
        apply_exponential_on_box_size=bool(_scalar(dec, 10, N.BoolFlags, False)),
        sigmoid_score=bool(_scalar(dec, 11, N.BoolFlags, False)),
    )
