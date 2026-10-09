"""MammalNet temporal behavior annotations (E11): checksum-verified fetch.

Source: https://mammalnet.s3.amazonaws.com/annotation.tar (MammalNet, Chen et al.,
CVPR 2023). The repository code is MIT; no dataset licence is published and the
videos are third-party YouTube uploads, so data and anything trained on it are
research-profile only. Videos are streamed frame by frame during E11 and never
stored.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from ..utils import download, sha256_file

URL = "https://mammalnet.s3.amazonaws.com/annotation.tar"
SHA256 = "055af7e2ae5184af1a292bd1605d7dd09d4937d25d2e81c1b333c654b97caee0"


def root(data_root: Path) -> Path:
    return data_root / "mammalnet"


def fetch(data_root: Path) -> dict:
    d = root(data_root) / "_download"
    tar = d / "annotation.tar"
    if not (tar.exists() and sha256_file(tar) == SHA256):
        download(URL, tar, expected_sha256=SHA256, timeout=600)
    out = d / "x"
    ann = out / "annotation" / "detection_annotations.json"
    if not ann.exists():
        with tarfile.open(tar) as tf:
            tf.extractall(out, filter="data")  # rejects absolute paths, '..' and links outside the target
    return {"annotation_tar_sha256": SHA256, "detection_annotations": str(ann), "present": ann.exists()}
