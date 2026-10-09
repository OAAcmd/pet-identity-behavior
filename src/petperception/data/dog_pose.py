"""Ultralytics dog-pose (Stanford Dogs images, 24 keypoint slots): checksum-verified fetch.

Terms: Ultralytics documents the data as Research-Only (Stanford Dogs /
ImageNet terms); the archive also ships the Ultralytics AGPL-3.0 licence text.
Research profile only. Images are never committed.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from ..utils import download, sha256_file

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/dog-pose.zip"
SHA256 = "0396a35f20d9c1fdf91a692e89467d909280c4058d5367a51f8dcba2aad4d667"
BYTES = 353152125


def root(data_root: Path) -> Path:
    return data_root / "dog_pose"


def fetch(data_root: Path) -> dict:
    d = root(data_root)
    z = d / "_download" / "dog-pose.zip"
    if not (z.exists() and sha256_file(z) == SHA256):
        download(URL, z, expected_sha256=SHA256, timeout=1800)
    out = d / "extracted"
    if not (out / "labels" / "train").exists():
        base = os.path.realpath(out)
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():  # refuse absolute paths and '..' components (zip slip)
                dest = os.path.realpath(os.path.join(base, info.filename))
                if not (dest == base or dest.startswith(base + os.sep)):
                    raise RuntimeError(f"unsafe path in archive: {info.filename}")
            zf.extractall(out)
    n_train = len(list((out / "labels" / "train").glob("*.txt")))
    n_val = len(list((out / "labels" / "val").glob("*.txt")))
    return {"dataset": "dog_pose", "sha256": SHA256, "train_label_files": n_train, "val_label_files": n_val, "path": str(out)}
