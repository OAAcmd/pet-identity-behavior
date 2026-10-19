"""AwA-Pose keypoints (MIT annotations) on Animals with Attributes 2 images, for cats and dogs.

Annotations: github.com/prinik/AwA-Pose at commit 7ba27c70 (MIT, Copyright
2021 Prianka Banik), one pickle per image with a dict per animal (`a1`,
`a2`, ...) of 39 named points plus `bbox` in pixels; missing points are -1.
The pickles contain only data opcodes and are read with an unpickler that
refuses every global, so no code in them can run.

Images: AwA2-data.zip (cvml.ista.ac.at, 13.9 GB), read member by member by
HTTP Range (`RemoteZip`, CRC-checked); never stored whole. Each image has a
licence file `licenses/<class>/<image>.txt` in the same archive. It is cached
and parsed, so a model can be restricted to licences that permit a release.

Classes used: persian+cat, siamese+cat (cats); chihuahua, collie,
dalmatian, german+shepherd (dogs). AwA-Pose points are mapped to the
24-point layout of `vision.keypoints.DOG_KEYPOINTS` (left/right from the
animal's point of view in both). Antlers, neck, belly and body-side points
have no counterpart and are dropped.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import pickle
import re
import subprocess
from pathlib import Path

import numpy as np

from ..vision import keypoints as KP
from .mirror import RemoteZip

REPO = "https://github.com/prinik/AwA-Pose.git"
COMMIT = "7ba27c70dca49b83f37770a16b64df19058f3f96"
ARCHIVE = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"
CLASSES = {"persian+cat": "cat", "siamese+cat": "cat", "chihuahua": "dog", "collie": "dog", "dalmatian": "dog", "german+shepherd": "dog"}
# DOG_KEYPOINTS name -> AwA-Pose name ("thai" is AwA-Pose's spelling of the upper-leg joint).
TO_AWA = {
    "front_left_paw": "front_left_paw", "front_left_knee": "front_left_knee", "front_left_elbow": "front_left_thai",
    "rear_left_paw": "back_left_paw", "rear_left_knee": "back_left_knee", "rear_left_elbow": "back_left_thai",
    "front_right_paw": "front_right_paw", "front_right_knee": "front_right_knee", "front_right_elbow": "front_right_thai",
    "rear_right_paw": "back_right_paw", "rear_right_knee": "back_right_knee", "rear_right_elbow": "back_right_thai",
    "tail_start": "tail_base", "tail_end": "tail_end", "left_ear_base": "left_earbase", "right_ear_base": "right_earbase",
    "nose": "nose", "chin": "lower_jaw", "left_ear_tip": "left_earend", "right_ear_tip": "right_earend",
    "left_eye": "left_eye", "right_eye": "right_eye", "withers": "back_base", "throat": "throat_base",
}


class _DataOnly(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: ARG002 - refuse every global
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def load_annotation(data: bytes) -> dict:
    return _DataOnly(io.BytesIO(data)).load()


def root(data_root: Path) -> Path:
    return data_root / "awa_pose"


def clone(data_root: Path) -> Path:
    repo = root(data_root) / "repo"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO, str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", COMMIT], check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        raise RuntimeError(f"AwA-Pose at {head}, expected {COMMIT}")
    return repo


def parse_licence(text: str) -> dict:
    """Licence class and source of one AwA2 licence file (free-text lines such as 'license: ...')."""
    t = text.lower()
    if "publicdomain/zero" in t or "cc0" in t:
        kind = "CC0"
    elif "publicdomain/mark" in t or "public domain" in t:
        kind = "PD"
    elif re.search(r"licenses/by-nc|by-nc|noncommercial|non-commercial", t):
        kind = "CC BY-NC" + ("-SA" if "-sa" in t or "sharealike" in t else "") + ("-ND" if "-nd" in t or "noderiv" in t else "")
    elif re.search(r"licenses/by-sa|by-sa|sharealike", t):
        kind = "CC BY-SA"
    elif re.search(r"licenses/by-nd|by-nd|noderiv", t):
        kind = "CC BY-ND"
    elif re.search(r"licenses/by/|creative commons attribution|\bcc by\b|cc-by", t):
        kind = "CC BY"
    else:
        kind = "other"
    url = re.search(r"https?://\S+", text)
    return {"kind": kind, "first_url": url.group(0) if url else None}


def fetch(data_root: Path, workers: int = 16) -> list[dict]:
    """Images and licence files of the annotated cat and dog images; one record per animal instance."""
    from concurrent.futures import ThreadPoolExecutor

    repo = clone(data_root)
    out = root(data_root) / "images"
    index = root(data_root) / "index.csv"
    if index.exists():
        return read_index(index)
    z = RemoteZip([ARCHIVE])
    jobs = []
    for cls in CLASSES:
        for p in sorted((repo / "Annotations" / cls).glob("*.pickle")):
            member = f"Animals_with_Attributes2/JPEGImages/{cls}/{p.stem}.jpg"
            if member in z.entries:
                jobs.append((cls, p, member))

    def get(job):
        cls, p, member = job
        img = z.read(member)
        lic_name = f"Animals_with_Attributes2/licenses/{cls}/{p.stem}.txt"
        lic = z.read(lic_name).decode("utf-8", "replace") if lic_name in z.entries else ""
        dst = out / cls / f"{p.stem}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(img)
        (dst.with_suffix(".licence.txt")).write_text(lic)
        return cls, p, dst, hashlib.sha256(img).hexdigest(), lic

    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for cls, p, dst, sha, lic in ex.map(get, jobs):
            ann = load_annotation(p.read_bytes())
            lp = parse_licence(lic)
            for aid, a in sorted(ann.items()):
                if not isinstance(a, dict) or "bbox" not in a:
                    continue
                rows.append({"id": f"{p.stem}#{aid}", "image": str(dst), "class": cls, "species": CLASSES[cls], "sha256": sha,
                             "licence": lp["kind"], "licence_url": lp["first_url"] or "",
                             "bbox": json.dumps([float(v) for v in a["bbox"]]),
                             "points": json.dumps({k: [float(v[0]), float(v[1])] for k, v in a.items() if k != "bbox"})})
    with open(index, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def read_index(index: Path) -> list[dict]:
    with open(index, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def to_record(row: dict, hw: tuple[int, int]) -> dict:
    """E9 record: normalised box (x1, y1, x2, y2) and keypoints [K, 3] (x, y, labelled) in DOG_KEYPOINTS order."""
    h, w = hw
    x1, y1, x2, y2 = json.loads(row["bbox"])
    pts = json.loads(row["points"])
    kp = np.zeros((len(KP.DOG_KEYPOINTS), 3))
    for i, name in enumerate(KP.DOG_KEYPOINTS):
        v = pts.get(TO_AWA[name])
        if v and v[0] >= 0 and v[1] >= 0:
            kp[i] = (v[0] / w, v[1] / h, 2.0)
    return {"id": row["id"], "image": row["image"], "breed": row["class"], "species": row["species"], "licence": row["licence"],
            "box": np.clip(np.array([x1 / w, y1 / h, x2 / w, y2 / h]), 0.0, 1.0), "kp": kp}
