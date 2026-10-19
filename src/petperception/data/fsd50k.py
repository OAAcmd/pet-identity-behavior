"""FSD50K v1.0 (Zenodo record 4060432): labels, per-clip licences, and audio members.

Files come through the mirror importer (`petperception mirror fetch --dataset fsd50k
--official`), which checks each file against the MD5 published on the Zenodo record
(manifests/mirror_files.json) and writes a provenance record. Zenodo ignores HTTP
Range requests, so the audio archives are downloaded whole; they are split ZIPs
(`.z01` + `.zip`) that Python's zipfile cannot open, and are read member by
member with `LocalSplitZip` (CRC-checked). Nothing is extracted to disk.

Labels are clip-level AudioSet-ontology labels, propagated to ancestors by the
publishers (a "Meow" clip is also "Cat", "Domestic_animals_and_pets", "Animal").
Two FSD50K labels are not species-specific: "Hiss" is mostly steam, fire and
vehicles, and "Growling" includes wild animals and people. Cat hiss and dog growl
are therefore defined as conjunctions with "Cat" and "Dog".

Terms: the collection is CC BY 4.0 (FSD50K.doc/LICENSE-DATASET); each clip
carries its own Freesound licence (CC0, CC BY, CC BY-NC or Sampling+) in
FSD50K.metadata/*_clips_info_FSD50K.json. Attribution is required.
"""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .mirror import LocalSplitZip

ARCHIVES = {"eval": ["FSD50K.eval_audio.z01", "FSD50K.eval_audio.zip"],
            "dev": [f"FSD50K.dev_audio.z0{i}" for i in range(1, 6)] + ["FSD50K.dev_audio.zip"]}
META_ZIPS = ["FSD50K.ground_truth.zip", "FSD50K.metadata.zip", "FSD50K.doc.zip"]
LICENCE_SHORT = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0",
    "http://creativecommons.org/licenses/by/3.0/": "CC BY",
    "http://creativecommons.org/licenses/by-nc/3.0/": "CC BY-NC",
    "http://creativecommons.org/licenses/sampling+/1.0/": "Sampling+",
}


@dataclass(frozen=True)
class Clip:
    fname: str  # the Freesound sound id
    split: str  # eval | train | val (dev is train + val)
    labels: frozenset[str]
    licence: str
    uploader: str


def root(data_root: Path) -> Path:
    return data_root / "fsd50k"


def prepare_meta(data_root: Path) -> Path:
    """Extract the three small label/metadata/doc archives (verified by the importer) into fsd50k/meta."""
    r = root(data_root)
    out = r / "meta"
    for name in META_ZIPS:
        src = r / name
        if not (r / f"{name}.provenance.json").exists():
            raise FileNotFoundError(f"{src} not fetched and verified: uv run petperception mirror fetch --dataset fsd50k --file {name} --official")
        with zipfile.ZipFile(src) as z:
            for m in z.namelist():
                p = (out / m).resolve()
                if not str(p).startswith(str(out.resolve())):
                    raise RuntimeError(f"unsafe path in {name}: {m}")
            z.extractall(out)
    return out


def load_clips(data_root: Path, split: str) -> list[Clip]:
    """split: 'eval' or 'dev' (dev rows carry their train/val split)."""
    meta = root(data_root) / "meta"
    if not (meta / "FSD50K.ground_truth" / f"{split}.csv").exists():
        prepare_meta(data_root)
    info = json.loads((meta / "FSD50K.metadata" / f"{split}_clips_info_FSD50K.json").read_text())
    out = []
    with open(meta / "FSD50K.ground_truth" / f"{split}.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            i = info[r["fname"]]
            out.append(Clip(r["fname"], r.get("split", "eval"), frozenset(r["labels"].split(",")),
                            LICENCE_SHORT.get(i["license"], i["license"]), i["uploader"]))
    return out


def audio_archive(data_root: Path, split: str) -> LocalSplitZip:
    parts = [root(data_root) / p for p in ARCHIVES[split]]
    missing = [p.name for p in parts if not (p.parent / f"{p.name}.provenance.json").exists()]
    if missing:
        raise FileNotFoundError(f"FSD50K {split} audio not fetched and verified: {missing} "
                                f"(uv run petperception mirror fetch --dataset fsd50k --file <name> --official)")
    return LocalSplitZip([str(p) for p in parts])


def member_name(split: str, fname: str) -> str:
    return f"FSD50K.{split}_audio/{fname}.wav"
