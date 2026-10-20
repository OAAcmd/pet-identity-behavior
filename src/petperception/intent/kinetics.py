"""Kinetics-700 pet clips as synchronized audio-visual data for situation (recorded context) recognition.

Splits (fixed before any model was fitted):
* train: the 600 Kinetics *train*-split clips already on disk (train_sample,
  clips 1-50 of each class, and train_holdout, clips 51-100; both fetched for
  E19/E19b, which tuned only the detector operating point);
* test: the 294 Kinetics *validation*-split clips (E18). No test clip shares a
  YouTube id with a training clip (checked at fetch time and again here).
* never read: train_final (clips 101-160), sealed for E23.

Unit of grouping: the YouTube id (one id = one upload; the uploader is not
published, so near-duplicate scenes across ids are audited by embedding
similarity instead).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data import kinetics_pets as KP

SPLIT_DIRS = {"train": ("train_sample", "train_holdout"), "test": ("clips",)}
SEALED = (KP.FINAL_SUBDIR,)


@dataclass(frozen=True)
class Clip:
    key: str
    path: Path
    split: str
    source_dir: str
    label: str
    species: str
    person_contact: bool
    youtube_id: str


def youtube_id(stem: str) -> str:
    # Kinetics file names: <11-char id>_<start 6 digits>_<end 6 digits>
    parts = stem.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else stem


def list_clips(data_root: Path) -> list[Clip]:
    root = KP.root(data_root)
    out: list[Clip] = []
    for split, dirs in SPLIT_DIRS.items():
        for d in dirs:
            if d in SEALED:
                raise RuntimeError("sealed final set must not be listed")
            for label, (_, species, contact) in KP.CLASSES.items():
                for p in sorted((root / d / label).glob("*.mp4")):
                    key = f"{d}__{label.replace(' ', '_')}__{p.stem}"
                    out.append(Clip(key, p, split, d, label, species, contact, youtube_id(p.stem)))
    tr = {c.youtube_id for c in out if c.split == "train"}
    te = {c.youtube_id for c in out if c.split == "test"}
    if tr & te:
        raise RuntimeError(f"YouTube ids shared by train and test: {sorted(tr & te)[:5]}")
    return out


def feature_dir(var_root: Path, variant: str = "clean") -> Path:
    return var_root / "intent" / "kinetics_av" / variant


def main(argv: list[str] | None = None) -> None:
    import argparse

    from ..config import load_settings
    from . import av_features as AF

    ap = argparse.ArgumentParser(description="Extract synchronized audio-visual features for the Kinetics pet clips (resumable).")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    a = ap.parse_args(argv)
    s = load_settings()
    clips = [c for c in list_clips(s.paths.resolve("data_root")) if a.split in ("all", c.split)]
    comp = AF.load_components(a.threads)
    AF.extract_many(comp, [(c.key, c.path) for c in clips], feature_dir(s.paths.resolve("data_root").parent))


if __name__ == "__main__":
    main()
