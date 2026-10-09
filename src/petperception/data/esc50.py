"""ESC-50 importer.

ESC-50 (https://github.com/karolpiczak/ESC-50) is an environmental sound
classification benchmark: 2000 five-second clips, 50 classes, 5 folds. The
authors placed clips extracted from the same original Freesound recording in
the same fold, so the official folds are source-preserving groups. ESC-50 is
NOT an intent or behavior dataset; here it supports cat/dog sound-event
evaluation (E1) only.

Terms: the collection as a whole is CC BY-NC 3.0; the ESC-10 subset is
CC BY 3.0; individual clips carry Freesound source licenses listed in the
repository LICENSE file, parsed here into a per-clip attribution table.
"""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..utils import sha256_file

CANONICAL_URL = "https://github.com/karolpiczak/ESC-50"
GIT_URL = "https://github.com/karolpiczak/esc-50"
PINNED_REVISION = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
PET_CLASSES = {"cat": "cat", "dog": "dog"}

# A few LICENSE lines are irregular (double spaces, bracketed titles, a
# "5-141683-Ad" typo), so the clip key tolerates a trailing lowercase letter.
_ATTR_RE = re.compile(
    r"^- \[(?P<clip>[0-9]+-[0-9]+-[A-Z])[a-z]?\.ogg\]:\s+clip derived from (?P<title>.*?)\s*"
    r"\((?P<url>https?://[^)]+)\) by (?P<author>.*?) \[(?P<license>[^\]]+)\]\s*$"
)


@dataclass(frozen=True)
class Esc50Clip:
    filename: str
    fold: int
    target: int
    category: str
    esc10: bool
    src_file: str
    take: str

    @property
    def pet_label(self) -> str:
        return PET_CLASSES.get(self.category, "other")

    @property
    def source_group(self) -> str:
        # Freesound source recording ID: all clips (takes) from one source
        # share this group and, by construction of ESC-50, one fold.
        return f"freesound:{self.src_file}"


def dataset_dir(data_root: Path) -> Path:
    return data_root / "esc50"


def fetch(data_root: Path) -> Path:
    """Clone ESC-50 at the pinned revision (explicit fetch only)."""
    d = dataset_dir(data_root)
    if (d / ".git").exists():
        rev = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        if rev != PINNED_REVISION:
            subprocess.run(["git", "-C", str(d), "fetch", "--depth", "1", "origin", PINNED_REVISION], check=True)
            subprocess.run(["git", "-C", str(d), "checkout", "--detach", PINNED_REVISION], check=True)
        return d
    d.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "remote", "add", "origin", GIT_URL], check=True)
    subprocess.run(["git", "-C", str(d), "fetch", "--depth", "1", "origin", PINNED_REVISION], check=True)
    subprocess.run(["git", "-C", str(d), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
    return d


def load_meta(root: Path) -> list[Esc50Clip]:
    rows: list[Esc50Clip] = []
    with open(root / "meta" / "esc50.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                Esc50Clip(
                    filename=r["filename"],
                    fold=int(r["fold"]),
                    target=int(r["target"]),
                    category=r["category"],
                    esc10=r["esc10"].strip().lower() == "true",
                    src_file=r["src_file"],
                    take=r["take"],
                )
            )
    return rows


def parse_attributions(root: Path) -> dict[str, dict[str, str]]:
    """Parse per-clip attribution lines from the repository LICENSE file."""
    out: dict[str, dict[str, str]] = {}
    for line in (root / "LICENSE").read_text(encoding="utf-8").splitlines():
        m = _ATTR_RE.match(line.strip())
        if m:
            out[m.group("clip")] = {
                "source_title": m.group("title"),
                "source_url": m.group("url"),
                "author": m.group("author"),
                "source_license": m.group("license"),
            }
    return out


def clip_key(filename: str) -> str:
    # "1-100032-A-0.wav" -> "1-100032-A"
    return filename.rsplit("-", 1)[0]


def build_file_manifest(root: Path, out_csv: Path) -> dict[str, object]:
    """Write a per-file manifest (bytes, sha256, fold, label, attribution)."""
    clips = load_meta(root)
    attrs = parse_attributions(root)
    agg = hashlib.sha256()
    missing_attr = 0
    total_bytes = 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "bytes", "sha256", "fold", "category", "pet_label", "esc10", "source_group", "take", "author", "source_license", "source_url"])
        for c in sorted(clips, key=lambda c: c.filename):
            p = root / "audio" / c.filename
            digest = sha256_file(p)
            size = p.stat().st_size
            total_bytes += size
            agg.update(f"{c.filename}:{digest}\n".encode())
            a = attrs.get(clip_key(c.filename))
            if a is None:
                missing_attr += 1
                a = {"author": "", "source_license": "", "source_url": ""}
            w.writerow([c.filename, size, digest, c.fold, c.category, c.pet_label, c.esc10, c.source_group, c.take, a["author"], a["source_license"], a["source_url"]])
    return {"files": len(clips), "bytes": total_bytes, "aggregate_sha256": agg.hexdigest(), "clips_missing_attribution": missing_attr}


def verify_against_manifest(root: Path, manifest_csv: Path) -> dict[str, object]:
    bad: list[str] = []
    n = 0
    with open(manifest_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n += 1
            p = root / "audio" / r["filename"]
            if not p.exists() or sha256_file(p) != r["sha256"]:
                bad.append(r["filename"])
    return {"checked": n, "mismatched_or_missing": bad}


def observed_counts(root: Path) -> dict[str, object]:
    clips = load_meta(root)
    by_label: dict[str, int] = {}
    by_fold: dict[int, dict[str, int]] = {}
    groups: dict[str, set[int]] = {}
    for c in clips:
        by_label[c.pet_label] = by_label.get(c.pet_label, 0) + 1
        by_fold.setdefault(c.fold, {}).setdefault(c.pet_label, 0)
        by_fold[c.fold][c.pet_label] += 1
        groups.setdefault(c.source_group, set()).add(c.fold)
    cross_fold = sorted(g for g, f in groups.items() if len(f) > 1)
    return {
        "clips": len(clips),
        "categories": len({c.category for c in clips}),
        "pet_label_counts": by_label,
        "per_fold_pet_label_counts": {str(k): v for k, v in sorted(by_fold.items())},
        "source_groups": len(groups),
        "source_groups_spanning_multiple_folds": cross_fold,
        "esc10_clips": sum(c.esc10 for c in clips),
    }
