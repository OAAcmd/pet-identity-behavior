"""The behaviour clip collection: Commons recordings, paper supplementary videos and Internet Archive clips, with provenance.

``labels`` per clip keep their source apart:
* ``source``  - uploader title / description / editor categories (human, lay);
* ``expert``  - the paper authors' description (case level; a case's videos may
  show different moments of the case);
* ``assistant`` - machine-assisted frame review (``annotations/behavior_clips.json``,
  ``label_source: assistant_frame_review``); never merged into the other two.
Grouping for splits: the uploader (Commons, Internet Archive) or the paper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import REPO_ROOT
from ..data import commons_pets as C

ANNOTATIONS = REPO_ROOT / "annotations" / "behavior_clips.json"


@dataclass
class Clip:
    key: str
    path: Path
    species: str
    source: str  # commons | paper | archive
    group: str
    title: str
    licence: str
    labels: dict[str, list[str]] = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    url: str = ""


def load_assistant() -> dict[str, dict]:
    if not ANNOTATIONS.exists():
        return {}
    return {a["key"]: a for a in json.loads(ANNOTATIONS.read_text())["clips"]}


def clips(data_root: Path) -> list[Clip]:
    out: list[Clip] = []
    m = json.loads((REPO_ROOT / "manifests" / "commons_pets.json").read_text())
    for r in m["records"]:
        p = C.media_path(data_root / "commons_pets", r)
        if p is None:
            continue
        out.append(Clip(f"commons_{C.key(r)}", p, r["species_from_text"], "commons", f"commons:{r.get('uploader')}", r["title"].removeprefix("File:"),
                        r.get("license") or "", {"source": list(r["source_labels"])}, {"description": r.get("description", "")[:400]},
                        "https://commons.wikimedia.org/wiki/" + r["title"].replace(" ", "_")))
    pv = REPO_ROOT / "manifests" / "paper_videos.json"
    if pv.exists():
        for v in json.loads(pv.read_text())["videos"]:
            p = data_root / "paper_videos" / v["file"]
            if p.exists():
                grp = "paper:" + v["doi"]
                out.append(Clip(f"paper_{v['id']}", p, v["species"], "paper", grp, v["citation"], v["licence"], {"expert": list(v["expert_labels"])},
                                {**v.get("context", {}), "caption": v["caption"]}, "https://doi.org/" + v["doi"]))
    ia = REPO_ROOT / "manifests" / "archive_pets.json"
    if ia.exists():
        for r in json.loads(ia.read_text())["records"]:
            p = data_root / "archive_pets" / r["file"]
            if p.exists():
                out.append(Clip(f"archive_{r['id']}", p, r["species"], "archive", f"archive:{r['uploader']}", r["title"], r["licence"],
                                {"source": list(r["source_labels"])}, {}, "https://archive.org/details/" + r["identifier"]))
    ann = load_assistant()
    for c in out:
        a = ann.get(c.key)
        if a:
            c.labels["assistant"] = list(a.get("behaviors", []))
            c.context["assistant_bouts"] = a.get("bouts", [])
    return out


def feature_path(var_root: Path, key: str, subdir: str = "features") -> Path:
    return var_root / "understand" / subdir / f"{key}.npz"


def main() -> None:
    import argparse
    import time

    import numpy as np

    from ..config import load_settings
    from .features import Extractor

    ap = argparse.ArgumentParser(description="Extract behaviour evidence for every clip (resumable).")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--only", default="", help="comma-separated source names")
    ap.add_argument("--keys", default="", help="comma-separated clip keys (default: all)")
    ap.add_argument("--detector", help="alternative detector .tflite (E31 part C); default: the configured detector")
    ap.add_argument("--detector-points", help='per-class thresholds for --detector as JSON, e.g. {"cat": 0.11, "dog": 0.28}')
    ap.add_argument("--subdir", default="features", help="output directory under var/understand/")
    a = ap.parse_args()
    s = load_settings()
    data_root = s.paths.resolve("data_root")
    keys = set(a.keys.split(",")) if a.keys else None
    todo = [c for c in clips(data_root) if (not a.only or c.source in a.only.split(",")) and (keys is None or c.key in keys)
            and not feature_path(data_root.parent, c.key, a.subdir).exists()]
    todo.sort(key=lambda c: (c.source != "paper", c.source != "archive", c.key))
    import json

    ex = Extractor.load(a.threads, a.detector, json.loads(a.detector_points) if a.detector_points else None)
    t0 = time.perf_counter()
    for n, c in enumerate(todo, 1):
        p = feature_path(data_root.parent, c.key, a.subdir)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            f = ex.extract(c.path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            f = {"error": np.array(str(exc)[:300])}
        np.savez(p.with_suffix(".tmp.npz"), **f)
        p.with_suffix(".tmp.npz").replace(p)
        if n % 10 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} clips, {(time.perf_counter() - t0) / n:.1f} s/clip", flush=True)


if __name__ == "__main__":
    main()
