"""Answers of the frozen E31 system (v1) for the E32 benchmark.

Run with the E31 code on the path, never this repository's package:
    PYTHONPATH=var/v1_worktree/src PETPERCEPTION_DATA_ROOT=$PWD/var/data PETPERCEPTION_MODELS_ROOT=$PWD/var/models \\
        python src/petperception/eval/e32_v1_runner.py --items benchmarks/behavior_questions_v2.json --features var/understand/features \\
        --out var/understand/e32_v1_answers.json
(``git worktree add var/v1_worktree 0c1f7c5``). Uses the v1 parser, recognizer, thresholds and knowledge base on version-1 features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import petperception
    from petperception.config import REPO_ROOT
    from petperception.intent import av_features as AF
    from petperception.understand import explain as X
    from petperception.understand import knowledge as K
    from petperception.understand import question as Qn
    from petperception.understand import recognize as R
    from petperception.understand.features import SiglipEncoder

    cfg = R.RecognizerConfig.load(REPO_ROOT / "artifacts" / "understand" / "recognizer.json")
    zsm = R.ZeroShot(SiglipEncoder(threads=2))
    yam = AF.load_components(1, with_audio=True).yamnet.labels
    out = {"system": "E31 (v1)", "package": str(Path(petperception.__file__).parent), "knowledge_version": K.KNOWLEDGE_VERSION, "answers": {}}
    for it in json.loads(Path(a.items).read_text())["questions"]:
        z = np.load(Path(a.features) / f"{it['clip']}.npz", allow_pickle=False)
        q = Qn.parse(it["species"], it["scene"], it["question"])
        readings, pattern = R.recognize(z, it["species"], zsm, yam if "a_scores" in z.files else None, None, cfg)
        ans = X.compose(q, readings, pattern, cfg)
        out["answers"][it["id"]] = {"answer": ans.to_dict(), "parsed": q.reported, "scores": {r.behavior: r.score for r in readings}}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
