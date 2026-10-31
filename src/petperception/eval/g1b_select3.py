"""G1b selection with all three candidates once run A finished (rule fixed in docs/TRAINING_PROGRAM.md; development data only).

Usage: python -m petperception.eval.g1b_select3 (run A read from its run directory; G1a and run B from models/g1/)."""
import json

from petperception.config import REPO_ROOT, load_settings
from petperception.eval import g1_select as G

s = load_settings()
s.runtime.num_threads = 4
c = {"g1a_full": (REPO_ROOT / "models/g1/g1a_full_mnv2_sb4096/export/model_float32.tflite", REPO_ROOT / "models/g1/g1a_full_mnv2_sb4096"),
     "g1b_warm_cat2_cop3d": (REPO_ROOT / "models/g1/g1b_warm_cat2_cop3d/export/model_float32.tflite", REPO_ROOT / "models/g1/g1b_warm_cat2_cop3d"),
     "g1b_warm_cat2": (REPO_ROOT / "models/g1/g1b_warm_cat2/export/model_float32.tflite", REPO_ROOT / "models/g1/g1b_warm_cat2")}
for k, (e, r) in c.items():
    assert e.exists() and (r / "final.json").exists(), k
out = G.run(s, REPO_ROOT / "results/g1/g1b_selection_3candidates.json", c)
print(json.dumps({k: {x: v[x] for x in ("oi_dev_ap50", "constraint_dog_dev_ap50", "constraint_dog_only_cat")} | {"points": v["points"]["thresholds"]} for k, v in out["candidates"].items()}, indent=1))
print("winner", out["winner"])
