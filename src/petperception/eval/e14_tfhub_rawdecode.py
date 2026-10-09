"""E14: a licence-confirmed detector that runs on the phone.

The TF Hub EfficientDet-Lite0 build (Apache-2.0 stated in its embedded
metadata) cannot run in LiteRT.js 2.5.3 because of its in-graph
``TFLite_Detection_PostProcess`` op. E14 removes that op and decodes the raw
box encodings and class scores outside the graph with the op's own anchor
table - the same decoder the phone uses for the MediaPipe builds. NMS becomes
per class (as for the MediaPipe builds) instead of the op's class-agnostic
top-25, so detections change and everything downstream is re-evaluated with
the E7 protocol (tuning on the Open Images test split, final on the E4
validation subset, verifier refitted; crop feature set exported for
deployment). The in-graph variant's E7 numbers are reported alongside.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import REPO_ROOT, Settings
from ..utils import utc_now, write_json
from . import e7_fp_rejection as E7

DETECTOR = "efficientdet_lite0_tfhub_rawdecode_v1"
IN_GRAPH = "efficientdet_lite0_tfhub_v1"


def _digest(e4: dict, key: str) -> dict:
    out = {}
    for c in E7.CLASSES:
        k = e4[key]["classes"][c]
        il = k["image_level"]
        out[c] = {"ap50": k["ap50"], "box_recall": k["recall_at_op"], "box_precision": k["precision_at_op"], "image_recall": il["recall"],
                  "specificity_on_verified_negatives": il["specificity_on_verified_negatives"],
                  "specificity_ci95": [il["specificity_ci95"]["ci_low"], il["specificity_ci95"]["ci_high"]] if "specificity_ci95" in il else None}
    return out


def run(settings: Settings, out_dir: Path) -> dict:
    summ = E7.run(settings, out_dir, detectors=[DETECTOR], filename="e7_tfhub_rawdecode.json")["results"][DETECTOR]
    params = REPO_ROOT / settings.vision.verifier.params_dir / f"verifier_{DETECTOR}.json"
    E7.export_feature_set(settings, DETECTOR, "crop", params, out_dir / "e7_tfhub_rawdecode.json")
    in_graph = json.loads((settings.paths.resolve("results_root") / "e7" / "metrics.json").read_text())["results"].get(IN_GRAPH, {}).get("e4")
    summary = {
        "experiment": "E14", "title": "Licence-confirmed TF Hub detector with post-processing removed (runs in LiteRT.js)",
        "protocol": __doc__.strip().splitlines(),
        "raw_decode": {"detector_only@0.35": _digest(summ["e4"], "detector_only@0.35"), "verifier[crop]": _digest(summ["e4"], "verifier[crop]"),
                       "tuning_selected": summ["selected"]},
        "in_graph_post_processing_e7": ({"detector_only@0.35": _digest(in_graph, "detector_only@0.35"), "verifier[crop]": _digest(in_graph, "verifier[crop]")}
                                        if in_graph else None),
        "verifier_params": str(params.relative_to(REPO_ROOT)),
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    write_json(out_dir / "metrics.json", summary)
    return summary
