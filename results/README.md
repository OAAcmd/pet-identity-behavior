# Experiment Results

Produced in the Phase 1 workspace with `uv run petperception evaluate --experiment <id>`.
Summaries and interpretation: [docs/ALGORITHM_REPORT.md](../docs/ALGORITHM_REPORT.md).

| Dir | Status | Files |
| --- | --- | --- |
| `e1/` | completed (real data: ESC-50) | `metrics.json` (protocol, per-fold info, all methods, CIs, false positives, calibration diagnostic), `predictions.csv` (per clip, per method), `run.log` |
| `e2/` | **blocked** (CatMeows unreachable) | `status.json` with reason and continuation command |
| `e3/` | **pending** (Barkopedia unreachable and unaudited) | `status.json` |
| `e4/` | completed (real data: Open Images V7 val subset) | `metrics.json`, `predictions_<detector>.json` (detections >= 0.2 per image), `run.log` |
| `e5/` | completed as behavior checks (synthetic pairings of real media) | `scenario_results.json` (14 scenarios x 3 scorers, invariants/expectations), `outputs/*.json` (full results, rule baseline), `run.log` |
| `e6/` | completed (runtime on this workspace) | `metrics.json`, `run.log` |

No metric from different datasets is combined into a product-level number.
Multimodal / hypothesis accuracy is `not_evaluated`.
