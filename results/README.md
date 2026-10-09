# Experiment Results

Produced with `uv run petperception evaluate --experiment <id>` (on-device
results: see the commands in [docs/ON_DEVICE.md](../docs/ON_DEVICE.md)).
Summaries and interpretation: [docs/ALGORITHM_REPORT.md](../docs/ALGORITHM_REPORT.md);
what each result does and does not show: [docs/CAPABILITY_MATRIX.md](../docs/CAPABILITY_MATRIX.md).

| Dir | Status | Data | Files |
| --- | --- | --- | --- |
| `e1/` | completed | ESC-50 (real clips, not household audio) | `metrics.json` (protocol, per-fold info, all methods, CIs, false positives, calibration diagnostic), `predictions.csv`, `run.log` |
| `e2/` | **blocked** (CatMeows on zenodo.org unreachable) | - | `status.json` with reason and continuation command |
| `e3/` | **pending** (Barkopedia on huggingface.co unreachable and unaudited) | - | `status.json` |
| `e4/` | completed | Open Images V7 validation subset (web photos) | `metrics.json`, `predictions_<detector>.json`, `run.log` |
| `e5/` | completed as behavior checks | synthetic pairings of real public media | `scenario_results.json`, `outputs/*.json`, `run.log` |
| `e6/` | completed | runtime on the cloud workspace (server path) | `metrics.json`, `run.log` |
| `e7/` | completed | Open Images V7 test split (tuning) and the E4 validation subset (final) | `metrics.json`: false-positive verifier, recall/specificity trade-off per detector and feature set |
| `e8/` | completed | semi-synthetic sequences (real animal pixels, scripted motion) | `metrics.json`: tracking (IDF1, ID switches) and behavior-segment detection |
| `e9/` | completed | Ultralytics dog-pose (Research-Only) | `metrics.json` (PCK, ear/tail cue agreement, gates), `training_log.json` (learning curves, seeds), `breed_breakdown.json` (per-breed PCK, heterogeneity test) |
| `e10/` | completed | Open Images V7 + Localized Narratives caption-derived weak labels | `metrics.json`: coarse posture agreement with weak labels, per species |
| `e11/` | completed | MammalNet dog-breed video segments (research only) | `metrics.json`: dog behavior classes from video, breed breakdown |
| `e12/` | completed | as E7 and E10 | `metrics.json` (quantized builds: detection, rejection, posture, agreement, latency, decision), `e7_<pair>.json` (full E7 protocol per detector/backbone pair) |
| `e13/` | completed | Open Images V7 relationships (person-pet pairs) | `metrics.json`: interaction AP/F1, coverage, baselines |
| `e14/` | completed | as E7 | `metrics.json` (TF Hub detector with raw decoding vs in-graph post-processing), `e7_tfhub_rawdecode.json` (full E7 protocol) |
| `ondevice/` | completed in **desktop headless Chromium only** | parity fixtures, scenario media | `runtime_parity_*.json` (model level: browser LiteRT.js vs Python LiteRT), `engine_parity_chromium.json` (engine level, Python vs TypeScript engine on identical files; `_int8` for the compact bundle), `device_check_desktop_chromium.json` (device-check report from an automated browser, **not a phone**) |

No metric from different datasets is combined into a product-level number.
Multimodal / hypothesis accuracy on real household recordings is
`not_evaluated` (no such labelled data exists yet; see
[docs/EVALUATION_PLAN_REAL_PAIRED.md](../docs/EVALUATION_PLAN_REAL_PAIRED.md)).
