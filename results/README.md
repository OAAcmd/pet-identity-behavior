# Experiment Results

Produced with `uv run petperception evaluate --experiment <id>` (on-device
results: see the commands in [docs/ON_DEVICE.md](../docs/ON_DEVICE.md)).
Summaries and interpretation: [docs/ALGORITHM_REPORT.md](../docs/ALGORITHM_REPORT.md);
what each result does and does not show: [docs/CAPABILITY_MATRIX.md](../docs/CAPABILITY_MATRIX.md).

| Dir | Status | Data | Files |
| --- | --- | --- | --- |
| `e1/` | completed | ESC-50 (real clips, not household audio) | `metrics.json` (protocol, per-fold info, all methods, CIs, false positives, calibration diagnostic), `predictions.csv`, `run.log`; `licence_subset.json` (E1-CC: thresholds refitted on the CC0 / CC BY clips only, nested comparison) |
| `e2/` | completed | CatMeows (440 meows, 21 cats, 3 contexts; unofficial GitHub mirror, per-file pins) | `metrics.json` (leave-one-cat-out and leave-one-owner-out, MFCC and YAMNet features, CIs), `status.json` |
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
| `e15/` | completed | Open Images V7 relationships (pet-object pairs, split by photographer) | `metrics.json`: per relation (on / under / inside a car) cross-validated selection, test AP/precision/recall with CIs, coverage, rule baseline, pre-set gate and whether it shipped, test false positives listed for audit; follow-up: `learned_vs_rule.json`, `coverage_by_object_threshold.json`, `rule_head.json` (shipped rule v2), `hard_negative_audit.json` |
| `e16/` | completed | Oxford-IIIT Pet (37 breeds, 7,349 images) | `metrics.json`: per species and breed found / wrong species / rejected / missed with Wilson CIs, breed heterogeneity permutation test |
| `e17/` | completed | MammalNet dog-breed segments (research only), as E11 | `metrics.json`: temporal model (job G3) vs pooled ablation vs E11, CV curves, 3 seeds; not adopted |
| `e18/` | completed | Kinetics-700-2020 validation, 6 pet-action classes, 294 clips with sound (YouTube; research only) | `metrics.json`: per class and species pet presence, species confusion, person-contact segments, vocalization and descriptive outputs; variants: reference detector, no verifier, phone-build detector |
| `e19/` | completed | Kinetics-700 train sample (300 clips, tuning) and the E18 validation clips (evaluation) | `metrics.json`: detector-threshold × verifier-scale grid replayed through the engine's tracker, selection, one-time paired evaluation, photo guard, replay fidelity |
| `e19b/` | completed | E19 tuning clips and 300 fresh train clips (holdout) | `per_species.json`: per-species thresholds, pre-specified adoption rule (not met) |
| `g1/` | data job completed | Open Images V7 train (G1 detector data) | `data_summary.json` (counts per class and split, leakage drops, licences, bytes, source checksums), `classes.json` |
| `data_access/` | completed | - | `network_probe.json`: reachability of every host the blocked sources need |
| `ondevice/` | completed in **desktop headless Chromium only** | parity fixtures, scenario media | `runtime_parity_*.json` (model level: browser LiteRT.js vs Python LiteRT), `engine_parity_chromium.json` (engine level, Python vs TypeScript engine on identical files; `_int8` for the compact bundle), `device_check_desktop_chromium.json` (device-check report from an automated browser, **not a phone**) |

No metric from different datasets is combined into a product-level number.
Multimodal / hypothesis accuracy on real household recordings is
`not_evaluated` (no such labelled data exists yet; see
[docs/EVALUATION_PLAN_REAL_PAIRED.md](../docs/EVALUATION_PLAN_REAL_PAIRED.md)).
