# Phase 1 Progress

Branch: `claude/dreamy-davinci-49d6ng`. Starting point: repository with a
two-line README only (no prior code, data or models to preserve).

## Done (verified in this workspace)

1. **Environment** - inspected Python/CPU/RAM/disk/FFmpeg/GPU/network
   (docs/ENVIRONMENT.md); Python 3.12 venv via uv; `uv.lock` + hash-pinned
   `requirements.lock.txt`. PyTorch not needed (frozen TFLite + scikit-learn).
2. **Register and manifests** - `manifests/resources.json` and
   docs/DATA_AND_MODEL_REGISTER.md with URLs, revisions, byte counts, SHA-256,
   observed counts, grouping, label provenance, separate code/data/weight terms,
   attribution, restrictions, and the exact errors for blocked sources.
   Per-file manifests for ESC-50 (2000 files, all attributed) and the Open
   Images subset (823 images, pinned SHA-256, per-image license/author).
3. **Ingestion** - probe-before-decode with limits, shared PTS timeline
   (verified < 0.4 ms A/V alignment across MP4/MKV-offset/WebM-Opus after fixing
   two real offset bugs), distinct missing/silent/corrupt/no-frame states,
   native-rate audio, frame sampling with original timestamps, dropped-frame scan.
4. **Audio** - YAMNet backend with verified interface (sigmoid tags, int8
   internals, label map, embedding tensor); ontology-derived cat/dog groups;
   timestamped events with raw scores; CPU MFCC baseline; JSON-persisted models.
5. **Vision** - EfficientDet-Lite decoding from embedded metadata (anchors,
   normalization, label map), NMS, cross-species merge, IoU/Hungarian tracker,
   ORB/RANSAC camera-motion compensation, quality flags, zones, pose interface
   with explicit unavailability.
6. **Fusion** - ontology, provenance roots, co-occurrence, attribution rules,
   abstention codes; rule baseline + weighted average + context-gated PoE
   (all experimental, unfitted); deterministic explanations and suggestions.
7. **Schema/CLI/API** - Pydantic schema 1.0.0 shared by CLI and API;
   `doctor`, `fetch-models`, `fetch-data`, `train`, `evaluate`, `analyze`,
   `serve`, `export-openapi`, `make-examples`; FastAPI service with token auth,
   loopback default, upload validation, private job dirs, deterministic deletion.
8. **Experiments** - E1 and E4 on real data; E5 behavior checks (14 scenarios x
   3 scorers, all passing); E6 runtime; E2/E3 implemented with recorded
   blocked/pending status.
9. **Tests** - 54 pytest tests (42 synthetic-only, 5 integration, 2 both,
   5 unmarked manifest/schema/contract checks), all passing.
10. **Docker** - CPU image built and tested (doctor, analyze, token-protected
    API round-trip, refusal of unauthenticated public bind).

## Bugs found and fixed during verification

- Video frame timestamps were re-based to the container start (fixed with `-copyts`).
- Audio sample-to-time anchor ignored Opus pre-skip / priming (now first decoded frame pts).
- AP bootstrap collapsed duplicate resampled images (CI was below the point estimate).
- Fusion ignored tracks referenced only by attributed audio (subject enumeration).
- Movement summaries contained "%" text that the no-percentage invariant flagged.
- ESC-50 LICENSE parser missed 5 irregular attribution lines (now 2000/2000).

## Commands actually run (outcomes)

| Command | Outcome |
| --- | --- |
| `uv lock --python 3.12 && uv sync --extra dev` | 60 packages resolved and installed |
| `git clone --depth 1 https://github.com/karolpiczak/esc-50` | revision 33c8ce9e, 2000 clips |
| `fetch_images` (Open Images subset) | 823/823 images after retrying sporadic S3 404s |
| `uv run petperception doctor --network` | models ready; zenodo/huggingface/ultralytics denied |
| `uv run petperception fetch-data catmeows` | exit 2, `blocked` (HTTP 403 tunnel) |
| `uv run petperception fetch-data barkopedia --audit-only` | exit 2, `blocked` |
| `evaluate --experiment e1` | completed; YAMNet tags macro-F1 0.895 |
| `evaluate --experiment e2` / `e3` | `blocked` / `pending` status files |
| `evaluate --experiment e4` | completed; Lite0 cat AP 0.923, dog AP 0.842 |
| `evaluate --experiment e5` | 14/14 per scorer |
| `evaluate --experiment e6` | 8 s video 0.95 s warm |
| `export-openapi`, `make-examples`, `analyze ...` | written / verified |
| `uv run pytest` | 54 passed |
| `docker build ...` + `docker run ...` | built; doctor/analyze/serve verified |

## Open items

See `blocked_or_optional_items` in docs/PHASE1_ACCEPTANCE.json: CatMeows (E2),
Barkopedia (E3), AST, dog pose, tracking ground truth, household paired data,
TLS/production auth (Phase 2), primary-source re-verification of MediaPipe
weight terms.

## Continuation point

With `zenodo.org` allowed:
`uv run petperception fetch-data catmeows && uv run petperception evaluate --experiment e2`,
then update the register (verified counts, sample rates, checksums, license
field) and the E2 section of the report. Phase 2 (PWA) has not been started.
