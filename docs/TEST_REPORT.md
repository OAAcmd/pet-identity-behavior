# Test report

Status date: 2026-10-20. **Everything below ran in automated environments
(this cloud workspace and GitHub Actions).** The browser inference
implementation has passed desktop Chromium tests. Physical Android, iPhone
and Safari validation remains pending. The phone test build for that is
published, with automatic report collection
([DEVICE_TEST_PROTOCOL.md](DEVICE_TEST_PROTOCOL.md)); no phone report has
arrived yet.

## Suites

| Suite | Count | Where | What it covers | Last result |
| --- | --- | --- | --- | --- |
| Python unit and integration (`pytest`) | 206 | workspace, CI | media ingestion and limits, splits and metrics, fusion and abstention, behavior segments, interaction geometry and segments, on-furniture features, segments and shipped-relation check, quantized I/O and in-graph dequantization, schema strictness, configuration strictness, pipeline with real models, backend (auth, CSRF, uploads, jobs, retention, deletion, logging without personal data, readiness incl. bundle, device reports, bundle route traversal, opt-in result backup: schema validation, isolation, export, deletion) | 206 passed, 0 skipped on 2026-10-31 (adds the E33 zero-shot evaluation path with the student and a text bank; per-class detector operating points: the shared helper, the committed G1b model resolving from the repository with research-only release, the phone bundle refusing per-class points, the engine loading G1b with its verifier; the E33 student exports: hashes match the run record, fixed batch of 1, unit-length 768-d outputs, float16 / int8 agree with float32 on the parity input; and behaviour explanation v2, `tests/test_understand_v2.py`: after/before/until context phrases in questions, the turning-in-place model (round trip, top-quarter clip score, bout merging, box-shape oscillation features), recording context and before/after events on synthetic detector and sound scores, recording cues counting half of the user's words, E32 benchmark integrity and disjointness from the development collection and the E31 benchmark, E32 per-question outcomes and the pre-registered decision rules, and the frozen E31 default system pinned by file hashes). 192 passed on 2026-10-20 (adds behaviour explanation, `tests/test_understand.py`: knowledge-base integrity, the per-class detector wrapper used for the G1b downstream test (every non-ordinary candidate cites a known source; every vocabulary behaviour has an entry per species), question parsing with pronoun normalization, answer composition (request vs no-request separation, no certainty phrases, reported-but-not-recognized status, the user's species kept against the detector), thresholds from the recognizer configuration, the turning-in-place descriptor, trained-head and configuration round trips, E31 scoring on a synthetic answer, benchmark file integrity, human labels never mixed with assistant labels). 177 passed earlier the same day (adds the intent layer: heads, calibration, abstention thresholds, late/early fusion with absent branches, label-taxonomy rules (a context is never an intent annotation), rule baselines, the combination (missing inputs, audio-video and species conflicts, context-only, below-gate channels), the engine's intent stage on synthetic media; result schema 1.5.0). 148 passed on 2026-10-19 with every dataset fetched (adds: E21 scoring and the vocal tag rule, the subtype gate (only validated subtypes become tags; unvalidated ones only block, e.g. a possible growl keeps "relaxed" abstained), E20 mask boxes and sequence statistics, AwA-Pose data-only unpickling, licence parsing and photographer splits, G1 hard negatives, crowd flags, edge boxes and label aliases, LocalSplitZip on split archives) |
| TypeScript unit (`vitest`) | 40 | workspace, CI | parity input generators, budgets, bundle compatibility, quantized input encoding (same bytes as Python), interaction geometry and segments, on-furniture features and segments (same values as Python), UI logic | 40 passed on 2026-10-20 (adds 13 intent parity cases, including the E29 divergence detector and the E30 degraded-video rule, against a Python fixture: pooled branches within 1e-5 relative; identical readings, hypotheses, statuses, conflicts and missing inputs). Earlier: vocal tags and conflict-only tags, per-tag context thresholds, same cases as Python; only validated subtypes are displayed |
| Model-level browser parity (`playwright -c playwright.runtime.config.ts runtime.spec.ts`) | 2 configurations (1 thread, 4 threads) | workspace | every shipped model and the quantized builds in LiteRT.js vs Python LiteRT on identical tensors | bit-exact or ≤ 3e-8; the TF Hub detector crashes with its in-graph post-processing and is bit-exact with it removed (E14). 2026-10-31, E33 student (not shipped): float32 / float16 bit-exact, int8 max abs diff 0.004; the first export (dynamic batch) failed in LiteRT.js and was re-exported with a fixed batch |
| Engine-level parity (`engine.spec.ts` + `engine-parity compare`) | 39 media cases (38 compared, 1 deliberately corrupt) | workspace | full TypeScript engine vs Python engine on identical files | 2026-10-19, current engine (subtype gate, result schema 1.4.0, asset format 0.2.2), reference bundle: decisions 38/38; tracks F1 0.95, sounds 1.00, segments 0.98, hypotheses 0.99 (one "insufficient evidence" candidate on the noise-added clip, which the browser decodes from WebM); 0 phone results fail the server result schema (`results/ondevice/engine_parity_chromium.json`) |
| Intent engine parity (`engine.spec.ts` with `PARITY_DIR`/`BUNDLE_DIR` + `engine-parity intent-compare`) | 12 held-out Kinetics clips (WebM) | workspace | the TypeScript engine with the intent stage (bundle exported with `--intent`) vs the Python engine with `intent.enabled` | 2026-10-20: situation label 12/12, species 11/12, hypotheses 11/12, overall status 9/12 (threshold-boundary cases), median situation-score difference 0.039, 0 schema errors (`results/ondevice/intent_engine_parity_chromium.json`) |
| PWA end-to-end (`playwright test`) | 17 runs of 15 tests (Pixel 7 emulation; layout test also with iPhone-size viewport and desktop, all Chromium) | workspace | registration, accessibility (axe, WCAG 2 A/AA serious/critical), pets and drawn zones, camera and microphone capture (fake devices) analysed on the device, on-device analysis without any upload request, offline analysis with the runtime asserted in Cache Storage, corrupt update rejected and previous bundle kept, incompatible bundle refused, opt-in result backup (only JSON sent, restore on a wiped device, deletion removes the backup), device check, server analysis (resumable chunked upload, offline pause and resume, corrupt file, cancellation, feedback, deletion), camera permission denied | 17 passed (2026-10-20 with the reference bundle, asset format 0.2.3, intent stage off; 2026-10-19 with the phone build bundle) |
| Phone test page (`playwright -c playwright.devicetest.config.ts`) | 1 run through every step | workspace (headless Chromium, desktop) | the published phone test build served as static files: download and verification of the models, engine start, model check, reference clip ×3, ten sustained analyses with a background/foreground switch, offline analysis with 0 network requests, a recorded clip, reload from saved models, report | passed with the licence-stated detector bundle (`var/device_test/desktop_chromium_report.json`): detector 74 ms per frame, 11.2 s clip in 5.6 s; **desktop figures, not phone figures** |
| Container image | 1 | CI, workspace | image builds from locked dependencies; starts and reports health; in the workspace additionally: bundle export inside the container and `/ready` with the bundle check | passed |
| CI (`.github/workflows/ci.yml`) | 3 jobs | GitHub Actions | lint, release gate, Python tests with real models, schema up to date, TypeScript typecheck/tests/build, image build and smoke | green on every push of this branch |

## What the automated tests do not establish

* Execution on phones: CPU speed, memory limits, thermal behaviour, iOS
  Safari (no WebKit run at all), Android Chrome on real hardware. The
  phone test page measures all of these once testers run it.
* Media from phone cameras (H.264/HEVC/MOV); the automated Chromium build
  has no H.264 decoder, so on-device tests use WebM copies.
* Behavioral validity: test media are public photos, synthetic pairings and
  semi-synthetic sequences ([CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md)).
* A public deployment behind TLS with real certificates.

## How to run

```bash
uv run ruff check src tests && uv run pytest -q
cd web && npm run typecheck && npm test && npm run build
bash tests/e2e/make_fixtures.sh            # needs var/examples (evaluate e5)
uv run petperception fetch-data openimages-tuning   # device-check reference images
uv run petperception export-ondevice --release-profile personal_noncommercial --detector efficientdet_lite0_tfhub_rawdecode_v1
npx playwright test                        # E2E (the phone build's bundle; without --detector the
                                           # personal profile excludes the MediaPipe detector, Q-M2)
uv run petperception export-parity && npx playwright test -c playwright.runtime.config.ts tests/runtime/runtime.spec.ts
uv run petperception engine-parity reference && npx playwright test -c playwright.runtime.config.ts tests/runtime/engine.spec.ts \
  && uv run petperception engine-parity compare --write ../results/ondevice/engine_parity_chromium.json
```
