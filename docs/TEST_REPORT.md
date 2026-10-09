# Test report

Status date: 2026-10-09. **Everything below ran in automated environments
(this cloud workspace and GitHub Actions). No test has run on a physical
phone or in Safari/WebKit.** Real-phone acceptance is pending
([DEVICE_TEST_PROTOCOL.md](DEVICE_TEST_PROTOCOL.md)).

## Suites

| Suite | Count | Where | What it covers | Last result |
| --- | --- | --- | --- | --- |
| Python unit and integration (`pytest`) | 97 | workspace, CI | media ingestion and limits, splits and metrics, fusion and abstention, behavior segments, interaction geometry and segments, on-furniture features, segments and shipped-relation check, quantized I/O and in-graph dequantization, schema strictness, configuration strictness, pipeline with real models, backend (auth, CSRF, uploads, jobs, retention, deletion, logging without personal data, readiness incl. bundle, device reports, bundle route traversal, opt-in result backup: schema validation, isolation, export, deletion) | 97 passed |
| TypeScript unit (`vitest`) | 19 | workspace, CI | parity input generators, budgets, bundle compatibility, quantized input encoding (same bytes as Python), interaction geometry and segments, on-furniture features and segments (same values as Python), UI logic | 19 passed |
| Model-level browser parity (`playwright -c playwright.runtime.config.ts runtime.spec.ts`) | 2 configurations (1 thread, 4 threads) | workspace | every shipped model and the quantized builds in LiteRT.js vs Python LiteRT on identical tensors | bit-exact or ≤ 3e-8; the TF Hub detector crashes with its in-graph post-processing and is bit-exact with it removed (E14) |
| Engine-level parity (`engine.spec.ts` + `engine-parity compare`) | 39 media cases (38 compared, 1 deliberately corrupt) | workspace | full TypeScript engine vs Python engine on identical files | decisions 38/38; tracks F1 0.96, sounds 1.00, segments 0.97 (incl. object of on-furniture segments), hypotheses 1.00 (float32 bundle); every phone result validates against the strict server result schema |
| PWA end-to-end (`playwright test`) | 17 runs of 15 tests (Pixel 7 emulation; layout test also with iPhone-size viewport and desktop, all Chromium) | workspace | registration, accessibility (axe, WCAG 2 A/AA serious/critical), pets and drawn zones, camera and microphone capture (fake devices) analysed on the device, on-device analysis without any upload request, offline analysis with the runtime asserted in Cache Storage, corrupt update rejected and previous bundle kept, incompatible bundle refused, opt-in result backup (only JSON sent, restore on a wiped device, deletion removes the backup), device check, server analysis (resumable chunked upload, offline pause and resume, corrupt file, cancellation, feedback, deletion), camera permission denied | 17 passed |
| Container image | 1 | CI, workspace | image builds from locked dependencies; starts and reports health; in the workspace additionally: bundle export inside the container and `/ready` with the bundle check | passed |
| CI (`.github/workflows/ci.yml`) | 3 jobs | GitHub Actions | lint, release gate, Python tests with real models, schema up to date, TypeScript typecheck/tests/build, image build and smoke | green on every push of this branch |

## What the automated tests do not establish

* Execution on phones: CPU speed, memory limits, thermal behaviour, iOS
  Safari (no WebKit run at all), Android Chrome on real hardware.
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
uv run petperception export-ondevice --release-profile personal_noncommercial
npx playwright test                        # E2E
uv run petperception export-parity && npx playwright test -c playwright.runtime.config.ts tests/runtime/runtime.spec.ts
uv run petperception engine-parity reference && npx playwright test -c playwright.runtime.config.ts tests/runtime/engine.spec.ts \
  && uv run petperception engine-parity compare --write ../results/ondevice/engine_parity_chromium.json
```
