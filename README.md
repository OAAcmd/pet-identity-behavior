# pet-identity-behavior

**Pet Observer** - an observation-first behavior app for cats and dogs whose
core analysis runs **on the phone**. A mobile web app (PWA) records or opens
a clip, analyses it locally with downloaded models (no upload, works offline
after the one-time download) and reports *observations* (animals found and
tracked, sounds, movement, posture, owner-drawn places, interaction with a
person, quality limits, all with timestamps), then *experimental,
uncalibrated interpretations* tied to that evidence, abstaining whenever the
evidence is insufficient. It does not translate animal language, read
emotions, identify individual pets, or diagnose anything.

> **Status: not complete.** Software correctness is established in
> automated environments; behavioral validity on real household recordings,
> execution on physical phones and a production deployment are not. See
> [docs/REQUIREMENTS_MATRIX.md](docs/REQUIREMENTS_MATRIX.md) for the
> acceptance status and what blocks each open item.

## Architecture

```
phone browser (PWA) ── one-time model download (/ondevice/, SHA-256 verified) ──┐
  capture / file ─► TypeScript engine + LiteRT.js (WebAssembly) ─► result (IndexedDB, on the phone)
                                                                    │
server (optional): accounts, pets, places, bundle distribution, device reports,
                   server analysis only when the user turns it on (Python engine, same result schema)
```

Models: YAMNet (audio events), EfficientDet-Lite0 (cats, dogs, people),
EfficientNet-Lite0 features with small heads trained here (false-positive
verifier, posture, person-pet interaction); rule-based tracking, behavior
segments and fusion. Details: [docs/ON_DEVICE.md](docs/ON_DEVICE.md).

## Documents

| Topic | Document |
| --- | --- |
| Acceptance status and requirements | [REQUIREMENTS_MATRIX.md](docs/REQUIREMENTS_MATRIX.md), [STATUS.json](docs/STATUS.json) |
| What each capability can and cannot do (data, model, evaluation, gaps) | [CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md) |
| Methods and measured results (E1-E13) | [ALGORITHM_REPORT.md](docs/ALGORITHM_REPORT.md), `results/` |
| On-device design, evidence, audit | [ON_DEVICE.md](docs/ON_DEVICE.md), [ON_DEVICE_AUDIT.md](docs/ON_DEVICE_AUDIT.md) |
| Phones: tiers, budgets, acceptance test | [DEVICE_TIERS_AND_BUDGETS.md](docs/DEVICE_TIERS_AND_BUDGETS.md), [DEVICE_TEST_PROTOCOL.md](docs/DEVICE_TEST_PROTOCOL.md) |
| Data, models, licences | [DATA_AND_MODEL_REGISTER.md](docs/DATA_AND_MODEL_REGISTER.md), `manifests/resources.json`, [artifacts/README.md](artifacts/README.md) |
| Training and reproduction, GPU jobs | [TRAINING_PROGRAM.md](docs/TRAINING_PROGRAM.md) |
| Validation still needed | [EVALUATION_PLAN_REAL_PAIRED.md](docs/EVALUATION_PLAN_REAL_PAIRED.md) |
| Tests | [TEST_REPORT.md](docs/TEST_REPORT.md) |
| API | [API_CONTRACT.md](docs/API_CONTRACT.md), [openapi.json](docs/openapi.json) |
| Deploy and operate | [DEPLOYMENT.md](docs/DEPLOYMENT.md), [OPERATIONS_RUNBOOK.md](docs/OPERATIONS_RUNBOOK.md), [CONFIGURATION.md](docs/CONFIGURATION.md) |
| Intended use and limits | [MODEL_CARD.md](docs/MODEL_CARD.md), [DATA_LIMITATIONS.md](docs/DATA_LIMITATIONS.md) |
| Phase 1 record | [PROGRESS.md](docs/PROGRESS.md), [PHASE1_ACCEPTANCE.json](docs/PHASE1_ACCEPTANCE.json), [ENVIRONMENT.md](docs/ENVIRONMENT.md) |

## Development quick start

Requirements: Linux, Python 3.12, FFmpeg, [uv](https://docs.astral.sh/uv/),
Node.js 22. No GPU needed.

```bash
uv sync --extra dev --extra train                 # locked Python environment
uv run petperception fetch-models                 # SHA-256-verified model downloads
uv run petperception export-ondevice --release-profile personal_noncommercial   # phone bundle
cd web && npm ci && npm run build && cd ..        # PWA
uv run petperception serve                        # http://127.0.0.1:8000 (the PWA needs HTTPS off localhost)
```

Analyse a file on the server engine from the command line:
`uv run petperception analyze clip.mp4 --zones configs/zones.example.json --out result.json`.

Experiments: `uv run petperception fetch-data <dataset>` then
`uv run petperception evaluate --experiment e1 ... e13` (data sources,
licences and blockers per experiment in the register). Tests:
`uv run pytest -q`, `cd web && npm test && npx playwright test`
([TEST_REPORT.md](docs/TEST_REPORT.md)).

Deployment with Docker Compose and TLS: [DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Repository layout

```
src/petperception/  ingest/ audio/ vision/ context/ fusion/ schemas/ explain/ ondevice/ app/ data/ eval/ pipeline.py cli.py
web/                PWA (Preact + TypeScript); web/src/engine = on-device engine; tests (unit, runtime parity, E2E)
configs/ deploy/    configuration files; compose stack and Caddy TLS proxy
manifests/          resource register, dataset manifests and pinned checksums
artifacts/          small models trained here (numeric JSON / NPZ, no pickle)
results/            metrics per experiment and on-device parity reports
docs/               documents listed above
```

Raw datasets, media, third-party model weights, caches and credentials are
never committed.

## Licence and terms

Project code: private. Third-party data and model terms differ per asset and
per release profile (research / personal non-commercial / commercial); they
are recorded in the register and enforced by `petperception release-check`,
the engines and the bundle export. Commercial use is not assumed for any
asset without the recorded legal sign-off.
