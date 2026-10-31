# pet-identity-behavior

**Pet Observer** - an observation-first behavior app for cats and dogs whose
core analysis is built to run **in the phone's browser** (TypeScript engine +
LiteRT.js WebAssembly; no upload, offline after the one-time model
download). **Device evidence so far: the browser inference implementation
has passed automated tests in desktop Chromium only; physical Android,
iPhone and Safari validation is pending** ([phone test](docs/DEVICE_TEST_PROTOCOL.md)).
A mobile web app (PWA) records or opens a clip, analyses it locally and
reports *observations* (animals found and
tracked, sounds, movement, posture, owner-drawn places, interaction with a
person, being on a bed, couch, chair or table, quality limits, all with
timestamps), then *experimental,
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
                   opt-in backup of result JSON (never recordings),
                   server analysis only when the user turns it on (Python engine, same result schema)
```

Models: YAMNet (audio events; only vocalization types that passed the
FSD50K validation gate, meow and bark, are shown or can trigger an
interpretation, E21), EfficientDet-Lite0 (cats, dogs, people, furniture),
EfficientNet-Lite0 features and small heads trained here (false-positive
verifier, posture, person-pet interaction, pet on furniture); rule-based
tracking, behavior segments and fusion. A pet-and-household detector trained
here (G1b) met acceptance 1 and finds far more dogs in real video, but not
more cats (acceptance 2 not met), so the phone detector is unchanged
([docs/TRAINING_PROGRAM.md](docs/TRAINING_PROGRAM.md)). Details: [docs/ON_DEVICE.md](docs/ON_DEVICE.md).

Behaviour / intent layer (`src/petperception/intent/`, optional `intent.enabled`):
trained channels on frozen encoders (situation from synchronized audio and video,
E28; dog vocal activity / intent from barks, E26-E27; bark and meow context) and a
combination that keeps observable behaviour, recorded context and intent apart,
lists missing inputs and reports conflicts. Try it on a file:
`uv run petperception intent clip.mp4` (label layers and data gaps:
[docs/BEHAVIOR_INTENT_DATA.md](docs/BEHAVIOR_INTENT_DATA.md)).

Behaviour explanation in context (`src/petperception/understand/`, research
profile, server/desktop only): give the species, the scene and your question
with a recording, and get what the recording shows (with times) plus
candidate explanations grounded in cited veterinary and ethological sources,
with requests kept apart from play, stress, compulsive and medical causes and
never a stated cause:
`uv run petperception explain clip.mp4 --species cat --scene "open living room" --question "my cat keeps turning in circles while biting its own tail - what could this mean?" --markdown answer.md`
(needs `uv sync --extra behavior` and `petperception fetch-models --only siglip_base_patch16_224`).
The default is the E31 system; `--system-version 2` runs the E32 system
(recording context, turning-in-place model, parser v2; also needs
`fetch-models --only minilm_l6_v2`), which was better on every paired measure
of the fresh E32 test but did not meet the pre-registered bar to become the
default. Measured on 37 real-clip questions in E31 and 37 fresh ones in E32;
what it can and cannot answer:
[docs/BEHAVIOR_EXPLANATION.md](docs/BEHAVIOR_EXPLANATION.md). It does not run
on the phone: phone-sized encoders tested so far (a distilled 4-15 MB
student, E33; TinyCLIP, E34) keep too little of the zero-shot behaviour
recognition. A cloud-assisted path (`--understanding cloud`, server only, off by default) and the
priority tail-chasing scenario are in §13 of that document: no recognizer is reliable yet on held-out sources, and
the hosted model has not been evaluated (no API key here).

## Try the user journey

* **Web**: `cd web && npm ci && npm run build`, then `uv run petperception serve --port 8765 --workers 1` and open
  http://127.0.0.1:8765 - create the first account, then **Ask about my pet's behaviour** (describe the concern, add a
  10-60 s video, read the answer, correct it). The answer comes from the server's pipeline; the cloud provider path is
  off unless configured (docs/OPERATIONS_RUNBOOK.md). Flow and screenshots: [docs/UX_FLOW.md](docs/UX_FLOW.md).
* **Android test build**: `android-app/dist/pet-observer-0.1.0-debug.apk` (install with `adb install`; on the emulator
  connect to `http://10.0.2.2:8765`). Built and linted here; not yet run on an emulator or a phone
  ([android-app/README.md](android-app/README.md)).
* **Capability totals**: [docs/CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md) §0.

## Documents

| Topic | Document |
| --- | --- |
| Acceptance status and requirements | [REQUIREMENTS_MATRIX.md](docs/REQUIREMENTS_MATRIX.md), [STATUS.json](docs/STATUS.json) |
| What each capability can and cannot do, per species (software, behavioural evidence, device, release) | [CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md) |
| Methods and measured results (E1-E31; intent layer in section 9, behaviour explanation in section 10) | [ALGORITHM_REPORT.md](docs/ALGORITHM_REPORT.md), `results/` |
| Behaviour / intent: label layers, data inventory, missing annotations | [BEHAVIOR_INTENT_DATA.md](docs/BEHAVIOR_INTENT_DATA.md) |
| Explaining a behaviour in context: existing work, data, method, E31 benchmark, answerable questions, gaps | [BEHAVIOR_EXPLANATION.md](docs/BEHAVIOR_EXPLANATION.md) |
| On-device design, evidence, audit | [ON_DEVICE.md](docs/ON_DEVICE.md), [ON_DEVICE_AUDIT.md](docs/ON_DEVICE_AUDIT.md) |
| Phones: tiers, budgets, acceptance test | [DEVICE_TIERS_AND_BUDGETS.md](docs/DEVICE_TIERS_AND_BUDGETS.md), [DEVICE_TEST_PROTOCOL.md](docs/DEVICE_TEST_PROTOCOL.md) |
| Data, models, licences | [ASSET_TERMS.md](docs/ASSET_TERMS.md) (per asset, open questions, replacements), [DATA_SOURCES.md](docs/DATA_SOURCES.md) (94-source survey, network blockers), [DATA_AND_MODEL_REGISTER.md](docs/DATA_AND_MODEL_REGISTER.md), `manifests/resources.json`, [artifacts/README.md](artifacts/README.md) |
| Training jobs G1-G5, I1 (intent heads) and reproduction | [TRAINING_PROGRAM.md](docs/TRAINING_PROGRAM.md), `training/g1/`, [G2 annotation spec](docs/training/G2_ANNOTATION_SPEC.md) |
| Validation still needed; household collection package | [EVALUATION_PLAN_REAL_PAIRED.md](docs/EVALUATION_PLAN_REAL_PAIRED.md), [docs/household/](docs/household/README.md) |
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

Experiments: `uv run petperception fetch-data <dataset>` (or `mirror fetch
--dataset <id> --official` for archives listed in `manifests/mirror_files.json`)
then `uv run petperception evaluate --experiment e1 ... e25` (data sources,
licences and blockers per experiment in the register). Detector training
(G1, CPU or GPU, resumable): [TRAINING_PROGRAM.md](docs/TRAINING_PROGRAM.md). Tests:
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
the engines and the bundle export. Per-asset terms, the specific open
questions and their replacements: [docs/ASSET_TERMS.md](docs/ASSET_TERMS.md).
Commercial use is not assumed for any asset whose question is open.
