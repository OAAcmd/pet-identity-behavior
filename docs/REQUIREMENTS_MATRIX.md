# Requirements matrix and acceptance status

Status date: 2026-10-09. Machine-readable copy: [STATUS.json](STATUS.json).

**The product is not complete.** Software correctness is established in
automated environments; behavioral validity, phone execution and production
readiness are not. Every open item names what blocks it.

## 1. Acceptance dimensions

| Dimension | What "accepted" requires | Status | Evidence | Blocking |
| --- | --- | --- | --- | --- |
| **A. Software correctness** | code does what it specifies; automated tests, lint, type checks and CI green; server and phone engines agree | **met in automated environments** | 92 Python tests, 16 TypeScript unit tests, 16 Playwright E2E tests (Chromium), CI green on every push, model-level browser parity bit-exact / ≤ 3e-8, engine parity 34/34 decisions ([TEST_REPORT.md](TEST_REPORT.md)) | Safari/WebKit and real phones untested (see C) |
| **B. Behavioral model validity** | each capability measured on held-out real data representative of use (household recordings), with acceptance thresholds met | **not met** | public-data validations only: E1, E4, E7, E9-E13 ([CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md)); rule-based behavior only on semi-synthetic video (E8); hypotheses unvalidated | real paired household recordings ([EVALUATION_PLAN_REAL_PAIRED.md](EVALUATION_PLAN_REAL_PAIRED.md)); cat keypoint/face/video data; licensed dog keypoint data |
| **C. On-device compatibility and performance** | the bundle runs offline on tier-B phones of both platforms within the budgets | **not met (pending)** | desktop headless Chromium only: device check passes, budgets measured there ([DEVICE_TIERS_AND_BUDGETS.md](DEVICE_TIERS_AND_BUDGETS.md)) | physical phones and testers ([DEVICE_TEST_PROTOCOL.md](DEVICE_TEST_PROTOCOL.md)) |
| **D. Complete product readiness** | deployed at a public HTTPS origin, release profile decided with legal sign-off, operations in place, A-C met | **not met** | image builds in CI, compose stack (bundle step verified in the image), runbook, release gating implemented | host/domain/TLS, legal review of weights and data terms, A-C |

The Phase 2 brief's dimensions map onto these: app-development readiness →
A; algorithm completeness → B (plus the capability gaps in section 3);
production deployment readiness → D; asset release suitability → D (release
profiles, section 4).

## 2. Requirements

Status columns: **Impl** implemented; **Test** covered by automated tests;
**Valid** validated on real, representative data or devices; **Blocked by**
what prevents the next status.

### 2.1 On-device architecture

| # | Requirement | Impl | Test | Valid | Blocked by / notes |
| --- | --- | --- | --- | --- | --- |
| D1 | Core analysis runs on the phone (audio events, detection, verification, tracking, posture, behavior segments, interaction candidates, fusion, explanations) | yes | engine parity, E2E | no | real phones |
| D2 | Works offline after the one-time model download | yes | E2E offline test (desktop Chromium; runtime cache asserted) | no | real phones (airplane-mode protocol) |
| D3 | No media upload in the default path; server analysis only by explicit setting | yes | E2E asserts no upload request | - | - |
| D4 | Capabilities that still need a server are listed | yes | - | - | [ON_DEVICE_AUDIT.md](ON_DEVICE_AUDIT.md) |
| D5 | Model loading, versioning, integrity (SHA-256), caching, atomic update with recovery, format compatibility | yes | E2E: corrupt update rejected, incompatible bundle refused, previous version keeps working | no | real phones |
| D6 | Browser runtime and operators validated before use | yes | model-level parity for every shipped model; unsupported builds documented (TF Hub detector crash) | desktop only | Safari/WebKit and phones |
| D7 | Preprocessing, timestamps, outputs and numerical differences validated between reference and deployed runtime | yes | runtime parity (identical tensors), engine parity (identical files) | desktop only | phones |
| D8 | No silent capability reduction or server substitution | yes | components report state and reason in every result; release gating notes in the bundle | - | - |
| D9 | Explanations without a remote language model | yes | template tests | - | - |
| D10 | Native path if the PWA is insufficient | not needed so far | - | - | decided by real-phone budgets |
| D11 | Opt-in sync of on-device results (no media) | **no** | - | - | not implemented; results stay on the device |

### 2.2 Device engineering

| # | Requirement | Impl | Test | Valid | Blocked by / notes |
| --- | --- | --- | --- | --- | --- |
| P1 | Device tiers and measurable budgets (size, memory, latency, sustained, battery/thermal), assumptions labelled | yes | budget unit tests; device check reports pass / fail / not measured | no | phones |
| P2 | Compression / quantization evaluated with re-evaluation incl. rejection behavior | yes (E12) | browser parity of int8/f16 builds | public photos only | int8 kept opt-in: per-image verifier decisions diverge from float32 |
| P3 | Real-phone offline end-to-end test | harness + protocol | desktop run of the harness | **pending** | physical phones |

### 2.3 Model and data program

| # | Requirement | Impl | Test | Valid | Blocked by / notes |
| --- | --- | --- | --- | --- | --- |
| M1 | Capability → dataset → model → evaluation matrix with missing labels and coverage | yes | - | - | [CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md) |
| M2 | Cats and dogs, breed and morphology variation | partial | E9 breed breakdown (dogs), E11 breed table | no | no cat breed metadata in any source; morphology not in any metadata |
| M3 | Facial cues | **no** | - | - | no usable data (cats or dogs) |
| M4 | Ear and tail cues | dogs, research profile | E9 | public photos | dog keypoint data is Research-Only; no cat data |
| M5 | Posture and temporal actions | posture (cats, dogs); rule segments; dog actions (research) | E10, E8, E11 | weak labels / semi-synthetic / YouTube | household video with labels; cat action data |
| M6 | Vocalizations | presence (YAMNet), types scored but unevaluated | E1 | ESC-50 clips | CatMeows and Barkopedia blocked (zenodo.org, huggingface.co); type-labelled pet audio |
| M7 | Interactions with people / animals / objects | people (E13, still images); animals: dog fights (research); objects: not trained | E13, E11 | public photos | interaction video; pet-pet data; object relations data exists but not trained |
| M8 | Environmental / scenario context | context sounds (unevaluated), owner zones | unit, E8 | no | household context labels |
| M9 | Multimodal interpretation with uncertainty and abstention | yes (rule-based) | E5 synthetic checks | **no** | real paired recordings |
| M10 | Splits by identity / household / source / breed where metadata supports; no invented metadata | yes | split tests (ESC-50 sources), protocols in each experiment | - | - |
| M11 | Real training with preserved configs, checkpoints, seeds, learning curves, independent evaluation | yes (E7, E9-E13) | experiments reproduce from commands | - | - |
| M12 | Compute requirements and reproducible jobs where CPU is insufficient | specified | - | - | GPU jobs G1-G5 not run ([TRAINING_PROGRAM.md](TRAINING_PROGRAM.md)) |
| M13 | Synthetic data only as integration/behavior checks | yes | E5, E8 labelled as such | - | - |
| M14 | Evaluation plan for real paired data | yes | - | - | [EVALUATION_PLAN_REAL_PAIRED.md](EVALUATION_PLAN_REAL_PAIRED.md); not started |

### 2.4 Algorithm gaps from Phase 1

| # | Requirement | Impl | Test | Valid | Notes |
| --- | --- | --- | --- | --- | --- |
| G1 | False positives on verified negatives reduced, with recall trade-off, separate tuning and evaluation data | yes (E7) | E7 protocol | public photos | dog specificity 0.56 → 0.87, recall 0.89 → 0.87 |
| G2 | Temporal behavior | rule-based segments | E8 | semi-synthetic only | household video |
| G3 | Tracking | Kalman tracker | E8 (IDF1 0.86) | semi-synthetic only | multi-pet household video |
| G4 | Posture / ear / tail | see M4, M5 | | | |
| G5 | Explicit uncertainty | reliabilities, abstention, validation status per segment, component reasons | schema tests | - | scores are uncalibrated and labelled so |

### 2.5 Backend, PWA, release

| # | Requirement | Impl | Test | Valid | Blocked by / notes |
| --- | --- | --- | --- | --- | --- |
| B1 | Accounts, pets, places (zones), feedback, export, account deletion | yes | backend tests, E2E | - | - |
| B2 | Server analysis with resumable chunked uploads, jobs, cancellation, retention | yes (opt-in) | backend tests, E2E | - | - |
| B3 | Health, readiness (incl. bundle), structured logs without personal data | yes | backend tests | - | - |
| W1 | Mobile PWA: capture (camera/microphone/file), on-device analysis, results timeline, history, settings, device check, accessibility | yes | E2E (3 viewports, axe checks) | no | real phones, Safari |
| R1 | Licence-confirmed assets before production; release profiles enforced | yes | release-check in CI | - | legal sign-off for `requires_legal_review` items (a browser-compatible licence-confirmed detector now exists: E14) |
| R2 | Locked dependencies, checksum-verified model retrieval | yes | CI (`uv sync --frozen`, `npm ci`, `fetch-models`) | - | - |
| R3 | Database migrations | yes | backend tests | - | - |
| R4 | CI | yes | green on every push | - | - |
| R5 | Deployment docs, runbook, configuration reference | yes | image build in CI | - | [DEPLOYMENT.md](DEPLOYMENT.md), [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md), [CONFIGURATION.md](CONFIGURATION.md) |
| R6 | Working public deployment | **no** | - | - | host, domain, DNS, TLS (operator) |

## 3. Capability gaps that block "complete"

1. No household validation of any capability; hypotheses unvalidated.
2. No cat keypoints, facial cues, cat video actions; no licensed dog
   keypoint data for product releases.
3. Vocal context datasets blocked by network policy.
4. No pet-pet interaction data; object relations not trained.
5. No real-phone results.
6. Result sync not implemented.
7. Commercial profile: every model still needs the recorded legal sign-off
   (training-data terms); the technical blocker (no browser-compatible
   licence-confirmed detector) is resolved by E14.

## 4. Asset release suitability (summary)

From `manifests/resources.json` (`petperception release-check` per profile):

| Asset | research | personal non-commercial | commercial |
| --- | --- | --- | --- |
| YAMNet | suitable | suitable | legal review |
| EfficientDet-Lite0 MediaPipe builds (float32 / int8 / float16) | suitable | suitable | not permitted |
| EfficientDet-Lite0 TF Hub build (raw decoding runs in the browser, E14) | suitable | suitable | legal review |
| EfficientNet-Lite0 (float32 / int8) | suitable | suitable | legal review |
| Audio thresholds (ESC-50 derived) | suitable | suitable | not permitted |
| Verifier, posture, interaction artifacts | suitable | suitable | suitable (backbone terms apply) |
| Dog keypoint head, dog behavior video head | suitable | not permitted | not permitted |
