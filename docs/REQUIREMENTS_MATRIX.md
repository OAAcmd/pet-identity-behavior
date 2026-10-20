# Requirements matrix and acceptance status

Status date: 2026-10-20. Machine-readable copy: [STATUS.json](STATUS.json).

**The product is not complete.** Software correctness is established in
automated environments; behavioral validity, phone execution and production
readiness are not. Every open item names what blocks it.

## 1. Acceptance dimensions

| Dimension | What "accepted" requires | Status | Evidence | Blocking |
| --- | --- | --- | --- | --- |
| **A. Software correctness** | code does what it specifies; automated tests, lint, type checks and CI green; server and phone engines agree | **met in automated environments** | 170 Python tests, 37 TypeScript unit tests (incl. 10 intent parity cases), 17 Playwright E2E tests (Chromium), model-level browser parity bit-exact / ≤ 3e-8, engine parity 38/38 decisions with the current engine (2026-10-19) ([TEST_REPORT.md](TEST_REPORT.md)) | Safari/WebKit and real phones untested (see C) |
| **B. Behavioral model validity** | each capability measured on held-out real data representative of use (household recordings), with acceptance thresholds met | **not met** | public-data validations only: E1, E1-CC, E2, E4, E7, E9-E22, E24; real pet video: YouTube (E18, the pet found in about half the clips) and home phone video (E20, cat 0.70 / dog 0.60 of frames, 14 % of dog frames only a 'cat'); audio types on FSD50K (E21: meow, bark pass; purr, growl fail); first cat keypoints (E24, research) ([CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md), per species); rule-based behavior only on semi-synthetic video (E8); hypotheses unvalidated | real paired household recordings ([EVALUATION_PLAN_REAL_PAIRED.md](EVALUATION_PLAN_REAL_PAIRED.md)); cat keypoint/face/video data; licensed dog keypoint data |
| **C. On-device compatibility and performance** | the bundle runs offline on tier-B phones of both platforms within the budgets | **not met (pending)** | the browser inference implementation has passed desktop Chromium tests; physical Android, iPhone and Safari validation remains pending; the phone test build is published with automatic report collection ([DEVICE_TEST_PROTOCOL.md](DEVICE_TEST_PROTOCOL.md)) | physical phones and testers ([DEVICE_TEST_PROTOCOL.md](DEVICE_TEST_PROTOCOL.md)) |
| **D. Complete product readiness** | deployed at a public HTTPS origin, release profile decided, each asset's terms cleared for it, operations in place, A-C met | **not met** | image builds in CI, compose stack (bundle step verified in the image), runbook, release gating implemented, asset terms recorded per artifact ([ASSET_TERMS.md](ASSET_TERMS.md)) | HTTPS origin (installed-app track); the specific open terms questions Q-M2, Q-T, Q-A2 in ASSET_TERMS.md; A-C |

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
| D1 | Core analysis implemented in the phone's browser engine (audio events, detection, verification, tracking, posture, behavior segments, person-interaction and on-furniture candidates, fusion, explanations; intent stage when the bundle carries it) | yes | engine parity, E2E; intent: feature-level parity (10 cases) and engine-level parity on 12 held-out clips (situation 12/12, status 9/12) | no | real phones |
| D2 | Works offline after the one-time model download | yes | E2E offline test (desktop Chromium; runtime cache asserted) | no | real phones (airplane-mode protocol) |
| D3 | No media upload in the default path; server analysis only by explicit setting | yes | E2E asserts no upload request | - | - |
| D4 | Capabilities that still need a server are listed | yes | - | - | [ON_DEVICE_AUDIT.md](ON_DEVICE_AUDIT.md) |
| D5 | Model loading, versioning, integrity (SHA-256), caching, atomic update with recovery, format compatibility | yes | E2E: corrupt update rejected, incompatible bundle refused, previous version keeps working | no | real phones |
| D6 | Browser runtime and operators validated before use | yes | model-level parity for every shipped model; unsupported builds documented (TF Hub detector crash) | desktop only | Safari/WebKit and phones |
| D7 | Preprocessing, timestamps, outputs and numerical differences validated between reference and deployed runtime | yes | runtime parity (identical tensors), engine parity (identical files) | desktop only | phones |
| D8 | No silent capability reduction or server substitution | yes | components report state and reason in every result; release gating notes in the bundle | - | - |
| D9 | Explanations without a remote language model | yes | template tests | - | - |
| D10 | Native path if the PWA is insufficient | not needed so far | - | - | decided by real-phone budgets |
| D11 | Opt-in sync of on-device results (no media) | yes (off by default) | backend tests (schema validation, isolation, CSRF, export, deletion); E2E (only JSON sent, restore on a wiped device, deletion removes the backup) | - | - |

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
| M4 | Ear and tail cues | dogs, research profile (E9); cats, research experiment (E24, not in the engines) | E9, E24 | public photos | cat ears meet the G2 PCK criterion (0.91), tails do not (0.51); all keypoint data research-only until image licences are verified (AwA2) or G2 annotation exists |
| M5 | Posture and temporal actions | posture (cats, dogs); rule segments; dog actions (research) | E10, E8, E11 | weak labels / semi-synthetic / YouTube | household video with labels; cat action data |
| M6 | Vocalizations | presence (YAMNet); types gated by validation (meow, bark shown; purr, growl, hiss not); bark context research-only (E22) | E1, E21, E21b, E22 | public audio: FSD50K eval (presence cat 0.69 / dog 0.73; meow 0.73 and bark 0.73 pass the gate; purr, growl fail; cat hiss not evaluable) | household audio with type labels; cat-hiss data; licensed bark-context data |
| M7 | Interactions with people / animals / objects | people (E13, still images); objects: pet on bed / couch / chair / table (E15, still images; "under" and "inside a car" failed the pre-set gate); animals: dog fights (research) | E13, E15, E11 | public photos | interaction video; pet-pet data; detector classes for pet objects (bowls, beds, cat trees, litter boxes) |
| M8 | Environmental / scenario context | context sounds; owner zones; scene (whole-frame) branch of the trained situation model; location as context for the bark channel | unit, E8, E27, E28 | public data | E28: the scene branch alone 0.643 macro-F1 and +0.073 [+0.026, +0.123] in the combined model; E27: location adds +0.068 [+0.016, +0.106] to the bark channel; household context labels missing |
| M9 | Multimodal interpretation with uncertainty and abstention | yes: rule-based fusion (default) and trained channels with a combination layer (intent stage, off by default): single-channel readings, combined situation (trained late fusion with availability flags), explicit missing inputs, conflicts | E5 synthetic checks; E27, E28 held-out public data; 9 combination unit tests; 10-case phone parity | situation only (E28 on YouTube clips); **intent: no** | no synchronized intent-labelled recordings (`taxonomy.MISSING`); the pre-registered audio-video conflict flag never fired (E28) |
| M10 | Splits by identity / household / source / breed where metadata supports; no invented metadata | yes | split tests (ESC-50 sources), protocols in each experiment | - | - |
| M11 | Real training with preserved configs, checkpoints, seeds, learning curves, independent evaluation | yes (E7, E9-E13, E15) | experiments reproduce from commands | - | - |
| M12 | Compute requirements and reproducible jobs where CPU is insufficient | specified and measured | G1 data-path tests | - | G1 trained on this CPU (full 50k-image run, 4.4 h): dog AP50 above the shipped detector, cat AP50 0.019 short of acceptance 1 at per-class operating points; much better on home video at its own operating point; durable checkpoint storage needs write access ([TRAINING_PROGRAM.md](TRAINING_PROGRAM.md)) |
| M13 | Synthetic data only as integration/behavior checks | yes | E5, E8 labelled as such | - | - |
| M14 | Evaluation plan for real paired data | yes | - | - | [EVALUATION_PLAN_REAL_PAIRED.md](EVALUATION_PLAN_REAL_PAIRED.md); not started |
| M15 | Observable behaviour, recorded context and intent kept as separate label layers; context never treated as a mental state | yes (`intent/taxonomy.py`, [BEHAVIOR_INTENT_DATA.md](BEHAVIOR_INTENT_DATA.md)) | taxonomy tests (a context is never an intent annotation; every link names an ontology hypothesis) | - | - |
| M16 | Trained behaviour/intent models compared with single branches, combined models and a transparent rule baseline, with audio-only / visual-only / combined, missing / noisy / conflicting inputs and coverage | yes (E27, E28; rules fixed before scoring) | experiment protocols; artifacts in `artifacts/intent/` | public data (Kinetics, Barkopedia) | intent-level combination unvalidated (no data); abstention thresholds optimistic on test (E28 0.764 vs 0.8 target) and not robust to poor video |

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
| R1 | Licence-confirmed assets before production; release profiles enforced | yes | release-check in CI | - | per asset in [ASSET_TERMS.md](ASSET_TERMS.md): the personal build is cleared (licence-stated detector since 2026-10-10); Q-M2 answered 2026-10-19 (no licence stated, the MediaPipe detector stays research-only); commercial needs Q-T (publisher decision on training-data terms) or G1b |
| R2 | Locked dependencies, checksum-verified model retrieval | yes | CI (`uv sync --frozen`, `npm ci`, `fetch-models`) | - | - |
| R3 | Database migrations | yes | backend tests | - | - |
| R4 | CI | yes | green on every push | - | - |
| R5 | Deployment docs, runbook, configuration reference | yes | image build in CI | - | [DEPLOYMENT.md](DEPLOYMENT.md), [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md), [CONFIGURATION.md](CONFIGURATION.md) |
| R6 | Working public deployment | **no** | - | - | host, domain, DNS, TLS (operator) |

## 3. Capability gaps that block "complete"

1. No household validation of any capability; hypotheses unvalidated.
2. No cat facial cues or cat video actions; cat ear keypoints exist only as
   a research model (E24, tails fail); no licence-verified keypoint data for
   product releases.
3. Vocalization types: only meow and bark pass validation (E21); cat hiss
   and growl need type-labelled data; vocal context is research-only (E2,
   E22).
4. No pet-pet interaction data; pet-object relations limited to "on furniture"
   (under-furniture and inside-car failed the gate; no detector for pet objects).
5. No real-phone results.
6. Commercial profile: the weights of YAMNet, the TF Hub detector and
   EfficientNet-Lite0 state Apache-2.0, but they were trained on ImageNet,
   COCO and AudioSet. Whether to ship them commercially is the publisher's
   decision on that specific question (ASSET_TERMS Q-T); G1b removes it for
   the detector.

## 4. Asset release suitability (summary)

From `manifests/resources.json` (`petperception release-check` per profile):

| Asset | research | personal non-commercial | commercial |
| --- | --- | --- | --- |
| YAMNet | suitable | suitable | Q-T (AudioSet training data) |
| EfficientDet-Lite0 MediaPipe builds (float32 / int8 / float16) | suitable | Q-M2 (no licence statement in the file) | not permitted |
| EfficientDet-Lite0 TF Hub build (raw decoding runs in the browser, E14; in the phone build) | suitable | suitable | Q-T (COCO training data) |
| EfficientNet-Lite0 (float32 / int8) | suitable | suitable | Q-T (ImageNet training data) |
| Audio thresholds (ESC-50 derived) | suitable | suitable | not permitted; the CC0/CC BY refit (E1-CC) is the candidate, open question Q-A2 |
| Verifier, posture, interaction, on-furniture artifacts | suitable | suitable | suitable (backbone terms apply) |
| Dog keypoint head, dog behavior video head | suitable | not permitted | not permitted |
