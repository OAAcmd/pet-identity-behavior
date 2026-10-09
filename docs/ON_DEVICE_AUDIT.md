# On-device audit (what runs where, and how it is validated)

Target architecture: models are trained and exported in a compute
environment; **core audio/video analysis runs on the phone**; the server
only distributes model bundles, hosts accounts and receives data the user
explicitly opts in to send. After the model bundle is installed, the core
analysis must work with no network connection.

Status date: 2026-10-09. "Desktop Chromium" below means Playwright headless
Chromium on a 4-core cloud VM, sometimes with a phone *viewport/user agent*
emulated. **No physical phone has run this software yet**
(see [DEVICE_TEST_PROTOCOL.md](DEVICE_TEST_PROTOCOL.md)).

Evidence files: `results/ondevice/runtime_parity_*.json` (model-level
parity), `results/ondevice/engine_parity_chromium.json` (engine-level
parity, 35 cases; `_int8` for the compact bundle),
`results/ondevice/device_check_desktop_chromium.json` (device-check report in
emulated Chromium), web E2E suite (`web/tests/e2e/flows.spec.ts`, 16 tests
passing in Chromium).

## Component table

| Component | What actually runs today | Server | Phone (PWA) | Origin | Validation status | Path to on-device | Unresolved |
|---|---|---|---|---|---|---|---|
| Media decoding / sampling | Server: FFmpeg (probe, PTS-exact 2 fps sampling, audio decode at native rate). Phone: browser decoders (video element seek + canvas; `decodeAudioData` into a 16 kHz `OfflineAudioContext`) | yes (server analysis only) | **yes** | library code | Engine parity on 30 cases (identical WebM inputs); E2E in Chromium | done | Dropped-frame check and native sample-rate (bandwidth) check are not available in the browser. Playwright's open-source Chromium lacks H.264/AAC, so automated on-device tests use WebM; MP4 on real Safari/Chrome is **untested**. |
| Audio events (YAMNet, MediaPipe build) | LiteRT on server; LiteRT.js WASM on phone | yes | **yes** | pretrained (AudioSet) | E1 on ESC-50 clips (not household audio). Browser output bit-identical to Python on identical input | done | No household-audio validation. Thresholds derive from ESC-50 (CC BY-NC): excluded from commercial bundles. Commercial use of the weights needs legal sign-off. |
| Cat/dog vocal context (CatMeows, Barkopedia) | nothing (blocked) | no | no | - | none | blocked by network policy (zenodo.org, huggingface.co) | Data access; dataset-context classification would still not be household intent inference. |
| Cat/dog detector (EfficientDet-Lite0, MediaPipe raw-anchor build) | server and phone (float32 by default; int8 build as opt-in compact bundle) | yes | **yes** | pretrained (COCO) | E4 on Open Images validation (web photos), E7, E12 (int8 / float16 re-evaluated) | done; float32 outputs within 3e-8 of Python, int8 bit-exact | Household-camera validation missing. MediaPipe builds have no primary-source weight licence (not permitted in the commercial profile). The licence-confirmed TF Hub build crashes LiteRT.js 2.5.3 (in-graph post-processing); fix path: strip the post-processing op and decode in TypeScript (not done). |
| False-positive verifier (crop classifier + per-class logistic regression) | server and phone | yes | **yes** | trained here (E7; refitted for the int8 pair in E12) | E7: tuning on Open Images test split, final on the untouched E4 subset; E12 for int8 | done (crop feature set) | ImageNet training-data terms need legal review for commercial use. Household validation missing. Verifier parameters are tied to the backbone build: using float parameters on int8 features fails the recall rule (E12). |
| Tracker (Kalman / SORT-style) | server and phone | yes | **yes** | rule-based algorithm | E8 semi-synthetic (real animal pixels, scripted motion); engine parity tracks F1 0.93 | done | No real multi-animal household tracking benchmark. |
| Camera-motion compensation | server: ORB + RANSAC; phone: block matching (different method) | yes | **yes (different algorithm)** | rule-based | Server: E8 camera-pan scripts. Phone: only indirectly through engine parity (segments F1 0.95) | done | Phone method not separately evaluated. |
| Behavior segments (stillness, movement, active bout, back-and-forth, zone approach/enter/dwell/leave, view entry/exit, vocal bout, startle candidate) | server and phone | yes | **yes** | rule-based, a-priori thresholds | E8 semi-synthetic only; engine parity F1 0.95 | done | **Not validated against labelled household behavior.** |
| Zones (owner-drawn places) | server and phone | yes | **yes** | rule-based | unit tests, E8 zone scripts | done | - |
| Coarse posture (lying / sitting / standing) | server and phone | yes | **yes** | trained here (E10) on weak caption labels | E10 (agreement with caption-derived labels; caption noise unmeasured); E12 on int8 features | done; identical labels whenever both engines report one | Weak labels only; no posture annotation; household validation missing. |
| Dog keypoints (ears, tail, legs) and derived cues | server, research profile only | research only | **no** | trained here (E9) | E9 on dog-pose validation (web photos): PCK@0.1 0.853 | TypeScript head port is straightforward (same backbone); not done because the training data are Research-Only | Production needs licensed dog keypoint data. |
| Cat keypoints / cat ear & tail cues | nothing | no | no | - | none | needs data | **No cat keypoint dataset available.** |
| Dog behavior from video (sleep, eat, drink, fight, eliminate) | experiment only (E11) | no | no | trained here (E11) on MammalNet dog videos | E11: test macro-F1 0.638 [0.537, 0.716] (official MammalNet splits) | same backbone + small head; not ported because the data are research-only | Research-only data (YouTube videos, no dataset licence); no cats in MammalNet. |
| Person-pet interaction candidates (holds / interacts with) | server and phone (person boxes are computed but never stored) | yes | **yes** | trained here (E13) on Open Images relationships | E13 on OI validation + test (still images): AP 0.907 on scored dog pairs, end-to-end recall 0.42; engine parity: identical segments on 4 person+pet clips | done | Still-image model applied per frame; temporal grouping unvalidated; cats not evaluated; person detection coverage is the main limit. |
| Fusion / hypotheses / abstention | server and phone | yes | **yes** | rule-based, unfitted | none on real paired data; engine parity hypotheses F1 1.0 | done | Needs real paired household recordings with behavior labels ([EVALUATION_PLAN_REAL_PAIRED.md](EVALUATION_PLAN_REAL_PAIRED.md)). |
| Explanations and suggestions | server and phone (templates) | yes | **yes** | rule-based templates | tests | done | Core explanations never need a remote language model. |
| Model distribution | server publishes `/ondevice/` bundle (SHA-256 manifest) | yes | client verifies, caches, switches atomically | - | E2E install / offline test | done | Signed bundles (beyond HTTPS + SHA-256 manifest) not implemented. |
| Accounts, pets, places | server; cached on device for offline use | yes | cached | - | backend tests, E2E | - | Pets/places created offline cannot be saved until online. |
| Result sync | **not implemented**; results stay on the device | - | - | - | - | opt-in upload of result JSON (no media) | Planned; until then on-device results are not backed up. |
| Server analysis (upload media) | optional path, **off by default**, explicit setting | yes | uploads only when enabled | - | backend and E2E tests | - | - |

## Capabilities that still require a remote service

* Installing or updating the model bundle (one-time download, ~37 MB plus the
  9.4 MB WebAssembly runtime).
* Creating the account and syncing pets/places edits (reads are cached).
* Server analysis, only if the user turns it on.
* Sending a device-check report (explicit button).

Nothing in the default analysis path uploads media. The automated test
`on-device analysis of an existing file uploads no media` asserts that no
upload or analysis request is made.
