# Device tiers and resource budgets

Status date: 2026-10-09. **Every limit in this document is an assumption
until it has been measured on physical phones of the tier.** The only
measurements so far come from headless Chromium on a 4-core cloud VM (not a
phone). The device check in the app (`#/device-check`, protocol in
[DEVICE_TEST_PROTOCOL.md](DEVICE_TEST_PROTOCOL.md)) measures each budget on a
real phone and reports it as `pass`, `fail` or `not_measured`. Limits live in
one place: `web/src/engine/budgets.ts`.

## Tiers

| Tier | Meaning | Examples (assumed, not yet tested) | Expected browser |
| --- | --- | --- | --- |
| A | current flagship | 2023 or newer flagships, ≥ 6 GB RAM: iPhone 15, Pixel 8, Galaxy S23 | Safari 17+, Chrome 120+ |
| B | mid-range, **reference tier** (the product must meet its budgets) | 2021-2023 mid-range, 4-8 GB RAM: Pixel 6a/7a, Galaxy A54, iPhone 12 | Safari 16.4+, Chrome 113+ |
| C | minimum supported | 2020-2022 budget phones, 3-4 GB RAM: Galaxy A14, Moto G 2022 | Chrome 113+ |

Tier B is the acceptance tier: the product is not "on-device ready" until a
tier-B phone passes every budget offline. Tier C may need the smaller
(quantized) bundle; see [E12 in TRAINING_PROGRAM.md](TRAINING_PROGRAM.md).

Browsers only partly reveal the hardware: Chromium exposes
`navigator.deviceMemory` (capped at 8) and core count, Safari exposes
neither. The device check therefore guesses a tier only where possible and the
tester confirms or overrides it.

## Budgets

| Budget | A | B | C | How it is measured | Desktop Chromium (not a phone) |
| --- | --- | --- | --- | --- | --- |
| One-time download (bundle + WebAssembly runtime) | 50 MB | 50 MB | 50 MB | bundle manifest bytes + cached `/litert/` runtime bytes | 44.8 MB (37.2 + 9.8) |
| Storage used by the origin | 120 MB | 120 MB | 120 MB | `navigator.storage.estimate()` | 44.8 MB (fresh profile, no saved results) |
| Model load from cache, cold (read, SHA-256 verify, compile) | 2 s | 4 s | 8 s | wall time of engine creation | 0.95-2.03 s |
| Analysis time / recording duration (audio + video, 2 frames per second) | 0.5 | 1.0 | 2.0 | median wall time over the runs / clip duration | 0.35 (idle VM) to 0.50 (VM busy with a training job) |
| Detector inference per frame (median) | 80 ms | 160 ms | 400 ms | 5 timed runs after a warm-up | 38 ms (4 threads) / 88-99 ms (1 thread) |
| Page memory after analysis | 400 MB | 300 MB | 250 MB | `performance.measureUserAgentSpecificMemory()` (Chromium, cross-origin isolated; resolves at the next garbage collection) | not measured (did not resolve within 30 s in headless Chromium) |
| Sustained slowdown (median of the last 3 / first 3 of 10 consecutive analyses) | 1.25 | 1.25 | 1.25 | sustained mode of the device check | not measured (VM has no thermal behaviour worth measuring) |
| Battery used per minute of analysed recording | 1.0 % | 1.5 % | 2.0 % | Battery Status API, unplugged, sustained mode (Chromium on Android only; 1 % resolution) | not applicable |

Why these numbers (assumptions):

* **Download 50 MB**: one-time, prompted, Wi-Fi recommended in the UI. The
  default bundle is dominated by two float32 models (detector 13.8 MB,
  backbone 18.6 MB). The int8 variant evaluated in E12 is 14.8 MB in total and
  roughly halves CPU latency; it is opt-in because individual verifier
  decisions can differ strongly from float32 (see E12 in
  [ALGORITHM_REPORT.md](ALGORITHM_REPORT.md)).
* **Real-time factor 1.0 on tier B**: a 60 s recording finishes in about a
  minute, which keeps the "record, then see the result" flow usable. Analysis
  processes 2 frames per second, so the per-frame work (decode + detector +
  verifier + posture crops) must stay under 500 ms on tier B.
* **Memory 300 MB on tier B, 250 MB on tier C**: mobile browsers terminate
  background tabs and sometimes foreground tabs well before physical RAM is
  exhausted. The engine holds three models (≈ 37 MB of weights) plus
  activations and decoded frames.
* **Sustained slowdown 1.25**: a phone that throttles more than 25 % within
  two minutes of continuous analysis will make long recordings feel broken.
* **Thermal**: browsers expose no temperature. Sustained slowdown is the
  measurable proxy; testers also note whether the phone became noticeably warm.

## Where the time goes (desktop measurement, 8 s clip)

From `results/ondevice/engine_parity_chromium.json` (per-stage timings of the
TypeScript engine): video decode + seek + detect + track ≈ 2.6 s, of which
detector inference ≈ 0.9 s (17 frames), verifier ≈ 0.35 s, posture ≈ 0.4 s,
quality and camera-motion ≈ 0.3 s; audio decode + YAMNet ≈ 0.1 s; fusion and
behaviour < 10 ms. Seeking the video element frame by frame is the largest
single cost and depends on the browser's decoder; a high-bitrate 6 Mbit/s clip
took 7.4 s for 8 s of video on the same VM.

Levers if a tier misses its budget (in order of preference): the int8
bundle (E12, evaluated, opt-in), fewer verifier/posture crops per frame (configurable), sampling 1 frame
per second for long clips (configurable, lowers temporal resolution and is
reported in the result), WebGPU acceleration (available in LiteRT.js, not yet
validated for these models).

## Numerical equivalence (part of every budget run)

The device check also verifies that models produce the reference outputs on
the phone (argmax equal, max absolute difference on the first values) and that
the end-to-end analysis of the bundled reference clip produces the expected
tracks, behaviour segments and decision. Functional correctness ("Passed") is
reported separately from budgets: a phone can be correct but too slow, and the
report says which.
