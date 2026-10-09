# On-device analysis: architecture, packaging and evidence

The core analysis (media decoding, audio events, detection, false-positive
verification, tracking, posture, behavior segments, fusion, explanations)
runs **in the phone's browser**. The server publishes a versioned model bundle,
hosts accounts and receives only what the user explicitly sends. After the
bundle is installed, analysis needs no network. What still needs a server is
listed in [ON_DEVICE_AUDIT.md](ON_DEVICE_AUDIT.md#capabilities-that-still-require-a-remote-service).

## Runtime

* **LiteRT.js** (`@litertjs/core` 2.5.3, Apache-2.0) executes the *same*
  `.tflite` files the Python reference engine uses (`ai_edge_litert`), on the
  CPU through WebAssembly + XNNPACK. WebGPU is available in the library but not
  validated for these models; it is not used.
* Multi-threading requires a cross-origin-isolated page (the server sends
  `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`) **and** relaxed WebAssembly
  SIMD. Otherwise the single-threaded build is loaded
  (`web/src/engine/runtime.ts`); the result's `engine.runtime` field and the
  Settings page show which one ran.
* The content security policy allows `'wasm-unsafe-eval'` (WebAssembly
  compilation) and `worker-src 'self' blob:` (runtime threads), nothing more.
* Why not alternatives: MediaPipe Tasks for Web was considered but not
  adopted: it runs fixed task graphs, while the product needs raw tensor
  access (intermediate backbone features for the verifier and posture head). TensorFlow.js would require converting every model and re-validating
  numerics. A native app is not required by any measurement so far; it stays
  the fallback if real phones miss the tier-B budgets
  ([DEVICE_TIERS_AND_BUDGETS.md](DEVICE_TIERS_AND_BUDGETS.md)).

### Operator compatibility (measured)

| Model | Browser result |
| --- | --- |
| YAMNet (MediaPipe build, float32) | runs; output bit-identical to Python on identical input |
| EfficientDet-Lite0 (MediaPipe build, float32, raw anchor outputs) | runs; max abs diff 3e-8 |
| EfficientNet-Lite0 backbone with 4 exposed intermediate tensors | runs; bit-identical |
| EfficientDet-Lite0 (MediaPipe int8 build) | runs; bit-exact |
| EfficientDet-Lite0 (MediaPipe float16 build) | runs; max abs diff 3e-8 |
| EfficientNet-Lite0 int8 backbone (int8 feature outputs exposed through appended DEQUANTIZE operators) | runs; bit-exact. Without the DEQUANTIZE ops LiteRT.js 2.5.3 refuses the model ("Element type INT8 is not supported") |
| EfficientDet-Lite0 (TF Hub build, uint8, in-graph `TFLite_Detection_PostProcess`) | **crashes** LiteRT.js 2.5.3 ("memory access out of bounds"); not shipped. Fix path: strip the post-processing op and decode in TypeScript |

Evidence: `results/ondevice/runtime_parity_*.json` (headless Chromium,
single- and multi-threaded).

### Intermediate tensors

The verifier and posture models need features from inside EfficientNet-Lite0
(three feature maps and the 7x7x1280 head). `petperception.ondevice.tflite_patch`
appends those tensors to the flatbuffer's subgraph outputs (no weights or
operators change); bundles refuse models with SignatureDefs, where this patch
would be ambiguous. Python and browser read the same patched file.

## Bundle

`petperception export-ondevice [--release-profile P]` writes
`var/ondevice/bundle/`, served by the backend at `/ondevice/`:

```
manifest.json              schema petperception.ondevice_bundle/1
engine_assets.json         decoding parameters, anchors, thresholds, verifier/posture
                           weights (base64 float32), behavior/fusion settings, limits
models/yamnet.tflite
models/efficientdet_lite0.tflite
models/efficientnet_lite0_features_v1.tflite
device_check/              reference clip (WebM + MP4), expected result, reference
                           model outputs, attribution
```

* `manifest.json` lists every file with SHA-256 and size, the
  `bundle_version` (hash of the files' SHA-256 digests), the asset format
  (`engine_version`, see below), the release profile and notes about
  excluded components (for example "dog_keypoints_head_v1 is not permitted
  under the 'personal_noncommercial' profile").
* Release gating happens at export: an asset not allowed in the chosen
  profile (`manifests/resources.json` + approvals file) is not written into the
  bundle, and the TypeScript engine reports that component as unavailable.
* Current size: 37.2 MB (personal_noncommercial profile, default float32
  builds) + 9.8 MB runtime. The opt-in int8 bundle (E12) is 14.8 MB.
* Quantized builds: `engine_assets.json` carries each model's input and
  output quantization (`input_quant`, `output_quant`); the TypeScript engine
  encodes inputs with NumPy-identical float32 arithmetic and dequantizes
  outputs, so both runtimes see identical bytes (asset format 0.2).

## Install, integrity, versioning and update recovery

Implemented in `web/src/engine/modelstore.ts` and `web/src/lib/ondevice.tsx`:

1. The app fetches `/ondevice/manifest.json` (never cached). A bundle whose
   asset format major.minor differs from what the app supports
   (`SUPPORTED_BUNDLE_ENGINE`) is refused with a message to reload the app;
   nothing is downloaded.
2. Each file is downloaded, its SHA-256 computed with Web Crypto and compared
   with the manifest; mismatches abort the install. Files are stored in Cache
   Storage under content-addressed keys (`/__ondevice__/files/<sha256>/<path>`),
   so files shared between versions are not downloaded twice.
3. The engine is started once **from the new files** before switching: this
   proves the models compile on this device and makes the service worker cache
   the WebAssembly runtime for offline use.
4. Only then is the active-manifest record replaced (single Cache Storage
   write) and unreferenced files of older versions deleted. Any failure in
   steps 1-3 leaves the previous version active and working
   (E2E test "a corrupt model update is rejected ...").
5. Every load re-verifies each file's SHA-256; a corrupted file is deleted
   and the user is asked to reinstall.
6. `navigator.storage.persist()` is requested so the browser does not evict
   the models under storage pressure (best effort; browsers may refuse).

Not implemented: signed bundles (integrity relies on HTTPS plus the
manifest's checksums, so a compromised server could ship a different
bundle); background or delta updates.

## Offline behavior

* Service worker (`web/public/sw.js`): app shell network-first with cached
  fallback; hashed assets and `/litert/` runtime cache-first. API responses
  and media are never cached by the service worker.
* The signed-in user's profile, pets and places are cached (no tokens) so the
  shell renders offline; edits need the network.
* On-device results and feedback are stored in IndexedDB on the device
  (`web/src/lib/localstore.ts`). They are not synced; deleting browser data
  deletes them. Opt-in result sync (JSON only, no media) is planned and not
  implemented.

## Recording-to-result workflow on the phone

Capture (camera/microphone via `MediaRecorder`, or a file) → *Analyze on this
device* → `LocalEngine.analyze()` (`web/src/engine/engine.ts`): probe and
hash → decode audio at 16 kHz (`OfflineAudioContext`) → YAMNet → sound
events → seek video at 2 fps (video element + canvas) → detector → verifier
→ tracker → posture → quality and camera motion → behavior segments → fusion
and explanation templates → result JSON (same schema as the server,
`docs/analysis_result.schema.json`) → IndexedDB → result page. The user can
cancel at any stage. The default path makes no upload request (asserted by
an E2E test). Server analysis exists only as an explicit setting.

## Evidence and how to reproduce it

| Evidence | Command | Result file |
| --- | --- | --- |
| Model-level parity (identical tensors) | `petperception export-parity`, then `cd web && npx playwright test -c playwright.runtime.config.ts tests/runtime/runtime.spec.ts` | `results/ondevice/runtime_parity_*.json` |
| Engine-level parity (identical media files) | `petperception engine-parity reference`, `npx playwright test -c playwright.runtime.config.ts tests/runtime/engine.spec.ts`, `petperception engine-parity compare --write results/ondevice/engine_parity_chromium.json` | `results/ondevice/engine_parity_chromium.json` |
| Product E2E incl. offline, update recovery, no-upload | `cd web && npm run build && npx playwright test` | Playwright report |
| Device check (automated, not a phone) | E2E test "device check passes ..." | `results/ondevice/device_check_desktop_chromium.json` |
| Real phones | [DEVICE_TEST_PROTOCOL.md](DEVICE_TEST_PROTOCOL.md) | **pending** |

Engine-parity summary (34 cases, default float32 bundle): decision 34/34;
tracks F1 0.96; sounds 1.00; behavior segments 0.97 (person-interaction
segments identical on the 4 person + dog clips); hypotheses 1.00; posture
labels identical whenever both engines report one, with about one frame in
five flipping across the report gate. With the int8 bundle on both sides: tracks
0.84, segments 0.875, hypotheses 0.99, decisions 30/30
(`results/ondevice/engine_parity_chromium_int8.json`). Media are decoded differently by design (FFmpeg vs browser), so
pixel-level differences propagate into small score differences.

## Differences between the phone engine and the server engine

| Aspect | Server (Python) | Phone (TypeScript) |
| --- | --- | --- |
| Decoding | FFmpeg, PTS-exact frames, native audio rate | browser decoders; video element seeking; audio resampled to 16 kHz |
| Dropped-frame and audio-bandwidth checks | yes | no (not observable in the browser) |
| Camera-motion compensation | ORB features + RANSAC | block matching (different method, validated only through engine parity) |
| Dog keypoints | research profile only | not shipped |
| Person context (`person_visible`) | optional, off by default | not used |
