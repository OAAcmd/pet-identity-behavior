# Real-phone acceptance test

**Status: PENDING - no physical phone has been tested.** The browser
inference implementation has passed automated tests in desktop headless
Chromium only (with phone-sized viewports and emulated touch, which is not
phone evidence). Android Chrome, iPhone Safari and every other phone browser
are untested. Device readiness is accepted only when the criteria below pass
on a tier-B phone of each platform ([DEVICE_TIERS_AND_BUDGETS.md](DEVICE_TIERS_AND_BUDGETS.md)).

Two test builds exist because no HTTPS server of ours is running:

| Track | Build | Where | Status |
| --- | --- | --- | --- |
| **A. Phone test** (engine, runtime, models, capture, offline analysis, reload) | static page that runs the app's TypeScript engine, LiteRT.js runtime and model bundle (`web/src/devicetest`, assembled by `petperception export-device-test`) | **published**: <https://claude.ai/artifact/NRG1a19X1ZkPbgdigJGaf2> (private claude.ai page; HTTPS) | ready to run; results collected automatically |
| **B. Installed app** (PWA: service worker, offline reopen, live camera page, history, update recovery) | the full PWA + `/ondevice/` bundle | needs an HTTPS origin we control (see [Infrastructure needed for track B](#infrastructure-needed-for-track-b)) | pending infrastructure |

## Coverage of the acceptance requirements

| Requirement | Track A (available now) | Track B (pending) |
| --- | --- | --- |
| Model download and integrity | yes: every file (and every Base64 part) SHA-256 checked | yes |
| Runtime backend recorded | yes: LiteRT.js build (threads / single-thread), accelerator | yes |
| Model versions recorded | yes: bundle version, model ids and SHA-256 in the report | yes |
| Device model, OS, browser | tester enters them; user agent, OS and browser guesses, in-app web view detection recorded | same |
| Camera input | yes, through the phone's own camera app (file input with `capture`); live camera preview is not available on this host | yes, live camera page (`getUserMedia` + `MediaRecorder`) |
| Microphone input | yes, the recorded video's sound track; separate audio recording via a saved audio file | yes, sound-only recording |
| Analysis and results | yes: reference clip 3x (must match the expected result), the tester's own recording (summary shown and reported, never uploaded) | yes, plus results stored in history |
| Offline after download | yes: analysis in airplane mode, with a count of network requests made during the analysis (must be 0) | yes, including **reopening the app** with no network |
| Memory | JS heap peak where the browser reports it (Chromium only; WebAssembly memory not included, so a lower bound); Safari: not measurable | same |
| Sustained latency | yes: 10 consecutive analyses, slowdown last-3 / first-3 | same |
| Battery | Battery Status API where available (Android Chrome), 1 % resolution | same |
| Interruptions | yes: app switch during the sustained run; page visibility, freeze, offline/online events logged with the runs | same, plus incoming call / lock screen in the installed app |
| Recovery | yes: reload the page, start from models saved on the phone without downloading (cold start time measured) | yes, plus interrupted model update keeps the previous version |

Track A does not test: service-worker caching of the app shell, reopening the
app offline from the home screen, the live camera and microphone page,
IndexedDB history, opt-in result backup, update recovery, and the
multi-threaded runtime (the page is not cross-origin isolated, so LiteRT.js
uses its single-threaded build, so latency from track A is a pessimistic
bound for an isolated install).

**What version 4 of the page tests (2026-10-10).** Bundle `2dd9e6b541aeffde`
(profile `personal_noncommercial`, 26.6 MB of model data):

* YAMNet;
* the TF Hub EfficientDet-Lite0 detector with raw-output decoding;
* EfficientNet-Lite0 features with the verifier, posture, interaction and
  object-relation parameters.

Every component is cleared for that profile (`docs/ASSET_TERMS.md`).
Versions 1-3 used the MediaPipe detector, whose weights state no licence.

Desktop Chromium run of version 4 (single-threaded WebAssembly; a desktop
figure, not a phone figure): download 26.6 MB, verified in 1.4 s; engine
start 750 ms; detector 74 ms per frame; a 11.2 s reference clip analysed in
5.6 s (median of 3); reload from saved models 562 ms.

## Track A: phone test (available now)

**Who can run it.** The page is private to the owner. To let others test,
share it from the page's Share menu. Only people who can **write** to its
database send reports automatically: the owner, members of the owner's
claude.ai Team or Enterprise organization with Contributor access, or people
invited by email as **Editor** (when the page is not also shared by a public
link). Anyone else can run every step and use **Copy report** or **Save
report** and send the JSON by any channel.

**Before you start.** Charge above 50 %, Wi-Fi on, about 60 MB of
storage free. Open the link in **Safari (iPhone)** or **Chrome (Android)**,
not in an app's built-in browser (the page warns when it detects one). If
the claude.ai app opens instead, use "Open in browser".

**Steps** (the page numbers them the same way):

1. Fill in *Phone model* (for example "Pixel 7a"), check *Operating system*
   and *Browser* (pre-filled from the browser), choose the tier.
2. *Download* (about 50 MB on the wire before compression, 26.6 MB of
   model data). Every file is verified.
3. *Start* the engine, then *Check* the models (outputs compared with the
   reference computer on fixed inputs).
4. *Analyse* the reference clip three times (each must match the expected
   result).
5. Unplug the charger. *Start ten analyses*; during the run, switch to
   another app for about 10 seconds once, then come back.
6. Turn on airplane mode (Wi-Fi off too) and tap *Analyse offline*. Turn the
   network back on afterwards.
7. Choose what you will record, tap *Record video*, record 10-30 s of your
   cat or dog (or anything) with the camera app, confirm. The clip is
   analysed on the phone; note whether the summary matches what you
   recorded. Optionally also *Choose a saved video* recorded earlier with the
   camera app (iPhone: HEVC `.mov`; Android: H.264/HEVC `.mp4`).
8. Reload the page (pull down), then *Start from saved models*.
9. Add notes (heat, reloads, anything odd) and tap *Send report*.

**Pass criteria (track A).** Steps 2-4, 6 and 8 report *passed*; step 6
reports 0 network requests; the tester's recording in step 7 decodes and
produces a result (a decode failure must give a clear message, not a hang);
budgets are reported per tier (a budget failure does not hide functional
success). A failure is recorded with its message and stack in the report.

**Collection.** Reports are stored in the page's database under
`reports/<tester id>/runs/<run id>`. Each tester writes only their own;
only the owner reads all. An index document `reports/<tester id>` lists the
tester's run ids and the latest phone, browser and pass state, so a listing
of `reports` finds every tester (added in version 4). The project copies each report into
`results/ondevice/devices/<date>_<phone model>/report.json` and updates
[DEVICE_TIERS_AND_BUDGETS.md](DEVICE_TIERS_AND_BUDGETS.md). The report
contains device, browser and timing data and analysis summaries; it never
contains video, audio or images.

**Rebuild and republish** (after a model or engine change):

```bash
uv run petperception export-ondevice --release-profile personal_noncommercial \
    --detector efficientdet_lite0_tfhub_rawdecode_v1 --out var/device_test/bundle
cd web && npm run build:devicetest && cd ..
uv run petperception export-device-test --bundle var/device_test/bundle   # writes var/device_test/site (not committed: contains model weights)
cd web && npx playwright test -c playwright.devicetest.config.ts   # desktop check of every step
```

Then publish `var/device_test/site/index.html` with the files listed in
`var/device_test/site_files.json` (the claude.ai host serves only web media
types, so model files are published as Base64 text parts and checked after
decoding; the host's per-publish limit requires two batches).

## Track B: installed app

### Infrastructure needed for track B

The PWA must be served from an HTTPS origin under our control (service
workers, Cache Storage, camera access and cross-origin isolation require a
secure context; cross-origin isolation additionally needs the
`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy:
require-corp` headers for the multi-threaded runtime). Either of:

1. **The existing compose deployment** ([DEPLOYMENT.md](DEPLOYMENT.md)) on
   any Linux host with Docker: 2 vCPU, 4 GB RAM, 20 GB disk (estimate;
   measured here: server models 123 MB, on-device bundle 27.9 MB with the
   licence-stated detector), a DNS name
   pointing to it and ports 80/443 open for automatic TLS (Caddy or any
   ACME-capable proxy). Gives the full product including accounts, result
   backup and opt-in server analysis.
2. **A static HTTPS host** for an account-free build of the PWA (not
   implemented yet: the PWA currently requires the server for sign-in, pets
   and places). GitHub Pages would need the repository owner to enable Pages
   (Settings > Pages > Source: GitHub Actions); it cannot set COOP/COEP
   headers, so the runtime would be single-threaded unless the service
   worker adds them.

### Procedure (track B)

1. **Install.** On Wi-Fi, open the app URL, sign in. Optionally "Add to Home
   Screen". Settings > On-device analysis > *Install models*. Expect
   "Installed: model bundle <version>" and a runtime line ("Multi-threaded" or
   "Single-threaded").
2. **Go fully offline.** Airplane mode (Wi-Fi and mobile data off). Close the
   browser completely, reopen the app (from the home-screen icon if
   installed). Expect the app to open with an "offline" notice.
3. **Device check** (Settings > *Run device check*): quick, then sustained
   (unplugged). Download the reports.
4. **Fresh recording** (*Record* tab, camera and microphone allowed): 15-30 s
   of a pet, *Analyze on this device*.
5. **Gallery file** (*Upload file* tab): a video from the phone's camera app.
6. **History** lists both analyses and opens them offline.
7. **Interruption**: start a long analysis, lock the screen for 10 s, unlock;
   record whether it completed, resumed or failed with a clear message.
8. **Update recovery** (needs a newer bundle): *Update models*, enable
   airplane mode during the download; expect the previous version still
   working.

### Pass criteria (track B)

| Area | Criterion |
| --- | --- |
| Functional | Steps 1-4, 6 and 8 succeed; device check `passed`; step 5 succeeds or fails with a clear unsupported-format message. |
| Offline | Steps 2-7 run in airplane mode. |
| Budgets | Every tier budget is `pass`, or listed with the reason it was not measured. |
| Privacy | No media leaves the phone in the default mode. |

## Known limitations going in (to confirm or refute on phones)

* Multi-threaded LiteRT.js needs cross-origin isolation **and** relaxed
  WebAssembly SIMD; otherwise the single-threaded build is used.
* Whether the claude.ai page frame allows WebAssembly compilation on phones
  is unverified; step 3 of track A reports it (an engine-start failure names
  the reason).
* Page memory is measurable only in Chromium; iOS Safari exposes neither
  memory nor battery.
* MP4/H.264, HEVC and MOV decoding have not been tested in any browser here
  (the automated Chromium build has no H.264 decoder).
* The reference clip is a semi-synthetic composition of public photos
  (`device_check/ATTRIBUTION.json`), so the reference steps verify the
  pipeline, not behavioral accuracy.
