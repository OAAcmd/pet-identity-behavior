# Real-phone offline acceptance test

**Status: PENDING.** This project has no physical phones. Everything below has
been exercised only in headless desktop Chromium with phone emulation
(viewport, user agent, touch), which is not evidence of phone execution. The
product cannot be declared on-device ready until this protocol passes on a
tier-B phone of each platform (iOS Safari, Android Chrome); see
[DEVICE_TIERS_AND_BUDGETS.md](DEVICE_TIERS_AND_BUDGETS.md).

## What is tested

1. The model bundle installs over the network, is verified (SHA-256) and the
   engine starts on the phone.
2. With **no network at all** (airplane mode), the app opens, analyses a
   reference clip, a fresh camera recording and a file from the phone's
   gallery, and shows and stores the results locally.
3. Numerical agreement of each model with the Python reference on the phone.
4. Resource budgets for the phone's tier (load time, real-time factor,
   detector latency, memory, sustained slowdown, battery).
5. Update and recovery: an interrupted update leaves the previous version
   working.

## Equipment

* At least one phone per platform in tier B (reference tier); tiers A and C
  are strongly recommended. Record the exact model, OS version and browser
  version.
* The server deployed over HTTPS ([DEPLOYMENT.md](DEPLOYMENT.md)) with a
  current on-device bundle (`petperception export-ondevice`). HTTPS is
  required: Cache Storage, service workers, camera and cross-origin isolation
  only work in a secure context.
* A test account (`petperception user create --username tester`).
* A cat or dog, or (if none is available) a second screen playing a pet video
  for the camera test. Note which was used.
* For diagnosis: Android: Chrome remote debugging over USB (`chrome://inspect`);
  iOS: Safari Web Inspector (Settings > Safari > Advanced > Web Inspector).

## Procedure

Record every step's outcome in the report template at the end.

1. **Install.** On Wi-Fi, open the app URL, sign in. Optionally use "Add to
   Home Screen". Settings > On-device analysis > *Install models*. Expect
   "Installed: model bundle <version>" and a runtime line ("Multi-threaded" or
   "Single-threaded"). Note the download time.
2. **Go fully offline.** Enable airplane mode (Wi-Fi and mobile data off).
   Close the browser completely (swipe it away), then reopen the app (from the
   home-screen icon if installed). Expect the app to open with an
   "offline" notice and the cached account.
3. **Device check, quick.** Open the device check (Settings > *Run device check*,
   or `#/device-check`). Enter the phone model, confirm the tier, tap *Run device
   check*. Expect **Passed**. Tap *Download report* (or *Copy report*).
4. **Device check, sustained.** Unplug the phone, battery above 50 %, screen
   on. Tick *Sustained check*, run again (1-3 minutes). Download the report.
   Note whether the phone became noticeably warm.
5. **Fresh recording.** *Record* tab: record 15-30 s of a pet (allow camera
   and microphone). Tap *Analyze on this device*. Expect a result page that
   says "Analysed on this device"; check tracks, timeline and the decision.
6. **Gallery file.** *Upload file* tab: pick a video recorded earlier with the
   phone's own camera app (iPhone: HEVC `.mov` by default; Android: usually
   H.264 or HEVC `.mp4`). Tap *Analyze on this device*. Record whether it
   decoded. A decode failure must produce a clear message, not a hang.
7. **History.** *History* lists both analyses under "Analysed on this
   device"; open one; it renders offline.
8. **Back online.** Disable airplane mode. Optionally *Send report to
   server* on the device-check page (sends the JSON report only, never media).
9. **Update recovery** (needs a newer bundle on the server): Settings >
   *Update models*; switch on airplane mode during the download. Expect an
   error message and the **previous** version still installed and working
   (repeat step 5 offline). Reconnect and complete the update.

## Pass criteria

| Area | Criterion |
| --- | --- |
| Functional | Steps 1-3, 5, 7 and 9 succeed; device check reports `passed: true` (integrity, numerical agreement, expected end-to-end result on the reference clip). Step 6 either succeeds or fails with a clear unsupported-format message. |
| Offline | Steps 3-7 run in airplane mode. |
| Budgets | Every tier budget in the report is `pass`; `not_measured` items are listed with the reason (for example no Battery API on iOS). A budget failure does not hide functional success: both are reported. |
| Privacy | No media leaves the phone in the default mode (airplane mode makes this physically certain for steps 3-7). |

## Report template

Store each phone's results as `results/ondevice/devices/<YYYY-MM-DD>_<model>/`
with the downloaded quick and sustained device-check JSON files and a
`notes.md`:

```
Phone model / OS version / browser version:
Tier (A/B/C):
Tester / date:
Step 1 install: ok | failed (message) ; download time:
Step 2 offline open: ok | failed
Step 3 device check quick: Passed | Not passed ; report file:
Step 4 sustained: report file ; warm: yes/no
Step 5 fresh recording: ok | failed ; pet or screen:
Step 6 gallery file: container/codec ; ok | failed (message)
Step 7 history offline: ok | failed
Step 9 update recovery: ok | failed | not run
Console errors (if any):
```

## Known limitations going in (to confirm or refute on phones)

* LiteRT.js loads its multi-threaded build only with cross-origin isolation
  **and** relaxed WebAssembly SIMD; otherwise the single-threaded build is
  used (fallback implemented in `web/src/engine/runtime.ts`). Browsers without
  relaxed SIMD are therefore expected to be slower.
* Page memory is measurable only in Chromium; iOS Safari exposes neither
  memory nor battery, so those budgets read `not_measured` there.
* MP4/H.264, HEVC and MOV decoding have not been tested in any browser here
  (the automated Chromium build has no H.264 decoder).
* The bundled reference clip is a synthetic composition of public photos
  (see `device_check/ATTRIBUTION.json` in the bundle), so step 3 verifies
  correctness of the pipeline, not behavioral accuracy.
