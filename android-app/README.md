# Pet Observer - Android test app

A thin native shell (Capacitor 7) around the same web app the server serves, so the journey, the business logic and
the stored data are shared with the browser version. What the shell adds is Android behaviour: runtime camera /
microphone permissions, the system camera for **Record video**, the system picker for **Choose a video**, the back
button walking the app's history, and a start screen that asks for the server address and reports an unreachable
server instead of a blank page.

Installable debug build: `dist/pet-observer-0.1.0-debug.apk` (SHA-256 in `dist/SHA256SUMS`; debug-signed, package
`org.petperception.observer`, min Android 6, target Android 15).

## Install and connect

1. Start a server with the analysis pipeline (repository root): `uv run petperception serve --port 8765 --workers 1`.
2. Emulator: `adb install dist/pet-observer-0.1.0-debug.apk`, open Pet Observer, keep the suggested address
   `http://10.0.2.2:8765`, Connect. Phone: allow installing from this source, install the APK, and enter the server's
   **HTTPS** address (the server refuses non-loopback plain HTTP; docs/DEPLOYMENT.md). Later: Settings > Server >
   Change server.
3. First account on a fresh server becomes the administrator; then Ask about my pet's behaviour.

## Build

```bash
cd android-app && npm install
export ANDROID_HOME=<sdk with platforms;android-35, build-tools;35.0.0>   # JDK 21
echo "sdk.dir=$ANDROID_HOME" > android/local.properties
npx cap sync android && cd android && ./gradlew assembleDebug lintDebug
```

## What has been checked, and what has not

| Check | Status |
| --- | --- |
| Gradle debug build, APK badging (permissions CAMERA, RECORD_AUDIO, INTERNET), signature | done (this environment) |
| Android lint | done: 0 errors, warnings are dependency-version and template notes |
| Start screen logic (unreachable server message, connect and open the web app) | done in mobile Chromium (`web/tests/journey/launcher.spec.ts`) |
| The web journey the shell loads (ask, record / choose, progress, answer, correction, history) | done in mobile Chromium against the real server (`web/tests/journey/journey.spec.ts`) |
| Emulator run | **not done**: this environment has no hardware virtualisation (/dev/kvm) |
| Physical phone: permission prompts, system camera, interruptions, resume after the app is killed | **not done** |

Behaviour designed for interruptions: the system camera app owns the recording, so an interrupted recording returns no
file and the screen stays on "Add a recording"; the question draft is kept on the phone; an upload that stops (app
closed, connection lost) stays in the phone's storage and resumes from Home ("Unfinished on this device").
