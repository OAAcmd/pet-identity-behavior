# Configuration

One YAML file configures the server engine, the API, and what goes into the
phone bundle: `configs/cpu.yaml` (development, the default), or the file in
`$PETPERCEPTION_CONFIG` (the container uses `configs/production.yaml`, copied
from `configs/production.example.yaml`). Values of the form
`${ENV_VAR:-default}` are expanded from the environment. The schema with all
defaults is `petperception.config.Settings`; `petperception doctor` prints
the effective state. Unknown keys are rejected.

## Environment variables used by the shipped configurations

| Variable | Effect | Default (production) |
| --- | --- | --- |
| `PETPERCEPTION_CONFIG` | configuration file | `/app/configs/production.yaml` |
| `PETPERCEPTION_HOME` | repository root inside the image | `/app` |
| `PETPERCEPTION_RELEASE_PROFILE` | `research` / `personal_noncommercial` / `commercial` | `personal_noncommercial` |
| `PETPERCEPTION_DETECTOR` | server detector model id | `efficientdet_lite0_mediapipe_v1` |
| `PETPERCEPTION_MODELS_ROOT` | fetched model files | `/models` |
| `PETPERCEPTION_ONDEVICE_DIR` | published on-device bundle (`/ondevice/`) | `/bundle/current` |
| `PETPERCEPTION_DB`, `PETPERCEPTION_MEDIA_ROOT` | SQLite database, retained media | under `/var/lib/petperception` |
| `PETPERCEPTION_WORKERS`, `PETPERCEPTION_NUM_THREADS` | server-analysis workers and threads per model | 2, 2 |
| `PETPERCEPTION_NEW_PASSWORD` | password for `user create / reset-password` without a prompt | - |

## Sections

| Section | What it controls | Notes |
| --- | --- | --- |
| `release_profile` | which assets may be used (`manifests/resources.json`, approvals file) | enforced by the engine, `release-check` and `export-ondevice` |
| `paths` | data, models, results, artifacts, manifests, job scratch | relative paths resolve from the repository root |
| `limits` | file size, duration, sampled frames, channels, decode timeouts | apply to server analysis; the phone engine reads `max_duration_s`, `max_video_frames`, `max_file_bytes` from the bundle |
| `audio` | YAMNet backend, thresholds file (E1), context-tag threshold, `subtype_validation_file` (E21: which vocalization types may be shown or trigger interpretations), `context_thresholds_file` (E21b: per-tag context thresholds) | thresholds file is CC BY-NC-derived: not used in commercial builds; without the validation file no subtype counts as validated |
| `vision` | detector id, 2 fps sampling, score threshold 0.35, NMS, target classes, tracker (Kalman, chosen in E8), quality thresholds, camera-motion compensation | `context_object_classes` (e.g. `[person]`) adds the generic "person visible" evidence on the server; COCO `bowl` is not validated for pet bowls |
| `vision.verifier` | second-stage false-positive rejection (E7) | parameters file is chosen per (detector, classifier) pair: `verifier_<detector>[__<classifier>].json` |
| `pose` | keypoints: `dog_keypoints_v1` (E9, dogs) or `catdog_keypoints_awa_v1` with `head_file: artifacts/keypoints/catdog_keypoints_head_awa_v1.npz` (E24, cats and dogs) | research profile only (data terms); `provider: none` elsewhere; never in the phone bundle |
| `posture` | coarse posture (E10) | `backbone` must match the features the parameters were validated on |
| `interaction` | person-pet interaction candidates (E13): model file, minimum person score, minimum consecutive frames | persons are detected but never stored in results |
| `object_relation` | on-furniture candidates (E15): model file, minimum furniture detection score (0.35, the value E15 was evaluated with), minimum consecutive frames | furniture is detected but never stored in results |
| `intent` | trained behaviour/intent stage (`src/petperception/intent/`, E26-E28): `enabled` (default false), `variant` (`phone_now`: YAMNet class scores, no keypoints, what the phone computes; `full`: YAMNet embedding and the research cat/dog keypoint head, needs `pose.provider: catdog_keypoints_awa_v1`) | research profile only (situation heads trained on YouTube clips); adds per frame a whole-frame embedding, detection of all COCO object classes and a crop of the largest cat/dog; `export-ondevice` puts the heads in the bundle only when enabled |
| `behavior` | operational definitions of behavior segments (speeds, durations, pacing, zones, vocal bouts, startle) | a-priori thresholds; validated only on semi-synthetic video (E8) |
| `fusion` | experimental hypothesis scoring and abstention | unfitted; not validated on paired data |
| `serve` | host/port, static PWA directory, on-device bundle directory, trusted proxy hops, accepted extensions | behind a proxy set `trusted_proxy_hops: 1` |
| `ondevice` | model builds exported into the phone bundle: `detector`, `backbone`, `posture_model` | default float32 builds; the int8 builds are an evaluated opt-in (E12) |
| `app.auth` | registration mode, session lifetime, cookie security, password length, login rate limit | `cookie_secure: true` requires HTTPS |
| `app.jobs` | server-analysis workers, leases, retries, timeouts, per-user and global queue limits | |
| `app.storage` | database and media paths, upload and chunk limits, quotas, retention, free-disk floor, cleanup interval | |
| `runtime` | threads per model, accelerator (`cpu`) | |
| `vlm` | disabled; no remote language model is used for explanations | |

## Changing models safely

* A different detector or backbone build needs its own verifier parameters
  (E12 showed that float-trained verifier parameters on int8 features fail
  the recall rule). Export them with the E7 procedure for that pair before
  switching.
* After any model or threshold change: re-run the affected experiment
  protocol, re-export the bundle, re-run engine parity and the E2E suite
  ([ON_DEVICE.md](ON_DEVICE.md#evidence-and-how-to-reproduce-it)).
