# Operations runbook

For the compose deployment in [DEPLOYMENT.md](DEPLOYMENT.md). Commands assume
`dc="docker compose -f deploy/compose.yaml --env-file deploy/.env"`; the
container entrypoint is `python -m petperception.cli`, so `$dc run --rm app
<command>` runs a CLI command.

## Health and readiness

| Endpoint | Meaning | Used by |
| --- | --- | --- |
| `GET /api/v1/health` | process is up | container health check |
| `GET /api/v1/ready` | 200 only when all checks pass: database, server models, disk space, worker heartbeat, queue length, **on-device bundle published** | monitoring, before switching traffic |
| `GET /api/v1/capabilities` | engine components and their state (ok / unavailable with reason) | troubleshooting |

A 503 from `/ready` names the failing check in `checks`. Monitor `/ready`
from outside (every minute) and alert after three consecutive failures.

Logs are JSON lines on stdout (`$dc logs app`), with a request ID per request
(`X-Request-ID` header; error responses quote it). User names, file names and
passwords are not logged (tested).

## Common incidents

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Phones show "On-device models are not published on this server" | bundle volume empty or `PETPERCEPTION_ONDEVICE_DIR` wrong; `/ready` shows `ondevice_bundle.ok: false` | `$dc run --rm bundle` (or redeploy the release bundle), check `$dc logs bundle` |
| Model install on a phone fails with "checksum mismatch" | bundle files changed while being served, or a proxy rewrote them | re-export or redeploy the bundle; never edit bundle files in place |
| "This app version cannot use model bundle ..." on phones | bundle asset format newer than the cached app | users reload the app online (the service worker fetches the new app); keep app and bundle from the same release |
| Phones report "The models were downloaded but could not be started" | runtime or operator incompatibility on that browser | collect a device-check report from the phone; see [DEVICE_TEST_PROTOCOL.md](DEVICE_TEST_PROTOCOL.md); the previous bundle stays active |
| `/ready` models check fails | server model files missing or checksum mismatch in the `models` volume | `$dc run --rm models` (re-fetch with checksum verification) |
| Server analyses stuck in `queued` | no worker heartbeat | `$dc restart app`; leased jobs are re-queued after the lease expires (`app.jobs.lease_s`) |
| Disk space check fails | retained media or uploads | `$dc run --rm app cleanup`; review `app.storage.max_retain_media_days` and per-user quota |
| Release check exit code 3 | an asset is not allowed for the profile | read the decision list; the engine already runs without that asset and says so in results |

## Routine tasks

* **Backups:** daily
  `$dc exec app python -m petperception.cli db backup --out /var/lib/petperception/backups/$(date +%F).sqlite3`,
  copy off the host, keep 30 days. Test a restore quarterly (stop `app`,
  replace the database file, start `app`, check `/ready`).
* **Cleanup** runs inside the app every `app.storage.cleanup_interval_s`
  (expired uploads, retained media past their date, expired sessions, orphan
  files); `$dc run --rm app cleanup` runs it once by hand.
* **Accounts:** `$dc run --rm app user list | create | disable | enable |
  reset-password --username NAME` (password from a prompt or
  `PETPERCEPTION_NEW_PASSWORD`; resetting a password revokes sessions).
* **Device reports:** users can send device-check reports (JSON only, no
  media). Review them to track which phones and browsers pass; they are the
  evidence for [DEVICE_TIERS_AND_BUDGETS.md](DEVICE_TIERS_AND_BUDGETS.md).
* **Model and bundle updates:** publish a new bundle (new `bundle_version`);
  phones offer the update in Settings, verify every file, start the engine
  from the new files and only then switch; an interrupted or failing update
  leaves the previous version working.
* **Behaviour / intent stage (research deployments only):** set
  `intent: {enabled: true, variant: phone_now}` in the server configuration
  and export the phone bundle with `petperception export-ondevice --intent`
  (release profile `research`; the situation heads are trained on YouTube
  clips). Results then carry the `intent` block (schema 1.5.0) and the
  result page shows the intent card. Check after a change:
  `petperception intent <clip>` on a known clip, the engine-level parity in
  [ON_DEVICE.md](ON_DEVICE.md), and the component state `intent: ok` in a
  result. If the component says `unavailable`, its reason names the missing
  artifact or the release-profile decision.
* **Behaviour explanation (command line, research profile):**
  `petperception explain <media> --species cat|dog --scene "..." --question "..." --out a.json --markdown a.md`
  needs the `behavior` extra (`uv sync --extra behavior`: CPU PyTorch,
  transformers) and SigLIP in `var/models/siglip_base_patch16_224`
  (`petperception fetch-models --only siglip_base_patch16_224`; checksums in
  `manifests/resources.json`). The default (E31 system, code frozen in
  `understand/e31/`) uses `artifacts/understand/recognizer.json` and falls
  back to the fixed zero-shot defaults when it is missing.
  `--system-version 2` (E32 system) uses `recognizer_v2.json` and
  `circling_v1.json`, and all-MiniLM-L6-v2 in `var/models/minilm_l6_v2`
  (`petperception fetch-models --only minilm_l6_v2`; without it the parser
  falls back to keywords). About 20-30 s for a 10-s clip on 2 CPU threads,
  most of it model loading. After changing `understand/knowledge.py` (v2),
  run `pytest tests/test_understand.py tests/test_understand_v2.py`
  (`knowledge.validate()` must stay empty); a change to the v2 system needs a
  new pre-registered test on fresh clips before it can become the default
  (E32 protocol). Never edit `understand/e31/` (a test pins its hashes). Not
  part of the phone bundle or the web app yet
  ([BEHAVIOR_EXPLANATION.md](BEHAVIOR_EXPLANATION.md)).
* **Cloud-assisted understanding (optional, off by default):** set `cloud.enabled: true` and put the provider key in
  the backend's environment variable named by `cloud.api_key_env` (default `ANTHROPIC_API_KEY`); never in the
  config file, the web app or Git. Check prices in `cloud.price_*_per_mtok` against the provider's list. Owners
  choose it per analysis; frames, pet crops and a sound summary of that clip leave the server for that answer.
  Research samples live under `storage.research_root` (back it up with the database; deleting an account deletes
  its samples). `petperception explain ... --understanding cloud --dry-run` shows the packet size and cost estimate.
* **Upgrades:** `git pull && $dc up -d --build`; migrations run at start.
  Read the release notes for asset-format changes (bundle `engine_version`):
  app and bundle must come from the same release.

## Data handling

* On-device analyses never reach the server unless the user turns on result
  backup; then the result JSON and feedback (never the recording) are stored
  in `synced_results` and included in the user's export and deletion.
* Server analysis (opt-in per user) uploads media; media are deleted after
  analysis unless the user chose to keep them (then at most
  `max_retain_media_days`). Users can export (`GET /api/v1/export`) and delete
  their account (`POST /api/v1/auth/delete-account`), which removes their
  analyses, media, feedback, backed-up on-device results and device reports.
* Person and furniture detection used for interaction and on-furniture
  candidates runs on the analysing device; person and furniture boxes are
  not stored in results.
