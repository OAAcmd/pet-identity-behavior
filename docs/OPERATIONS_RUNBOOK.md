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
