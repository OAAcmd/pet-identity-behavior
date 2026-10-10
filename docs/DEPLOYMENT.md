# Deployment

**Status:** the container image builds and its smoke test passes in this
workspace; the full stack has **not** been deployed to a public host, because
none (and no domain, DNS or credentials) is available to this project. What
an operator needs to provide is listed at the end.

## Architecture

```
phone browser (PWA) ──HTTPS──► Caddy (TLS, HSTS) ──► app container
                                                      ├─ API (/api/v1), accounts, optional server analysis + workers
                                                      ├─ static PWA (web/dist)
                                                      └─ /ondevice/  model bundle (read-only volume)
```

The core analysis is built to run in the phone's browser with the bundle
downloaded from `/ondevice/` (verified in desktop Chromium only; physical
phones pending, [DEVICE_TEST_PROTOCOL.md](DEVICE_TEST_PROTOCOL.md)); the server does accounts, pets and places, bundle distribution,
device-check reports and (only when a user enables it) server analysis.
See [ON_DEVICE.md](ON_DEVICE.md).

Compose services (`deploy/compose.yaml`):

| Service | Runs | Purpose |
| --- | --- | --- |
| `models` | once | `petperception fetch-models`: checksum-verified model download into the `models` volume |
| `bundle` | once | `petperception export-ondevice --out /bundle/current --no-device-check --fetch-missing`: the on-device bundle for the release profile into the `bundle` volume |
| `app` | always | API, workers and PWA on port 8000 (not published); health check `/api/v1/health` |
| `proxy` | always | Caddy 2.10 (pinned digest): automatic TLS for `PETPERCEPTION_DOMAIN`, ports 80/443 |

Volumes: `data` (SQLite database, retained media), `models`, `bundle`,
Caddy state. Containers are otherwise disposable.

## Requirements

* Linux host with Docker Engine 24+ and the compose plugin; 4 vCPU and 4-8 GB
  RAM for server analysis with 2 workers (phones do the default analysis, so
  a smaller host works if server analysis stays off); 10 GB disk plus
  retained media.
* A domain whose DNS A/AAAA record points to the host, ports 80 and 443
  reachable from the internet (Caddy obtains the certificate). HTTPS is not
  optional: the PWA needs a secure context for Cache Storage, service worker,
  camera and cross-origin isolation.
* Outbound HTTPS from the host to `storage.googleapis.com` during install
  (model downloads), plus the base-image registry (`mirror.gcr.io` by
  default) and PyPI/npm during the image build.

## Install

```bash
git clone <repository> && cd pet-identity-behavior
cp deploy/env.example deploy/.env       # set PETPERCEPTION_DOMAIN, release profile
docker compose -f deploy/compose.yaml --env-file deploy/.env up -d --build
docker compose -f deploy/compose.yaml --env-file deploy/.env logs -f bundle app
```

First account: open `https://<domain>/` and register; with
`registration: first_user_only` the first account becomes the administrator
and later accounts are created by the administrator:

```bash
docker compose -f deploy/compose.yaml --env-file deploy/.env run --rm app user create --username alice
```

(password from a prompt or `PETPERCEPTION_NEW_PASSWORD`).

Behind a TLS-intercepting proxy, build with the extra CA:
`docker build --secret id=extra_ca,src=/path/ca.pem -t petperception:local .`

## Release profile and asset approvals

`PETPERCEPTION_RELEASE_PROFILE` is `personal_noncommercial` by default
(`research` is refused by the image; use a source checkout). Check what the
profile allows before going live:

```bash
docker compose -f deploy/compose.yaml --env-file deploy/.env run --rm app release-check
```

Exit code 3 means a configured asset is not allowed; the engine and the
bundle export already leave such assets out and the result says so.
`commercial` additionally needs legal sign-off recorded in an approvals
file (`configs/release_approvals.example.yaml`) for every asset marked
`requires_legal_review`, and the detector must be
`efficientdet_lite0_tfhub_rawdecode_v1` (Apache-2.0 in its embedded
metadata; runs in the browser engine (desktop Chromium) with its post-processing decoded outside the graph,
evaluated in E14): set `PETPERCEPTION_DETECTOR` and add `--detector
efficientdet_lite0_tfhub_rawdecode_v1` to the bundle export. Without the
approvals the commercial bundle contains no models and says why.

## On-device bundle

The `bundle` service exports the bundle without the device-check reference
clip (the clip is built from datasets that are not in the image). For a
release, build the full bundle on a build machine that has the datasets and
publish it as a versioned artifact:

```bash
uv run petperception fetch-data openimages-tuning   # device-check clip sources (backgrounds are fetched by the export)
uv run petperception export-ondevice --release-profile personal_noncommercial --out dist/bundle
tar -C dist -czf bundle-$(jq -r .bundle_version dist/bundle/manifest.json).tgz bundle
```

Deploy it by replacing the `bundle` volume with a read-only bind mount of the
unpacked directory (mount it so that `manifest.json` is at
`/bundle/current/manifest.json`) and removing the `bundle` service. Clients
see the new `bundle_version`, download and verify it, and switch only after
the engine starts from the new files.

Compact variant for constrained phones (14.8 MB instead of 37.2 MB; evaluated
in E12, opt-in because individual verifier decisions can differ from the
float32 default): add `--detector efficientdet_lite0_mediapipe_int8_v1
--backbone efficientnet_lite0_mediapipe_int8_v1` to the export command (the
matching refitted verifier is selected automatically; posture_v1 is valid on
int8 features per E12).

## Upgrade, backup, restore

* Upgrade: `git pull`, then `docker compose ... up -d --build`. Database
  migrations run at start (`db.migrate()`); running analyses are cancelled
  and re-queued on restart (`stop_grace_period: 60s`).
* Backup (online, WAL-safe):
  `docker compose ... exec app python -m petperception.cli db backup --out /var/lib/petperception/backups/$(date +%F).sqlite3`,
  then copy it off the host. Retained media are in the same volume
  (`/var/lib/petperception/media`). On-device results live on users' phones
  and are not backed up by the server (result sync is not implemented).
* Restore: stop `app`, replace the database file in the `data` volume, start
  `app`.

Operational procedures: [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md).
Configuration reference: [CONFIGURATION.md](CONFIGURATION.md).

## What the operator must provide (not available to this project)

| Item | Why |
| --- | --- |
| A Linux host with Docker | run the stack |
| A domain and DNS record, ports 80/443 open | TLS certificate and secure context for the PWA |
| Decision on the release profile and, for `commercial`, legal sign-off per asset | data and weight terms |
| Off-host backup storage | database backups |
| Physical phones for the acceptance test | [DEVICE_TEST_PROTOCOL.md](DEVICE_TEST_PROTOCOL.md) |
