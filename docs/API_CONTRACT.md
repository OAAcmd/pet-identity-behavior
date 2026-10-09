# API contract (`/api/v1`, result schema 1.2.0)

Generated, always-current artifacts (CI fails if they are stale):

- OpenAPI 3.1: [`docs/openapi.json`](openapi.json)
- Analysis result JSON Schema: [`docs/analysis_result.schema.json`](analysis_result.schema.json)
- Regenerate: `uv run petperception export-openapi`

The phone engine (TypeScript) and the server engine (Python) produce the
**same result schema**; on-device results reach the server only if the user
turns on result backup (result JSON and feedback, never media).

## Service posture

| Aspect | Behavior |
| --- | --- |
| Origin | API, PWA and on-device bundle are served from one origin behind a TLS proxy ([DEPLOYMENT.md](DEPLOYMENT.md)); no CORS. Headers: strict CSP (`'wasm-unsafe-eval'` for WebAssembly only), COOP `same-origin`, COEP `require-corp`, `Permissions-Policy: camera=(self), microphone=(self)`. |
| Authentication | Session cookie (HttpOnly, `SameSite=Strict`, `Secure` in production; 32 random bytes, only the SHA-256 stored). Passwords hashed with scrypt. Login attempts rate-limited. Registration: first user only (becomes admin), then admin-created accounts (`app.auth.registration`). |
| CSRF | State-changing requests must send `X-CSRF-Token` (returned by login/register and `/auth/me`). |
| Errors | JSON `{"error": {"code", "message", "action", "details"?}}` with a stable `code`; every response carries `X-Request-ID`. |
| Limits | per-user open uploads, active analyses and quota; global queue length; upload size and chunk size; media duration and frame limits (`app.*`, `limits.*`). |
| Privacy | Media only arrive through explicit server analysis. Uploaded media are deleted after analysis unless the user keeps them (bounded retention). Logs contain no user names, file names or passwords. Account deletion removes analyses, media, feedback, backed-up on-device results and device reports. |

## Endpoints

| Method and path | Purpose |
| --- | --- |
| `GET /health`, `GET /ready`, `GET /capabilities` | liveness; readiness with checks (database, models, disk, worker, queue, on-device bundle); engine components and their state |
| `GET /auth/status` | whether registration is open |
| `POST /auth/register`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me` | account and session |
| `POST /auth/password`, `POST /auth/delete-account` | password change (revokes other sessions), account deletion |
| `GET /admin/users`, `POST /admin/users` | administrator account management |
| `PATCH /settings`, `GET /export` | media retention, research-consent default; export of the user's data |
| `GET/POST /pets`, `GET/PATCH/DELETE /pets/{id}` | pet profiles (cat/dog, name, birth year, notes) |
| `GET/POST /places`, `PATCH/DELETE /places/{id}` | owner-drawn places (`petperception.zones/1` polygons) |
| `POST /uploads`, `GET /uploads/{id}`, `PUT /uploads/{id}?offset=N`, `DELETE /uploads/{id}` | resumable chunked upload (server analysis only): the client resumes from the server's offset after interruptions; SHA-256 checked when given |
| `POST /analyses`, `GET /analyses`, `GET /analyses/{id}` | queue a server analysis of a completed upload; list; status and result |
| `POST /analyses/{id}/cancel`, `POST /analyses/{id}/retry`, `DELETE /analyses/{id}` | job control and deletion (media removed with it) |
| `GET /analyses/{id}/media`, `DELETE /analyses/{id}/media` | retained media of a server analysis |
| `GET/POST /analyses/{id}/feedback`, `DELETE /feedback/{id}` | user feedback on observations and hypotheses, with explicit research consent |
| `GET/POST /device-reports` | device-check reports (JSON only, never media; size-limited) |
| `PUT/GET/DELETE /synced-results/{id}`, `GET /synced-results` | opt-in backup of on-device results: result JSON (validated against the result schema, id must match) and feedback; never media; per-user limits |

Outside `/api/v1`: `GET /ondevice/<path>` serves the checksummed on-device
bundle (`manifest.json` → 503 `ondevice_bundle_missing` when not published;
path traversal rejected); `/` serves the PWA.

## Analysis result (schema 1.2.0)

Top level: `schema_version`, `analysis_id`, `created_at`, `engine_version`,
`engine` (runtime, model ids and versions, component states with reasons),
`media` (container, duration, audio/video streams, timeline), `observations`
(`animal_tracks`, `sound_events`, `context_sounds`, `behavior_events`),
`evidence` (each item with provenance root and reliability), `hypotheses`
(experimental, uncalibrated, with supporting and conflicting evidence),
`decision` (`supported_observation`, `abstention`, ...), `explanations`
(deterministic templates), `suggestions`, `quality` flags, `timings_ms`.

Per animal track: species and scores, time span, detections (boxes),
movement summary, zone occupancy, `behavior_segments`, posture frames,
verification summary, quality flags, pose features (with `available: false`
and a reason when no keypoint model is present).

Behavior segment kinds: `stationary`, `moving`, `active_bout`,
`pacing_candidate`, `approach_zone`, `enter_zone`, `dwell_zone`,
`leave_zone`, `entered_view`, `left_view`, `vocal_bout`,
`startle_candidate`, and (1.2.0) `person_interaction_candidate`. Each carries
`reliability` and `validation_status`
(`synthetic_and_semi_synthetic_only` or, for interaction candidates,
`per_frame_model_public_still_images`).

Versioning: additive changes bump the minor version (1.1.0 added behavior
segments, keypoints and verification; 1.2.0 added interaction candidates);
readers accept older minor versions. Scores are never presented as
probabilities: every score has a `score_kind` such as `detector_sigmoid_raw`
or `posture_softmax_uncalibrated`.
