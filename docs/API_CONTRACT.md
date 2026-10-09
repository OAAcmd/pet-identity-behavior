# API Contract (v1, result schema 1.0.0)

Generated artifacts (kept in sync by `tests/test_pose_zones_schema.py`):

- OpenAPI 3.1 document: [`docs/openapi.json`](openapi.json)
- Result JSON Schema: [`docs/analysis_result.schema.json`](analysis_result.schema.json)
- Regenerate: `uv run petperception export-openapi`

The CLI (`petperception analyze`) and the API return the same `AnalysisResult`
model produced by the same `AnalysisEngine`.

## Service posture (Phase 1)

| Aspect | Behavior |
| --- | --- |
| Bind address | `127.0.0.1:8000` by default (`configs/cpu.yaml: serve`). `petperception serve` refuses a non-loopback host unless a token is set. |
| Authentication | If `$PETPERCEPTION_API_TOKEN` is set, all `/v1/*` endpoints except `/v1/health` need `Authorization: Bearer <token>` (constant-time compare). Without a token, non-loopback clients get 403. TLS and production auth are Phase 2. |
| Inputs | Multipart upload only. **No server-side URL fetching.** Extension allow-list: `.wav .flac .ogg .mp3 .m4a .mp4 .mov .mkv .webm .avi`. Size cap `serve.max_upload_bytes` (100 MiB) enforced from `Content-Length` and again while streaming. `ffprobe` validation before queueing; duration/size limits from `limits`. |
| Storage | Private `0700` directory per job under `paths.jobs_root`. Uploaded media is deleted as soon as the analysis finishes. Results stay in memory until `DELETE` or TTL (`serve.job_ttl_s`, 1 h). |
| Concurrency | One worker thread, in-process queue, `serve.max_jobs` (32) cap (503 when full). |
| Models | Loaded at startup; missing models produce `unavailable` components and abstention reasons, never fabricated predictions. Nothing is downloaded at inference time. |

## Endpoints

| Method | Path | Success | Errors |
| --- | --- | --- | --- |
| GET | `/v1/health` | 200 `{status, schema_version, engine_version, engine_loaded}` | - |
| GET | `/v1/capabilities` | 200 components with readiness, modalities, detector classes, fusion status (`validated: false`, `multimodal_accuracy: not_evaluated`), limits, upload policy | 401, 403 |
| GET | `/v1/schema/analysis-result` | 200 JSON Schema of `AnalysisResult` | 401, 403 |
| POST | `/v1/analyses` | 202 `{analysis_id, status: "queued", status_url, result_url}` | 401, 403, 413 `upload_too_large`, 415 `unsupported_extension`, 422 (`empty_upload`, `invalid_zones`, `invalid_modalities`, `corrupt_input`, `duration_too_long`, ...), 503 `job_store_full` |
| GET | `/v1/analyses/{id}` | 200 `{analysis_id, status: queued|running|succeeded|failed, created_at, updated_at, error}` | 404 |
| GET | `/v1/analyses/{id}/result` | 200 `AnalysisResult` | 404, 409 `not_ready`, 422 `AnalysisError` for failed jobs |
| DELETE | `/v1/analyses/{id}` | 200 `{status: "deleted"}` (files removed before the response) or `{status: "deletion_scheduled"}` (job running; removed by the worker right after it returns; the job is unreadable from this moment) | 404 |

`POST /v1/analyses` form fields:

- `file` (required): the media file.
- `zones` (optional): JSON string, `petperception.zones/1` (see `configs/zones.example.json`).
- `modalities` (optional): `audio,video` (default), `audio`, or `video`.

Error body: `{"error_code": "<stable code>", "message": "<text>"}`.

## Example

```bash
export PETPERCEPTION_API_TOKEN=change-me
uv run petperception serve                        # 127.0.0.1:8000
curl -s -H "Authorization: Bearer $PETPERCEPTION_API_TOKEN" \
     -F file=@clip.mp4 -F zones="$(cat configs/zones.example.json)" \
     http://127.0.0.1:8000/v1/analyses
curl -s -H "Authorization: Bearer $PETPERCEPTION_API_TOKEN" http://127.0.0.1:8000/v1/analyses/<id>
curl -s -H "Authorization: Bearer $PETPERCEPTION_API_TOKEN" http://127.0.0.1:8000/v1/analyses/<id>/result
curl -s -X DELETE -H "Authorization: Bearer $PETPERCEPTION_API_TOKEN" http://127.0.0.1:8000/v1/analyses/<id>
```

## `AnalysisResult` (schema 1.0.0) - field guide for clients

| Field | Meaning | Client guidance |
| --- | --- | --- |
| `schema_version` | constant `"1.0.0"` | reject unknown major versions |
| `analysis_id`, `created_at`, `engine_version`, `config_profile`, `config_hash` | provenance of the run | show in a details view |
| `media` | sanitized filename, size, SHA-256, container, duration, audio/video stream facts; `timeline = container_pts_seconds` | all timestamps in the result use this timeline |
| `modalities` | requested vs available; `audio_state`, `video_state` | display missing/silent/corrupt states distinctly |
| `components[]` | per-component `state` (`ok`, `unavailable`, `disabled`, ...) with model IDs and reasons | surface unavailable components; never hide them |
| `observations.animal_tracks[]` | per-analysis tracks (`identity_note`: not a pet identity), species scores (raw detector scores), detections with normalized boxes and timestamps, movement with explicit units and reliability, zone dwell, pose features (currently all unavailable), quality flags, evidence IDs | primary content of the UI |
| `observations.sound_events[]` | cat/dog/growl events with time spans, **raw** YAMNet scores, threshold and its source, `calibrated_probability: null`, attribution status and note | never render scores as percentages |
| `observations.context_sounds[]` | environmental tags (speech, doorbell, vacuum, ...) | context only, not animal behavior |
| `observations.evidence[]` | atomic evidence items with provenance root and reliability | used for explanations and audit |
| `quality_flags[]` | codes such as `low_light`, `camera_motion`, `bandwidth_limited_audio`, `audio_silent`, `still_image`, `dropped_or_missing_frames` | show as limitations near the affected time span |
| `hypotheses[]` | `name` (food/outside/attention/play/alert/tense/warning/relaxed), `semantic_type` (need/activity/state/communicative_signal), `status` (`experimental_hypothesis` / `insufficient_evidence` / `abstained`), supporting and conflicting evidence IDs, provenance roots, `score` (uncalibrated, unitless), `calibration_status: uncalibrated`, `calibrated_confidence: null`, reason codes, explanation | always label as experimental; show reasons; never show `score` as a confidence |
| `decision` | `state` (`supported_observation` / `experimental_hypothesis` / `abstention`), `overall_interpretation` (`observations_only` / `experimental_hypotheses_present` / `unknown` / `insufficient_evidence`), reason codes, summary | default UI framing |
| `suggestions[]` | restrained guidance citing evidence IDs; kinds `observation_followup`, `safety_general`, `data_quality` | display with the cited evidence |
| `limitations[]`, `usage_restrictions` | static and input-specific limitations; research-only notices | always visible |
| `fusion` | `mode: experimental`, `validated: false`, `multimodal_accuracy: not_evaluated`, `weights_fitted: false` | show an "experimental" badge |
| `timings_ms`, `models[]` | stage timings; model revisions, checksums, terms | diagnostics view |

Reason codes used in `decision.reason_codes` and `hypotheses[].reason_codes`:
`missing_modality_audio`, `missing_modality_video`, `audio_silent`, `audio_corrupt`,
`audio_model_unavailable`, `model_unavailable`, `no_decodable_video_frames`,
`no_pet_detected`, `no_pet_sound_detected`, `multiple_animals_visible`,
`fusion_not_validated`, `single_provenance_root`, `context_only_evidence`,
`stale_evidence`, `zones_not_configured`, `pose_unavailable`,
`person_detection_disabled`, `uncertain_attribution_multiple_animals`,
`no_visible_animal_for_attribution`, `conflicting_evidence`, `low_support`,
`audio_attribution_unverified`, `low_reliability_evidence`,
`silence_not_used_as_evidence`, `purr_does_not_rule_out_discomfort`.

## Stability for Phase 2

Additive changes (new optional fields, new reason codes, new quality flags)
keep `1.x`. Removing/renaming fields or changing score semantics requires
`2.0.0`. The PWA must treat unknown reason codes and flags as displayable text.
