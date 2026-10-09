# Analysis report `example-s02_food_zone_silent`

*Schema 1.0.0 - engine 0.1.0 - profile `cpu` - research use only.*

> Observations first; hypotheses are experimental and uncalibrated. This is not a translation of animal language, an emotion reading, or a medical assessment.

## Input
- File: `s02_food_zone_silent.mp4` (33899 bytes, sha256 `c5b22bd68bca3a2e...`), container `mov,mp4,m4a,3gp,3g2,mj2`, duration 8.0
- Audio: silent (aac, 44100 Hz, 1 ch)
- Video: present (h264, 640x480)

## Decision
- State: **supported_observation** (insufficient_evidence)
- Observations only: 1 animal track(s), 0 pet sound event(s). Candidate interpretations lacked sufficient independent evidence.
- Reason codes: audio_silent

## Observations
- **track-1** cat: 0.00-7.50 s, 16 detections; median apparent speed 0.000 image_diagonals_per_second, moving fraction 0.00 (reliability high)
  - in zone `food_area` (food_area) for 7.5 s
  - pose features unavailable: ear_angle_deg, tail_height, play_bow_candidate, tail_lateral_oscillation

## Hypotheses (experimental)
- `food` [need] -> **insufficient_evidence**: Insufficient evidence 'food' (need) for track-1. Supporting observations: track-1 in owner-marked zone 'owner-marked food area (synthetic test zone)' (food_area) for 7.5s [ev-zone-track-1-food_area]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; only scene context (an owner-marked zone) supports it; the audio track is silent (silence is not evidence either way). Score 0.60 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `relaxed` [state] -> **insufficient_evidence**: Insufficient evidence 'relaxed' (state) for track-1. Supporting observations: low apparent movement of track-1 (median 0.000 image diagonals/s; moving in 0 of 15 sampled steps) [ev-move-track-1]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; the audio track is silent (silence is not evidence either way); silence is never used as evidence of relaxation. Score 0.60 is an uncalibrated heuristic_support_0_1 value, not a probability.

## Quality flags
- `audio_silent` (info): audio is silent; silence is not evidence of relaxation or of absence of need

## Limitations
- Phase 1 research engine: outputs are observations plus unvalidated heuristics. They are not translations of animal language, readings of thoughts or emotions, or medical assessments.
- Track IDs are per-analysis labels for linked detections, not verified pet identities.
- Hypothesis scores are uncalibrated; multimodal accuracy has not been evaluated (no paired household validation data).
- Sound events use raw YAMNet scores with thresholds selected on ESC-50 clips, which are not household recordings.
- Cat/dog detection uses a COCO-trained EfficientDet-Lite model; ear, tail and posture features are unavailable (no verified animal pose model).
- For veterinary or behavioral concerns, consult a qualified professional; this system does not diagnose.

## Timings (ms)
probe: 66.9, hash: 0.1, decode_audio: 142.5, video_decode_detect_track: 661.3, video_detector_inference: 373.9, video_quality_and_camera_motion: 198.2, frame_gap_check: 57.5, fusion: 0.2, model_load_ms_engine_startup: 313.6
