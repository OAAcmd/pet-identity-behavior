# Analysis report `example-s04_no_pet_vacuum`

*Schema 1.0.0 - engine 0.1.0 - profile `cpu` - research use only.*

> Observations first; hypotheses are experimental and uncalibrated. This is not a translation of animal language, an emotion reading, or a medical assessment.

## Input
- File: `s04_no_pet_vacuum.mp4` (136803 bytes, sha256 `d4148858700c4c14...`), container `mov,mp4,m4a,3gp,3g2,mj2`, duration 8.0
- Audio: present (aac, 44100 Hz, 1 ch)
- Video: present (h264, 640x480)

## Decision
- State: **abstention** (unknown)
- No cat or dog was detected in the available modalities; no interpretation is offered.
- Reason codes: no_pet_detected, no_pet_sound_detected

## Observations
- No cat or dog observations.
- Context sound `context:vacuum_cleaner` 1.92-7.21 s (raw 0.99)

## Hypotheses (experimental)
- None considered.

## Quality flags
- `audio_end_padded` (info): final analysis window zero-padded

## Limitations
- Phase 1 research engine: outputs are observations plus unvalidated heuristics. They are not translations of animal language, readings of thoughts or emotions, or medical assessments.
- Track IDs are per-analysis labels for linked detections, not verified pet identities.
- Hypothesis scores are uncalibrated; multimodal accuracy has not been evaluated (no paired household validation data).
- Sound events use raw YAMNet scores with thresholds selected on ESC-50 clips, which are not household recordings.
- Cat/dog detection uses a COCO-trained EfficientDet-Lite model; ear, tail and posture features are unavailable (no verified animal pose model).
- For veterinary or behavioral concerns, consult a qualified professional; this system does not diagnose.

## Timings (ms)
probe: 58.2, hash: 0.2, decode_audio: 165.4, audio_inference: 25.1, video_decode_detect_track: 665.8, video_detector_inference: 362.6, video_quality_and_camera_motion: 210.5, frame_gap_check: 59.6, fusion: 0.0, model_load_ms_engine_startup: 313.6
