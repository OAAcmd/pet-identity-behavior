# Analysis report `example-s03_two_animals_bark`

*Schema 1.0.0 - engine 0.1.0 - profile `cpu` - research use only.*

> Observations first; hypotheses are experimental and uncalibrated. This is not a translation of animal language, an emotion reading, or a medical assessment.

## Input
- File: `s03_two_animals_bark.mp4` (172286 bytes, sha256 `a1c9d280de356da7...`), container `mov,mp4,m4a,3gp,3g2,mj2`, duration 8.0
- Audio: present (aac, 44100 Hz, 1 ch)
- Video: present (h264, 640x480)

## Decision
- State: **supported_observation** (insufficient_evidence)
- Observations only: 2 animal track(s), 1 pet sound event(s). Candidate interpretations lacked sufficient independent evidence.
- Reason codes: multiple_animals_visible

## Observations
- **track-1** cat: 0.00-7.50 s, 16 detections; median apparent speed 0.000 image_diagonals_per_second, moving fraction 0.00 (reliability high)
  - pose features unavailable: ear_angle_deg, tail_height, play_bow_candidate, tail_lateral_oscillation
- **track-2** dog: 0.00-7.50 s, 16 detections; median apparent speed 0.000 image_diagonals_per_second, moving fraction 0.00 (reliability high)
  - pose features unavailable: ear_angle_deg, tail_height, play_bow_candidate, tail_lateral_oscillation
- Sound `dog_vocalization` 1.92-7.21 s, peak raw score 0.99 (threshold 0.80, uncalibrated); attribution: unknown - several animals visible; no evaluated audio-visual association method, so attribution is unknown

## Hypotheses (experimental)
- `food` [need] -> **insufficient_evidence**: Insufficient evidence 'food' (need). Supporting observations: dog vocalization sound 1.92-7.21s (peak raw score 0.99; bark 0.92, bow_wow 0.26, growl 0.26) [ev-snd-1]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; no scene zones were configured; several animals were visible, so the sound could not be attributed. Score 0.65 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `outside` [need] -> **insufficient_evidence**: Insufficient evidence 'outside' (need). Supporting observations: dog vocalization sound 1.92-7.21s (peak raw score 0.99; bark 0.92, bow_wow 0.26, growl 0.26) [ev-snd-1]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; no scene zones were configured; several animals were visible, so the sound could not be attributed. Score 0.65 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `attention` [need] -> **insufficient_evidence**: Insufficient evidence 'attention' (need). Supporting observations: dog vocalization sound 1.92-7.21s (peak raw score 0.99; bark 0.92, bow_wow 0.26, growl 0.26) [ev-snd-1]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; person detection is disabled; several animals were visible, so the sound could not be attributed. Score 0.60 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `alert` [state] -> **insufficient_evidence**: Insufficient evidence 'alert' (state). Supporting observations: dog vocalization sound 1.92-7.21s (peak raw score 0.99; bark 0.92, bow_wow 0.26, growl 0.26) [ev-snd-1]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; several animals were visible, so the sound could not be attributed. Score 0.60 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `relaxed` [state] -> **insufficient_evidence**: Insufficient evidence 'relaxed' (state) for track-1. Supporting observations: low apparent movement of track-1 (median 0.000 image diagonals/s; moving in 0 of 15 sampled steps) [ev-move-track-1]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; no scene zones were configured; silence is never used as evidence of relaxation. Score 0.60 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `relaxed` [state] -> **insufficient_evidence**: Insufficient evidence 'relaxed' (state) for track-2. Supporting observations: low apparent movement of track-2 (median 0.000 image diagonals/s; moving in 0 of 15 sampled steps) [ev-move-track-2]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; no scene zones were configured; silence is never used as evidence of relaxation. Score 0.60 is an uncalibrated heuristic_support_0_1 value, not a probability.

## Suggestions
- Several animals were visible, so sounds were not assigned to a specific animal.

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
probe: 61.4, hash: 0.3, decode_audio: 212.2, audio_inference: 32.3, video_decode_detect_track: 655.9, video_detector_inference: 336.9, video_quality_and_camera_motion: 220.7, frame_gap_check: 61.0, fusion: 0.3, model_load_ms_engine_startup: 313.6
