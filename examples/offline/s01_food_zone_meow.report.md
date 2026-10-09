# Analysis report `example-s01_food_zone_meow`

*Schema 1.0.0 - engine 0.1.0 - profile `cpu` - research use only.*

> Observations first; hypotheses are experimental and uncalibrated. This is not a translation of animal language, an emotion reading, or a medical assessment.

## Input
- File: `s01_food_zone_meow.mp4` (118590 bytes, sha256 `e1e2c22ef904add9...`), container `mov,mp4,m4a,3gp,3g2,mj2`, duration 8.0
- Audio: present (aac, 44100 Hz, 1 ch)
- Video: present (h264, 640x480)

## Decision
- State: **experimental_hypothesis** (experimental_hypotheses_present)
- 1 experimental hypothesis(es) with co-occurring independent evidence; scores are uncalibrated.

## Observations
- **track-1** cat: 0.00-7.50 s, 16 detections; median apparent speed 0.000 image_diagonals_per_second, moving fraction 0.00 (reliability high)
  - in zone `food_area` (food_area) for 7.5 s
  - pose features unavailable: ear_angle_deg, tail_height, play_bow_candidate, tail_lateral_oscillation
- Sound `cat_vocalization` 1.44-7.21 s, peak raw score 0.98 (threshold 0.74, uncalibrated); attribution: single_visible_candidate - exactly one species-consistent animal visible; attribution is unverified

## Hypotheses (experimental)
- `food` [need] -> **experimental_hypothesis**: Experimental hypothesis 'food' (need) for track-1. Supporting observations: cat vocalization sound 1.44-7.21s (peak raw score 0.98; meow 0.85) [ev-snd-1]; track-1 in owner-marked zone 'owner-marked food area (synthetic test zone)' (food_area) for 7.5s [ev-zone-track-1-food_area]. Notes: the fusion rules have not been validated on paired household data; attributing the sound to this animal is unverified (one visible candidate). Score 0.63 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `outside` [need] -> **insufficient_evidence**: Insufficient evidence 'outside' (need) for track-1. Supporting observations: cat vocalization sound 1.44-7.21s (peak raw score 0.98; meow 0.85) [ev-snd-1]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; attributing the sound to this animal is unverified (one visible candidate). Score 0.65 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `attention` [need] -> **insufficient_evidence**: Insufficient evidence 'attention' (need) for track-1. Supporting observations: cat vocalization sound 1.44-7.21s (peak raw score 0.98; meow 0.85) [ev-snd-1]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; person detection is disabled; attributing the sound to this animal is unverified (one visible candidate). Score 0.60 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `relaxed` [state] -> **insufficient_evidence**: Insufficient evidence 'relaxed' (state) for track-1. Supporting observations: low apparent movement of track-1 (median 0.000 image diagonals/s; moving in 0 of 15 sampled steps) [ev-move-track-1]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; silence is never used as evidence of relaxation. Score 0.60 is an uncalibrated heuristic_support_0_1 value, not a probability.

## Suggestions
- The animal vocalized while in the area you marked for feeding. If it is close to the usual feeding time, you could check whether food and water are available. This is an unvalidated heuristic, not a determination of hunger.

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
probe: 55.8, hash: 0.2, decode_audio: 157.6, audio_inference: 23.8, video_decode_detect_track: 640.6, video_detector_inference: 351.8, video_quality_and_camera_motion: 201.2, frame_gap_check: 62.5, fusion: 0.3, model_load_ms_engine_startup: 313.6
