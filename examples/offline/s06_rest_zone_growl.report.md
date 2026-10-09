# Analysis report `example-s06_rest_zone_growl`

*Schema 1.0.0 - engine 0.1.0 - profile `cpu` - research use only.*

> Observations first; hypotheses are experimental and uncalibrated. This is not a translation of animal language, an emotion reading, or a medical assessment.

## Input
- File: `s06_rest_zone_growl.mp4` (111018 bytes, sha256 `9fe9c762a9e1284f...`), container `mov,mp4,m4a,3gp,3g2,mj2`, duration 8.0
- Audio: present (aac, 44100 Hz, 1 ch)
- Video: present (h264, 640x480)

## Decision
- State: **experimental_hypothesis** (experimental_hypotheses_present)
- 2 experimental hypothesis(es) with co-occurring independent evidence; scores are uncalibrated.

## Observations
- **track-1** cat: 0.00-7.50 s, 16 detections; median apparent speed 0.000 image_diagonals_per_second, moving fraction 0.00 (reliability high)
  - in zone `rest_area` (rest_area) for 7.5 s
  - pose features unavailable: ear_angle_deg, tail_height, play_bow_candidate, tail_lateral_oscillation
- Sound `cat_vocalization` 2.40-3.38 s, peak raw score 0.80 (threshold 0.74, uncalibrated); attribution: single_visible_candidate - exactly one species-consistent animal visible; attribution is unverified
- Sound `growl` 5.28-6.25 s, peak raw score 0.50 (threshold 0.50, uncalibrated); attribution: single_visible_candidate - exactly one species-consistent animal visible; attribution is unverified

## Hypotheses (experimental)
- `food` [need] -> **insufficient_evidence**: Insufficient evidence 'food' (need) for track-1. Supporting observations: cat vocalization sound 2.40-3.38s (peak raw score 0.80; meow 0.59) [ev-snd-1]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; attributing the sound to this animal is unverified (one visible candidate). Score 0.65 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `outside` [need] -> **insufficient_evidence**: Insufficient evidence 'outside' (need) for track-1. Supporting observations: cat vocalization sound 2.40-3.38s (peak raw score 0.80; meow 0.59) [ev-snd-1]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; attributing the sound to this animal is unverified (one visible candidate). Score 0.65 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `attention` [need] -> **insufficient_evidence**: Insufficient evidence 'attention' (need) for track-1. Supporting observations: cat vocalization sound 2.40-3.38s (peak raw score 0.80; meow 0.59) [ev-snd-1]. Notes: the fusion rules have not been validated on paired household data; all supporting evidence comes from a single source; person detection is disabled; attributing the sound to this animal is unverified (one visible candidate). Score 0.60 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `tense` [state] -> **experimental_hypothesis**: Experimental hypothesis 'tense' (state) for track-1. Supporting observations: growl sound 5.28-6.25s (peak raw score 0.50; growl 0.50, bark 0.11) [ev-snd-2]; low apparent movement of track-1 (median 0.000 image diagonals/s; moving in 0 of 15 sampled steps) [ev-move-track-1]. Notes: the fusion rules have not been validated on paired household data; attributing the sound to this animal is unverified (one visible candidate). Score 0.57 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `warning` [communicative_signal] -> **experimental_hypothesis**: Experimental hypothesis 'warning' (communicative_signal) for track-1. Supporting observations: growl sound 5.28-6.25s (peak raw score 0.50; growl 0.50, bark 0.11) [ev-snd-2]; cat detected (track-1) from 0.00s to 7.50s in 16 sampled frames [ev-track-1]. Notes: the fusion rules have not been validated on paired household data; attributing the sound to this animal is unverified (one visible candidate). Score 0.60 is an uncalibrated heuristic_support_0_1 value, not a probability.
- `relaxed` [state] -> **abstained**: Abstained 'relaxed' (state) for track-1. Supporting observations: low apparent movement of track-1 (median 0.000 image diagonals/s; moving in 0 of 15 sampled steps) [ev-move-track-1]; track-1 in owner-marked zone 'owner-marked rest area (synthetic test zone)' (rest_area) for 7.5s [ev-zone-track-1-rest_area]. Conflicting observations: growl sound 5.28-6.25s (peak raw score 0.50; growl 0.50, bark 0.11) [ev-snd-2]. Notes: the fusion rules have not been validated on paired household data; co-occurring evidence points in a different direction; silence is never used as evidence of relaxation. Score 0.58 is an uncalibrated heuristic_support_0_1 value, not a probability.

## Suggestions
- Hiss- or growl-like sounds were observed together with a visible animal. Giving animals space and not reaching toward them is a common precaution. If such signals are new or frequent, consider asking a veterinarian or a qualified behavior professional.
- Hiss- or growl-like sounds were observed together with a visible animal. Giving animals space and not reaching toward them is a common precaution. If such signals are new or frequent, consider asking a veterinarian or a qualified behavior professional.

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
probe: 55.2, hash: 0.2, decode_audio: 159.8, audio_inference: 23.8, video_decode_detect_track: 629.2, video_detector_inference: 349.9, video_quality_and_camera_motion: 194.1, frame_gap_check: 63.5, fusion: 0.3, model_load_ms_engine_startup: 313.6
