# Model Card - Pet Perception Engine, Phase 1 (engine 0.1.0, result schema 1.0.0)

## Summary

A modular, CPU-only research engine that turns a local audio/video/image file
into timestamped **observations** of cats and dogs (visual tracks, sound events,
movement, owner-zone occupancy, quality limits) and, only when independent
co-occurring evidence exists, **experimental, uncalibrated hypotheses** about
possible needs, activities, states or signals. It abstains by default.

It is **not** a pet translator, emotion detector, identity recognizer or
diagnostic tool. Track IDs are per-analysis labels. No hypothesis has been
validated against household outcomes.

## Components

| Component | Model / method | Trained here? | Evaluated here |
| --- | --- | --- | --- |
| Sound events | YAMNet (MediaPipe TFLite, AudioSet-trained), frozen; ontology-derived cat/dog groups; thresholds from ESC-50 | thresholds only | E1 (ESC-50, clip level) |
| CPU audio baseline | MFCC/spectral stats + logistic regression | yes (ESC-50) | E1 |
| Visual detection | EfficientDet-Lite0 (MediaPipe TFLite, COCO-trained), frozen | no | E4 (Open Images V7 val subset) |
| Tracking | IoU + Hungarian, time-gap limit, camera-translation compensation | no (rules) | functional only (E5 scenarios) |
| Pose | interface only; no verified animal pose model | - | not evaluated |
| Fusion | rule baseline, weighted average, context-gated PoE (unfitted) | no | behavior checks only (E5); accuracy not evaluated |

## Intended use

- Research and prototyping of an observation-first interface (Phase 2 PWA).
- Offline analysis of short clips supplied by an owner, with optional zones.
- Out of scope: medical or welfare decisions, claims about what an animal
  "says" or "feels", identity verification, surveillance of people, commercial
  deployment (third-party terms not cleared).

## Measured performance (see docs/ALGORITHM_REPORT.md)

- E1 cat/dog/other on ESC-50 (5 source-preserving folds, nested thresholds):
  YAMNet tags macro-F1 0.895 [0.846, 0.935]; cat F1 0.79, dog F1 0.90.
  CPU MFCC baseline 0.617 [0.551, 0.677]; majority 0.327.
- E4 detection on Open Images V7 validation subset (Lite0, threshold 0.35):
  cat AP@0.5 0.923 [0.897, 0.947], dog 0.842 [0.789, 0.886]; specificity on
  human-verified hard negatives cat 0.62, dog 0.56 (foxes, lions and other
  animals trigger detections).
- E6 runtime: 8 s 640x480 video in 0.95 s on 4 vCPUs (real-time factor 0.12).
- Not measured: household audio, cat-context (CatMeows blocked), dog context,
  tracking accuracy, pose, multimodal/hypothesis accuracy, calibration of any
  hypothesis score.

## Known failure modes

- Sound tags: snoring, crying babies, door creaks and farm animals can trigger
  cat/dog events; TV/speaker audio is indistinguishable from a present animal
  unless a visible-animal check fails (`no_visible_animal`).
- 8 kHz or heavily compressed audio is out of domain (flagged).
- Small (< 10% of frame), occluded, or poorly lit animals are often missed;
  foxes and other carnivores are confused with dogs/cats.
- Camera motion compensation is translation-only; zooms, rotation and parallax
  leave residual apparent motion.
- Zone membership uses the box bottom-center; perspective can place an animal
  "in" a floor zone it is behind.
- Event boundaries are quantized to 0.975 s windows (onsets up to ~0.5 s early).

## Ethical considerations

- Outputs are phrased as observations; hypotheses are labelled experimental with
  reasons; suggestions avoid diagnoses and advise professional help for concerns.
- Uploaded media is deleted after analysis; results are deletable on demand.
- People appearing in frames are not analysed unless `person` context detection
  is explicitly enabled (off by default); no face or identity processing.

## Licensing and restrictions

Research profile. YAMNet and EfficientDet weights are third-party (Apache-2.0
claimed; EfficientDet MediaPipe build lacks a standalone model card). Thresholds
and the MFCC model derive from ESC-50 (CC BY-NC 3.0). See
docs/DATA_AND_MODEL_REGISTER.md. Do not assume commercial permission.
