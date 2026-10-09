# Model card - Pet Observer analysis engine (result schema 1.2.0)

## Summary

Two implementations of one analysis engine - TypeScript in the phone's
browser (default) and Python on the server (opt-in server analysis) - turn a
short audio/video clip into timestamped **observations** of cats and dogs
(tracks, sounds, movement, posture, behavior segments, owner-place
occupancy, person interaction candidates, quality limits) and, only when
independent co-occurring evidence exists, **experimental, uncalibrated
hypotheses** about possible needs, activities, states or signals. It
abstains by default.

It is **not** a pet translator, emotion detector, identity recognizer or
diagnostic tool. Track IDs are per-analysis labels. No hypothesis has been
validated against household outcomes.

## Components

| Component | Model / method | Trained here? | Evaluated here | On the phone |
| --- | --- | --- | --- | --- |
| Sound events | YAMNet (AudioSet), ontology-derived cat/dog groups; thresholds from ESC-50 | thresholds | E1 (ESC-50) | yes |
| Detection (cats, dogs, people) | EfficientDet-Lite0 (COCO), frozen | no | E4, E7, E12 | yes |
| False-positive verifier | EfficientNet-Lite0 ImageNet group masses + per-class logistic regression | yes (E7) | E7, E12 | yes |
| Tracking | Kalman (SORT-style) with camera-motion compensation | no (rules; selected on E8) | E8 (semi-synthetic) | yes |
| Behavior segments | operational definitions with a-priori thresholds | no | E8 (semi-synthetic) | yes |
| Posture (lying / sitting / standing) | logistic regression on EfficientNet-Lite0 crop embedding | yes (E10, weak labels) | E10, E12 | yes |
| Person-pet interaction | logistic regression on box geometry | yes (E13) | E13 (dogs) | yes |
| Dog keypoints (ears, tail, limbs) | JAX head on frozen EfficientNet-Lite0 features | yes (E9) | E9 | no (research-only data) |
| Dog actions from video | logistic regression on segment features | yes (E11) | E11 | no (research-only data) |
| Fusion and abstention | rule baseline (unfitted) | no | E5 synthetic checks only | yes |
| Explanations | deterministic templates; no language model | no | tests | yes |

## Intended use

- Owners recording short clips of their own cat or dog at home, to see what
  was observable and, cautiously, what it might relate to.
- Out of scope: medical or welfare decisions, claims about what an animal
  "says" or "feels", identity verification, surveillance of people,
  commercial deployment without the recorded legal sign-off.

## Measured performance (public data; none of it household)

- Sound events, ESC-50: macro-F1 0.895 [0.846, 0.935].
- Detection, Open Images: AP50 cat 0.92, dog 0.84; with the verifier,
  specificity on verified-negative photos cat 0.70, dog 0.87 at image recall
  0.91 / 0.87.
- Tracking, semi-synthetic: IDF1 0.86.
- Posture vs caption-derived labels: macro-F1 0.65 (38 % of animals reported
  at 85 % agreement with the gate).
- Person-pet interaction, Open Images (dogs, scored pairs): AP 0.91;
  end-to-end recall 0.42 because people or pets are often not detected.
- Dog keypoints (research): PCK@0.1 0.85; dog actions from video (research):
  macro-F1 0.64.
- Phone vs server on identical files (desktop browser): decisions 34/34,
  segments F1 0.97.
- Not measured: anything in real homes, on real phones, cat interaction,
  vocalization types, hypothesis accuracy, calibration.

## Known failure modes

- Sound tags: snoring, crying babies, door creaks and farm animals can trigger
  cat/dog events; TV audio is indistinguishable from a present animal unless
  no animal is visible.
- Small, occluded or poorly lit animals are often missed; foxes and other
  carnivores can be confused with dogs or cats (the verifier rejects many,
  and rejects some real pets - reported as a quality flag).
- Posture labels are unstable near the report gate (about one frame in five
  flips between the phone and server decoders).
- Interaction candidates are still-image judgements per frame; a person
  standing next to a pet can trigger them; most held pets are missed when
  the detector does not separate the person and the animal.
- Quantized (int8) builds can make different per-image verification
  decisions than float32 builds (E12); the default uses float32.
- Camera-motion compensation is approximate; zone membership uses the box
  bottom centre.

## Ethical considerations

- Analysis runs on the user's phone by default; media are uploaded only
  through an explicit server-analysis setting.
- People in frames are detected as boxes to find interaction candidates on
  the analysing device; person boxes are not stored in results, and there is
  no face or identity processing.
- Outputs are observations first; interpretations are labelled experimental
  with their evidence; suggestions avoid diagnoses and point to a
  veterinarian for health concerns.

## Licensing and restrictions

Per-asset terms and release-profile suitability are in
[DATA_AND_MODEL_REGISTER.md](DATA_AND_MODEL_REGISTER.md) and enforced in
code. Research-only data (dog keypoints, MammalNet) never enters product
profiles.
