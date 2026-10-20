# Model card - Pet Observer analysis engine (result schema 1.2.0)

## Summary

Two implementations of one analysis engine - TypeScript in the phone's
browser (default) and Python on the server (opt-in server analysis) - turn a
short audio/video clip into timestamped **observations** of cats and dogs
(tracks, sounds, movement, posture, behavior segments, owner-place
occupancy, person interaction and on-furniture candidates, quality limits) and, only when
independent co-occurring evidence exists, **experimental, uncalibrated
hypotheses** about possible needs, activities, states or signals. It
abstains by default.

It is **not** a pet translator, emotion detector, identity recognizer or
diagnostic tool. Track IDs are per-analysis labels. No hypothesis has been
validated against household outcomes.

## Components

| Component | Model / method | Trained here? | Evaluated here | In the phone engine (browser; phones untested) |
| --- | --- | --- | --- | --- |
| Sound events | YAMNet (AudioSet), ontology-derived cat/dog groups; thresholds from ESC-50 | thresholds | E1 (ESC-50) | yes |
| Detection (cats, dogs, people) | EfficientDet-Lite0 (COCO), frozen | no | E4, E7, E12 | yes |
| False-positive verifier | EfficientNet-Lite0 ImageNet group masses + per-class logistic regression | yes (E7) | E7, E12 | yes |
| Tracking | Kalman (SORT-style) with camera-motion compensation | no (rules; selected on E8) | E8 (semi-synthetic) | yes |
| Behavior segments | operational definitions with a-priori thresholds | no | E8 (semi-synthetic) | yes |
| Posture (lying / sitting / standing) | logistic regression on EfficientNet-Lite0 crop embedding | yes (E10, weak labels) | E10, E12 | yes |
| Person-pet interaction | logistic regression on box geometry | yes (E13) | E13 (dogs) | yes |
| Pet on furniture (bed, couch, chair, table) | logistic regression on box geometry, species, object group | yes (E15) | E15 | yes |
| Dog keypoints (ears, tail, limbs) | JAX head on frozen EfficientNet-Lite0 features | yes (E9) | E9 | no (research-only data) |
| Dog actions from video | logistic regression on segment features | yes (E11) | E11 | no (research-only data) |
| Fusion and abstention | rule baseline (unfitted) | no | E5 synthetic checks only | yes |
| Situation from synchronized audio + video (intent stage, optional) | logistic heads on frozen YAMNet / EfficientNet-Lite0 / detector features + trained late fusion with availability flags | yes (E28) | E28 (Kinetics, YouTube) | yes, when exported with `--intent` (research profile) |
| Dog vocal activity / intent (intent stage) | logistic heads on YAMNet features, optional location context | yes (E27) | E26, E27 (Barkopedia) | yes (class-score head) |
| Behaviour/intent combination | taxonomy relations (annotated / consistent / context), thresholds fitted per channel, explicit missing inputs and conflicts | heads yes; combination no (no intent-labelled synchronized data) | channel level only; end-to-end status on held-out clips | yes |
| Explanations | deterministic templates; no language model | no | tests | yes |
| Behaviour explanation in context (`petperception explain`, research profile) | frozen SigLIP base zero-shot behaviour recognition on the animal crop (thresholds from E31), YAMNet for sounds, deterministic question parser, source-grounded knowledge base (24 sources) | thresholds only (a trained head was evaluated and not adopted) | E31 (300 real clips, 37-question benchmark) | no (SigLIP 812 MB) |

## Intended use

- Owners recording short clips of their own cat or dog at home, to see what
  was observable and, cautiously, what it might relate to.
- Out of scope: medical or welfare decisions, claims about what an animal
  "says" or "feels", identity verification, surveillance of people,
  commercial deployment before the per-asset questions in
  [ASSET_TERMS.md](ASSET_TERMS.md) (Q-T, Q-A2) are decided and recorded.

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
- Pet on furniture, Open Images (test split by photographer, scored pairs):
  AP 0.97, precision 0.97; end-to-end recall 0.40 because pets (mostly
  small dogs) or the furniture are often not detected. "Under furniture"
  and "inside a car" failed the pre-set gate and are not shipped.
- Dog keypoints (research): PCK@0.1 0.85; dog actions from video (research):
  macro-F1 0.64.
- Phone vs server on identical files (desktop browser): decisions 38/38,
  segments F1 0.98 (2026-10-19, current engine).
- Home phone videos of pets (CoP3D, E20): the phone build finds the cat in
  0.70 and the dog in 0.60 of frames; 14 % of dog frames show only a "cat".
- Independent sound clips (FSD50K, E21, nothing tuned): cat sound present
  F1 0.69, dog 0.73. Vocalization types: meow 0.73 and bark 0.73 pass the
  pre-set gate. Purr (0.70, interval too wide) and growl (0.49) do not. Cat
  hiss could not be measured (two clips). Only meow and bark are shown, and
  only they can start or support an interpretation. Unvalidated types stay
  in the result file and can only make an interpretation abstain.
- Cat keypoints (research, AwA-Pose, E24): PCK@0.1 0.76; ears 0.91, tail 0.51.
- Situation from synchronized audio + video (Kinetics validation, 294
  clips, E28): macro-F1 0.730 combined, 0.689 video only, 0.456 audio only;
  the scene is the strongest cue. Coverage 0.88 at selective accuracy 0.76.
- Dog activity / intent from a single bark (Barkopedia, E27): macro-F1
  0.22 (8 activities), 0.29 with the location as context; no activity
  passes the gate, so bark readings are at most "tentative".
- No intent hypothesis reaches "supported" on held-out data; what the
  intent stage adds is the situation as context, missing inputs and
  conflicts.
- Not measured: anything in real homes with labels, on real phones, cat
  interaction, hypothesis (intent) accuracy, calibration on new data.

## Known failure modes

- Sound tags: snoring, crying babies, door creaks and farm animals can trigger
  cat/dog events; TV audio is indistinguishable from a present animal unless
  no animal is visible. On FSD50K, purr-like scores fire on engines and
  motorcycles and the hiss score on trains and traffic, which is why those
  types are not shown (E21).
- Dogs in home videos are often reported as cats (E20).
- Small, occluded or poorly lit animals are often missed; foxes and other
  carnivores can be confused with dogs or cats (the verifier rejects many,
  and rejects some real pets - reported as a quality flag).
- Posture labels are unstable near the report gate (about one frame in five
  flips between the phone and server decoders).
- Interaction candidates are still-image judgements per frame; a person
  standing next to a pet can trigger them; most held pets are missed when
  the detector does not separate the person and the animal.
- On-furniture candidates come from 2-D box containment: a pet on the floor
  *in front of* a sofa or bed can be reported as on it (seen in the E15
  audit); furniture detected near the threshold makes the segments
  fragmentary.
- Quantized (int8) builds can make different per-image verification
  decisions than float32 builds (E12); the default uses float32.
- Camera-motion compensation is approximate; zone membership uses the box
  bottom centre.
- Behaviour explanation (E31): tail chasing is recognized in only 3 of 8
  real clips (a small cat on a suitcase and a hyperaesthesia cat on a play
  mat were missed), so answers about it rest on the owner's description;
  threat displays, tail or flank biting, staring and tail flicking were not
  recognized; a dog rubbing itself on the carpet was read as scratching.
  The question parser misses paraphrases (27 of 34 parsed) and context words
  such as "after eating" can be taken for a reported behaviour. Behaviours
  or explanations outside the knowledge base get the nearest entry, which
  can be wrong (scratch reflex, trained barking).

## Ethical considerations

- Analysis runs on the user's phone by default; media are uploaded only
  through an explicit server-analysis setting.
- People in frames are detected as boxes to find interaction candidates on
  the analysing device; person boxes are not stored in results, and there is
  no face or identity processing. Furniture boxes are likewise not stored.
- Outputs are observations first; interpretations are labelled experimental
  with their evidence; suggestions avoid diagnoses and point to a
  veterinarian for health concerns.

## Licensing and restrictions

Per-asset terms and release-profile suitability are in
[DATA_AND_MODEL_REGISTER.md](DATA_AND_MODEL_REGISTER.md) and enforced in
code. Research-only data (dog keypoints, MammalNet) never enters product
profiles.
