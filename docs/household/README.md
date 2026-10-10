# Household validation package

Status: **ready to start collection; no household data exists yet.** This
package turns [EVALUATION_PLAN_REAL_PAIRED.md](../EVALUATION_PLAN_REAL_PAIRED.md)
(thresholds and splits, fixed in advance) into what a study team needs:

| File | Content |
| --- | --- |
| [COLLECTION_PROTOCOL.md](COLLECTION_PROTOCOL.md) | households, situations and counts, how to record, safety rules, intake checks |
| [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) | what annotators label and how: observed behaviour (operational definitions) separate from interpretation, timing rules, quality control |
| [CONSENT_AND_PRIVACY.md](CONSENT_AND_PRIVACY.md) | consent elements, people and voices in recordings, retention, withdrawal, decisions the data controller must make before collection |
| `recording.schema.json`, `annotation.schema.json` | the exact files per recording (generated from `src/petperception/validation/household.py`; unknown fields are rejected) |
| `uv run petperception household-eval --dataset DIR` | scores app results against the adjudicated annotations with every pre-registered metric, household-level 95 % intervals, and annotator agreement (`--write-schemas` regenerates the schemas) |

Only synchronized audio and video recorded in real homes count as evidence
of behavioural accuracy here. Synthetic pairings (E5), semi-synthetic
sequences (E8) and software scenario tests check that the software behaves
as specified; they are never reported as accuracy.

## What households must answer, and what public recordings already cover

| Capability | Cats: public evidence today | Dogs: public evidence today | Needs household recordings because |
| --- | --- | --- | --- |
| Pet present / detected | web photos (E4, E7); 12 breeds (E16, portraits) | web photos (E4, E7); 25 breeds (E16) | phone angles, low light, clutter, pet-free rooms of the same homes |
| Tracking several pets | none (VidOR labels blocked, no licence) | none | multi-pet homes, identity switches |
| Activity over time (still, moving, active, pacing, entering/leaving) | semi-synthetic only (E8) | semi-synthetic (E8); MammalNet actions (E11, research) | real movement in homes |
| Eating, drinking, grooming, sleeping, play, scratching, litter or elimination | none for cats (no public cat behaviour video) | MammalNet sleep/eat/drink/fight/eliminate (E11, research, YouTube) | cats have no public data at all; dogs only research-licensed YouTube |
| Posture | caption-derived weak labels (E10) | weak labels (E10) | human posture labels on phone video |
| Ears and tail | none | research-only keypoints (E9) | licensed labels; cats have none |
| Facial cues | none | none | expert-defined facial labels on video |
| Vocalization present | ESC-50 clips (E1) | ESC-50 (E1) | home acoustics, distance, TV, other pets |
| Vocalization type | none reachable (AudioSet, VGGSound, FSD50K blocked) | none reachable | type-labelled home audio (or unblocking zenodo.org / YouTube for public data) |
| Vocal context | CatMeows: 21 cats, 3 contexts (E2, macro-F1 0.46 across cats) | none usable (terms unresolved) | owner-reported context at recording time |
| Interaction with a person | web photos (E13; cats not scorable) | web photos (E13) | interaction over time, the animal's response |
| On furniture | web photos (E15); "in front of" cases almost absent | same | in-front-of / beside / under negatives at phone height |
| Pet-pet interaction | none | MammalNet fights (research) | multi-pet homes |
| Household context sounds (doorbell, TV, vacuum) | not evaluated | not evaluated | labelled home audio |
| Interpretations (food, outside, attention, play, alert, tense, warning, relaxed) | none | none | only owner context plus blind annotation can test them |
| Phone result equals reference result | desktop Chromium only | same | phones of the participants (also the phone test, [DEVICE_TEST_PROTOCOL.md](../DEVICE_TEST_PROTOCOL.md)) |

## Workflow

1. Data controller decisions and consent text ([CONSENT_AND_PRIVACY.md](CONSENT_AND_PRIVACY.md)).
2. Recruit households ([COLLECTION_PROTOCOL.md](COLLECTION_PROTOCOL.md));
   assign each household to development or test at enrolment (50/50,
   stratified by species and multi-pet status; never split a household).
3. Participants record with the app (or the phone camera), enter the owner
   context **before** seeing any result, and transfer the recording.
4. Intake checks; the project writes `recordings/<id>.json`.
5. Two annotators label each recording blind to the app output
   (`annotations/<id>.<annotator>.json`); an adjudicator writes
   `adjudicated/<id>.json`.
6. Run the app's engine on every recording: on the participant's phone (the
   result JSON from the app) and with the reference engine
   (`petperception analyze`), stored as `results/<id>.json`.
7. `uv run petperception household-eval --dataset DIR` on development
   households for calibration and thresholds; test households once, at the
   end.

## What exists and what does not

Exists: the result schema both engines produce, the annotation and recording
schemas, the evaluator with tests on a hand-built fixture, in-app feedback
with a research-consent flag, the phone test.

Does not exist yet: a research upload flow in the app (media with consent
to a study server), an annotation tool configuration (any tool that exports
time intervals works, for example ELAN or the VGG Image Annotator; a
converter to `annotation.schema.json` is written per tool), recruited
participants, trained annotators, and the controller decisions listed in
the consent document.
