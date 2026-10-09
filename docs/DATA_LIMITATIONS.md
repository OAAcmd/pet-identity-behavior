# Data Limitations

## No paired household data

There is no verified paired household audio + video + context + outcome dataset
for this project. Consequently:

- multimodal accuracy, intent/need accuracy and hypothesis calibration are
  `not_evaluated`;
- fusion weights, temperature, priors and thresholds are unfitted configuration;
- unrelated public audio and images are **never** paired to claim multimodal
  training or accuracy. The E5 scenarios pair them only for clearly marked
  behavior tests.

## Per-dataset limitations

| Dataset | What it supports | What it cannot support |
| --- | --- | --- |
| ESC-50 | clip-level cat/dog/other sound detection on curated Freesound clips | event timing, household acoustics, cat/dog intent, cross-animal claims (no animal IDs); 4 non-pet sources span two folds |
| Open Images V7 val subset | still-image cat/dog box detection with verified labels | video tracking, behaviour, pose, household camera angles; hard negatives over-represent other animals; possible Flickr overlap with COCO training data |
| Open Images V7 test-split tuning subset (E7) | fitting and thresholding the false-positive verifier | any final metric (never used for one) |
| Ultralytics dog-pose (E9) | dog keypoints on web photos, 112 breeds (Stanford Dogs synsets) | product releases (Research-Only); cats; household frames; puppies; eyes/withers/throat (never annotated) |
| Localized Narratives captions (E10) | weak posture labels for cats and dogs | posture accuracy (caption noise unmeasured); breed analysis (no breed metadata) |
| MammalNet dog-breed segments (E11) | dog behavior classes in real (YouTube) video | product releases (no dataset licence); cats (none); household framing |
| Open Images V7 relationships (E13) | person-pet "holds / interacts with" in still images with reliable negatives | interaction over time; the animal's response; cats (4 scorable final pairs); pet-pet interaction (no such relations) |
| Open Images V7 relationships (E15) | pet on a bed / couch / chair / table in still images, split by photographer | home camera angles (few "in front of" cases in web photos); under-furniture and inside-car (too few positives); objects without a detector class (dog beds, cat trees, crates, bowls); whether the animal is allowed there |
| CatMeows (blocked) | would support cat recording-context classification with cat/owner grouping | not fetched; labels are recording contexts (brushing, isolation, waiting for food), not emotions or verified needs; 8 kHz audio |
| Barkopedia (blocked, unreviewed) | potentially dog activity/environment context | provenance, splits and animal/source grouping unverified; no cross-dog claims |

## Timing and sampling caveats

- Video frames are selected at least `1/sample_fps` apart from the source
  frames, so the effective rate depends on the source frame rate (e.g. 4 fps
  requested on a 10 fps source gives ~3.3 fps).
- Audio events are unions of 0.975 s windows with a 0.48 s hop.
- Dwell times are measured on sampled frames only.

## Label semantics

- `waiting_for_food`, `brushing`, `isolation` (CatMeows) and any Barkopedia
  activity labels keep their original meaning. `isolation` is not fear.
- AudioSet tags such as "Hiss" or "Growling" are acoustic categories, not
  verified emotional states, and "Growling" is shared by cats and dogs.

## Identity

A video track ID is a per-analysis label for linked boxes. Recognizing the same
animal across sessions is a later task and is not attempted.
