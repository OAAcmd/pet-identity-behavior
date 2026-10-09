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
