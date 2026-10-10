# Asset usage terms, artifact by artifact

One row per model, dataset, derived artifact and shipped software component:
what its terms are, the evidence for them, what it is used for, which release
profiles it is cleared for, and, where something is open, the exact question
and the replacement that would make the question moot. Machine-readable
counterparts: `manifests/resources.json` (enforced by
`petperception release-check` and by the bundle exporter) and
`manifests/data_sources.json` (the 94-source survey).

Evidence levels used below:

* **Stated (primary)**: read from the asset itself or its publisher's page
  (for example the licence field embedded in a `.tflite` file).
* **Secondary**: from third-party pages, mirrors or search snippets; the
  primary page could not be read here (the blocked host is named).
* **Not stated**: no licence statement found at the primary source.

Profiles: `research` (internal evaluation), `personal_noncommercial` (the
phone test build and a free personal-use app), `commercial`.
Checked on 2026-10-10.

## What the phone test build actually ships

The build published for real-phone acceptance
(https://claude.ai/artifact/NRG1a19X1ZkPbgdigJGaf2, version 4, profile
`personal_noncommercial`, bundle `2dd9e6b541aeffde`, 26.6 MB) contains
exactly:

* models M1 (YAMNet), M3 (TF Hub EfficientDet-Lite0, raw decoding) and M4
  (EfficientNet-Lite0 features);
* parameters A1, A3, A4, A7 and A8;
* the LiteRT.js runtime and Preact (S1, S2);
* one reference clip made from two CC BY 2.0 photos (P1).

Every one of these is cleared for this profile. The page lists their
credits.

Until 2026-10-10 the build carried M2 (the MediaPipe detector). It was
replaced when M2's missing licence statement was confirmed (Q-M2). M2 stays
available for research and comes back if Q-M2 is answered. The
research-only artifacts A5, A6 and A10 are excluded by the exporter for
this profile, as the bundle manifest notes.

## Models

| # | Asset | Weights licence | Training data and its terms | Used for | research | personal_noncommercial | commercial |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | YAMNet, MediaPipe Audio Classifier build v1 (`yamnet_mediapipe_v1`) | **Apache-2.0, stated (primary)**: licence field of the TFLite metadata embedded in the pinned file (author "MediaPipe"). Google also publishes YAMNet in the Apache-2.0 `tensorflow/models` repository. | AudioSet: YouTube audio, not redistributed; AudioSet labels CC BY 4.0, ontology CC BY-SA 4.0 | sound events (cat, dog, sub-types) on both engines; phone bundle | yes | yes | open: Q-T |
| M2 | EfficientDet-Lite0, MediaPipe Object Detector build v1, float32 (`efficientdet_lite0_mediapipe_v1`; same for the int8 / float16 builds and Lite2) | **Not stated.** The embedded metadata names author "MediaPipe" (metadata v1.5.0) and has no licence field. The MediaPipe model-card bucket (`gs://mediapipe-assets`, 9 cards listed) has no card for it. The MediaPipe Object Detector page on ai.google.dev is blocked here. | COCO 2017 (per MediaPipe documentation): annotations CC BY 4.0, images under per-image Flickr licences, including non-commercial ones | cat/dog/person/furniture detection in the Python reference engine and evaluations E4-E18; phone bundle until 2026-10-10 | yes | open: Q-M2 | not permitted (licence not stated) |
| M3 | EfficientDet-Lite0, TF Lite Task Library build with raw-output decoding (`efficientdet_lite0_tfhub_rawdecode_v1`) | **Apache-2.0, stated (primary)**: licence field of the embedded TFLite metadata | COCO 2017, as M2 | detection in the phone bundle since 2026-10-10; runs in the browser bit-exact (E14) | yes | yes | open: Q-T |
| M4 | EfficientNet-Lite0, MediaPipe Image Classifier build v1 (float32; int8 likewise) | **Apache-2.0, stated (primary)**: embedded metadata | ImageNet-1k (ILSVRC 2012). The ImageNet terms of access limit the images to non-commercial research and education. | detection verifier, posture features; phone bundle | yes | yes | open: Q-T |
| M5 | AST AudioSet candidate (`ast_audioset_candidate`) | not checked (huggingface.co blocked) | AudioSet | not used | - | - | - |

**Q-M2 (one question, one owner action).** Does Google state terms for
`storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/*`?
The answer would be on the "Models" section of
`https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector`
(blocked here, `manifests/network_requirements.json`). Anyone with a browser
can read it in a minute.

* If it states Apache-2.0 (as for the classifier and YAMNet builds), M2 is
  cleared for `personal_noncommercial` and can return to the phone build.
* If it states nothing, the phone build keeps M3. The build already uses
  it.

What switching costs, measured on the final E4 split with the verifier (E14):

| | Cat recall | Cat specificity | Dog recall | Dog specificity | AP50 cat / dog | Engine parity, segments |
| --- | --- | --- | --- | --- | --- | --- |
| M2 (MediaPipe) | 0.906 | 0.702 [0.57, 0.83] | 0.872 | 0.865 [0.80, 0.92] | 0.847 / 0.758 | 0.966 |
| M3 (TF Hub, raw decoding) | 0.885 | 0.660 [0.52, 0.79] | 0.866 | 0.919 [0.86, 0.97] | 0.833 / 0.748 | 0.91 |

The confidence intervals overlap. M3 is internally uint8, so its phone
timings differ from M2's. The phone test now runs the M3 bundle:

```
export-ondevice --release-profile personal_noncommercial --detector efficientdet_lite0_tfhub_rawdecode_v1 --out var/device_test/bundle
export-device-test --bundle var/device_test/bundle
```

In desktop Chromium the detector takes 74 ms per frame, single-threaded
WebAssembly. That is a desktop figure, not a phone figure.

**Q-T (one question for M1, M3 and M4; commercial profile only).** The
weights carry Apache-2.0. Their training data carries narrower terms:
ImageNet's non-commercial terms of access, the per-image Flickr licences of
COCO (some non-commercial), and YouTube audio for AudioSet. The question is
whether the publisher of a commercial app accepts shipping weights trained
on such data. This is a publisher's risk decision, recorded in
`configs/release_approvals.yaml` (`approved_by`, `reference`). It does not
affect research or the personal non-commercial build. The replacement that
removes the question is job G1 in `docs/TRAINING_PROGRAM.md`: a detector
and backbone trained only on Open Images images under CC BY 2.0.

## Datasets

| # | Dataset | Terms and evidence | How obtained here | Used for | Profiles of what is derived from it |
| --- | --- | --- | --- | --- | --- |
| D1 | ESC-50 | Collection CC BY-NC 3.0 (stated, repository LICENSE). Per-clip Freesound licences, parsed into `manifests/esc50_files.csv`: CC0 869, CC BY 901, CC BY-NC 170, Sampling+ 60. | GitHub, pinned commit and per-file SHA-256 | E1 evaluation; thresholds A1, A2 | research, personal_noncommercial; commercial only via A2 |
| D2 | CatMeows v1.0.2 (Zenodo 4008297) | The Zenodo record (search snippet; zenodo.org blocked) says "open access for scientific research and non-commercial purposes", with citation required. The University of Milan repository says CC BY 4.0 (secondary). | Unofficial GitHub mirror at a pinned commit, 440 files with per-file SHA-256 pins. Whether the mirror matches Zenodo is unverified, because Zenodo's checksums cannot be read here. | E2 evaluation (G4 cats); artifact A10 | research. A10 is not shipped for accuracy reasons, so its terms block nothing now. |
| D3 | Open Images V7 subsets (validation detection set; test tuning set; visual relationships person-pet and pet-object; Localized Narratives captions) | Annotations CC BY 4.0, images CC BY 2.0 (stated, Google). The licence and author of every image used are recorded in the subset manifests. | storage.googleapis.com and open-images-dataset S3, SHA-256 pinned | E4, E7, E10, E13, E15; artifacts A3, A4, A7, A8; reference clip P1 | all three |
| D4 | Ultralytics dog-pose (Stanford Dogs images, StanfordExtra-style keypoints) | Documented "Research-Only" by Ultralytics. The images are ImageNet-derived (Stanford Dogs). | GitHub release asset, pinned | E9; A5 | research only |
| D5 | MammalNet (domestic-dog videos) | Code MIT. No dataset licence is published. Videos are third-party YouTube uploads. | annotations from mammalnet S3; video streamed frame by frame, never stored | E11; A6 | research only |
| D6 | Oxford-IIIT Pet | Annotations CC BY-SA 4.0 (secondary; robots.ox.ac.uk blocked). Image copyright stays with the owners. | fast.ai S3 mirror, SHA-256 pinned | E16 breed robustness (evaluation only) | nothing derived is shipped |
| D7 | Kinetics-700-2020, 6 pet-action validation classes (294 clips) | Annotations CC BY 4.0 (secondary; the DeepMind page is blocked). Clips are YouTube uploads whose rights stay with the uploaders. | CVDF mirror on S3, SHA-256 pinned; not committed | E18 full-engine evaluation (evaluation only) | nothing derived is shipped |
| D8 | COCO 2017 | Annotations CC BY 4.0. Images under per-image Flickr licences (ids 1-3 non-commercial; 4, 5, 7, 8 allow commercial use). | not obtained (images.cocodataset.org blocked) | training data of M2 and M3 only | - |
| D9 | AudioSet | Labels CC BY 4.0. Media is YouTube. | label CSVs only | training data of M1 | - |

## Derived artifacts (committed under `artifacts/`)

| # | Artifact | Fitted on | research | personal_noncommercial | commercial | Note |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | `audio_event_thresholds.json` (cat 0.738, dog 0.801) | D1, all 2,000 clips | yes | yes | no | collection is non-commercial |
| A2 | `audio_event_thresholds_cc.json` (cat 0.5, dog 0.801), new | D1, the 1,770 CC0 / CC BY clips only (E1-CC) | yes | yes | open: Q-A2 | see below |
| A3 | `verifier/verifier_<detector>.json` | D3 test tuning set (CC BY) | yes | yes | yes; inherits M2/M3/M4 status | |
| A4 | `posture/posture_v1.npz` | D3 Localized Narratives + images | yes | yes | yes; inherits M4 status | |
| A5 | `keypoints/dog_keypoints_head_v1.npz` | D4 | yes | no | no | replacement: G2 (new annotation) |
| A6 | `behavior_video/dog_behavior_video_v1.npz` | D5 | yes | no | no | replacement: G3/G5 on household video |
| A7 | `interaction/person_pet_interaction_v1.npz` | D3 relationships | yes | yes | yes | |
| A8 | `object_relation/pet_object_relation_v2.npz` | none: a fixed containment rule, evaluated on D3 | yes | yes | yes | v1 (learned on D3) has the same terms; not shipped |
| A9 | `e1_mfcc_catdogother_logreg.json` | D1 | yes | yes | no | baseline only |
| A10 | `e2_catmeows_context_logreg.json` | D2 | yes | not shipped | no | weak model (E2) |

**Q-A2.** A2's audio is CC0 or CC BY clip by clip, with every author and
URL listed in `manifests/esc50_files.csv`. But the cat/dog/other labels used
to fit the two numbers are ESC-50's own, and the collection is CC BY-NC 3.0.
The question is whether two scalar thresholds fitted against those labels
are bound by the collection's NC term.

The label-independent route avoids the question: re-label the 1,770 clips
from the uploaders' own Freesound titles and tags, which are published with
each sound under its licence, and listen only where the tags are ambiguous
or mention a cat or dog. One person can do this in a few hours. A second
rater on the cat/dog candidates gives an agreement figure.

Measured cost of A2 against A1, with nested fold-wise thresholds pooled over
the 5 held-out ESC-50 folds (`results/e1/licence_subset.json`):

* Dog: identical. All 40 dog clips are CC0 or CC BY; F1 0.902 for both.
* Cat: F1 0.763 against 0.790 on all clips (precision 0.81 / 0.78, recall
  0.73 / 0.80). Only 29 of the 40 cat clips are CC0 or CC BY.

## Software in the phone build

| # | Component | Licence | Evidence |
| --- | --- | --- | --- |
| S1 | `@litertjs/core` 2.5.3, including the LiteRT WebAssembly runtime | Apache-2.0 | package metadata (`web/node_modules/@litertjs/core/package.json`) |
| S2 | `preact` 10.29.8 | MIT | package metadata |

The server and reference engine (not part of the phone build) use numpy,
SciPy, scikit-learn, soundfile and psutil (BSD); librosa (ISC); OpenCV
headless, ai-edge-litert and python-multipart (Apache-2.0); and pydantic,
PyYAML and FastAPI (MIT). uvicorn is BSD-3-Clause. The read-out is from the
installed package metadata.

The server container image installs Ubuntu's `ffmpeg` package, which is a
GPL build. Running the image as a service is not distribution. Handing the
image to someone else is, and then requires the GPL source offer for
ffmpeg. The phone build contains no ffmpeg.

## Media in the phone build

| # | Item | Licence | Attribution |
| --- | --- | --- | --- |
| P1 | Reference clip: an Open Images cat photo (00dd6d717c50e78b) moved over an Open Images room photo (00a55e292e46550a) | CC BY 2.0, both photos | "hakim azahari" (flickr.com/photos/stapakspetak/3794735607) and "Andrew Michael Nathan" (flickr.com/photos/digitaldrew/4929845469). Shown in the page's Credits section and in `device_check/ATTRIBUTION.json`. Added in this round: the background author was missing before (`manifests/e8_backgrounds.csv` now records the author, licence and source page of all 60 background photos). |

## Release blockers stated precisely

| Profile | Blocker | What clears it |
| --- | --- | --- |
| personal_noncommercial | none for the shipped build (M1, M3, M4, A1, A3, A4, A7, A8, S1, S2, P1) | - |
| personal_noncommercial, optional | Q-M2: using M2 instead of M3 (better parity, slightly better cat specificity) needs its licence statement | read the ai.google.dev Models section (one minute in a browser) and record it in `configs/release_approvals.yaml` |
| commercial | Q-T: training-data terms of M1, M3, M4 | a publisher decision recorded in `configs/release_approvals.yaml`, or G1 (Open-Images-only detector and backbone) |
| commercial | A1 not permitted | A2 plus Q-A2 (or its re-labelling route) |
| commercial and personal_noncommercial | dog ears/tail/keypoints (A5) and dog video behaviour (A6) are research-only | G2 and G3/G5 in `docs/TRAINING_PROGRAM.md` (new licensed annotation; household video) |
