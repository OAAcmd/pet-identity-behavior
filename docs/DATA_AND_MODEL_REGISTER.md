# Data and Model Register

Machine-readable source of truth: [`manifests/resources.json`](../manifests/resources.json).
Per-file manifests: [`manifests/esc50_files.csv`](../manifests/esc50_files.csv),
[`manifests/openimages_catdog_images.csv`](../manifests/openimages_catdog_images.csv),
[`manifests/openimages_catdog_boxes.csv`](../manifests/openimages_catdog_boxes.csv),
[`manifests/openimages_catdog_sha256.csv`](../manifests/openimages_catdog_sha256.csv).

All facts below were checked on the actual files in this workspace on **2026-10-09**
unless marked *unverified*. The Phase 1 sections are kept as written; everything
acquired or trained afterwards is in [Phase 2 additions](#phase-2-additions) at the end
(including the dog-pose data, which became reachable). Counts were computed from the files, not copied
from papers or dataset cards.

## Retrieval environment

Phase 1 ran in a Claude Code cloud workspace whose egress goes through a
policy proxy. Reachability observed on 2026-10-09:

| Host | Result | Effect |
| --- | --- | --- |
| `pypi.org`, `files.pythonhosted.org` | reachable | dependencies installed and locked |
| `storage.googleapis.com` | reachable | YAMNet, EfficientDet-Lite, Open Images annotations |
| `raw.githubusercontent.com`, GitHub git (anonymous clone) | reachable | ESC-50, YAMNet class map, AudioSet ontology, MediaPipe/TFLite schemas |
| `open-images-dataset.s3.amazonaws.com` | reachable (sporadic 404s, retried) | Open Images images |
| `zenodo.org` | **denied (HTTP 403 at CONNECT)** | CatMeows blocked |
| `huggingface.co` | **denied (HTTP 403 at CONNECT)** | AST candidate and Barkopedia blocked |
| `docs.ultralytics.com`, GitHub release assets | **denied / not served** | dog-pose dataset and checkpoints blocked |
| `download.pytorch.org`, `cocodataset.org`, `*.wikimedia.org`, `www.robots.ox.ac.uk`, `deb.debian.org` | denied | not needed after substitutions below |

To unblock: an environment owner can allow `zenodo.org` and `huggingface.co`
(and optionally `docs.ultralytics.com`, `github.com` release assets) in the
environment's network settings, then run the `next_command` listed per
resource.

## Summary table

| ID | Kind | Status | Use in Phase 1 | Data / weights terms | Code terms |
| --- | --- | --- | --- | --- | --- |
| `yamnet_mediapipe_v1` | model | available | default frozen audio-event backend | weights: Apache-2.0 per the tensorflow/models YAMNet release (MediaPipe build: no standalone card, re-verify); labels: AudioSet ontology CC BY-SA 4.0 | Apache-2.0 |
| `efficientdet_lite0_mediapipe_v1` | model | available | default cat/dog detector | no standalone model card; Apache-2.0 by secondary evidence; COCO-trained; research profile only | Apache-2.0 (MediaPipe) |
| `efficientdet_lite2_mediapipe_v1` | model | available (optional) | E4 comparison only | as Lite0 | Apache-2.0 |
| `ast_audioset_candidate` | model | **blocked** | none | unverified | unverified |
| `esc50` | dataset | available | E1 | CC BY-NC 3.0 (collection), CC BY 3.0 (ESC-10), per-clip Freesound licenses | upstream repo |
| `openimages_v7_val_catdog` | dataset | available | E4, scenario media | annotations CC BY 4.0; images CC BY 2.0 (per-image) | n/a |
| `catmeows` | dataset | **blocked** | E2 implemented, not run | research / non-commercial (Zenodo text) vs CC BY 4.0 (Univ. Milan page): discrepancy, treated as research-only | n/a |
| `barkopedia_dog_act_env` | dataset | **blocked + pending review** | E3 disabled | card claims MIT (*unverified*); media provenance unreviewed | n/a |
| `ultralytics_dog_pose` | dataset/model | **blocked** | none (pose unavailable) | images and checkpoint terms unverified; Ultralytics code AGPL-3.0 | AGPL-3.0 |

## Models

### YAMNet (MediaPipe Audio Classifier build) - `yamnet_mediapipe_v1`

- Canonical: <https://github.com/tensorflow/models/tree/master/research/audioset/yamnet>
- Files (pinned SHA-256):
  - `yamnet.tflite` - <https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite>,
    4,126,810 bytes, `4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf`, GCS generation 1682545011401633.
  - `yamnet_class_map.csv` - tensorflow/models@`34a21326906b9574fa11c4d6d0a5c534ff039267`, 14,096 bytes, `cdf24d19...6df2`.
  - `audioset_ontology.json` - audioset/ontology@`d417d32bf59c711abb5910fd2f76a0eb44697991`, 342,780 bytes, `9c685f44...2d15d`.
- Verified interface (by inspecting the graph): input `[15600]` float32 16 kHz mono waveform;
  log-mel frontend inside the graph (`RFFT2D`, `LOG`); output `[1, 521]` after an in-graph
  `LOGISTIC` op, i.e. **independent sigmoid scores, never softmax**. Internal activations
  are **int8-quantized** despite the "float32" path, so scores are multiples of 1/256.
  The 521 embedded labels match `yamnet_class_map.csv` in order (verified).
  The 1024-d pooled embedding (`tower0/network/layer28/reduce_mean`, int8, dequantized)
  is readable with `experimental_preserve_all_tensors`.
- Preprocessing implemented here: resample to 16 kHz (polyphase), clip to [-1, 1],
  0.975 s windows with 0.48 s hop, zero-pad the final partial window.
- Training data: AudioSet (YouTube). Contamination vs ESC-50 (Freesound) unaudited, believed low.
- Terms: code Apache-2.0; weights released in the Apache-2.0 YAMNet project (the MediaPipe
  page that would state terms for this exact build was not reachable); AudioSet ontology/labels
  CC BY-SA 4.0. Attribution: YAMNet by M. Plakal and D. Ellis (Google).

### EfficientDet-Lite0 / Lite2 (MediaPipe Object Detector builds)

- Canonical: <https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector> (page not reachable from the workspace; content confirmed via web search).
- `efficientdet_lite0.tflite`: 13,836,895 bytes, `40338edf5ec70d43e318b0a716a84d4564cd1802759a7a07170c7e43796dbf58`, GCS generation 1682636012161435.
- `efficientdet_lite2.tflite`: 23,096,891 bytes, `ad2abbf2b4e10585e15176fd7b5ef03c28dda959ae26fc142549fdd1814db91d`.
- Verified interface: input `[1, 320, 320, 3]` (Lite2: 448) float32 RGB normalized
  `(x - 127.5) / 127.5` (parsed from the embedded TFLite `NormalizationOptions`); outputs
  raw anchor tensors `[1, 19206, 4]` (YXHW offsets) and `[1, 19206, 90]` (sigmoid already
  applied in-graph); **no in-graph NMS**. Decoding parameters (19,206 fixed anchors,
  unit scales, exponential box size) are parsed from the embedded MediaPipe
  `DETECTOR_METADATA` flatbuffer using MediaPipe's own schema
  (`object_detector_metadata_schema.fbs`) and reproduce `TensorsToDetectionsCalculator`.
- Exact class map: embedded `labels.txt`, 90 COCO slots with `???` placeholders.
  Index 16 = `cat`, 17 = `dog`, 0 = `person`, 50 = `bowl`, 87 = `teddy bear`.
  There is **no** door, leash, litter box, pet bowl or toy class; `bowl`, `frisbee`,
  `sports ball` and `teddy bear` exist but are not validated for pet contexts and are off by default.
- Training data: COCO 2017. Terms: MediaPipe framework Apache-2.0; no standalone model card
  states the weight license; secondary sources state Apache-2.0. **Research profile only;
  do not assume commercial permission.** Weights are fetched from Google, never committed.

### Blocked / not implemented

- **AST (`MIT/ast-finetuned-audioset-10-10-0.4593`)**: huggingface.co denied. Model card,
  input length/normalization, label map and revision were not verified, so no adapter was
  written (writing one against unverified preprocessing would risk silently wrong inputs).
- **Dog pose (Ultralytics)**: docs and release assets unreachable. No verified cat/dog pose
  checkpoint exists in the workspace; a generic human-pose model is not a substitute.
  Pose features (ear angle, tail height, play-bow candidate, tail oscillation) report
  `available: false` with a reason.

## Datasets

### ESC-50 - `esc50`

- Canonical: <https://github.com/karolpiczak/ESC-50>; retrieved by anonymous git clone at
  revision `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6` (2023-10-13 commit).
- Integrity: 2000 WAV files, 882,088,000 bytes; per-file SHA-256 in
  `manifests/esc50_files.csv`; aggregate `2bcd3b1c5d39f93f16449f6e79fc5e51bfa89d35798c235a6f013cc70797fb79`
  (SHA-256 over sorted `filename:sha256` lines). All 2000 verified before E1.
- Observed: 50 classes x 40 clips; `cat` 40, `dog` 40, other 1920; each fold has 8 cat,
  8 dog, 384 other; 44.1 kHz mono, 5 s; 1524 Freesound source groups; ESC-10 subset 400 clips.
- Grouping fields: `fold`, `src_file` (Freesound recording ID), `take`.
- **Verified grouping finding:** 4 Freesound sources appear in two folds each, always
  under different classes (clock_tick/clock_alarm: 131943, 209698; hen/rooster: 134049, 234879).
  No cat or dog source crosses folds (enforced by `tests/test_splits_and_metrics.py`).
- Labels: manual clip-level labels by the dataset author. ESC-50 is not an intent dataset.
- Terms: collection CC BY-NC 3.0; ESC-10 CC BY 3.0; per-clip Freesound licenses parsed
  from the repository LICENSE for all 2000 clips: CC0 869, CC-BY 901, CC-BY-NC 169,
  CC-Sampling+ 60 (author, license, source URL per clip in the manifest).
- Restrictions: non-commercial research; artifacts fitted on ESC-50
  (`artifacts/audio_event_thresholds.json`, `artifacts/e1_mfcc_catdogother_logreg.json`)
  carry the CC BY-NC 3.0 obligations. Audio is never committed.

### Open Images V7 validation, cat/dog subset - `openimages_v7_val_catdog`

- Canonical: <https://storage.googleapis.com/openimages/web/index.html>.
- Annotation files (SHA-256 pinned in resources.json): `validation-annotations-bbox.csv`
  (25,105,048 B), `oidv7-val-annotations-human-imagelabels.csv` (28,392,297 B),
  `oidv7-class-descriptions-boxable.csv` (12,064 B), `validation-images-with-rotation.csv` (15,245,485 B).
- Observed in the full validation split: 345 images with Cat boxes (389 boxes), 1586 with Dog boxes (1937 boxes).
- Subset (seed 20261009, `petperception fetch-data openimages-catdog`): 823 images =
  all 339 rotation-safe cat-positive images + 350 sampled dog-positive images + all 134
  human-verified negatives (no cat and no dog positive, at least one verified negative);
  16 images excluded because their `Rotation` field makes the box frame ambiguous; 820 boxes.
- Images come from the S3 mirror, which re-encodes the Flickr originals, so `OriginalMD5`
  cannot be used; our SHA-256 of each downloaded file is pinned in `openimages_catdog_sha256.csv`.
- Label provenance: human-verified image labels and human-drawn boxes (Open Images team).
- Grouping: image ID; Flickr author (bootstrap group).
- Terms: annotations CC BY 4.0; every subset image lists CC BY 2.0 with author and landing
  page URL in `openimages_catdog_images.csv`. Images are not committed; reproduce with attribution.

### CatMeows - `catmeows` (BLOCKED)

- Canonical: <https://zenodo.org/records/4008297> (v1.0.2, DOI 10.5281/zenodo.4008297).
- Error: `zenodo.org:443` CONNECT rejected with HTTP 403 (session egress policy), also via
  WebFetch; no complete official mirror reachable (huggingface.co denied; GitHub copies found
  by search contain 3 samples only).
- Documented but **unverified** here: 440 meows, 21 cats (Maine Coon, European Shorthair),
  contexts brushing / isolation in an unfamiliar environment / waiting for food, filename
  scheme `C_NNNNN_BB_SS_OOOOO_RXX`, `extra.zip` with excluded recordings, reported 8 kHz.
- Terms discrepancy: Zenodo text says scientific research / non-commercial; the University
  of Milan record says CC BY 4.0. Treated as research-only, non-commercial until resolved.
- Implemented: MD5-verified fetch via the Zenodo API, strict filename parser (unparseable
  names are reported, never guessed), `extra.zip` kept separate, E2 protocol.
- Next: `uv run petperception fetch-data catmeows && uv run petperception evaluate --experiment e2`.

### Barkopedia Dog Act/Env - `barkopedia_dog_act_env` (BLOCKED, PENDING REVIEW)

- Canonical: <https://huggingface.co/datasets/ArlingtonCL2/Barkopedia_Dog_Act_Env>.
- Error: huggingface.co denied. Card license (MIT per the brief), media provenance, split
  formats, and the existence of animal/source IDs are all unverified.
- Importer is audit-only: `fetch-data barkopedia --audit-only` fetches repository metadata;
  media ingestion stays disabled until `var/data/barkopedia/audit_review.json` marks
  provenance, license, splits, grouping and test labels as reviewed. No dog-intent claim.

## Artifacts produced in Phase 1

| Path | Content | Terms | Durability |
| --- | --- | --- | --- |
| `artifacts/audio_event_thresholds.json` | cat/dog thresholds on raw YAMNet scores (ESC-50, 5 folds) | derived from CC BY-NC 3.0 data | committed (private repo) |
| `artifacts/e1_mfcc_catdogother_logreg.json` | MFCC + logistic regression coefficients (numeric JSON, no pickle) | derived from CC BY-NC 3.0 data | committed (private repo) |
| `var/models/*` | fetched model files | see above | session only; re-fetch with `fetch-models` (checksums pinned) |
| `var/data/*` | raw datasets | see above | session only; re-fetch with `fetch-data` |
| `var/examples/*` | synthetic-pairing scenario media | inherits source terms | session only; regenerate with `evaluate --experiment e5` |
| Docker image `pet-perception:phase1` | CPU runtime | as dependencies | built and tested in session only; not pushed to a registry |

## Phase 2 additions

Reachability changes observed in Phase 2: `github.com/ultralytics/assets` release
downloads and `mammalnet.s3.amazonaws.com` are reachable; `zenodo.org` and
`huggingface.co` remain denied (CatMeows, Barkopedia still blocked).

### Release suitability

Every resource now carries `release_suitability` per release profile
(`research`, `personal_noncommercial`, `commercial`: `suitable`,
`requires_legal_review` or `not_permitted`) in `manifests/resources.json`.
`petperception release-check` reports the decision for every configured asset;
the analysis engine and `export-ondevice` refuse assets that are not allowed in
the active profile. Legal approvals for `requires_legal_review` items are
recorded by a person in a separate approvals file
(`configs/release_approvals.example.yaml`), never by code.

### Models

| ID | Use | File (bytes, SHA-256 prefix) | Weights terms | Profiles (research / personal NC / commercial) |
| --- | --- | --- | --- | --- |
| `efficientnet_lite0_mediapipe_v1` | crop classifier for the verifier; shared feature backbone (posture, keypoints, interaction, video) | 18,582,189, `6c7ab0a6` | Apache-2.0 in the embedded ModelMetadata; ImageNet training-data question open | suitable / suitable / legal review |
| `efficientdet_lite0_tfhub_v1` | licence-confirmed detector candidate (TF Lite Task Library build) | 4,563,519, `2e04c53b` | Apache-2.0 in the embedded ModelMetadata; COCO training-data question open | suitable / suitable / legal review |
| `efficientdet_lite0_mediapipe_int8_v1` | int8 detector evaluated for phones (E12) | 4,602,795, `0720bf24` | as the float MediaPipe build: no primary-source weight licence (none embedded; checked) | suitable / suitable / not permitted |
| `efficientdet_lite0_mediapipe_f16_v1` | float16-weight detector evaluated for phones (E12) | 7,254,339, `4b591000` | as above | suitable / suitable / not permitted |
| `efficientnet_lite0_mediapipe_int8_v1` | int8 backbone evaluated for phones (E12) | 5,434,517, `bc2ffe19` | Apache-2.0 in the embedded ModelMetadata | suitable / suitable / legal review |

Interfaces of the quantized builds (verified by inspection): the int8 detector
takes uint8 input (scale 1/128, zero point 127) and returns float32 raw anchors
like the float build; the int8 classifier takes uint8 input (scale 0.012566,
zero point 131), returns a uint8 softmax, and its intermediate feature tensors
are int8 at the same tensor indices as the float build.

### Datasets

| ID | Use | Content and size (computed here) | Splits used | Terms | Profiles |
| --- | --- | --- | --- | --- | --- |
| `openimages_v7_test_tuning` | E7 verifier fitting and threshold selection | 2,111 Open Images V7 test-split images (cat/dog positives, verified negatives, confusable animals); `manifests/oi_test_tuning_*.csv` | tuning only (never a final metric); disjoint from E4 (asserted) | annotations CC BY 4.0, images CC BY 2.0 | all suitable |
| `ultralytics_dog_pose` | E9 dog keypoints | `dog-pose.zip` 353,152,125 B, SHA-256 `0396a35f...`; 24 keypoint slots, of which 4 (eyes, withers, throat) are never annotated; 6,773 train instances (6,096 fit + 677 development), 1,703 val; Stanford Dogs synsets give 112 breeds | dog-pose train / val | Research-Only (Stanford Dogs / ImageNet terms; archive ships the Ultralytics AGPL-3.0 text) | research only |
| `localized_narratives_captions` | E10 weak posture labels | caption files for OI train/test/validation (SHA-256 pinned); 2,834 candidate images in `manifests/posture_weak_images.csv`, 2,119 with exactly one accepted detection | OI train / test / validation | CC BY 4.0 (captions), CC BY 2.0 (images) | all suitable |
| MammalNet dog-breed segments (`petperception fetch-data mammalnet`) | E11 dog behavior from video | `annotation.tar` SHA-256 `055af7e2...`; genus Canis with domestic-breed keywords (bulldog, terrier, malamute, hound, husky dog, shepherd dog, eskimo dog, foxhound), classes sleeps / eats food / drinks water / fights / pees or poops; up to 3 segments per video, at most 30 s each; 813 segments requested | MammalNet's own training / validation / testing | code MIT; **no dataset licence published**; videos are third-party YouTube uploads, streamed and never stored | research only |
| Open Images V7 visual relationships, person-pet pairs (`fetch-data openimages-vrd`) | E13 person-pet interaction | source files pinned in `manifests/oi_vrd_sources_sha256.csv`; 3,089 images, 11,598 (person, pet) box pairs; positives: train 2,219 / validation 47 / test 145 | OI train (10 % image hold-out for development) / OI validation + test (final) | annotations CC BY 4.0, images CC BY 2.0 (attribution in `manifests/oi_vrd_interaction_images.csv`) | all suitable |

Notes on the relationship data: relationship boxes are drawn separately from
the box annotations; each relation was mapped to the annotated boxes of the
same classes by IoU (all matched with IoU ≥ 0.91). Open Images states that
relationship annotations are exhaustive for annotated pairs, which is what
makes unrelated person-pet pairs reliable negatives. There are no pet-pet
relationships and no pet attributes in the files.

### Trained artifacts (this repository)

See [`artifacts/README.md`](../artifacts/README.md) for checksums and
[TRAINING_PROGRAM.md](TRAINING_PROGRAM.md) for reproduction.

| Artifact | Trained on | Profiles |
| --- | --- | --- |
| `verifier/verifier_<detector>.json` | OI test-split tuning images | all suitable (backbone terms apply at runtime) |
| `keypoints/dog_keypoints_head_v1.npz` | dog-pose | research only |
| `posture/posture_v1.npz` | OI train + Localized Narratives weak labels | all suitable (backbone terms apply) |
| `behavior_video/dog_behavior_video_v1.npz` | MammalNet dog segments | research only |
| `interaction/person_pet_interaction_v1.npz` | OI visual relationships | all suitable (backbone terms apply) |
