# Data and Model Register

Machine-readable source of truth: [`manifests/resources.json`](../manifests/resources.json).
**Current terms, artifact by artifact, with the open questions and their
replacements: [ASSET_TERMS.md](ASSET_TERMS.md).** Where a Phase 1 or Phase 2
row below differs from it (for example CatMeows and dog-pose were later
obtained, and the MediaPipe detector licence was checked in the file
itself on 2026-10-10), ASSET_TERMS.md and `manifests/resources.json` are
current. The 94-source survey is [DATA_SOURCES.md](DATA_SOURCES.md).
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
| `barkopedia_dog_act_env` | dataset | available (research) | E26, E27 (dog audio channel) | MIT (card at revision 60a721a7, read 2026-10-20); source-audio provenance not stated (unresolved permission) | n/a |
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

### Barkopedia Dog Act/Env - `barkopedia_dog_act_env` (available, research)

- Canonical: <https://huggingface.co/datasets/ArlingtonCL2/Barkopedia_Dog_Act_Env>, revision
  `60a721a7f9477634971f6b3df29dc2a5e56994d8`; archives match the published LFS SHA-256.
- Card (read 2026-10-20): licence MIT; labels "initially generated using video-assisted
  inference via a visual-language model (Janus-Pro-7B), and later manually verified".
- Unresolved permission: the card does not state where the source audio/video came from or
  under which rights; recorded as unresolved, research use only until clarified.
- No dog, video or uploader ids: held-out evaluation uses embedding pseudo-groups
  (E26 shows they cannot establish independence). Test split (`test/test.zip`) has no labels.
- Used by E26 (activity from a bark) and E27 (dog audio channel of the intent layer:
  `artifacts/intent/dog_audio_activity_v1/`). Label layers: `docs/BEHAVIOR_INTENT_DATA.md`.

### Intent-layer artifacts (2026-10-20)

| Resource id | Path | Trained on | Release suitability (research / personal NC / commercial) |
| --- | --- | --- | --- |
| `intent_situation_kinetics_v1` | `artifacts/intent/kinetics_situation_v1/{full,phone_now}/` | Kinetics-700 pet clips (YouTube; E28) | suitable / not permitted / not permitted |
| `intent_dog_audio_activity_v1` | `artifacts/intent/dog_audio_activity_v1/` | Barkopedia (MIT, provenance unresolved; E27) | suitable / legal review / legal review |
| `intent_cat_audio_context_v1` | `artifacts/intent/cat_audio_context_v1/` | CatMeows (CC BY-NC 4.0; E2 evaluation) | suitable / legal review / not permitted |
| `intent_dog_bark_context_v1` | `artifacts/intent/dog_bark_context_v1/` | Yin & McCowan barks via BEANS (no licence; E22 evaluation) | suitable / not permitted / not permitted |

All heads are JSON (`petperception.intent_head/1`, `petperception.intent_late_fusion/1`):
standardization, coefficients, temperature and abstention threshold, plus metadata naming
the data, label layer and held-out evaluation. New dataset entries: `kinetics700_pet_clips`
(research only) and `yin_mccowan_barks_beans` (research only).

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
| `efficientdet_lite0_tfhub_rawdecode_v1` | the TF Hub build above with its `TFLite_Detection_PostProcess` op removed and raw outputs decoded outside the graph, so it runs in LiteRT.js (E14) | same file as `efficientdet_lite0_tfhub_v1` | Apache-2.0 in the embedded ModelMetadata; COCO training-data question open | suitable / suitable / legal review |
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
| Open Images V7 visual relationships, pet-object pairs (`fetch-data openimages-vrd-objects`) | E15 pet on / under furniture, inside a car | same pinned source files; 526 images, 1,125 (pet, object) pairs for bed, couch (incl. sofa bed, loveseat, studio couch), chair, table (incl. coffee table, desk, dining table) and car; duplicate object boxes merged (IoU >= 0.7); "on" positives fit 269 / development 29 / test 77 | all three OI splits pooled, split by **photographer** (Flickr author): fit 369 / development 52 / test 105 images | annotations CC BY 4.0, images CC BY 2.0 (attribution in `manifests/oi_vrd_object_images.csv`) | all suitable |

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
| `object_relation/pet_object_relation_v1.npz` | OI visual relationships (pet-object pairs) | all suitable (detector terms apply) |

## Phase 3 additions (2026-10-10)

### Datasets

| ID | Use | Content and size (computed here) | Splits | Terms | Profiles |
| --- | --- | --- | --- | --- | --- |
| CatMeows via mirror (`fetch-data catmeows`) | E2 cat vocal context | 440 WAV files, 21 cats, 12 owners, 3 contexts; GitHub mirror `jinjingleayi/meow-by-meow` at commit `0fe6d4c2`, per-file SHA-256 in `manifests/catmeows_mirror_sha256.csv`; not verifiable against Zenodo (blocked) | leave-one-cat-out, leave-one-owner-out | research and non-commercial (Zenodo text); CC BY 4.0 per the University of Milan page | research only |
| Oxford-IIIT Pet (`fetch-data oxford-pets`) | E16 breed robustness | 7,349 images, 12 cat and 25 dog breeds; fast.ai S3 mirror tarball 811,706,944 B, SHA-256 in `manifests/oxford_pets_sha256.csv` | evaluation only | annotations CC BY-SA 4.0 (secondary); image rights with owners | evaluation only, nothing derived ships |
| Kinetics-700-2020, 6 pet-action validation classes (`fetch-data kinetics-pets`) | E18 full engine on real video with sound | 294 clips (petting cat 49, grooming cat 50, grooming dog 50, bathing dog 47, training dog 49, walking the dog 49); CVDF S3 tarballs SHA-256 in `manifests/kinetics_pets_sha256.csv` | evaluation only | annotations CC BY 4.0 (secondary); YouTube clips | evaluation only |
| Kinetics-700-2020 train, same 6 classes: tuning sample (clips 1-50 per class) and holdout (clips 51-100) | E19 / E19b video operating point | 300 + 300 clips streamed from the train tarballs (transfer stopped after the needed clips); per-clip SHA-256 in `manifests/kinetics_pets_train_sample_sha256.csv` and `kinetics_pets_train_holdout_sha256.csv`; no YouTube video shared with each other or with validation | tuning / holdout | as above | evaluation only |
| Open Images V7 train, G1 detector set (`train-prep g1`) | G1 detector training data | see `results/g1/data_summary.json`; source CSVs streamed with SHA-256 recorded, compact manifests in `var/training/g1/` | train / development by Flickr author; evaluation authors excluded | annotations CC BY 4.0, images CC BY 2.0 (per image) | all suitable (data) |

### Artifacts

| Artifact | Trained on | Profiles |
| --- | --- | --- |
| `object_relation/pet_object_relation_v2.npz` (shipped) | none (rule; evaluated on OI relationships) | all suitable |
| `audio_event_thresholds_cc.json` | ESC-50 CC0 / CC BY clips (E1-CC) | research, personal; commercial pending ASSET_TERMS Q-A2 |
| `e2_catmeows_context_logreg.json` | CatMeows | research only; not shipped |

## Behaviour-explanation additions (2026-10-20)

Machine-readable entries: `siglip_base_patch16_224`, `xclip_base_patch32`,
`commons_pet_videos`, `paper_supplement_videos`, `internet_archive_pet_clips`
in [`manifests/resources.json`](../manifests/resources.json). Use:
[BEHAVIOR_EXPLANATION.md](BEHAVIOR_EXPLANATION.md), E31.

### Models

| ID | Revision (pinned) | Files and checksums | Terms | Use |
| --- | --- | --- | --- | --- |
| `siglip_base_patch16_224` | `google/siglip-base-patch16-224@7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | 7 files, `model.safetensors` 812,672,320 B, SHA-256 `2c63cb7d…2af8` (all in resources.json); `petperception fetch-models --only siglip_base_patch16_224` | Apache-2.0 (model card); WebLI training data, terms not published | frozen encoder: zero-shot behaviour prompts, inputs of the trained behaviour head |
| `xclip_base_patch32` | `microsoft/xclip-base-patch32@a2e27a78a2b5d802e894b8a1ef14f3a8ce490963` | 8 files, `model.safetensors` 786,414,772 B, SHA-256 `abf286e8…dbb` | MIT (model card); Kinetics-400 (YouTube) | retrieved as a temporal candidate; **not evaluated, not used** |

### Datasets

| ID | Content (counted here) | Licence / permission | Labels | Grouping |
| --- | --- | --- | --- | --- |
| `commons_pet_videos` (`manifests/commons_pets.json`, `python -m petperception.data.commons_pets`) | 430 records (cat 237, dog 193) from the Commons categories "Videos of cats" / "Videos of dogs" (depth 3) and searches; first 60 s of 287 files downloaded (stream copy) to `var/data/commons_pets/` (about 1.2 GB), 4 failed with HTTP 429 | per file, recorded with author and URL (mostly CC BY, CC BY-SA, CC0) | *source*: parsed from title, description and categories by keyword rules (lay, noisy: for example a chemistry demonstration called "Barking Dog Reaction") | uploader |
| `paper_supplement_videos` (`manifests/paper_videos.json`, `python -m petperception.data.paper_videos`) | 8 videos with SHA-256: Tiira et al. 2012 Movies S1-S3; Vet Res Commun 2026 M1-M3; Frontiers 2023 Dachshund; Frontiers 2026 focal seizures | CC BY 4.0 (articles) | *expert*: the authors' captions (case level; M3 shows string play, not the case's tail chasing) | paper |
| `internet_archive_pet_clips` (`manifests/archive_pets.json`) | 5 videos with SHA-256 (one cartoon found by search excluded) | **none stated**: research analysis only, never redistributed | *source*: title | uploader |
| Frame review (`annotations/behavior_clips.json`) | 38 clips: behaviours, coarse bouts, absent confusers, setting, quality | - | *assistant* (machine-assisted, reviewed by the AI assistant from timestamped contact sheets; never merged with human labels) | - |
| Benchmark (`benchmarks/behavior_questions.json`) | 37 questions on 37 clips with references, confusers, source interpretations, request expectations | inherits the clips' terms | references combine frame review with source / expert labels (basis per item) | - |

Features (`var/understand/features/<clip>.npz`, not committed): target
track, optical-flow motion, appearance descriptors, SigLIP crop and scene
embeddings, YAMNet scores. Rebuilt with
`python -m petperception.understand.dataset` (resumable).

### G1b detector (2026-10-20)

| Path | Content | Trained on | Terms |
| --- | --- | --- | --- |
| `models/g1/g1b_warm_cat2_cop3d/` | checkpoint `ckpt-21` (weights and optimizer state, weights SHA-256 `1a53f4b3…6aad`), exports float32 (SHA-256 `8e47751e…7fbe`), float16, int8 PTQ, run record; `MANIFEST.json` | G1 Open Images CC BY images with cat images repeated twice, plus 4,538 CoP3D home-video frames (CC BY-NC 4.0; 25 contributors disjoint from every CoP3D evaluation) | Open Images annotations CC BY 4.0 / images CC BY 2.0; **CoP3D CC BY-NC 4.0 frames in training: research and non-commercial only** |
