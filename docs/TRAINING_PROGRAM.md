# Training program

How every model in this repository is trained and reproduced, and the five
training jobs G1-G5 that close the measured gaps, each specified so it can
be run. Phones only run exported models ([ON_DEVICE.md](ON_DEVICE.md));
training happens in the workspace (CPU) or in a GPU container.

Every number below is marked **measured** (with the file it comes from)
or **estimate** (with its basis). No job waits for all datasets. A job runs
as soon as its own data exists, and a learned model replaces a rule only
when held-out evaluation supports it (E15: the learned "on furniture" head
was not better than the containment rule, so the rule ships, and both
results are kept).

## Status at a glance (2026-10-19)

| Job | Capability | Ran here | Blocked on |
| --- | --- | --- | --- |
| G1 | pet and household detector (presence, location, objects) | data job **ran** (52,599 CC BY images); CPU benchmark and a 6k-image resumable training run **ran** (`results/g1/cpu_training.json`): within about 0.03 AP50 of M2/M3 on the E4 subset; three data-path defects fixed | the full-data CPU run (several hours, resumable), its own threshold and verifier, then E18 and parity |
| G2 | licensed cat and dog keypoints: ears, tail, body | candidate job **ran** (4,813 cat / 6,348 dog images, split by photographer); annotation spec written | human annotation (estimate 70-100 hours for 2,000 + 2,000) |
| G3 | temporal behaviour from video | E17 **ran on CPU**: temporal model 0.590 vs pooled 0.575 vs E11 0.638 macro-F1; not adopted | household or licensed video (cats; release) |
| G4 | vocal context and vocalization type | cat context (E2) **ran**: macro-F1 0.46, not shipped | dog context data (huggingface.co blocked, terms); type-labelled audio (zenodo.org blocked) |
| G5 | distilled multi-task student for phones | not started | household video and G1 |

## Rules for every job

* Selection (hyperparameters, thresholds, epochs, gates) uses only training
  and development data. The test split is scored once, after freezing.
* Splits follow the strongest grouping the metadata supports: Flickr author
  for Open Images, cat for CatMeows, video for MammalNet, household for
  household recordings.
* Leakage control against every evaluation set in the repository. G1
  removes from training every image whose photographer also took an
  evaluation image (26,929 images removed; one photographer alone had 947).
* Confidence intervals resample the grouping unit. Rule-vs-learned and
  old-vs-new comparisons are paired on the same items.
* Every phone model is exported to TFLite with builtin ops only (LiteRT.js
  cannot run `TFLite_Detection_PostProcess`, E14). Then the job runs:
  model-level parity (browser vs Python, identical tensors), engine-level
  parity (38 media cases), the phone test page in desktop Chromium, and
  finally physical phones (`docs/DEVICE_TEST_PROTOCOL.md`).
* Asset terms are recorded before release (`docs/ASSET_TERMS.md`).

## G1 - Pet and household detector

**Capability.** Cats, dogs and people, plus the household objects the
relation features use: bed, couch, chair, table, car, bowl, toy, teddy
bear, door, window, stairs. It replaces EfficientDet-Lite0, which was
trained on COCO.

**Why (measured).**
* E18: on real pet video the detector finds the pet in only 0.53-0.60 of
  clips. Without the verifier it still misses 32 % of cat clips and 36 % of
  dog clips, so the detector itself is the limit.
* E18, paired over the same clips: the licence-stated detector loses 17
  dog clips out of 195 (p = 0.0095).
* E19 / E19b: re-choosing the detector threshold and verifier scale on
  real video (300 tuning clips, then 300 fresh clips) does not help both
  species. Lower thresholds add cat-versus-dog confusion. The operating
  point cannot fix the detector.
* E4: small-animal recall 0.27.
* E15: the furniture detector misses a third of the furniture under
  detected pets.
* COCO training data is the commercial question Q-T.

**Data (measured, `results/g1/data_summary.json`).** Open Images V7 train,
streamed (14.6 M box rows, source SHA-256 recorded):

| | Train | Development |
| --- | --- | --- |
| Images | 50,007 | 2,592 |
| cat / dog images | 5,819 / 9,249 | 317 / 557 |
| cat / dog boxes | 6,725 / 11,711 | 371 / 691 |
| person boxes | 82,850 | 4,174 |
| bed / couch / chair / table boxes | 2,356 / 3,356 / 14,838 / 8,615 | 125 / 141 / 621 / 381 |
| bowl / toy / teddy bear boxes | 3,035 / 7,481 / 1,118 | 117 / 377 / 27 |
| door / window / stairs / car boxes | 3,172 / 22,036 / 2,546 / 6,306 | 144 / 828 / 112 / 337 |

* The selection takes every pet image (depicted pets removed: 2,607),
  11,971 person-only images, 18,742 household-object images (per-class
  quotas, rarest class first), and 6,045 hard negatives with confusable
  animals or dolls (fox, raccoon, lynx, plush toys ...).
* Licences: all 52,599 are CC BY 2.0. Original files total 112 GB
  (measured from metadata). Stored at 640 px, an estimated 6-7 GB
  (about 125 KB per image).
* 496 negative images carry human-verified cat and dog absence. The rest
  rely on Open Images' box exhaustiveness, a known label-noise risk.

**Remaining gaps.**
* No box classes exist for litter box, food or water bowl as such, pet
  bed, crate, cat tree, scratching post, leash or collar. They need new
  collection (household annotation guide, objects field).
* No night (IR) or wide-angle home-camera frames.
* No motion-blurred video frames.
* 698 small pet boxes (under 1 % of the image) are few for small-animal
  recall.

**Split and leakage.** Development holds 5 % of photographers. Evaluation
sets keep their own splits:
* E4 final split with the E7 protocol;
* E16 breeds;
* E18 real video;
* later, household.
Photographers of all of them are excluded from training.

**Baseline.** M2 (MediaPipe EfficientDet-Lite0) and M3 (TF Hub
EfficientDet-Lite0, raw decoding, in the phone build). Both are measured
on E4/E7, E16 and E18.

**Proposed model.**
* G1a: MediaPipe Model Maker RetinaNet with a MobileNet-MultiHW-AVG 384
  backbone (alternative: MobileNetV2 256/320), fine-tuned from TF Model
  Garden's COCO checkpoint on the 14 G1 classes.
* G1b (licence-clean, removes Q-T): the same network trained from
  scratch on G1 data only. This needs a longer schedule and a Model Maker
  change (no checkpoint initialisation).

**Phone format (measured on the smoke export).**
* TFLite with only builtin ops (CONV_2D, DEPTHWISE_CONV_2D, ADD, MUL,
  RELU6, LOGISTIC, CONCATENATION, RESHAPE, RESIZE_NEAREST_NEIGHBOR).
* Input 256×256 RGB, normalised to [-1, 1].
* Outputs: 12,276 raw box encodings and 15 sigmoid class scores; anchors
  and box-coder scales (1.0) are in `metadata.json`.
* The product's detector class loads the export unchanged. Anchors,
  box-coder scales (1.0, exponential sizes) and the 14-class label map are
  read from the MediaPipe detector metadata that Model Maker embeds; this
  was verified on the smoke export, float32 and int8. The phone bundle
  exporter writes the same decoding for the TypeScript engine. Browser
  parity of a G1 export is still to run, once a trained model exists.
* Exports: float32, float16 and int8 post-training quantization (10.96 /
  5.74 / 3.38 MB measured for MobileNetV2-256 with 14 classes).
* Quantization-aware training fails for MobileNetV2 in this toolchain
  (found by the smoke test) and is opt-in (`--qat`).

**Compute, memory, storage, runtime.**

| Item | Value | Kind |
| --- | --- | --- |
| Inference, float32 MobileNetV2-256, 1 thread, this workspace's CPU (native LiteRT) | 19.7 ms per frame, against 20.4 ms for the current phone detector (M3, 320 px) on the same CPU | measured |
| CPU smoke run (39 + 10 images, 1 epoch, evaluation, float32 / float16 / int8 exports; `results/g1/smoke_test.json`) | completes in 334 s on 4 shared CPU cores (training 61 s); scores meaningless | measured |
| Exported sizes, MobileNetV2-256, 14 classes | float32 10.96 MB, float16 5.74 MB, int8 3.38 MB | measured |
| int8 inference, 1 thread, this CPU | 12.9 ms per frame (uint8 in and out; builtin ops only) | measured |
| Full run: 50k images × 50 epochs at 384 px | 4-8 h on one 24 GB GPU (L4/A10 class), plus 1 h for evaluation and exports | estimate (throughput 100-200 images/s for a MobileNet RetinaNet at 384 px) |
| GPU memory, batch 32 at 384 px | under 16 GB | estimate |
| Disk | 7 GB images + 2 GB caches + 1 GB checkpoints | estimate |
| CPU training throughput, MobileNetV2-256, batch 16, 4 vCPUs (Xeon 2.8 GHz), idle machine (`results/g1/cpu_training.json`) | head only 31.3, decoder + head 28.3, all layers 12.2 images/s; data pipeline alone 81 images/s; validation 14.2 images/s; peak RSS 4.6 GB | measured |
| Checkpoint | 20.8 MB; written in 0.59 s, restored in 0.51 s, weights bit-identical | measured |
| Full G1 set (50,007 train images) on this CPU | 68 min per all-layer epoch, 29 min per decoder + head epoch, 3 min per validation; image store at 320 px 1.24 GB | estimate from the measured throughput |
| Phone latency | must be measured on phones; in the browser on desktop the current detector takes 74 ms per frame single-threaded | measured (desktop) |

**CPU training, measured (2026-10-19).** The plan above assumed a GPU.
The resumable trainer was first benchmarked, then run for real on a
seeded 6,000 / 800-image sample of the G1 set (`views/bench6k`):
one decoder + head epoch, then three all-layer epochs, COCO-initialised,
batch 16, learning rate 0.3 × 16 / 256 with warm-up and cosine decay. About
35 minutes of training.

| Epoch | Stage | Train loss | Dev mAP50 (14 classes) | Dev AP50 cat | Dev AP50 dog |
| --- | --- | --- | --- | --- | --- |
| 0 | decoder + head | 1.413 | 0.225 | 0.477 | 0.610 |
| 1 | all (interrupted at step 82 and resumed) | 0.841 | 0.301 | 0.542 | 0.692 |
| 2 | all | 0.781 | 0.340 | 0.593 | 0.732 |
| 3 | all | 0.750 | 0.382 | 0.640 | 0.737 |

The export was then scored with E4's own code on the E4 subset (823
images; every E4 photographer is excluded from G1 training), next to the
shipped detectors (`results/g1/light6k_export_e4.json`; detector only, no
verifier):

| Detector | AP50 cat [95 % CI] | AP50 dog [95 % CI] | Dog image specificity at 0.35 | ms per image (CPU, incl. JPEG decode) |
| --- | --- | --- | --- | --- |
| G1 light run, float32 (6k images, 4 epochs) | 0.889 [0.858, 0.922] | 0.820 [0.768, 0.864] | 0.37 | 28.4 |
| G1 light run, int8 post-training quantization | 0.824 [0.779, 0.867] | 0.760 [0.706, 0.810] | 0.74 | 21.3 |
| M3, phone build (TF Hub, raw decoding) | 0.923 [0.895, 0.947] | 0.851 [0.800, 0.889] | 0.67 | 33.3 |
| M2, reference (MediaPipe) | 0.923 [0.897, 0.947] | 0.842 [0.789, 0.886] | 0.56 | 48.8 |

* A CPU run on 12 % of the data comes within about 0.03 AP50 of both
  COCO-trained detectors; the intervals overlap. A GPU is not required to
  train a usable G1a. Training on all 50k images on this CPU is a
  several-hour, resumable job (table above).
* At the shared operating threshold (0.35, set for M2) the G1 model
  over-fires on dog-negative images. G1 needs its own operating threshold
  and its own verifier, both selected on the tuning data (E7 protocol),
  before acceptance 1 can be checked.
* int8 post-training quantization costs 0.06-0.07 AP50 here, more than for
  M2 (E12). The float16 build (5.7 MB) is the phone candidate until
  quantization-aware training works for this backbone.
* The E4-subset AP50 of M3 (0.923 / 0.851) differs from acceptance 1's
  values (0.833 / 0.748): those are on the E4 final split with the
  verifier.
* Acceptance 1 check (`petperception g1-eval --e7`): a verifier was fitted
  for the light model on the tuning set, and the E4 split scored once with
  it (E7 protocol). AP50 is then computed from the detections kept at the
  operating point, so it is lower than the full-curve AP above
  (`results/g1/light6k_e7/`). The light model is close but does not pass:

  | E7 protocol | AP50 cat | AP50 dog | Image specificity cat / dog |
  | --- | --- | --- | --- |
  | M3 with its verifier (acceptance thresholds) | 0.833 | 0.748 | 0.660 / 0.919 |
  | G1 light run with its verifier | 0.713 (fail) | 0.737 (fail, by 0.011) | 0.766 / 0.820 |

  One tuning image's upstream bytes no longer match the pin recorded at
  first retrieval. It is excluded and listed in the result, not silently
  replaced.
* **On home phone video (E20, CoP3D, detector only) the light model is
  clearly worse** (`results/g1/light6k_e20/metrics.json`):
  - cat frames found 0.565 against M3's 0.713 (paired difference -0.148
    [-0.182, -0.115]);
  - dog frames found 0.539 against 0.675 (-0.135 [-0.185, -0.086]);
  - dog frames with only a "cat": 0.154 against 0.118.
  Photo AP50 does not predict video performance here. Likely factors: the
  256-px input (M3: 320), little training, and no motion-blurred or
  off-centre frames in the training data. E20 and E18 therefore decide G1
  acceptance (criterion 2), and the full-data run is judged on them. If
  MobileNetV2-256 cannot match M3 on video, the next candidate is the
  MobileNet-MultiAVG 384-px model. At about 2.3 times the pixels, it is a
  multi-session CPU job or a GPU job.

**Data-path defects found by the run and fixed** (details in
`results/g1/cpu_training.json`; tests in `tests/test_g1_data_loader.py`):
* Model Maker 0.2.1.4's COCO cache writer skips every image without a box.
  That silently removed 518 of the 707 hard-negative training images
  (confusable animals, dolls) in the sample. At full scale it would remove
  most of the 6,045 that G1 selects to reduce false cat and dog detections.
  The trainer now installs its own writer that keeps them, under a new
  cache layout name so old caches are never reused.
* The same writer never stored `is_crowd`. Validation therefore scored
  group-of boxes as ordinary boxes, although its docstring said they were
  ignored. Now it is stored. Training is unchanged: Model Maker trains on
  group-of boxes as ordinary boxes either way.
* Boxes touching the right or bottom image edge could exceed the width by
  one rounding step and were dropped (15 in the sample). Fixed in
  `prepare_coco.py`.
* The export step had never run before and crashed (missing directory).
  Fixed. The crashed run was recovered by re-running the same command,
  which restored the final checkpoint and exported.

**Metrics and acceptance (fixed now).** G1a replaces the phone detector
only if all of these hold:
1. E4 final split with the verifier (E14 protocol): AP50 cat ≥ 0.833 and
   dog ≥ 0.748 (M3's values) with non-overlapping-worse intervals for
   specificity.
2. E18 real video, paired with M3: more clips with the right species for
   both species, with the lower 95 % bound of the paired difference above
   0, and no increase in only-wrong-species clips.
3. E16 lowest-breed found rate not below M3's.
4. Engine parity: decisions 38/38, segments F1 ≥ 0.9.
5. Phone latency within the tier budgets (`docs/DEVICE_TIERS_AND_BUDGETS.md`),
   measured on phones.

**Checkpoints and artifacts.**
* `train_resumable.py` writes a full checkpoint every `--ckpt-minutes`, at
  every epoch end and on SIGTERM / SIGINT (see Recoverability below).
* The development AP50 (fast NumPy validation per epoch) selects the epoch.
* Exports: `model_float32.tflite`, `model_float16.tflite`,
  `model_int8_ptq.tflite`, `metadata.json`, `metrics.json` (versions,
  hyperparameters, data sizes, development COCO metrics, history, wall
  time).
* Registered as `pet_detector_oi_g1a_v1` in `manifests/resources.json`.
  The weights are never committed.

**Launch (reproducible).**
```bash
uv run petperception train-prep g1                 # main env: manifests in var/training/g1 (streams Open Images metadata)
uv venv --python 3.11 var/venv-g1 && VIRTUAL_ENV=var/venv-g1 uv pip install -r training/g1/requirements.lock.txt
# download once into a shared 320 px store and write a COCO view (all images, or a seeded sample)
var/venv-g1/bin/python -I training/g1/prepare_coco.py --root var/training/g1 --view full --max-side 320
var/venv-g1/bin/python -I training/g1/prepare_coco.py --root var/training/g1 --view bench6k --train-n 6000 --dev-n 800
# benchmark, then train (re-running the same command resumes; a changed configuration is refused)
var/venv-g1/bin/python -I training/g1/train_resumable.py --view var/training/g1/views/bench6k --run-dir var/training/g1/runs/bench \
    --model MOBILENET_V2 --batch-size 16 --stages 1:head,1:decoder_head,1:all --benchmark 10
var/venv-g1/bin/python -I training/g1/train_resumable.py --view var/training/g1/views/full --run-dir var/training/g1/runs/<name> \
    --model MOBILENET_V2 --batch-size 16 --stages 1:decoder_head,5:all --ckpt-minutes 10
var/venv-g1/bin/python -I training/g1/train_resumable.py ... --validate-only   # restore the latest checkpoint, check its weights hash, validate
# score an export with the E4 protocol next to the shipped detectors (main env)
uv run petperception g1-eval --model g1=var/training/g1/runs/<name>/export/model_float32.tflite --out results/g1/<name>_e4.json
# GPU container (Model Maker's one-shot trainer): docker build -f training/g1/Dockerfile -t petperception-g1 .
```

**Recoverability.** A run can be interrupted at any point and continued
in this or a new environment:
* Checkpoint contents: model weights; optimizer slots and iteration (which
  drive the learning-rate schedule); epoch and step within the epoch;
  NumPy and Python RNG states; the per-epoch data-order seed. Also the
  SHA-256 of the configuration, the training and development label files,
  the script, and the weights themselves.
* Resume: re-run the same command. The data order is replayed and the
  completed steps are skipped. The run refuses to continue if the
  configuration or data changed.
* Verify: `--validate-only` restores the latest checkpoint, checks that
  the weights hash matches the recorded one, and validates.
* Durable copy: the workspace disk does not outlive the environment, and
  weights are never committed. `training/g1/vault.py pack` turns a
  checkpoint into SHA-256-verified text shards for object storage
  (`docs/DATA_TRANSFER_REQUEST.md` section 6). `vault.py unpack` restores
  it into a new run directory, which resumes or validates like the
  original.
* Measured on 2026-10-19 (`results/g1/cpu_training.json`):
  - SIGTERM mid-epoch wrote a `signal` checkpoint (epoch 1, step 82) and
    exited cleanly;
  - a resume with a changed batch size was refused;
  - `--validate-only` reported bit-identical weights;
  - the vault round trip (2 shards, 20.8 MB) restored identical weights and
    an identical development mAP50;
  - the resumed run continued at optimizer iteration 458;
  - a crash in the export step was recovered by re-running the command.
* Not yet available: an object-store prefix with write access. Until one
  exists, a checkpoint survives only as long as this environment.

## G2 - Licensed cat and dog keypoints (ears, tail, body)

**Capability.** Ear position, tail position and body keypoints, for both
species, under terms that allow release. Today dogs have this only in the
research profile (E9, dog-pose data), and cats have none.

**Data.**
* Candidates (measured, `results/g2/candidates_summary.json`): 4,813 cat
  and 6,348 dog Open Images images. Each has one pet, at least 10 % of the
  image, no depictions, CC BY 2.0, and train / development / test split by
  photographer:

  | | Train | Dev | Test |
  | --- | --- | --- | --- |
  | Cats | 3,364 | 566 | 883 |
  | Dogs | 4,425 | 608 | 1,315 |

  1,193 cats and 585 dogs are fully visible.
* Labels do not exist yet. The schema (25 points plus ear, tail, coat,
  posture and view fields), procedure and agreement checks are in
  `docs/training/G2_ANNOTATION_SPEC.md`.
* **Gap:** the annotation work itself.

**Split and leakage.** Photographer-disjoint splits, and G1's evaluation
photographers are already excluded. The test split is double-annotated
and adjudicated.

**Baseline.**
* Dogs: the E9 head (research-only data) scored on the new test split.
  That is a cross-dataset check only; it is never shipped.
* Cats: none.

**Proposed model.** First the E9 architecture: spatial-softmax heatmap
and offset heads on the frozen EfficientNet-Lite0 features the phone
already computes, trained per species or jointly with a species
embedding. CPU training took 24 min for E9 (measured). If it misses the
acceptance criteria, a small top-down pose network trained end to end on
a GPU (estimate 8-24 GPU hours).

**Phone format.** Head parameters go into `engine_assets.json`, like
posture today. The extra phone cost is the head alone, because the
backbone pass is shared. Estimate: under 2 ms per crop, from E9's head
size; to be measured.

**Metrics and acceptance.**
* PCK@0.1 per keypoint group (ears, tail, legs, head) per species on
  test.
* Ear and tail elevation sign agreement (as in E9).
* Kappa for the categorical fields against the adjudicated labels.
* Ship ears for a species only if ear PCK@0.1 ≥ 0.85 and ear-position
  kappa ≥ 0.6. Ship tail only if tail PCK@0.1 ≥ 0.75. Each must also hold
  on the fully-visible subset and for every breed or coat group with
  ≥ 30 test animals.

**Artifacts.** `keypoints/{cat,dog}_keypoints_head_g2_v1.npz` and a
`metrics.json` in the E9 format. The release profile becomes
personal_noncommercial and commercial, subject to the backbone (Q-T).

**Launch.** `uv run petperception train-prep g2`, then annotation, then
the importer to the E9 format (not written: about half a day). Then
`uv run petperception evaluate --experiment e9` with the new data source.

**Compute.** CPU: under 1 h, measured for E9 on 6,773 instances.
Annotation: 70-100 annotator hours for 2,000 + 2,000 animals (estimate,
60-90 s per instance).

## G3 - Temporal behaviour from video

**Capability.** Timed actions (sleep, eat, drink, fight, eliminate for
now) from the per-frame features the phone already computes.

**Data.** MammalNet dog-breed segments, as E11: 813 segments,
MammalNet's own video-level splits, research only. For cats and for
release: none. That needs household video (`docs/household/`) or
licensed video.

**Split and leakage.** MammalNet's training / validation / testing splits.
Selection by 5-fold cross-validation grouped by video on training +
validation. 3 seeds. The test split is scored once per seed.

**Baseline.**
* E11: logistic regression on pooled features, macro-F1 0.638
  [0.537, 0.716], measured.
* E17's pooled ablation: the same network without temporal convolutions.

**Proposed model (E17).** Per-frame projection (64) → two dilated 1-D
convolutions → masked attention pooling + max pooling → 5 classes. JAX,
CPU.

**Result (measured, `results/e17/metrics.json`).** Test macro-F1, mean of
3 seeds:

| Model | Macro-F1 (sd) |
| --- | --- |
| Temporal model | 0.590 (0.036) |
| Pooled ablation | 0.575 (0.030) |
| E11 logistic regression | 0.638 [0.537, 0.716] |

The temporal model does not beat E11, so it is **not adopted** and nothing
is exported. The acceptance rule was set before the run: adopt only if it
beats E11. Seed variance (0.548-0.636) exceeds the temporal-vs-pooled gap.
The test needs more, and more representative, video, not a bigger model.

**Phone format.** Numeric parameters in `engine_assets.json`, evaluated
in TypeScript over the track's per-frame features (about 0.1 M
parameters; estimate under 1 ms per window).

**Compute (measured).** About 50 min of wall time to stream 813 segments
and extract features, and about 30 min to train (5 folds × 3 weight
decays × 2 models, plus 6 final runs). That was on 4 CPU cores shared with
other jobs: 87 CPU minutes in total.

**Next.** The same model on household video, for both species, with the
household split.

## G4 - Vocal context and vocalization type

**Capability.**
* Cats: meow context.
* Dogs: bark context.
* Both: vocalization type (meow, purr, hiss, growl; bark, growl, whine,
  howl).

**Done.** E2 on CatMeows (measured): leave-one-cat-out macro-F1
0.46 [0.38, 0.55] (MFCC) and 0.46 [0.40, 0.54] (YAMNet), against 0.34 at
random. Not shipped. The acceptance criterion was a lower bound above 0.5.

**Blocked.**
* Dog context: Barkopedia and DogSpeak are on huggingface.co (blocked;
  terms unverified). Yin & McCowan is reachable, but no data licence is
  stated.
* Vocalization type: FSD50K is on zenodo.org (blocked). Its per-clip CC
  licences allow a commercial subset.

**Proposed model.** Logistic heads on YAMNet embeddings (E2 code), and
type thresholds on YAMNet sub-type scores selected with E1's protocol.
CPU, under 30 min each (estimate from E1/E2).

**Phone format.** The YAMNet build in the bundle outputs scores only.
Context heads need its 1,024-d embedding exposed, a graph-output patch
like the one already applied to the backbone. Its size is unchanged.

**Acceptance.**
* Type: per-type F1 with a source-grouped interval. A type is shown on
  the phone only if F1 ≥ 0.7 with the lower bound ≥ 0.6.
* Context: as E2.

**Resume.** Allow zenodo.org, then `fetch-data fsd50k` (fetcher not
written: about half a day).

## G5 - Distilled multi-task student

**Capability.** One compact network for detection, verifier, posture,
keypoints and temporal features, sized for tier-C phones.

**Data.** Unlabelled, consented household video (`docs/household/`
consent includes training use as a separate tick box) plus G1 data.

**Teachers.** G1, the verifier, G2 and G3.

**Model.** A MobileNet-class backbone with multiple heads,
quantization-aware.

**Compute.** Estimate 1-2 days on one GPU.

**Acceptance.** No metric worse than the teacher pipeline beyond its
interval. Phone latency and memory within the tier-C budget, measured on
phones.

**Status.** Not started: depends on household data and G1.

## Already-trained artifacts and how to reproduce them

All commands: `uv sync --extra dev --extra train`, then
`uv run petperception fetch-models` and the listed `fetch-data`. Manifests
and checksums are in `manifests/`. Seeds are fixed in the experiment
modules.

| Artifact | Experiment | Data | Model | Selection | Final evaluation | CPU time (measured) |
| --- | --- | --- | --- | --- | --- | --- |
| `audio_event_thresholds.json` | E1 | ESC-50 | thresholds on YAMNet scores | dev folds | held-out fold | ~3 min |
| `audio_event_thresholds_cc.json` | E1-CC | ESC-50, CC0/CC BY clips | same | same | same | < 1 min (cached scores) |
| `verifier/verifier_<detector>.json` | E7 / E12 / E14 | OI test tuning | logistic regression per class | out-of-fold by author | E4 split, once | ~3 min per detector |
| `keypoints/dog_keypoints_head_v1.npz` | E9 | dog-pose (research) | JAX heatmap head | 10 % hold-out | dog-pose val | 24 min |
| `posture/posture_v1.npz` | E10 | OI + Localized Narratives | multinomial LR | OI test | OI validation | ~10 min |
| `behavior_video/dog_behavior_video_v1.npz` | E11 | MammalNet (research) | multinomial LR | validation | test | ~1 h |
| `interaction/person_pet_interaction_v1.npz` | E13 | OI relationships | LR on geometry | 10 % hold-out | OI val + test | ~15 min |
| `object_relation/pet_object_relation_v2.npz` (shipped) and `_v1` | E15 | OI relationships | containment rule (v2); LR (v1) | grouped CV by photographer | test, once | ~5 min |
| `e2_catmeows_context_logreg.json` | E2 | CatMeows (research) | LR on MFCC | inner grouped CV | leave-one-cat-out | ~14 min |

The heads are small enough that the final parameters are the checkpoint
(numeric JSON / NPZ without pickle). E9 keeps the best development epoch.
A run whose loss becomes non-finite is aborted and not exported.
