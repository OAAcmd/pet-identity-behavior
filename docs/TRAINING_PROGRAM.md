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

## Status at a glance (2026-10-30)

| Job | Capability | Ran here | Blocked on |
| --- | --- | --- | --- |
| G1 | pet and household detector (presence, location, objects); G1b run B **finished 2026-10-20** (dev AP50 cat 0.774 / dog 0.821, selected over G1a) | data job **ran** (52,599 CC BY images); CPU benchmark and a 6k-image resumable training run **ran** (`results/g1/cpu_training.json`): within about 0.03 AP50 of M2/M3 on the E4 subset; three data-path defects fixed | the full-data CPU run (several hours, resumable), its own threshold and verifier, then E18 and parity |
| G2 | licensed cat and dog keypoints: ears, tail, body | candidate job **ran** (4,813 cat / 6,348 dog images, split by photographer); annotation spec written | human annotation (estimate 70-100 hours for 2,000 + 2,000) |
| G3 | temporal behaviour from video | E17 **ran on CPU**: temporal model 0.590 vs pooled 0.575 vs E11 0.638 macro-F1; not adopted | household or licensed video (cats; release) |
| G4 | vocal context and vocalization type | cat context (E2) **ran**: macro-F1 0.46, not shipped | dog context data (huggingface.co blocked, terms); type-labelled audio (zenodo.org blocked) |
| G5 | distilled multi-task student for phones | not started | household video and G1 |
| B1 | behaviour head on frozen SigLIP (explanations) | **ran** in E31 (5-fold grouped CV on 278 clips): 0.248 vs zero-shot 0.416 macro AP, not adopted; re-run on feature version 2 in E32: head -0.19, not adopted | more labelled clips per behaviour |
| B3 | phone-sized behaviour encoder: SigLIP -> MobileNetV3-Large student (E33) | **ran** on CPU (20,313 crops, 6 epochs, about 57 min on 4 threads plus about 45 min of teacher embeddings; resumed after a machine reboot): zero-shot macro AP 0.273 vs teacher 0.416, retains 58 % above chance (bar 80 %): not adopted | behaviour-rich licensed video frames for distillation; a larger student |
| B2 | turning-in-place (tail chasing) window model (E32) | **ran** on CPU (20,822 windows of 294 clips, grouped 5-fold CV, under 2 min): out-of-fold clip AP 0.30, mean with zero-shot 0.43 (adopted in v2); fresh test 2/2 positives, 2/2 confusers (n tiny); v2 not the default | more tail-chasing recordings with usable terms; one consistent threshold |

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

**Full-data CPU run, measured (2026-10-19).** All 50,007 training images,
one decoder + head epoch and three all-layer epochs, otherwise as the light
run (`results/g1/full_run.json`). Wall time 4.4 h on the 4 vCPUs; other jobs
shared the CPU during epochs 0-1. Peak RSS 5.6 GB, with the shuffle buffer
bounded at 4,096. The checkpoint and the three exports are kept outside git
as vault shards (`results/g1/g1a_full_vault_manifest.json`).

| Epoch | Stage | Train loss | Minutes | Images/s | Dev mAP50 | Dev AP50 cat | Dev AP50 dog |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | decoder + head | 1.037 | 38.6 | 21.6 | 0.363 | 0.605 | 0.650 |
| 1 | all | 0.760 | 80.3 | 10.4 | 0.429 | 0.665 | 0.727 |
| 2 | all | 0.728 | 65.7 | 12.7 | 0.446 | 0.693 | 0.765 |
| 3 | all | 0.713 | 67.9 | 12.3 | 0.457 | 0.741 | 0.765 |

Evaluation against the acceptance rules (float32 export):

* E4 subset, full-curve AP50 (detector only; `results/g1/g1a_full_export_e4.json`):
  - cat 0.898 [0.867, 0.923] against M3's 0.923;
  - dog **0.864 [0.817, 0.903]**, above both M3 (0.851) and M2 (0.842);
  - 24.7 ms per image on this CPU (M3: 27.3).
* The G1 scores are calibrated lower than the COCO detectors'. At M3's
  operating threshold (0.35), G1's cat image recall is only 0.60.
  Comparing at another detector's threshold is unfair to G1, so an
  operating point was selected on the tuning set before scoring E4 at it
  (`E7.select_operating_point`, recall-matched to M3's tuning recall;
  `results/g1/g1a_full_e7/operating_point.json`). Selected: 0.16 (tuning
  recall cat 0.85, dog 0.96, against M3's 0.92 / 0.88 at 0.35). G1 ranks dogs
  higher than cats, which a single threshold cannot equalise.
* **Acceptance 1 (E7 protocol on E4, own verifier fitted at 0.16): not met.**

  | | AP50 cat | AP50 dog | Specificity cat [95 % CI] | Specificity dog [95 % CI] |
  | --- | --- | --- | --- | --- |
  | M3 + verifier (thresholds) | 0.833 | 0.748 | 0.660 [0.52, 0.79] | 0.919 [0.865, 0.966] |
  | G1 full @ 0.16 + own verifier | 0.766 (fail) | **0.828** (pass) | 0.702 [0.568, 0.826] | 0.703 [0.615, 0.793] (fail: worse, intervals do not overlap) |
  | G1 full @ 0.35 + own verifier (for reference) | 0.543 | 0.700 | 0.936 | 0.946 |
  | **G1 full, per-class points (cat 0.11, dog 0.28) + own verifier** | 0.814 (fail by 0.019) | 0.749 (pass) | 0.574 [0.43, 0.71] (overlaps: pass) | 0.901 [0.84, 0.95] (overlaps: pass) |

  The per-class points were chosen on the tuning set with the same rule,
  applied per class (each class's tuning recall matched to M3's;
  `results/g1/g1a_full_e7_perclass/`). They remove the dog-specificity
  failure. Acceptance 1 then fails only on cat AP50, by 0.019.

* Home phone video (E20, CoP3D):
  - at 0.35, detector only (`results/g1/g1a_full_e20/metrics.json`), against
    M3 detector only: cat frames found 0.528 vs 0.713 (paired -0.185
    [-0.218, -0.153]); dog 0.699 vs 0.675 (+0.024 [-0.022, +0.071]); dog
    frames with only a cat 0.090 vs 0.118 (better); cat frames with only a
    dog 0.165 vs 0.078 (worse);
  - at the selected point 0.16 with its own verifier
    (`results/g1/g1a_full_e20/at_selected_operating_point.json`; a second
    E20 evaluation, the operating point having been chosen on the tuning set
    only), against the phone build with its verifier:
    - cat frames found **0.811** [0.778, 0.842] vs 0.697 (paired +0.113
      [0.086, 0.142]);
    - dog frames found **0.856** [0.822, 0.887] vs 0.598 (+0.259
      [0.221, 0.300]);
    - dog frames with only a cat 0.059 vs 0.137; cat frames with only a dog
      0.039 vs 0.025.
    Detector only at 0.16: cat 0.889, dog 0.957.
  - at the per-class points (cat 0.11, dog 0.28) with their verifier
    (`results/g1/g1a_full_e20/at_per_class_operating_points.json`, a third
    E20 evaluation): cat frames found **0.908** [0.887, 0.929] vs 0.697
    (paired +0.211), with only a dog in 0.006; dog frames 0.720 vs 0.598
    (+0.122 [0.081, 0.164]), but only a cat in **0.167** vs 0.137. The low
    cat threshold now catches dogs as cats: the per-class points trade dog
    confusion for cat recall.
  - Every CoP3D frame contains a pet, so E20 rewards lower thresholds and
    cannot show false positives.
    The specificity cost appears on E4's verified negatives (table above),
    which is why acceptance rests on E4 and E18, not on E20.
* Decision: **G1a (MobileNetV2-256, 4 epochs) does not replace the phone
  detector yet.** It is the first detector trained here on CC BY images. At
  its own operating point it finds far more pets on home phone video than
  the shipped detector, with less species confusion for dogs, and it has
  the higher dog AP50 on photos. But it fails acceptance 1: cat AP50 at the
  operating point is too low, and dog specificity on verified negative
  photos is worse at matched recall.
  With per-class operating points it misses acceptance 1 only on cat AP50
  (0.814 against 0.833). Next, in order:
  1. more epochs. Cat development AP50 was still rising (0.693 → 0.741 in
     the last epoch). Continuing from these weights needs a warm-start
     option in `train_resumable.py` (`--init-from <checkpoint>`; not
     written yet), because the configuration hash rightly refuses to
     stretch a finished schedule. Alternatively, a fresh run with a longer
     schedule takes about 70 minutes per epoch on this CPU;
  2. acceptance 2 (E18) for the best candidate, with the engine wired to a
     G1 export, its per-class points and its verifier;
  3. the 384-px MobileNet-MultiAVG model, a multi-session CPU job or a GPU
     job.
  Acceptance 2 (E18) was not run for G1. E18 needs the Kinetics clips and
  the engine wired to a G1 export plus its verifier; it is only worth
  running once acceptance 1 passes.

**G1b: closing the cat gap and the dog-to-cat confusion (started
2026-10-20; rule fixed before any result).** Two warm starts from the full
run's checkpoint (`--init-from models/g1/g1a_full_mnv2_sb4096/ckpt/ckpt-27`),
2 all-layer epochs at learning rate 0.1 (Model Maker convention; the full
run used 0.3), 2 threads each, in parallel:
* `g1b_warm_cat2`: every training image with a cat appears twice
  (`train-prep g1-extra --cat-repeat 2`; 55,826 training entries);
* `g1b_warm_cat2_cop3d`: the same, plus 4,538 CoP3D home-video frames from
  25 contributors (`--cop3d`; 60,364 entries; boxes from the model-generated
  masks). The contributors are disjoint from E20's archives and from the
  CoP3D development frames (448 cat and 857 dog frames, 9 contributors).

Selection among G1a, `g1b_warm_cat2` and `g1b_warm_cat2_cop3d` uses
development data only:
1. per-class operating points for each candidate, chosen on the Open Images
   tuning set (each class's tuning recall matched to M3's at 0.35);
2. constraints, relative to G1a:
   - Open Images development dog AP50 not lower by more than 0.01;
   - on the CoP3D development frames (detector only, at the candidate's
     points), the rate of dog frames with only a cat not higher by more
     than 0.02;
3. among the candidates that meet both, the highest Open Images development
   cat AP50 wins.

The winner is then scored once: acceptance 1 (E7 protocol on E4, its own
verifier at its points) and E20 (home video, test). If acceptance 1 passes,
E16 and E18 follow. The acceptance thresholds are unchanged.

*Status: paused, 2026-10-20.* The project priority moved to trained
behaviour/intent inference (`docs/BEHAVIOR_INTENT_DATA.md`), where detection
work continues only when it measurably helps downstream. Both runs were
stopped with SIGTERM; each wrote its signal checkpoint
(`g1b_warm_cat2`: epoch 0, step 106 of 3,489; `g1b_warm_cat2_cop3d`: epoch
0, step 230 of 3,772) under `var/training/g1/runs/` and resumes with the
same command. No G1b result exists; the selection rule above stays fixed
for when the runs are resumed. Later the same day `g1b_warm_cat2_cop3d` was
resumed (2 threads, same command; it restored the signal checkpoint at step
230), because the intent stage measured a downstream cost of the detector's
species confusion: 50 of 294 Kinetics test clips end in a species conflict
(`results/intent/end_to_end_status.json`). `g1b_warm_cat2` stays paused.

*Status: run B finished, 2026-10-20.* `g1b_warm_cat2_cop3d` completed its
two epochs with one interruption (a machine reboot; it resumed from the
10-minute checkpoint at step 1,012 with the same command, no restart from
scratch): epoch 0 70 min, epoch 1 90 min (the CPU was shared with feature
extraction), peak RSS 5.5 GB. Final Open Images development AP50 cat 0.774,
dog 0.821, mAP50 0.483 (G1a: 0.741 / 0.765 / 0.457; identical development
file). Checkpoint `ckpt-21` (with optimizer state), float32 / float16 / int8
exports and the run record are in `models/g1/g1b_warm_cat2_cop3d/`
(`MANIFEST.json`); a fresh GitHub clone verified all 8 files, the exports
load in the product detector, and `vault.py restore` rebuilt a resumable run
directory from the clone (`results/g1/github_restore_check_g1b.json`).
`g1b_warm_cat2` (run A) stays paused at step 106: the selection below
compares the completed candidates G1a and B only (the candidate set is
reduced, the rule is unchanged). *Update 2026-10-31 (machine clock, see
STATUS.json):* run A was resumed in the E32 milestone with the same command
(it restored its signal checkpoints at steps 106 and 1,086; 2 threads,
nice 15); when it finishes, the selection is re-run with the three
candidates under the same rule, and a different winner would be scored once
on acceptance 1 and E20.

Selection on development data (`results/g1/g1b_selection.json`,
`src/petperception/eval/g1_select.py`):

| Candidate | Per-class points (tuning, recall-matched to M3) | OI dev AP50 cat / dog | CoP3D dev cat found / only a dog | CoP3D dev dog found / only a cat | Constraints |
| --- | --- | --- | --- | --- | --- |
| G1a | 0.11 / 0.28 | 0.741 / 0.765 | 0.998 / 0.000 | 0.807 / 0.154 | - |
| **G1b run B** | 0.24 / 0.30 | **0.774 / 0.821** | 0.931 / 0.027 | 0.839 / **0.086** | both met |

Winner: **G1b run B** (constraints met; higher cat AP50). On the CoP3D
development frames it nearly halves the dog frames reported only as a cat.

Scored once, as fixed above (per-class points cat 0.24 / dog 0.30, collection floor 0.19 as for
G1a's per-class run):
* **Acceptance 1 (E7 protocol on E4, own verifier fitted on the tuning set):
  met** (`results/g1/g1b_e7_perclass/`).

  | E7 protocol | AP50 cat | AP50 dog | Specificity cat [95 % CI] | Specificity dog [95 % CI] |
  | --- | --- | --- | --- | --- |
  | M3 + verifier (thresholds) | 0.833 | 0.748 | 0.660 [0.52, 0.79] | 0.919 [0.865, 0.966] |
  | G1a, per-class points + own verifier | 0.814 (fail) | 0.749 | 0.574 [0.43, 0.71] | 0.901 [0.84, 0.95] |
  | **G1b run B, per-class points + own verifier** | **0.852 [0.809, 0.890] (pass)** | **0.7481 [0.685, 0.805] (pass, at the threshold)** | 0.681 [0.541, 0.812] (overlaps: pass) | 0.883 [0.821, 0.942] (overlaps: pass) |

  The dog margin is nil (0.7481 against 0.748; the phone build's own
  unrounded value is 0.7481 too).
* **E20, CoP3D home video (test), per-class points with the verifier, paired
  with the phone build and its verifier** (`results/g1/g1b_e20/`): cat frames
  found 0.766 [0.732, 0.802] vs 0.697 (paired +0.069 [+0.047, +0.089]),
  only a dog 0.040 vs 0.025; dog frames found 0.762 [0.730, 0.789] vs 0.598
  (+0.164 [+0.147, +0.184]), only a cat **0.072 vs 0.137**. Against G1a at
  its per-class points (cat 0.908, dog 0.720, dog frames only a cat 0.167),
  G1b gives up cat recall for far less dog-to-cat confusion.
* Next, as fixed above: E16 and E18 (acceptance criteria 2 and 3). Both need
  the engine wired to a G1 export with its per-class points and its
  verifier, which is not implemented yet; then engine parity (criterion 4)
  and phone latency (criterion 5, needs phones). **The phone detector is
  unchanged until then.** The downstream effect on behaviour explanation is
  E31 part C ([BEHAVIOR_EXPLANATION.md](BEHAVIOR_EXPLANATION.md)).

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
* The per-epoch shuffle buffered the whole training set as decoded images.
  On the full set that reached 12.4 GB of the 15.7 GB of RAM within two
  minutes. `--shuffle-buffer` (seeded, part of the configuration hash)
  bounds it: the full run uses 4,096 and peaks at about 4.4 GB.

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
* The export is made from the final epoch. The per-epoch development AP50
  (fast NumPy validation) is recorded in the history for monitoring. The
  learning-rate schedule anneals to the last epoch, and the last three
  checkpoints are kept.
* Exports: `model_float32.tflite`, `model_float16.tflite`,
  `model_int8_ptq.tflite`, `metadata.json` (anchors, labels), and
  `final.json` (configuration, per-epoch history with development AP50 per
  class, SHA-256 of each export, peak memory).
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
* Durable copy in the repository (since 2026-10-20): `training/g1/vault.py
  publish` copies a run's final checkpoint (weights and optimizer state, about
  21 MB), its exports (3-11 MB each) and its run record into `models/g1/<run>/`.
  It writes a SHA-256 manifest, and the folder is committed with ordinary Git.
  Each file is far below GitHub's 50 MB warning size, so no LFS or release host
  is needed; GitHub's API is not reachable from this workspace anyway.
  `vault.py restore` checks the manifest and recreates a resumable run directory.
  Verified from a fresh clone (`results/g1/github_restore_check.json`): all 8
  files match, the three exports load in the product detector, and
  `--validate-only` reports identical weights and the same development
  mAP50 (0.4569). `--init-from models/g1/<run>/ckpt/ckpt-27` warm-starts a
  new run from these weights.

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

## I1 - Behaviour and intent heads (intent layer, 2026-10-20)

**Capability.** What a cat or dog may want or be doing, from trained
channels: situation from synchronized audio and video (both species), dog
vocal activity / intent (barks), dog bark context, cat meow context; and
their combination with explicit missing inputs and conflicts
(`src/petperception/intent/`, label layers in
[BEHAVIOR_INTENT_DATA.md](BEHAVIOR_INTENT_DATA.md)).

**Encoders reused, frozen.** YAMNet (521 class scores and the 1,024-d
embedding), the EfficientNet-Lite0 backbone (pooled 1,280-d embedding of the
largest cat/dog crop and of the whole frame), the COCO detector (77 object
classes, person boxes), the crop verifier, the posture head, the person-pet
interaction model and (research variant only) the AwA cat/dog keypoint head.
Features are cached once per clip (`python -m petperception.intent.kinetics`,
resumable per clip; Barkopedia features cached by E26/E27).

**Heads.** Multinomial logistic regression per branch (class-balanced),
C by grouped 5-fold cross-validation on training data only, temperature
(NLL on out-of-fold scores) and an abstention threshold (out-of-fold
selective accuracy 0.8). Late fusion: logistic regression on the branches'
centred log-probabilities times availability flags, trained with modality
dropout (every subset of branches), so one stacker serves any combination of
present inputs. Early fusion for comparison. Transparent rule baselines
fixed before scoring (`intent/rules.py`, committed in 48e0c82).

| Job | Data (train / test) | Grouping | Command | CPU time (measured) | Artifacts |
| --- | --- | --- | --- | --- | --- |
| E27 dog audio channel | Barkopedia 12,480 / 500 | near-duplicate pseudo-groups (no ids) | `petperception evaluate --experiment e27` | 40 min (incl. YAMNet scores for 13k clips) | `artifacts/intent/dog_audio_activity_v1/` |
| E28 situation, synchronized A+V | Kinetics 600 train-split / 294 validation clips | YouTube id + embedding audit | `python -m petperception.intent.kinetics` (features, 894 clips, 3.3-4.5 s/clip with 2 threads), then `petperception evaluate --experiment e28` | see results/e28 | `artifacts/intent/kinetics_situation_v1/{full,phone_now}/` |
| cat meow / dog bark context heads | CatMeows 440 (21 cats); Yin barks 693 (10 dogs) | cat; dog | `python -m petperception.intent.train_heads` | < 2 min | `artifacts/intent/{cat_audio_context_v1,dog_bark_context_v1}/` |

Held-out evaluation of the last row is E2 / E22 (leave-one-animal-out); the
training script records only out-of-fold figures.

**Recovery.** Feature extraction skips clips whose cache file exists (atomic
rename per clip). E28 caches its fitted models with a digest of the training
inputs (`var/intent/e28_models.pkl`) and reuses them after an interruption;
a changed training set refits.

**Acceptance for use as evidence.** A channel's top label counts as
*supported* only above its own fitted threshold; a channel below the
per-class gate (F1 >= 0.7, lower bound >= 0.6) is reported with its gate
status. Context channels never support a hypothesis alone.

## B1 - Behaviour head for explanations (2026-10-20)

**Capability.** Recognizing what the animal does (and when) as evidence for
the behaviour-explanation answers (`src/petperception/understand/`,
[BEHAVIOR_EXPLANATION.md](BEHAVIOR_EXPLANATION.md)).

**What is trained and why.** The SigLIP base encoder stays frozen (812 MB;
fine-tuning it on a few hundred clips on this CPU is neither feasible nor
justified). The trainable part is a one-vs-rest logistic regression per
behaviour on clip-level evidence (mean animal-crop embedding, scene
embedding, species flag; `understand/head.py`), C chosen by inner grouped CV
on training folds. Its competitor is the untrained zero-shot reading of the
same encoder.

| Job | Data | Grouping | Command | CPU time (measured) | Outcome |
| --- | --- | --- | --- | --- | --- |
| Features | 300 clips (Commons, paper supplements, Internet Archive) | - | `python -m petperception.understand.dataset --threads 2` (resumable per clip) | 15-19 s per clip of up to 60 s with 2 threads (about 95 min for 298 clips while the G1b run shared the CPU) | `var/understand/features/*.npz` |
| E31 part A | 278 clips with a located animal, title / caption labels, 9 eligible behaviours | uploader or paper; 5-fold GroupKFold | `petperception evaluate --experiment e31` | about 2 min including the benchmark | head 0.248 vs zero-shot 0.416 macro AP: **not adopted** |

**Artifacts.** `artifacts/understand/recognizer.json`: the adopted method
(zero-shot), per-behaviour thresholds chosen on out-of-fold scores, the
fixed defaults for the other behaviours, and provenance (commit, pool,
eligible behaviours). No head file is written while the head is not
adopted; `petperception explain` falls back to the fixed zero-shot defaults
when the configuration is missing.

**Recovery.** Feature extraction skips clips whose file exists (atomic
rename per clip); E31 itself is deterministic and re-runs from the features
in about two minutes.

**Next.** More labelled clips per behaviour (household recordings); the
head is re-evaluated with the same rule when the data change. The temporal
recognizer for tail chasing became B2 (below).

## B2 - Turning-in-place window model (E32, 2026-10-30)

**Capability.** Tail chasing and other circling, recognized from the
animal's track over time rather than single frames
(`src/petperception/understand/temporal.py`).

**What is trained.** A logistic regression on 14 features of 2-s windows
(hop 0.5 s) from feature version 2: detection and fill rates, displacement
and path relative to the animal's size, flow speed and rotation, horizontal
motion flips, box aspect and area variation, appearance periodicity, and the
zero-shot circling and neutral probabilities. Window labels: inside a
frame-review tail-chasing bout = 1, outside every bout = 0, partial overlap
ignored; other development clips all 0 unless a human label says tail
chasing (then left out). C from {0.01, 0.1, 1} by grouped 5-fold CV
(uploader / paper); clip score = mean of the top quarter of window
probabilities, combined with the zero-shot clip score by their mean (chosen
on out-of-fold clip AP: window 0.30, zero-shot 0.35, mean 0.43).

| Job | Data | Command | CPU time (measured) | Outcome |
| --- | --- | --- | --- | --- |
| Features v2 | 300 development clips, then the 37 test clips after pre-registration | `python -m petperception.understand.dataset --threads 2 --feature-version 2 --subdir features_v2` (resumable per clip) | about 11-24 s per clip with 2 threads (37 test clips in 14 min with v1 extraction in parallel) | `var/understand/features_v2/*.npz` |
| Development | the E31 collection only | `python -m petperception.eval.e32_behavior_v2 develop` | a few minutes (SigLIP text prompts and CV) | `artifacts/understand/circling_v1.json`, `recognizer_v2.json`, `results/e32/development.json` |

**Known defect (found after the test, not fixed in the tested system).** The
v2 recognizer compares the combined clip score with the tail-chasing
threshold that the part A rule chose for zero-shot scores (0.64), not with
the development threshold for the combined score (0.81); the circling
threshold only sets the bouts. A v3 must use one threshold chosen for the
score it is applied to, and be tested on new clips.

**Recovery.** Feature extraction skips finished clips; development is
deterministic from the features.

## B3 - Phone-sized behaviour encoder (E33, 2026-10-31)

**Capability.** Zero-shot behaviour recognition on the phone: a student
that maps a 224 x 224 animal crop into SigLIP's image space, so that the
SigLIP text prompts (embedded offline) can score it.

| Job | Data | Command | CPU time (measured) | Outcome |
| --- | --- | --- | --- | --- |
| Crops | G1 Open Images cat/dog boxes (area >= 0.04; train 15,775 / development 927) and G1b CoP3D frames (train 4,538 / 400 development) | `python -m petperception.eval.e33_phone_student build` (resumable per file) | about 3 min | `var/student/crops/`, `index.json` |
| Teacher | SigLIP base on every crop | `... e33_phone_student teacher --threads 4` (resumable in chunks of 512) | about 4.8 crops/s on 2 threads while G1 trained, 10.7 crops/s on 4 threads | `var/student/teacher/*.npy`, `text_e31.npz` |
| Student | MobileNetV3-Large + Dense(768) + L2; cosine + KL on the E31 prompt logits; Adam 3e-4 cosine decay, batch 32, 6 epochs | `var/venv-g1/bin/python training/student/train_student.py --work var/student --run-dir var/student/run_e33 --export-dir models/student/e33_mnv3l --threads 4` | about 8 min per epoch, 57 min in total | validation cosine 0.863, KL 0.555 (epoch 5 selected) |
| Fixed-batch export | the selected weights | `training/student/export_student.py --weights models/student/e33_mnv3l/student_best.weights.h5 --out models/student/e33_mnv3l` | under 1 min | float32 / float16 / int8 TFLite, outputs identical to the evaluated files |
| Evaluation | E31 pool (278 clips) | `python -m petperception.eval.e33_phone_student evaluate --threads 2` | about 25 min | `results/e33/metrics.json` |

**Recovery (tested).** A SIGTERM mid-epoch and a resume with the same
command continued at the saved step (smoke run); the real run was killed by
a machine reboot at epoch 5, step about 560, and resumed from its interval
checkpoint at step 1 (about 7 minutes of work lost; the interval timer is
not reset by an epoch-end checkpoint, which a later trainer version should
fix). A resume refuses to continue if the configuration, the crop index,
the text bank or the trainer changed.

**Artifacts.** `models/student/e33_mnv3l/` (exports, selected weights,
`run_record.json` with hashes and history; the optimizer-state checkpoint
stays in `var/student/run_e33/`). Release: research and non-commercial only
(CoP3D CC BY-NC frames in training; SigLIP as teacher).

**Next.** Distillation on behaviour-rich video frames, a larger student, or
per-behaviour heads on the student's embeddings, each evaluated with the
same E31-pool protocol and bar. E34 measured off-the-shelf TinyCLIP 8M and
39M on the same protocol: 0.267 and 0.308 (bar 0.347), so a published small
CLIP is not a shortcut either.

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
