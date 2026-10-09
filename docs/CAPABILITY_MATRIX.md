# Capability → dataset → model → evaluation matrix

Status date: 2026-10-09. This matrix lists every behavioral capability the
product is meant to have, for cats and dogs, and for each one: the labels it
needs, the real data we have (or lack), the model that implements it, the
evaluation that exists, whether it runs on the phone, and what is missing.

Rules applied throughout:

* Only real, licence-recorded data is used for training and evaluation
  (register: [DATA_AND_MODEL_REGISTER.md](DATA_AND_MODEL_REGISTER.md),
  machine-readable: `manifests/resources.json`). No metadata is invented:
  where a source has no breed, identity or household field, no split or
  breakdown by that field is claimed.
* Splits follow the strongest grouping the source metadata supports
  (identity > household > video/source > image). Synthetic and semi-synthetic
  media (E5, E8) are integration and behavior checks, **not** evidence of
  real-world accuracy.
* Every number is tied to its dataset and split. Agreement with weak labels is
  reported as agreement, not accuracy.
* "Phone" means the TypeScript engine in the PWA running LiteRT.js models
  locally ([ON_DEVICE_AUDIT.md](ON_DEVICE_AUDIT.md)). So far it has been
  exercised only in desktop headless Chromium; **no physical phone has run it**
  ([DEVICE_TEST_PROTOCOL.md](DEVICE_TEST_PROTOCOL.md)).

Legend for "Status": **validated (public data)** = trained or selected and
tested on held-out real public data, not household video; **rule-based** =
operational definition with a-priori thresholds; **weak** = trained and
scored against weak (automatically derived) labels; **research only** = data
terms forbid the product release profiles; **missing** = no usable labelled
data available to this project; **blocked** = data identified but the host is
unreachable from this environment.

## 1. Matrix

| # | Capability | Species | Labels needed | Data we have (licence) | Model | Evaluation (split, result) | Phone | Status | Missing labels / coverage |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Animal presence and location | cat, dog | boxes, verified negatives | Open Images V7 validation boxes + verified image-level negatives (CC BY 4.0 annotations, CC BY 2.0 images) | EfficientDet-Lite0 (COCO, pretrained) | E4: AP50 cat 0.92 [0.90, 0.95], dog 0.84 [0.79, 0.89]; box recall 0.88 / 0.82 at the 0.35 operating point | yes | validated (public data) | Household camera frames (wide angle, low light, IR night mode, motion blur, partial views from floor-level cameras). Small animals (1-10 % of frame) have recall 0.27 (n = 11). Licence-confirmed alternative for commercial releases: E14. |
| 2 | False-positive rejection | cat, dog | detections labelled TP/FP; verified negative images | Open Images V7 test split (tuning) and the E4 validation subset (final), disjoint | crop verifier: EfficientNet-Lite0 ImageNet-group features + per-class logistic regression (E7) | E7 final, Lite0: image specificity on verified negatives cat 0.62 → 0.70 [0.57, 0.83], dog 0.56 → 0.87 [0.80, 0.92]; image recall cat 0.93 → 0.91, dog 0.89 → 0.87 (98 % TP-retention target set on the tuning split) | yes | validated (public data) | Household negatives (pet beds, plush toys, coats, pet-shaped decor) are the real failure population; cat specificity CI is wide (47 verified negative images). |
| 3 | Identity across frames (tracking) within a clip | cat, dog | per-frame identities in multi-animal video | none real; E8 semi-synthetic sequences (real animal pixels from Open Images, scripted motion, real indoor backgrounds) | Kalman (SORT-style) tracker, rule-based | E8 test: IDF1 0.859 (Kalman) vs 0.784 (IoU baseline) | yes | rule-based, semi-synthetic only | Real multi-pet household video with identity labels. Cross-clip re-identification of a specific pet is **not implemented** (no labelled per-animal identity data). |
| 4 | Facial cues (eye narrowing, mouth, whiskers, facial action units) | cat, dog | facial landmarks or action units | **none** (dog-pose has eye keypoints in its schema but 0 % are annotated) | - | - | no | **missing** | Cat facial landmarks / Feline Grimace Scale-style units; DogFACS-style units. Any such dataset must have a licence permitting the release profile. |
| 5 | Ear position | dog | ear base/tip keypoints | Ultralytics dog-pose (Stanford Dogs images; Research-Only) | trained head on frozen EfficientNet-Lite0 features (E9) | E9 val (1,703 instances, CIs resample breeds): ear-tip elevation sign agreement with annotation 0.93, Spearman 0.84; ear keypoint PCK@0.1 per breed: median 0.89, min 0.66 | **no** (research only) | research only | Licensed dog keypoint data for production; ear-carriage morphology (pricked / folded / drop) is not in the metadata, so its effect is unmeasured. |
| 5c | Ear position | cat | ear keypoints | **none** | - | - | no | **missing** | Any cat keypoint dataset with a usable licence. |
| 6 | Tail position | dog | tail start/end keypoints | dog-pose (Research-Only) | E9 head | E9 val: tail-elevation sign agreement 0.89, Spearman 0.87; tail keypoints per-breed PCK@0.1 median 0.79, min 0.0 (breeds with docked or curled tails) | **no** (research only) | research only | Licensed data; tail-type morphology labels; docked tails. |
| 6t | Tail wagging (temporal) | dog | timestamped wag intervals | **none** | keypoint-based lateral-oscillation feature (`tail_lateral_oscillation`) | none | no | **missing** (feature exists, unvalidated) | Video with wag labels. |
| 6c | Tail position / movement | cat | tail keypoints | **none** | - | - | no | **missing** | Cat keypoint data. |
| 7 | Static posture (lying / sitting / standing) | cat, dog | per-animal posture label | Open Images V7 train images + Localized Narratives captions (CC BY 4.0) → caption-derived **weak** labels | logistic regression on EfficientNet-Lite0 crop embedding + box shape (E10) | E10 final (OI validation split, 229 animals): macro-F1 0.648 [0.584, 0.704] vs weak labels (cat 0.75, n = 41; dog 0.61, n = 188); box-shape-only baseline 0.45. Gated output (score ≥ 0.84): 38 % coverage, 85 % agreement | yes | weak | Human-verified posture labels (caption noise is unmeasured); "standing" has only 3 cat examples in validation; no breed metadata in Open Images. On-device: identical labels whenever both engines report one; 18 % of frames flip across the report gate between engines (see §3). |
| 8 | Body keypoints (limbs) | dog | 20 keypoints | dog-pose (Research-Only) | E9 head | E9 val PCK@0.1 0.853 [0.843, 0.863]; end-to-end with the product detector 0.844; per-breed PCK@0.1 0.73-0.94, breed heterogeneity significant (permutation p = 0.0005, `results/e9/breed_breakdown.json`) | **no** (research only) | research only | Licensed data; puppies; household frames. |
| 8c | Body keypoints | cat | keypoints | **none** | - | - | no | **missing** | - |
| 9 | Locomotion and activity over time (still, moving, active bout, back-and-forth, entered/left view) | cat, dog | timestamped activity intervals in real video | none real; E8 semi-synthetic | rule-based segments on tracks | E8 test: moving 16/18, stationary 25/30, active bout 6/6, pacing 6/6 segments recovered | yes | rule-based, semi-synthetic only | Real household video with timestamped activity labels. |
| 10 | Zone behavior (approach / enter / dwell / leave owner-drawn places) | cat, dog | owner zones + intervals | E8 semi-synthetic | rule-based | E8: zone segments 6/6 | yes | rule-based, semi-synthetic only | Real recordings with owner-drawn zones and labelled visits. |
| 11 | Actions from video (sleep, eat, drink, fight, eliminate) | dog | segment-level action labels | MammalNet dog-breed segments (YouTube videos, no dataset licence; research only) | logistic regression on detector-crop + frame embeddings + box motion (E11) | E11 on MammalNet's own test split (154 segments): macro-F1 0.638 [0.537, 0.716] vs majority 0.113, motion-only 0.369; sleeps 0.86, fights 0.77, eats 0.58, drinks 0.53, eliminates 0.46; breed accuracy 0.50-0.75 (5-49 segments per breed) | no (research only; not wired in) | research only | Licensed household video; play, greeting, scratching, chewing, digging, door-waiting. Eat / drink / eliminate are mostly confused with each other. |
| 11c | Actions from video | cat | segment-level labels | **none** (MammalNet has no domestic cats) | - | - | no | **missing** | Cat video with action labels (eating, grooming, kneading, litter use, scratching, play, hunting). |
| 12 | Vocalization present (cat / dog / other) | cat, dog | clip labels | ESC-50 (CC BY-NC 3.0 collection, per-clip Freesound licences) | YAMNet (AudioSet, pretrained) + thresholds selected on ESC-50 dev folds | E1 5-fold CV: macro-F1 0.895 [0.846, 0.935] (tags) / 0.956 [0.920, 0.982] (embedding + LR, not deployed) | yes | validated (public data) | Household audio (room reverb, TV, distance); ESC-50 clips are clean, foreground sounds. Commercial thresholds need non-NC data. |
| 13 | Vocalization type (meow, purr, hiss, growl, bark, howl, whimper, yip) | cat, dog | type-labelled pet audio | none audited (ESC-50 only has "cat" and "dog") | YAMNet AudioSet subtype scores | **not evaluated** per type | yes (scores computed) | pretrained, unevaluated | Type-labelled pet vocalization corpus; purr detection in particular. |
| 14 | Vocal context (e.g. meow while waiting for food / brushing / isolation) | cat | context labels | CatMeows (Zenodo) | planned E2 | - | no | **blocked** (zenodo.org) | Dataset context ≠ household intent; even with access it would be dataset-context classification. |
| 14d | Vocal context / bark type | dog | context labels | Barkopedia (Hugging Face) | planned E3 | - | no | **blocked** (huggingface.co), terms unaudited | - |
| 15 | Interaction with a person (holds / interacts with) | cat, dog | person-pet pair labels with reliable negatives | Open Images V7 visual relationships (exhaustive for annotated pairs; CC BY 4.0): 3,089 images, 11,598 pairs, positives train 2,219 / validation 47 / test 145 | logistic regression on detection-box geometry (E13; a union-crop embedding did not help) | E13 final (OI validation + test), pairs where both boxes are detected: AP 0.907 [0.86, 0.95] vs chance 0.47, precision 0.87, recall 0.68; only 249 of 531 annotated pairs scorable (person or pet undetected); end-to-end recall 0.42; 'boxes overlap' baseline F1 0.82 vs model 0.76 | yes (server and phone; `person_interaction_candidate` segments, identical on both engines in engine parity) | validated (public data), dogs only | Cats: 4 scorable final pairs - not evaluated. Still images only: no temporal interaction, nothing about the animal's response. Person detection coverage is the main limit. |
| 16 | On furniture (bed, couch, chair, table) | cat, dog | pet-object relation labels with reliable negatives | Open Images V7 visual relationships (CC BY 4.0): 526 images, 1,125 pairs, split by photographer; furniture boxes from the product detector's COCO classes | logistic regression on detection-box geometry + species + object group (E15) | E15 test (79 photographers, scored once): AP 0.974 [0.90, 1.00] vs chance 0.67, precision 0.97, recall 1.00 on the 46 pairs where pet and furniture are detected (31 positives: 17 cat, 14 dog); end-to-end recall 0.40; a "pet box inside furniture box" rule is as good (AP 0.995) | yes (server and phone; `on_furniture_candidate` segments with `related_object`) | validated (public photos) | "In front of" vs "on" is the observed failure mode and is likely more common in home video; "on a table" rests on 1 scored positive; whether the animal is allowed there is not inferred. |
| 16u | Under furniture (table, chair) | cat, dog | as row 16 | as row 16 (37 development, 8 test positives) | E15 head | cross-validated AP 0.18; no test positive scorable | no | **failed the E15 gate** | More data with hiding / under-furniture labels. |
| 16v | Inside a car | cat, dog | as row 16 | as row 16 | E15 head | E15 test: AP 0.931 [0.75, 1.00], precision 0.88 on 15 scored positives (dogs only) | no | **failed the E15 gate** (< 20 scored positives) | More scored positives, cats. |
| 16o | Other objects (toys, bowls, litter boxes, crates, cat trees, dog beds, doors, windows) | cat, dog | relation labels | Open Images has dog beds and cat furniture, but the detector has no class for them | - | - | no | **missing** | A detector for pet objects; COCO "bowl" is not validated for pet bowls. |
| 17 | Interaction with other animals (play, chase, fight, groom) | cat, dog | pair/interval labels | MammalNet "fights against other animals" (dog, research only); Open Images relationships contain **no** pet-pet triplets | E11 class "fights" only | E11 | no | research only / **missing** | Multi-pet household video with labelled social interactions. |
| 18 | Environmental context sounds (doorbell, knock, vacuum, TV, speech, thunder, fireworks, alarm, baby cry ...) | - | clip labels | none audited | YAMNet AudioSet classes | not evaluated in this repository | yes | pretrained, unevaluated | Household recordings with context labels. |
| 19 | Scenario context: owner-drawn places, time stamps, view coverage | - | owner input | owner input | deterministic | unit and E2E tests | yes | deterministic | - |
| 20 | Multimodal interpretation (food / outside / attention / play / alert / tense / warning / relaxed hypotheses) with abstention | cat, dog | real paired recordings with behavior and context labels | **none** | rule-based fusion with explicit evidence, conflicts and abstention | **not evaluated**; E5 checks behavior on synthetic pairings only | yes | rule-based, unvalidated | Everything in [EVALUATION_PLAN_REAL_PAIRED.md](EVALUATION_PLAN_REAL_PAIRED.md). |
| 21 | Breed / morphology robustness | dog | breed field | dog-pose (Stanford Dogs synsets, 112 breeds), MammalNet breed keywords | - | E9 breed breakdown (89 breeds with ≥ 10 instances); E11 breed table | - | partial | Cats: no breed metadata in any acquired source. Morphology (coat colour/length, ear carriage, tail type, brachycephaly, body size) is not in any metadata; not inferred. |

## 2. What the matrix says about coverage

* **Cats are under-covered.** Cat capabilities with real trained models:
  presence (1), false-positive rejection (2), coarse posture (7, weak labels)
  and vocalization presence (12). Cat ears, tail, face, body keypoints and
  video actions have **no usable data**.
* **Dog body-language cues exist only in the research profile** (E9 data are
  Research-Only). Product releases (personal non-commercial, commercial) ship
  without ear/tail/limb keypoints and the result says so.
* **Temporal behavior in real video** is covered only by E11 (dogs, YouTube,
  research only; macro-F1 0.64 over five classes). Everything the product reports over time on the phone is
  rule-based and validated only on semi-synthetic sequences.
* **Interactions**: Open Images relationship annotations are exhaustive for
  annotated pairs, so they give reliable negatives for person-pet interaction
  in still images (E13: AP 0.91 on scored dog pairs, but two thirds of pairs
  are not scorable because a box is missed) and for pets on furniture (E15:
  "on" shipped; "under" and "inside a car" failed the pre-set gate). Pet-pet
  interactions have no labelled source.
* **No capability has household validation.** All public data are web photos,
  YouTube video or curated clips.

## 3. On-device equivalence of each shipped capability

Engine parity (`results/ondevice/engine_parity_chromium.json`, 38 cases,
identical WebM/MP4 inputs, Python reference engine vs TypeScript engine in
headless Chromium; decoders differ by design; default float32 bundle):

| Output | Agreement |
| --- | --- |
| Decision state | 38/38 |
| Tracks (species + interval within 1.1 s) | F1 0.96 |
| Sound events | F1 1.00 |
| Behavior segments (kind, object and interval; incl. person interaction and on-furniture) | F1 0.97 |
| Hypotheses (name + status) | F1 1.00 |
| Posture | identical label whenever both engines report one; about one frame in five flips across the report gate (one engine just below 0.84) |

With the opt-in int8 bundle (38 cases): tracks 0.87, segments 0.83,
hypotheses 0.99, decisions 38/38 (`engine_parity_chromium_int8.json`); with
the TF Hub detector: tracks 0.91, segments 0.91, hypotheses 0.96, decisions
38/38 (`engine_parity_chromium_tfhub_rawdecode.json`).

Model-level parity on identical tensors is bit-exact or within 3e-8
(`results/ondevice/runtime_parity_*.json`). The posture gate flips show that
frame-level posture labels are sensitive to small pixel differences (FFmpeg
vs browser decoding), and by extension to real-world image noise. The result
view therefore summarises posture per track as frame counts per label,
including "uncertain", rather than presenting a single frame as a fact.
On-furniture segments are identical on the four E15 clips; in one scenario
clip a couch scored 0.34-0.37 against the 0.35 detector threshold, so the
dog's on-couch segments appear only in the Python run - furniture near the
threshold makes this output fragile.

## 4. Data acquisition priorities (what would change the matrix most)

1. **Real paired household recordings** with timestamped behavior, context
   and owner-reported situation labels, for both species
   ([EVALUATION_PLAN_REAL_PAIRED.md](EVALUATION_PLAN_REAL_PAIRED.md)). Unlocks
   validation of rows 3, 9, 10, 13, 18 and 20, and household validation of all
   others.
2. **Cat keypoints / facial landmarks** under a licence compatible with the
   release profile (rows 4, 5c, 6c, 8c).
3. **Licensed dog keypoint data** to move E9 out of research-only.
4. **Cat video actions** (row 11c) and **pet-pet interaction video** (row 17).
5. **Network access** to zenodo.org and huggingface.co for CatMeows and
   Barkopedia (rows 14, 14d), plus a terms audit.

Compute: all trainings so far fit on the 4-core CPU workspace because they
fit small heads on frozen pretrained features. Fine-tuning backbones or
training a video model is a GPU job; see [TRAINING_PROGRAM.md](TRAINING_PROGRAM.md).
