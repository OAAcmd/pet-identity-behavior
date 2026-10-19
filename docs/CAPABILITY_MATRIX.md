# Capability and evidence matrix

Status date: 2026-10-19. Every required capability is listed for cats and
for dogs separately. Each is assessed on four independent dimensions;
progress on one never counts for another.

| Dimension | Question | Evidence that counts |
| --- | --- | --- |
| **Software** | Is it implemented, and does the implementation do what it specifies? | unit and end-to-end tests; parity between the Python reference engine and the phone (TypeScript + LiteRT.js) engine |
| **Behavioural validity** | Does it detect the behaviour in real recordings of real animals? | held-out real data with labels; numbers with their split and interval. Semi-synthetic clips (E8), synthetic pairings (E5) and software scenario tests are **not** evidence of behavioural accuracy. |
| **Device readiness** | Does it run on phones within budget? | measurements on physical phones |
| **Release readiness** | May it ship, and under which profile? | per-asset terms (`docs/ASSET_TERMS.md`) |

**Device status, the same for every capability marked "phone".** The browser
inference implementation has passed desktop Chromium tests: engine parity
on 38 cases, and the phone test page run end to end in headless Chromium.
Physical Android, iPhone and Safari validation remains pending. The phone
test build is published and collects reports automatically
(`docs/DEVICE_TEST_PROTOCOL.md`); no report has arrived yet. No latency,
memory or battery figure in this repository comes from a phone.

**Household validity, the same for every capability.** None has been
validated on household recordings. The collection and annotation package
is in `docs/household/`. Public data evaluates what it can (the tables
below). The list of what only household recordings can establish is in
§3.

Abbreviations:
* Python: the reference engine.
* Phone: the TypeScript engine with LiteRT.js models.
* PNC: release profile `personal_noncommercial`.
* COM: release profile `commercial`.
* "research only": the data terms exclude PNC and COM.

## 1. Cats

| Capability | Software | Behavioural validity (real data) | Release (PNC / COM) | Next step |
| --- | --- | --- | --- | --- |
| Presence and location | Python + phone; parity tracks F1 0.91 with the phone detector | Photos, E4/E7, Open Images validation (phone detector, E14): image recall 0.885, image specificity 0.660 [0.52, 0.79] on 47 verified negative images. Breeds, E16, Oxford-IIIT Pet, 2,371 images: found 0.935 [0.924, 0.944]; Sphynx 0.55, Bombay and Russian Blue lowest, breed heterogeneity p = 0.0005. **Real video, E18, Kinetics (99 clips with sound): a cat track in 0.566 [0.47, 0.66] of clips (reference detector); 0.596 with the phone build's detector; 0.677 without the verifier, but then 15 of 99 cat clips show only a dog.** | yes / Q-T | G1 detector: the detector misses a third of clips even without the verifier, and E19/E19b found no better threshold on 600 more real clips |
| False-positive rejection | Python + phone | E7 final: cat specificity 0.62 → 0.70 with the verifier, recall 0.93 → 0.91 (MediaPipe detector); household negatives (plush toys, pet beds) unmeasured | yes / Q-T | household negatives (`docs/household/`) |
| Identity across frames (one clip) | Python + phone (Kalman tracker) | none real. E8 semi-synthetic IDF1 0.86 is a software check. | yes / yes | household multi-cat clips |
| Facial cues (eyes, mouth, whiskers, ear tips in the face) | **not implemented** | none. CatFLW (48 landmarks, CC BY-NC 4.0) is blocked on kaggle.com. | - | network (kaggle.com) for research; new annotation for PNC/COM |
| Ears | Python research experiment (E24); not in the engines | **E24, AwA-Pose (MIT annotations), photographer-grouped test (79 cats, 51 photographers): ear PCK@0.1 0.914 (joint head) / 0.898 (cat head), meeting the G2 PCK criterion; ear-tip elevation sign agreement 0.93.** Overall cat PCK@0.1 0.760 [0.711, 0.802]. Ear-position kappa not evaluable (no categorical labels). | research only (image licence variants unverified) | verify the AwA2 image licences per image; engine integration (research profile); G2 for a CC BY set |
| Tail (position, movement) | Python research experiment (E24) | E24: tail PCK@0.1 0.51 (joint head), 0.39 (cat head): fails the G2 criterion (≥ 0.75); tail elevation computable on only 11 test cats | research only | more cat tail annotation (G2) |
| Posture (lying / sitting / standing) | Python + phone | E10 against caption-derived **weak** labels, Open Images validation, 41 cats: F1 0.75. Human-verified accuracy unknown; "standing" has 3 cat examples. | yes / Q-T (backbone) | household annotation (posture field) |
| Temporal: locomotion, activity, entered/left view | Python + phone (rule-based) | none real. E8 is semi-synthetic. E18 shows the segments are produced on real video (moving in 50 of 99 clips), but these are not labels. | yes / yes | household timestamps |
| Temporal: actions (eat, drink, groom, litter, scratch, knead, play, sleep) | **not implemented** for cats | none: no public cat action set with usable terms (A2D is blocked and research-only; MammalNet has no domestic cats) | - | household collection, then G3 for cats |
| Vocalization present | Python + phone | E1, ESC-50 (40 cat clips, 5-fold CV): F1 0.79 nested (cat), deployed thresholds. E1-CC on CC0/CC BY clips only: 0.76. **E21, FSD50K eval (129 cat clips, 2,284 uploaders, nothing tuned): F1 0.688 [0.623, 0.754], precision 0.65, recall 0.74.** E21b: a threshold re-selected on 41k FSD50K dev clips (0.855) gives 0.702, not a significant gain, so not adopted. Real video, E18: own-species vocalization in 9 of 99 cat clips (unlabelled; Kinetics audio often has music). | yes / A2 + Q-A2 | household audio with sound intervals |
| Vocalization type (meow, purr, hiss, caterwaul) | Python + phone (YAMNet sub-type scores, gated by `artifacts/audio_subtype_validation.json`) | E21, FSD50K eval, deployed point: **meow F1 0.725 [0.637, 0.806], passes** the pre-set gate (F1 ≥ 0.7, lower bound ≥ 0.6); purr 0.701 [0.595, 0.795], fails on its interval (false positives on engines); hiss not evaluable (2 cat-hiss clips; FSD50K "Hiss" is mostly steam and traffic); caterwaul has no label. Only meow is shown; purr and hiss stay in the JSON and can only block interpretations. | yes / Q-T | household audio with type labels; cat-hiss data |
| Vocal context (brushing, isolation, waiting for food) | Python research experiment (E2) only | E2, CatMeows, leave-one-cat-out: macro-F1 0.46 [0.38, 0.55] (MFCC) / 0.46 [0.40, 0.54] (YAMNet) vs 0.34 stratified random. Weak, so not shipped. | research only | more data per context; household owner-reported situations |
| Environmental context (doorbell, vacuum, TV, speech, thunder) | Python + phone (YAMNet classes) | E21, FSD50K eval at the shared threshold 0.3: music 0.770, thunder 0.797; speech 0.560, door 0.413, doorbell 0.378, knock 0.412, alarm 0.413, siren 0.432, fireworks 0.315, dishes 0.277, cutlery 0.241. E21b per-tag thresholds improve door, knock, thunder (0.840), fireworks (0.569); not yet applied. TV, vacuum and baby cry have no FSD50K label. | yes / Q-T | household audio labels |
| Interaction with a person | Python + phone | E13 photos: only 4 scorable cat pairs, not evaluated. **Real video, E18: in petting/grooming clips where a cat is tracked, a person-contact segment in 35 of 56 (0.63).** No negative label is available on Kinetics. | yes / yes | household recordings (person interaction field) |
| On furniture (bed, couch, chair, table) | Python + phone (containment rule v2) | E15, Open Images, 17 cat positives in test: rule precision and recall 0.97 (both species pooled); end-to-end recall 0.40 | yes / yes | household; pet-specific objects need new collection |
| Under furniture, inside a car | not shipped (failed the E15 gate) | E15: under AP 0.18 (cross-validated) | - | more labelled data |
| Interaction with other animals | **not implemented** | none: no labelled pet-pet source obtainable (VidOR, EgoPet and Visual Genome are blocked or have unclear terms) | - | household multi-pet recordings |
| Multimodal interpretation (food, outside, attention, play, alert, tense, warning, relaxed) | Python + phone (rule-based fusion with abstention) | **none**. E5 uses synthetic pairings. E18: no non-abstained interpretation on any of 99 cat clips (no labels to score). | yes / yes | household recordings with owner-reported situations |
| Breed / morphology | - | E16 per breed (above); no coat or ear-type metadata in any source | - | household metadata (breed, coat) |

## 2. Dogs

| Capability | Software | Behavioural validity (real data) | Release (PNC / COM) | Next step |
| --- | --- | --- | --- | --- |
| Presence and location | Python + phone | Photos, E14 phone detector: image recall 0.866, specificity 0.919 [0.86, 0.97]. Breeds, E16, Oxford-IIIT Pet, 25 breeds: found 0.924; Newfoundland 0.71, keeshond 0.84; heterogeneity p = 0.0005. **Real video, E18, Kinetics (195 clips): a dog track in 0.533 [0.46, 0.60] of clips with the reference detector, but 0.446 [0.38, 0.52] with the phone build's licence-stated detector (paired p = 0.0095); 0.636 without the verifier; walking-the-dog clips are worst.** | yes / Q-T | answer Q-M2 (restores the reference detector); G1 |
| False-positive rejection | Python + phone | E7 final: dog specificity 0.56 → 0.87, recall 0.89 → 0.87 | yes / Q-T | household negatives |
| Identity across frames (one clip) | Python + phone | none real (E8 semi-synthetic) | yes / yes | household multi-dog clips |
| Facial cues | **not implemented** | none. DogFLW (CC BY-NC 4.0) is blocked on kaggle.com; eye keypoints in dog-pose are 0 % annotated. | - | network for research; new annotation |
| Ears | Python research profile (E9 keypoint head) | E9, dog-pose validation, 1,703 instances: ear-tip elevation sign agreement 0.93, Spearman 0.84; per-breed ear PCK@0.1 median 0.89, minimum 0.66 | **research only** (dog-pose) | G2 (licensed annotation) |
| Tail position | Python research profile (E9) | E9: tail elevation sign 0.89, Spearman 0.87; tail PCK@0.1 per breed median 0.79, minimum 0.0 (docked or curled tails) | research only | G2 |
| Tail wagging (temporal) | feature only (`tail_lateral_oscillation`) | none (no wag labels anywhere) | research only | household (tail field in the annotation guide) |
| Body keypoints | Python research profile (E9) | E9 PCK@0.1 0.853 [0.843, 0.863]; per breed 0.73-0.94. **Cross-dataset (E24, AwA-Pose dogs, 459 test instances): the E9 head scores 0.507; a head trained on AwA-Pose scores 0.712 [0.675, 0.807]** (ears 0.76, tail 0.63): keypoint figures do not transfer between datasets | research only | G2 |
| Posture | Python + phone | E10, weak labels, 188 dogs: F1 0.61 | yes / Q-T | household annotation |
| Temporal: locomotion, activity | Python + phone (rules) | none real (E8 semi-synthetic) | yes / yes | household |
| Temporal: actions (sleep, eat, drink, fight, eliminate) | Python research experiments E11 and E17 (G3) | E11, MammalNet test, 154 segments: macro-F1 0.638 [0.537, 0.716] (pooled features, logistic regression). E17 temporal network: 0.590 (mean of 3 seeds), pooled network 0.575: modelling time did not help, not adopted. | research only (MammalNet) | household video (play, greeting, scratching, chewing, door-waiting have no labels) |
| Vocalization present | Python + phone | E1, ESC-50 40 dog clips: F1 0.90 nested; identical on the CC0/CC BY subset (all dog clips are CC0/CC BY). **E21, FSD50K eval (170 dog clips): F1 0.728 [0.667, 0.789], precision 0.91, recall 0.61.** E21b: a threshold from FSD50K's CC0 / CC BY dev clips (0.67) is non-inferior (+0.015 [-0.009, +0.042]), a licence-clean candidate for the commercial profile. E18: own-species vocalization in 7 of 195 clips (unlabelled). | yes / A2 | household audio |
| Vocalization type (bark, growl, whine, howl, yip) | Python + phone (YAMNet sub-type scores, gated) | E21, FSD50K eval: **bark F1 0.728 [0.656, 0.797], passes**; growl 0.489 [0.349, 0.607] (dog growl 0.466), fails, and no threshold fixes it (E21b); whine, howl and yip have no FSD50K label. Only bark is shown; a possible growl can only block interpretations (it no longer opens "tense" or "warning"). | yes / Q-T | household audio with type labels |
| Vocal context (bark context, arousal) | Python research experiment (E22) only | E22, Yin & McCowan barks (693 barks, 10 dogs, 6 breeds): leave-one-dog-out macro-F1 **0.722 [0.641, 0.785]** (YAMNet embeddings), leave-one-breed-out 0.696, against 0.33 at random. No stated data licence, one study, recording situations not states: research only. Barkopedia is CC BY-NC-SA 4.0 with emotion (arousal / valence) labels, which the product does not output. | research only | licensed household barks with owner-reported situations |
| Environmental context sounds | Python + phone | E21 (as for cats, the same detector): music and thunder pass the 0.7 bar; door, doorbell, knock, alarm, siren, dishes, cutlery do not | yes / Q-T | household |
| Interaction with a person | Python + phone | E13 photos (dog pairs, Open Images validation + test): AP 0.907 [0.86, 0.95], precision 0.87, recall 0.68 on scorable pairs; end-to-end recall 0.42. **Real video, E18: in bathing/grooming clips where a dog is tracked, a person-contact segment in 36 of 50 (0.72)**; in training/walking clips, 19 of 54 (contact possible, not a false-positive rate). | yes / yes | household |
| On furniture | Python + phone (rule v2) | E15: 14 dog positives in test, pooled precision and recall 0.97; end-to-end recall 0.40 | yes / yes | household |
| Inside a car | not shipped (failed gate: under 20 scored positives) | E15 test AP 0.93 on 15 dog positives | - | more data |
| Interaction with other animals | research class "fights" in E11 only | E11 "fights" F1 0.77 (MammalNet, research) | research only | household multi-pet; VidOR if its terms clear |
| Multimodal interpretation | Python + phone (rules, abstention) | **none**. E18: 2 of 195 dog clips produced a non-abstained hypothesis (alert, warning; both in "training dog"; unlabelled). | yes / yes | household |
| Breed / morphology | - | E16 (above); E9 per-breed keypoints; E11 per-breed actions (5-49 segments per breed) | - | household metadata |

## 3. What needs household recordings, and what public data already covers

| Question | Public data that answers it now | Needs new household recordings |
| --- | --- | --- |
| Is the pet found in its own home footage (wide angle, low light, floor-level camera, partial views)? | E18 real video (YouTube, person-centred) gives a first real-video figure | yes: home cameras and phones, the owner's own pets |
| Are household objects mistaken for pets? | E7 (web photos) | yes: plush toys, pet beds, coats, decor |
| Ears, tail and face of cats | none obtainable | yes, or new annotation of CC BY photos (G2) |
| Ear and tail of dogs in a releasable model | none (E9 is research only) | new annotation (G2) |
| Timestamped activities and actions (both species) | E11/E17 for 5 dog actions (research only) | yes (eating, drinking, grooming, litter, scratching, play, greeting, waiting at the door) |
| Vocalization type and context in the home | CatMeows (3 cat contexts, research) | yes (with owner-reported situation) |
| Person interaction over time, and the pet's response | E13 (photos), E18 (contact clips) | yes |
| Pet-pet interactions | none | yes |
| Interpretations (food, outside, play, tense ...) | none: public data has no owner-reported situations | yes; observed behaviour and inferred intention are annotated in separate layers (`docs/household/ANNOTATION_GUIDE.md`) |

## 4. Engine parity of the shipped capabilities

Python reference engine vs phone engine in headless Chromium, 38 cases,
identical WebM/MP4 inputs (`results/ondevice/engine_parity_chromium*.json`):

| Bundle | Decisions | Tracks | Segments | Hypotheses |
| --- | --- | --- | --- | --- |
| MediaPipe detector (float32), used by E4-E18 | 38/38 | F1 0.96 | 0.97 (0.966 after rule v2) | 1.00 |
| TF Hub detector, raw decoding (the phone test build since 2026-10-10) | 38/38 | 0.91 | 0.91 | 0.96 |
| int8 builds (opt-in) | 38/38 | 0.87 | 0.83 | 0.99 |

Model-level parity on identical tensors is bit-exact or within 3e-8.
Posture: when both engines report a label it is identical, but about one
frame in five flips across the 0.84 report gate. Posture is therefore
summarised per track, never shown as a single-frame fact.
