# Capability and evidence matrix

Status date: 2026-10-20. Every required capability is listed for cats and
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
| Presence and location | Python + phone; parity tracks F1 0.91 with the phone detector | Photos, E4/E7, Open Images validation (phone detector, E14): image recall 0.885, image specificity 0.660 [0.52, 0.79] on 47 verified negative images. Breeds, E16, Oxford-IIIT Pet, 2,371 images: found 0.935 [0.924, 0.944]; Sphynx 0.55, Bombay and Russian Blue lowest, breed heterogeneity p = 0.0005. **Real video, E18, Kinetics (99 clips with sound): a cat track in 0.566 [0.47, 0.66] of clips (reference detector); 0.596 with the phone build's detector; 0.677 without the verifier, but then 15 of 99 cat clips show only a dog.** **Home phone video, E20, CoP3D (209 cat sequences, 2,090 frames): phone build finds the cat in 0.697 [0.66, 0.74] of frames, only a dog in 0.025; the pet is found in at least half the frames of 81 % of sequences. Engine level (E20b): a cat track in 0.646 of cat videos, only a dog in 0.053.** | yes / Q-T | **G1b (CC BY-trained, run B finished): acceptance 1 met (E4 cat AP50 0.852 with its verifier); home video cat frames found 0.766 vs 0.697 (paired +0.069), only a dog 0.040 vs 0.025.** Next: engine wiring for E18 / E16, parity, phones |
| False-positive rejection | Python + phone | E7 final: cat specificity 0.62 → 0.70 with the verifier, recall 0.93 → 0.91 (MediaPipe detector); household negatives (plush toys, pet beds) unmeasured | yes / Q-T | household negatives (`docs/household/`) |
| Identity across frames (one clip) | Python + phone (Kalman tracker) | none real. E8 semi-synthetic IDF1 0.86 is a software check. | yes / yes | household multi-cat clips |
| Facial cues (eyes, mouth, whiskers, ear tips in the face) | **not implemented** | none. CatFLW (48 landmarks, CC BY-NC 4.0) is blocked on kaggle.com. | - | network (kaggle.com) for research; new annotation for PNC/COM |
| Ears | Python research profile (E24 joint head, `pose.provider: catdog_keypoints_awa_v1`); not on the phone | **E24, AwA-Pose (MIT annotations), photographer-grouped test (79 cats, 51 photographers): ear PCK@0.1 0.914 (joint head) / 0.898 (cat head), meeting the G2 PCK criterion; ear-tip elevation sign agreement 0.93.** Overall cat PCK@0.1 0.760 [0.711, 0.802]. Ear-position kappa not evaluable (no categorical labels). | research only (image licence variants unverified) | verify the AwA2 image licences per image; engine integration (research profile); G2 for a CC BY set |
| Tail (position, movement) | Python research experiment (E24) | E24: tail PCK@0.1 0.51 (joint head), 0.39 (cat head): fails the G2 criterion (≥ 0.75); tail elevation computable on only 11 test cats | research only | more cat tail annotation (G2) |
| Posture (lying / sitting / standing) | Python + phone | E10 against caption-derived **weak** labels, Open Images validation, 41 cats: F1 0.75. Human-verified accuracy unknown; "standing" has 3 cat examples. | yes / Q-T (backbone) | household annotation (posture field) |
| Temporal: locomotion, activity, entered/left view | Python + phone (rule-based) | none real. E8 is semi-synthetic. E18 shows the segments are produced on real video (moving in 50 of 99 clips), but these are not labels. | yes / yes | household timestamps |
| Temporal: actions (eat, drink, groom, litter, scratch, knead, play, sleep) | Python research profile: zero-shot SigLIP behaviour recognition with bouts (`understand/`, `petperception explain`); not on the phone | **E31, Wikimedia Commons / paper / Internet Archive clips (both species pooled, uploader-grouped CV, title labels): macro AP 0.416 [0.351, 0.496] over 9 behaviours (chance 0.073); drinking 0.757, kneading 0.687, eating 0.613, grooming 0.392, object play 0.285, social play 0.190; a trained head was worse (0.248), not adopted.** Benchmark clips: kneading 2/2, object play, grooming recognized; litter and scratching-post use have no data | research (Internet Archive clips, SigLIP data terms) | household labels; temporal model for tail chasing |
| Explaining a behaviour in context (recording + scene + question) | Python research profile: `petperception explain` (default: E31 system, knowledge base of 24 sources; `--system-version 2`: E32 system with recording context, turning-in-place model, parser v2, knowledge base v2 of 33 sources) | **E31 benchmark, 37 questions on real clips: explanation checks 34/34 (grounding, distinguishing observations, no certainty claims, species kept), request separation 33/34, paper authors' interpretation covered 7/7, ranked first 6/7; parsing 27/34 (below the 0.9 target). Cat tail chasing recognized in 1 of 4 clips (the bathtub cat; the task-example clip and both hyperaesthesia videos missed); tail flicking 0/2, tail licking 0/1. E32, 37 fresh questions (both species), one scoring run: v2 vs E31 system - asked behaviours recognized 24 vs 21 of 33, parse 16 vs 13 of 33 (target 0.9 not met), request separation 28 vs 25 of 33 (target 0.9 not met), expert interpretation covered 6 vs 3 of 7; the fresh cat tail-chasing clip recognized by both; v2 not made default (pre-registered rule); post hoc, v2's advantage comes only from the five questions on conditions its knowledge base was extended for (on the other 32 the systems tie on explanation checks; recognition 20 vs 21 of 28).** | research | v3 parser on independently written paraphrases; thresholds for new behaviours; household questions |
| Vocalization present | Python + phone | E1, ESC-50 (40 cat clips, 5-fold CV): F1 0.79 nested (cat), deployed thresholds. E1-CC on CC0/CC BY clips only: 0.76. **E21, FSD50K eval (129 cat clips, 2,284 uploaders, nothing tuned): F1 0.688 [0.623, 0.754], precision 0.65, recall 0.74.** E21b: a threshold re-selected on 41k FSD50K dev clips (0.855) gives 0.702, not a significant gain, so not adopted. Real video, E18: own-species vocalization in 9 of 99 cat clips (unlabelled; Kinetics audio often has music). | yes / A2 + Q-A2 | household audio with sound intervals |
| Vocalization type (meow, purr, hiss, caterwaul) | Python + phone (YAMNet sub-type scores, gated by `artifacts/audio_subtype_validation.json`) | E21, FSD50K eval, deployed point: **meow F1 0.725 [0.637, 0.806], passes** the pre-set gate (F1 ≥ 0.7, lower bound ≥ 0.6); purr 0.701 [0.595, 0.795], fails on its interval (false positives on engines); hiss not evaluable (2 cat-hiss clips; FSD50K "Hiss" is mostly steam and traffic); caterwaul has no label. Only meow is shown; purr and hiss stay in the JSON and can only block interpretations. | yes / Q-T | household audio with type labels; cat-hiss data |
| Vocal context (brushing, isolation, waiting for food) | Python + phone intent stage (`intent.enabled`), as a hint below the gate (`artifacts/intent/cat_audio_context_v1/`) | E2, CatMeows, leave-one-cat-out: macro-F1 0.46 [0.38, 0.55] (MFCC) / 0.46 [0.40, 0.54] (YAMNet) vs 0.34 stratified random. Weak: the head's fitted threshold is never reached, so it never supports a hypothesis. | research only (CC BY-NC 4.0) | more data per context; household owner-reported situations |
| Environmental context (doorbell, vacuum, TV, speech, thunder) | Python + phone (YAMNet classes) | E21, FSD50K eval at the shared threshold 0.3: music 0.770, thunder 0.797; speech 0.560, door 0.413, doorbell 0.378, knock 0.412, alarm 0.413, siren 0.432, fireworks 0.315, dishes 0.277, cutlery 0.241. E21b per-tag thresholds improve door (0.466), knock (0.526), thunder (0.840) and fireworks (0.569); applied in both engines. TV, vacuum and baby cry have no FSD50K label. | yes / Q-T | household audio labels |
| Interaction with a person | Python + phone | E13 photos: only 4 scorable cat pairs, not evaluated. **Real video, E18: in petting/grooming clips where a cat is tracked, a person-contact segment in 35 of 56 (0.63).** No negative label is available on Kinetics. | yes / yes | household recordings (person interaction field) |
| On furniture (bed, couch, chair, table) | Python + phone (containment rule v2) | E15, Open Images, 17 cat positives in test: rule precision and recall 0.97 (both species pooled); end-to-end recall 0.40 | yes / yes | household; pet-specific objects need new collection |
| Under furniture, inside a car | not shipped (failed the E15 gate) | E15: under AP 0.18 (cross-validated) | - | more labelled data |
| Interaction with other animals | **not implemented** | none: no labelled pet-pet source obtainable (VidOR, EgoPet and Visual Genome are blocked or have unclear terms) | - | household multi-pet recordings |
| Situation from synchronized audio + video (petted, groomed) | Python + phone intent stage (trained late fusion, E28; off by default) | **E28, Kinetics validation (YouTube), both species pooled: macro-F1 0.730 [0.678, 0.777] combined, 0.689 visual only, 0.456 audio only; petting cat F1 0.81, grooming cat 0.66.** Coverage 0.88 at selective accuracy 0.76. Replicated on 300 fresh clips (E29): 0.702 combined, 0.705 visual only. | research only (YouTube training clips) | household recordings with owner-reported situations |
| Multimodal interpretation (food, outside, attention, play, alert, tense, warning, relaxed) | Python + phone: rule-based fusion with abstention (default) and the trained intent stage (`intent.enabled`) | **none for intent.** No cat intent annotation exists in any modality; cat links are contexts only (petting -> attention/relaxed as `context_for`). E5 uses synthetic pairings. End to end on the 99 cat clips of E28's test set no hypothesis is supported (situation and context only). | yes / research only (intent stage) | cat intent annotations; household recordings with owner-reported situations (`taxonomy.MISSING`) |
| Breed / morphology | - | E16 per breed (above); no coat or ear-type metadata in any source | - | household metadata (breed, coat) |

## 2. Dogs

| Capability | Software | Behavioural validity (real data) | Release (PNC / COM) | Next step |
| --- | --- | --- | --- | --- |
| Presence and location | Python + phone | Photos, E14 phone detector: image recall 0.866, specificity 0.919 [0.86, 0.97]. Breeds, E16, Oxford-IIIT Pet, 25 breeds: found 0.924; Newfoundland 0.71, keeshond 0.84; heterogeneity p = 0.0005. **Real video, E18, Kinetics (195 clips): a dog track in 0.533 [0.46, 0.60] of clips with the reference detector, but 0.446 [0.38, 0.52] with the phone build's licence-stated detector (paired p = 0.0095); 0.636 without the verifier; walking-the-dog clips are worst.** **Home phone video, E20, CoP3D (147 dog sequences): phone build 0.598 [0.55, 0.64] of frames, better than the reference detector there (paired -0.068 for M2); but 0.137 of dog frames show only a "cat", and the verifier costs dogs 7.7 points without reducing that confusion. Engine level (E20b): a dog track in 0.605 of dog videos, only a cat track in 0.163. Relabelling by the crop classifier (E25) fixes dogs but breaks cats on this footage: not adopted.** | yes / Q-T | **G1b (run B): acceptance 1 met (E4 dog AP50 0.748 with its verifier, at the threshold); home video dog frames found 0.762 vs 0.598 (paired +0.164), only a cat 0.072 vs 0.137.** Next: engine wiring for E18 / E16, parity, phones |
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
| Explaining a behaviour in context (recording + scene + question) | Python research profile: `petperception explain` | **E31 benchmark (shared with cats, see above): dog tail chasing recognized in 2 of 4 clips (Tiira S2, S3; the Bull Terrier puppy and a dog behind a fence missed), flank biting 0/1, staring 0/1; barking and howling from sound 4/4 when present; the authors' interpretation of the three Tiira dogs, the Dachshund and the seizure dog covered 5/5.** Zero-shot pool AP as for cats. **E32 (fresh dog clips): the sensory-impaired compulsive dog's tail chasing recognized by v2 only (score 0.637 at threshold 0.636), a corgi running in snow falsely called tail chasing by both; head tremor and narcolepsy collapse recognized by v2 (entries new in v2), with 7 head-tremor false alarms on other clips; the authors' reading ranked first by v2 in 4 of the 6 dog questions with an expert interpretation (hamartoma, head tremor, epilepsy, narcolepsy; missed: the compulsive dog, where a medical candidate came first, and sleep movements, not parsed).** | research | as for cats |
| Vocalization present | Python + phone | E1, ESC-50 40 dog clips: F1 0.90 nested; identical on the CC0/CC BY subset (all dog clips are CC0/CC BY). **E21, FSD50K eval (170 dog clips): F1 0.728 [0.667, 0.789], precision 0.91, recall 0.61.** E21b: a threshold from FSD50K's CC0 / CC BY dev clips (0.67) is non-inferior (+0.015 [-0.009, +0.042]), a licence-clean candidate for the commercial profile. E18: own-species vocalization in 7 of 195 clips (unlabelled). | yes / A2 | household audio |
| Vocalization type (bark, growl, whine, howl, yip) | Python + phone (YAMNet sub-type scores, gated) | E21, FSD50K eval: **bark F1 0.728 [0.656, 0.797], passes**; growl 0.489 [0.349, 0.607] (dog growl 0.466), fails, and no threshold fixes it (E21b); whine, howl and yip have no FSD50K label. Only bark is shown; a possible growl can only block interpretations (it no longer opens "tense" or "warning"). | yes / Q-T | household audio with type labels |
| Vocal context (bark context, arousal) | Python research experiment (E22) only | E22, Yin & McCowan barks (693 barks, 10 dogs, 6 breeds): leave-one-dog-out macro-F1 **0.722 [0.641, 0.785]** (YAMNet embeddings), leave-one-breed-out 0.696, against 0.33 at random. No stated data licence, one study, recording situations not states: research only. Barkopedia is CC BY-NC-SA 4.0 with emotion (arousal / valence) labels, which the product does not output. | research only | licensed household barks with owner-reported situations |
| Vocal activity / intent from a bark (rest, alerting, seeking attention, begging for food, playing x3, taking a shower) | Python + phone intent stage (`artifacts/intent/dog_audio_activity_v1/`; off by default) | **E26/E27, Barkopedia validation (500 barks; labels by a vision-language model, verified; no dog ids): macro-F1 0.222 [0.184, 0.269] (8 activities) from audio, 0.290 [0.251, 0.328] with the location as context (+0.068 paired); no activity passes the gate; validation coverage 0-4 % at the fitted thresholds; no transfer to Yin barks (0.226, below chance).** | research / legal review (MIT card, audio provenance unresolved) | barks with dog and household ids; household recordings with owner-reported situations |
| Situation from synchronized audio + video (groomed, bathed, trained, walked) | Python + phone intent stage (E28; off by default) | **E28, Kinetics validation: combined macro-F1 0.730 (both species); dogs: walking 0.80, training 0.73, grooming 0.71, bathing 0.67. The scene (environment) is the strongest branch (0.643); audio alone 0.457.** | research only (YouTube training clips) | household recordings |
| Environmental context sounds | Python + phone | E21 (as for cats, the same detector): music and thunder pass the 0.7 bar; door, doorbell, knock, alarm, siren, dishes, cutlery do not | yes / Q-T | household |
| Interaction with a person | Python + phone | E13 photos (dog pairs, Open Images validation + test): AP 0.907 [0.86, 0.95], precision 0.87, recall 0.68 on scorable pairs; end-to-end recall 0.42. **Real video, E18: in bathing/grooming clips where a dog is tracked, a person-contact segment in 36 of 50 (0.72)**; in training/walking clips, 19 of 54 (contact possible, not a false-positive rate). | yes / yes | household |
| On furniture | Python + phone (rule v2) | E15: 14 dog positives in test, pooled precision and recall 0.97; end-to-end recall 0.40 | yes / yes | household |
| Inside a car | not shipped (failed gate: under 20 scored positives) | E15 test AP 0.93 on 15 dog positives | - | more data |
| Interaction with other animals | research class "fights" in E11 only | E11 "fights" F1 0.77 (MammalNet, research) | research only | household multi-pet; VidOR if its terms clear |
| Multimodal interpretation | Python + phone: rules with abstention (default) and the trained intent stage (`intent.enabled`) | **none for intent.** E18: 2 of 195 dog clips produced a non-abstained rule hypothesis (unlabelled). Intent stage end to end on E28's 195 dog test clips and 500 Barkopedia barks: no hypothesis reaches *supported*; the bark channel gives *tentative* hypotheses whose top choice matches the label at the base rate (43.9 % vs 44 %). | yes / research only (intent stage) | synchronized household recordings with intent annotations (`taxonomy.MISSING`) |
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
