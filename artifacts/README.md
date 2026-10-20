# Artifacts

Small fitted artifacts trained in this repository, stored as plain numeric
JSON or NumPy `.npz` (numeric and fixed-width string arrays only, loaded with
`allow_pickle=False`; no pickle or other executable serialization). They are
committed to this **private** research repository; that is their durable
storage. Third-party model weights (YAMNet, EfficientDet-Lite,
EfficientNet-Lite) are not stored here: `petperception fetch-models` downloads
them into `var/models/` and checks pinned SHA-256 digests.

Every artifact is regenerated deterministically by the command in the table
(pinned data manifests, fixed seeds). Release suitability per profile is in
`manifests/resources.json` and enforced by `petperception release-check`,
the analysis engine and `petperception export-ondevice`.

| File | SHA-256 | Produced by | Trained on | Used at runtime | Release |
| --- | --- | --- | --- | --- | --- |
| `audio_event_thresholds.json` | `5ae0a08e...6c` | `petperception train --experiment e1` | ESC-50 (CC BY-NC 3.0) | yes: cat/dog decision thresholds on raw YAMNet scores (server and phone) | non-commercial only |
| `e1_mfcc_catdogother_logreg.json` | `8116f0ee...f3` | same | ESC-50 | no (CPU baseline, reproducibility) | non-commercial only |
| `verifier/verifier_<detector>.json` | see below | `petperception evaluate --experiment e7` | Open Images V7 test split (tuning) images, CC BY 2.0 | yes: false-positive verifier, "crop" feature set (server and phone) | all profiles (backbone weights need legal review for commercial use) |
| `verifier/verifier_<detector>_e7selected.json` | see below | same | same | no (protocol-selected feature set, for comparison) | same |
| `keypoints/dog_keypoints_head_v1.npz` + `.json` | `814b0348...7a` | `petperception evaluate --experiment e9` | Ultralytics dog-pose (Stanford Dogs images) | server, research profile only | **research only** |
| `posture/posture_v1.npz` | `aa9bc30c...14` | `petperception evaluate --experiment e10` | Open Images V7 train images (CC BY 2.0) with weak labels from Localized Narratives captions (CC BY 4.0) | yes: coarse posture (server and phone) | all profiles (backbone weights need legal review for commercial use) |
| `verifier/verifier_efficientdet_lite0_mediapipe_int8_v1__efficientnet_lite0_mediapipe_int8_v1.json` | see below | E12 run, then `E7.export_feature_set` (crop set, same procedure as E7) | OI test split (tuning) | yes: verifier of the int8 phone bundle | all profiles (backbone terms apply) |
| `verifier/verifier_efficientdet_lite0_tfhub_rawdecode_v1.json` | see below | `petperception evaluate --experiment e14` (crop set, E7 procedure) | OI test split (tuning) | yes, when the bundle uses the licence-confirmed detector (commercial) | all profiles (backbone terms apply) |
| `behavior_video/dog_behavior_video_v1.npz` | see below | `petperception evaluate --experiment e11` | MammalNet dog-breed videos (third-party YouTube uploads, no dataset licence) | not wired into analysis | **research only** |
| `interaction/person_pet_interaction_v1.npz` | see below | `petperception evaluate --experiment e13` | Open Images V7 relationships (CC BY 4.0), images CC BY 2.0 | see docs/ON_DEVICE_AUDIT.md | all profiles |
| `object_relation/pet_object_relation_v1.npz` | see below | `petperception evaluate --experiment e15` (contains only relations that passed the E15 gate: "on") | Open Images V7 relationships (CC BY 4.0), images CC BY 2.0 | no: evaluated alternative to v2 (learned head) | all profiles |
| `object_relation/pet_object_relation_v2.npz` | `6dec2380...0c` | `petperception evaluate --experiment e15-analysis` (containment rule, tau 0.5 fixed beforehand) | none fitted; evaluated on Open Images V7 relationships | yes: on-furniture candidates (server and phone) | all profiles |
| `audio_event_thresholds_cc.json` | `2f959355...73` | `petperception evaluate --experiment e1-cc` | ESC-50 clips whose own licence is CC0 or CC BY (1,770 of 2,000) | no (candidate for the commercial profile; docs/ASSET_TERMS.md A2) | research, personal; commercial pending Q-A2 |
| `e2_catmeows_context_logreg.json` | `3cee1826...07` | `petperception evaluate --experiment e2` | CatMeows (research, non-commercial) | no (macro-F1 0.46, not shipped) | research only |

Full digests:

```
5ae0a08ea06c9488fe374fcef9768b0603420de2c7fb04f7195e60ad6e53056c  audio_event_thresholds.json
8116f0ee581f5504eb4928de6f0ffc90e1011d6e3632808ecbfcac46dc5541f3  e1_mfcc_catdogother_logreg.json
cef6d6398a1bdd52dd56d2d4a50153c0d1ca4843ee389aaa6444087d68baddd1  verifier/verifier_efficientdet_lite0_mediapipe_v1.json
a513230c6b0a744a8917ea5a52d99f42681e7fe0131138f571e0e2f089260075  verifier/verifier_efficientdet_lite0_mediapipe_v1_e7selected.json
7fd134e2f7f88f4bc58b646689903308be13d6420b2c2ec67c2c4f785100127d  verifier/verifier_efficientdet_lite0_tfhub_v1.json
31e32a1d8aabd454e84d97408fb90ae907e5c0ef63f2a51bc05126ea94270238  verifier/verifier_efficientdet_lite0_tfhub_v1_e7selected.json
9f56ef9ebd79dc28e506493c0ce9e5f2a777d94a876a83d74e7967091011e6eb  verifier/verifier_efficientdet_lite2_mediapipe_v1.json
86014d8b688c4d9ffa9e13bb755f28abb0d9369e6e4c601f9fd7a7378cf51288  verifier/verifier_efficientdet_lite2_mediapipe_v1_e7selected.json
814b0348575624f0a28d573064bbce5216c75d78f2a8f26fbad0b5eb16e5567a  keypoints/dog_keypoints_head_v1.npz
aa9bc30c15285c1a9b76e336bf4d5bef11b1ef1b0027e654497816829dd14d72  posture/posture_v1.npz
2a5d747eb5e27e7b7edfb9cd435c05af0d28e37a937fa2fdde248918d30d52d0  verifier/verifier_efficientdet_lite0_mediapipe_int8_v1__efficientnet_lite0_mediapipe_int8_v1.json
a7f9e8f5a539d156138007a7dedf79c12b54edcdeb0ba92f409828d6750f32e4  behavior_video/dog_behavior_video_v1.npz
7997042681687e864b731c1b518b40940cd73b3e90dfdcb93c3936736e8bca3d  interaction/person_pet_interaction_v1.npz
e98c1b1649d62a1fa94b3e08bd14054423e681da7ee731a440a7a0d72bf70219  verifier/verifier_efficientdet_lite0_tfhub_rawdecode_v1.json
1aecfeadd21d821fecbdb00ce74e42cc52009f7cd263e336aba424a667f62d21  object_relation/pet_object_relation_v1.npz
6dec2380aee529b675ebf66aa9ae9f243453ba8aeddcd99fcdb703423257e90c  object_relation/pet_object_relation_v2.npz
2f959355c0ddb892359e7763872d1a3b75a09304e50a4364c1b53396f1103d73  audio_event_thresholds_cc.json
3cee1826eceb702dadf9e8ea42c62e1157d0498d7ab4b4ae30c5384e94e14307  e2_catmeows_context_logreg.json
```

The phone does not read these files directly: `petperception export-ondevice`
converts the runtime artifacts into `engine_assets.json` inside the
checksummed on-device bundle (see [docs/ON_DEVICE.md](../docs/ON_DEVICE.md)).

## Behaviour / intent heads (`intent/`, 2026-10-20)

JSON heads (`petperception.intent_head/1`: standardization, coefficients,
temperature, abstention threshold, metadata naming data, label layer and
held-out evaluation) and late-fusion stackers
(`petperception.intent_late_fusion/1`). SHA-256 of every file, the training
configuration and the producing commands: `intent/MANIFEST.json`.

| Directory | Produced by | Trained on | Used at runtime | Release (resources.json) |
| --- | --- | --- | --- | --- |
| `intent/kinetics_situation_v1/{full,phone_now}/` | `petperception evaluate --experiment e28` | Kinetics-700 pet clips, 600 train-split clips (YouTube) | intent stage, server (`intent.enabled`) and phone (bundle exported with `--intent`) | **research only** |
| `intent/dog_audio_activity_v1/` | `petperception evaluate --experiment e27` | Barkopedia (MIT card; audio provenance unresolved) | intent stage | research; legal review otherwise |
| `intent/cat_audio_context_v1/` | `python -m petperception.intent.train_heads` | CatMeows (CC BY-NC 4.0) | intent stage (hint only, below the gate) | research; non-commercial review |
| `intent/dog_bark_context_v1/` | `python -m petperception.intent.train_heads` | Yin & McCowan barks via BEANS (no licence stated) | intent stage, `full` variant only | **research only** |


## Behaviour explanation (`understand/`, 2026-10-20)

| File | Produced by | Content | Used at runtime | Release |
| --- | --- | --- | --- | --- |
| `understand/recognizer.json` (checksums: `understand/MANIFEST.json`) | `petperception evaluate --experiment e31` | method `zero_shot` (the trained head was not adopted), per-behaviour thresholds chosen on out-of-fold scores of 278 clips, fixed defaults (0.30 zero-shot, 0.30 YAMNet vocal), provenance | `petperception explain` | research (thresholds chosen on Commons / paper / Internet Archive clips; SigLIP terms) |

The knowledge base (sources, candidates, distinguishing observations) is
code with its own version (`understand/knowledge.py`, `behavior-kb-v1`),
validated by `knowledge.validate()` and the tests.
| `understand/candidates/behavior_head_e31_not_adopted.{json,npz}` | same command | the trained head E31 evaluated and did not adopt (logistic regression per behaviour on 1,537 inputs, fitted on all 278 pool clips; metadata names data and decision) | never (kept for reproducibility and later comparison) | research |
