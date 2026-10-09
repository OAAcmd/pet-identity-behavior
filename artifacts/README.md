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
| `behavior_video/dog_behavior_video_v1.npz` | in `results/e11/metrics.json` | `petperception evaluate --experiment e11` | MammalNet dog-breed videos (third-party YouTube uploads, no dataset licence) | not wired into analysis | **research only** |

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
```

The phone does not read these files directly: `petperception export-ondevice`
converts the runtime artifacts into `engine_assets.json` inside the
checksummed on-device bundle (see [docs/ON_DEVICE.md](../docs/ON_DEVICE.md)).
