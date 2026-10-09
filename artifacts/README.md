# Artifacts

Small fitted artifacts, stored as plain numeric JSON (no pickle or other
executable serialization). Both derive from ESC-50 (collection license
CC BY-NC 3.0, per-clip Freesound licenses in `manifests/esc50_files.csv`):
non-commercial research use with attribution to K. J. Piczak (ESC-50) and the
clip authors. They are committed to this **private** research repository; that
is their durable storage.

| File | SHA-256 | Produced by | Used at runtime |
| --- | --- | --- | --- |
| `audio_event_thresholds.json` | `5ae0a08ea06c9488fe374fcef9768b0603420de2c7fb04f7195e60ad6e53056c` | `petperception train --experiment e1` | yes: cat/dog decision thresholds on raw YAMNet scores |
| `e1_mfcc_catdogother_logreg.json` | `8116f0ee581f5504eb4928de6f0ffc90e1011d6e3632808ecbfcac46dc5541f3` | same | no (CPU baseline, reproducibility) |

Model weights (YAMNet, EfficientDet-Lite) are third-party files fetched by
`petperception fetch-models` into `var/models/` with pinned SHA-256; they are not
redistributed here. Rerunning E1 rewrites these files (deterministic given the
pinned data and seeds).
