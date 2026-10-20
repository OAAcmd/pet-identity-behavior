# Models trained in this project

Weights and training state that this project produced, committed with
ordinary Git. Each file is under 50 MB, so neither Git LFS nor a release
host is needed. Third-party pretrained weights are not stored here: they are
downloaded with checksums by `petperception fetch-models`. Raw datasets are
never committed; their manifests and checksums live in `manifests/`.

| Folder | What | Evaluation |
| --- | --- | --- |
| `g1/g1a_full_mnv2_sb4096/` | G1 pet-and-household detector (MobileNetV2-256 RetinaNet, 14 classes), full-data CPU run, final checkpoint (epoch 4 of 4) with optimizer state, and exports `model_float32.tflite`, `model_float16.tflite`, `model_int8_ptq.tflite` | `results/g1/full_run.json`, `results/g1/g1a_full_*`; docs/TRAINING_PROGRAM.md G1 |

Each folder has a `MANIFEST.json` with the SHA-256 of every file, the
training position, and the configuration and data hashes.

Use an export directly:

```python
from pathlib import Path
from petperception.vision.detector import EfficientDetDetector
det = EfficientDetDetector(Path("models/g1/g1a_full_mnv2_sb4096/export/model_float32.tflite"))
```

Restore the training state, check it, and continue (G1 environment;
`docs/TRAINING_PROGRAM.md`, "Launch"):

```bash
var/venv-g1/bin/python -I training/g1/vault.py restore --src models/g1/g1a_full_mnv2_sb4096 \
    --run-dir var/training/g1/runs/g1a_full_mnv2_sb4096
var/venv-g1/bin/python -I training/g1/train_resumable.py --view var/training/g1/views/full \
    --run-dir var/training/g1/runs/g1a_full_mnv2_sb4096 --model MOBILENET_V2 --batch-size 16 \
    --stages 1:decoder_head,3:all --shuffle-buffer 4096 --validate-only
```

Terms: trained on Open Images CC BY 2.0 images, initialised from Model
Maker's COCO-trained checkpoint. For research and non-commercial use
(docs/ASSET_TERMS.md).
