# Training program

How models are trained here, how to reproduce every trained artifact, and
which training needs compute this workspace does not have. Training happens
in a compute environment; phones only run exported models
([ON_DEVICE.md](ON_DEVICE.md)).

## Design choice: small trained heads on shared frozen backbones

Every model trained so far is a small head on top of pretrained, frozen
networks that already ship to the phone:

* EfficientDet-Lite0 (COCO) finds cats, dogs and people;
* EfficientNet-Lite0 (ImageNet) is run once per crop and its outputs
  (1000-class softmax, three feature maps, pooled 1280-d embedding) feed every
  visual head: false-positive verifier, posture, dog keypoints, interaction,
  video behavior;
* YAMNet (AudioSet) produces audio event scores.

Reasons: (1) the phone runs one backbone pass per crop for all heads, which
keeps the per-frame budget ([DEVICE_TIERS_AND_BUDGETS.md](DEVICE_TIERS_AND_BUDGETS.md));
(2) the labelled data available is small (hundreds to a few thousand
examples per task), where frozen features plus a regularised head is the
appropriate capacity; (3) it trains on the 4-core CPU of this workspace in
minutes. The cost: backbones are not adapted to pets or household footage,
and the heads inherit the backbones' blind spots.

## Trained artifacts and how to reproduce them

All commands: `uv sync --extra dev --extra train`, then
`uv run petperception fetch-models` and the `fetch-data` command listed.
Data manifests (image lists, checksums) are committed under `manifests/`;
downloaded files are verified against them before use. Seeds are fixed in
the experiment modules. Each run writes its configuration, split sizes,
selected hyperparameters and metrics to `results/<exp>/metrics.json`.

| Artifact | Experiment | Data (fetch-data) | Model | Selection data | Final evaluation | Seed(s) | CPU time here |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `artifacts/audio_event_thresholds.json` | E1 | `esc50` | per-class thresholds on YAMNet scores | ESC-50 dev folds | held-out fold (5-fold CV) | fixed folds | ~3 min |
| `artifacts/verifier/verifier_<detector>.json` | E7 | `openimages-tuning`, `openimages-catdog` | logistic regression per class on crop features | OI test split, out-of-fold by Flickr author | E4 OI validation subset, once | n/a (deterministic LR) | ~3 min per detector (features cached) |
| `artifacts/keypoints/dog_keypoints_head_v1.npz` | E9 | `dog-pose` | JAX/optax head: spatial softmax heatmaps + offsets + visibility on 3 feature maps | 10 % image hold-out of dog-pose train (seed 20261011) | dog-pose val (1,703 instances), once | 20261011 | 24 min (30 epochs, best epoch 13 by dev PCK@0.1; learning curve in `results/e9/training_log.json`) |
| `artifacts/posture/posture_v1.npz` | E10 | `posture-weak` | multinomial logistic regression | OI test split | OI validation split, once | n/a | ~10 min (detection + features) |
| `artifacts/behavior_video/dog_behavior_video_v1.npz` | E11 | MammalNet annotations (see register) | multinomial logistic regression on segment features | MammalNet validation | MammalNet test, once | n/a | ~1 h (video streaming dominates) |
| quantized-variant verifiers / posture refits | E12 | as E7 / E10 | as E7 / E10 | as E7 / E10 | as E7 / E10 | as E7 / E10 | ~30 min |
| `artifacts/interaction/person_pet_interaction_v1.npz` | E13 | `openimages-vrd` | logistic regression on pair geometry (+ union-crop embedding) | 10 % image hold-out of OI train (seed 20261013) | OI validation + test, once | 20261013 | ~15 min |

Checkpoints: for the heads above the final parameters are the checkpoint;
they are small enough to commit (numeric JSON / NPZ without pickle). E9 keeps
the best development epoch; a run whose loss becomes non-finite aborts and is
not exported (this happened once and is recorded in the E9 history).

Rules that every training run follows: selection decisions (hyperparameters,
thresholds, feature sets, gates) use only the selection data; the final split
is evaluated once after everything is frozen; confidence intervals resample
the strongest available grouping unit (image, Flickr author, breed, video).

## What the CPU cannot do, and the prepared GPU jobs

The following jobs are justified by measured gaps but exceed this workspace
(4 CPU cores, no GPU) or need data that is not available. They are specified
so they can be run reproducibly once compute and data exist. **None has been
run.** Code that does not exist yet is marked as such.

| Job | Why (measured gap) | Data | Approach | Compute estimate | Status |
| --- | --- | --- | --- | --- | --- |
| G1 Detector adaptation to household footage | E4/E7 measured only on web photos; small-animal recall 0.27; household false positives unknown | labelled household frames (to be collected, [EVALUATION_PLAN_REAL_PAIRED.md](EVALUATION_PLAN_REAL_PAIRED.md)) | fine-tune EfficientDet-Lite0 (TF Model Maker-style recipe) on cat/dog/person, re-export float and int8, re-run E4/E7/E12 protocols | 1 GPU (16 GB), 6-12 h | blocked on data; training script not written |
| G2 End-to-end dog keypoints | E9 head on frozen features: per-breed PCK@0.1 down to 0.73, tail keypoints down to 0.0 for some breeds | licensed dog keypoint data (dog-pose is Research-Only) | mobile top-down pose network (e.g. a MobileNet/EfficientNet-Lite backbone, heatmap + offset head) trained end-to-end, int8 export, LiteRT.js parity | 1 GPU, 8-24 h | blocked on licensed data; script not written |
| G3 Temporal action model | rule-based segments validated only on semi-synthetic video; E11 is segment-level mean/max pooling | licensed cat and dog video with action labels | streaming mobile video model (e.g. MoViNet-A0-stream class) or temporal head over per-frame embeddings; int8 export | 1-4 GPUs, 1-3 days | blocked on data; script not written |
| G4 Audio context heads | CatMeows / Barkopedia blocked | CatMeows (zenodo.org), Barkopedia (huggingface.co) | logistic head on YAMNet embeddings (E2/E3 code exists) | CPU is enough | blocked on network access |
| G5 Distilled multi-task student | phone budget; several heads share one backbone | outputs of larger teachers on unlabelled household video (consented) | distil detector + backbone heads into one compact model, quantization-aware training, re-run all protocols including rejection behaviour | 1 GPU, 1-2 days | not started |

Container for GPU jobs: the repository's `pyproject.toml` defines the Python
environment; GPU jobs add `jax[cuda12]` (JAX heads) or a TensorFlow/Keras
toolchain for detector fine-tuning (not part of this repository yet). Each
job must write the same `results/<exp>/metrics.json` structure and export
through `petperception export-ondevice` so browser parity can be re-measured.
