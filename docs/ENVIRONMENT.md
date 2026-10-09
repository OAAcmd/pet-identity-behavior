# Phase 1 Workspace Environment Report (2026-10-09)

Observed at the start of Phase 1 and re-checked with `uv run petperception doctor --network`.
This is a development workspace (ephemeral cloud container), not the future hosting environment.

| Item | Observed |
| --- | --- |
| OS | Ubuntu 24.04.5 LTS, Linux 6.18 (KVM guest) |
| CPU | Intel Xeon @ 2.10 GHz, 4 vCPUs (AVX-512, AMX flags present) |
| RAM | 15.7 GB, no swap |
| Disk | ~31 GB available at start |
| GPU | none (no `/dev/nvidia*`, no `nvidia-smi`) |
| Python | system 3.11, 3.12.3, 3.13.16 available; project venv uses **3.12.3** |
| Package manager | uv 0.11.32; lock `uv.lock` (60 packages), export `requirements.lock.txt` (hash-pinned) |
| Key packages | numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, librosa 1.0.0, soundfile 0.14.0, opencv-python-headless 5.0.0.93, ai-edge-litert 2.3.0, pydantic 2.14.0, fastapi 0.143.0, uvicorn 0.54.0, pytest 9.1.1 |
| PyTorch | not installed: no component needs it (frozen TFLite models + scikit-learn); CPU wheels host `download.pytorch.org` is denied anyway |
| FFmpeg | 6.1.1 (ffmpeg + ffprobe, with libx264, libvpx, libopus, libmp3lame) |
| Docker | CLI 29.8.2; daemon started manually in the session; image `pet-perception:phase1` built (1.82 GB) and tested |
| Webcam / microphone / owner LAN | not available and not assumed |

## Network (egress via policy proxy)

| Reachable | Denied (HTTP 403 at CONNECT or DNS failure) |
| --- | --- |
| pypi.org, files.pythonhosted.org, storage.googleapis.com, raw.githubusercontent.com, anonymous git clone of public GitHub repos, open-images-dataset.s3.amazonaws.com, archive.ubuntu.com, Docker Hub | zenodo.org, huggingface.co, docs.ultralytics.com, github.com release assets/API (unattached repos), download.pytorch.org, cocodataset.org, *.wikimedia.org, www.robots.ox.ac.uk, deb.debian.org, kaggle.com, ai.google.dev / developers.google.com (pages) |

Consequences are recorded per resource in `manifests/resources.json` and
`docs/DATA_AND_MODEL_REGISTER.md`. No paid API or paid compute was used.
