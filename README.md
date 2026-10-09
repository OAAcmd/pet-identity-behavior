# pet-identity-behavior

Personal experiments in pet identification and everyday behavior observation.

**Phase 1: Pet Perception and Intent Research Engine** - an observation-first,
CPU-only engine for cats and dogs. Given a local audio, video or image file it
reports *observations* (detected animals, sound events, movement, owner-zone
occupancy, timestamps, quality limits), then *experimental, uncalibrated
hypotheses* tied to evidence, then *restrained suggestions*. It abstains by
default. It does not translate animal language, read emotions, identify
individual pets, or diagnose anything.

| Document | Content |
| --- | --- |
| [docs/ALGORITHM_REPORT.md](docs/ALGORITHM_REPORT.md) | methods, protocols, measured results (E1-E6), error analysis |
| [docs/DATA_AND_MODEL_REGISTER.md](docs/DATA_AND_MODEL_REGISTER.md) | every dataset/model: URL, revision, checksums, counts, terms, blockers |
| [docs/API_CONTRACT.md](docs/API_CONTRACT.md) | HTTP API, result schema field guide, security posture |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md), [docs/DATA_LIMITATIONS.md](docs/DATA_LIMITATIONS.md) | intended use, failure modes, data limits |
| [docs/PROGRESS.md](docs/PROGRESS.md), [docs/PHASE1_ACCEPTANCE.json](docs/PHASE1_ACCEPTANCE.json) | what was done, verified, blocked; phase gate |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | workspace hardware/network report |

## Install

Requirements: Linux, Python 3.11-3.12 (3.12 tested), FFmpeg/ffprobe on `PATH`,
[uv](https://docs.astral.sh/uv/). No GPU needed.

```bash
uv sync --extra dev                       # creates .venv from uv.lock (reproducible)
uv run petperception doctor               # environment, models, datasets (add --network to probe hosts)
uv run petperception fetch-models         # YAMNet + EfficientDet-Lite0, SHA-256 verified (~18 MB)
uv run petperception fetch-models --include-optional   # also EfficientDet-Lite2 (E4 comparison)
```

`requirements.lock.txt` is a hash-pinned export of `uv.lock` (used by the Dockerfile).
Models go to `var/models/`, datasets to `var/data/` (both git-ignored; override with
`PETPERCEPTION_MODELS_ROOT` / `PETPERCEPTION_DATA_ROOT`). Inference never downloads.

## Analyze a file

```bash
uv run petperception analyze clip.mp4 --zones configs/zones.example.json \
    --out result.json --report report.md
uv run petperception analyze meow.wav --modalities audio
uv run petperception analyze clip.mp4 --fusion-method product_of_experts   # experimental scorer
```

Zones are optional owner-drawn polygons in normalized image coordinates with a
fixed vocabulary (`food_area`, `exit_door`, `rest_area`, `play_area`, ...); see
[`configs/zones.example.json`](configs/zones.example.json). Output follows
schema 1.0.0 ([docs/analysis_result.schema.json](docs/analysis_result.schema.json)).
Worked examples (synthetic pairings of real public media):
[`examples/offline/`](examples/offline/).

## Serve the API

```bash
uv run petperception serve                              # 127.0.0.1:8000, loopback only
PETPERCEPTION_API_TOKEN=change-me uv run petperception serve --host 0.0.0.0   # non-loopback requires a token
uv run petperception export-openapi                     # docs/openapi.json + result JSON schema
```

Endpoints: `GET /v1/health`, `GET /v1/capabilities`, `GET /v1/schema/analysis-result`,
`POST /v1/analyses`, `GET /v1/analyses/{id}`, `GET /v1/analyses/{id}/result`,
`DELETE /v1/analyses/{id}`. Details: [docs/API_CONTRACT.md](docs/API_CONTRACT.md).

## Reproduce the experiments

```bash
uv run petperception fetch-data esc50                   # git clone @33c8ce9e + checksum check (~880 MB)
uv run petperception fetch-data openimages-catdog       # annotations + 823 images, pinned SHA-256 (~260 MB)
uv run petperception evaluate --experiment e1           # ESC-50 cat/dog sound events (also: train --experiment e1)
uv run petperception evaluate --experiment e4           # Open Images cat/dog detection (Lite0 + Lite2)
uv run petperception evaluate --experiment e5           # fusion/abstention scenarios (builds var/examples)
uv run petperception evaluate --experiment e6           # runtime (needs e5 media)
uv run petperception evaluate --experiment e2           # CatMeows: writes 'blocked' status until data is fetched
uv run petperception evaluate --experiment e3           # Barkopedia: 'pending' until audited
uv run petperception make-examples                      # regenerate examples/offline/
uv run pytest                                           # 54 tests; markers: synthetic, integration
uv run pytest -m "not integration"                      # without fetched models/data
```

Blocked in the Phase 1 workspace (network policy): CatMeows (`zenodo.org`),
AST and Barkopedia (`huggingface.co`), Ultralytics dog-pose. Continuation
commands are in [docs/PHASE1_ACCEPTANCE.json](docs/PHASE1_ACCEPTANCE.json).

Configuration: [`configs/cpu.yaml`](configs/cpu.yaml) (default, bounded CPU) and
[`configs/accelerator.example.yaml`](configs/accelerator.example.yaml)
(larger CPU host; no GPU path was available or tested). Select with
`--config` or `PETPERCEPTION_CONFIG`.

## Docker (CPU)

```bash
docker build -t pet-perception:phase1 .                 # behind a TLS proxy add: --network host --secret id=extra_ca,src=ca.pem
docker run --rm -v $PWD/var/models:/models:ro pet-perception:phase1 doctor
docker run --rm -e PETPERCEPTION_API_TOKEN=change-me -p 127.0.0.1:8000:8000 \
    -v $PWD/var/models:/models:ro pet-perception:phase1   # serve
```

Built and tested in the Phase 1 workspace (doctor, analyze, token-protected
serve round-trip). Models are mounted, not baked in (or build with
`--build-arg FETCH_MODELS=1`).

## Repository layout

```
src/petperception/   ingest/ audio/ vision/ context/ fusion/ schemas/ explain/ eval/ serve/ data/ pipeline.py cli.py
configs/             cpu.yaml, accelerator.example.yaml, zones.example.json
manifests/           resources.json (register), per-file dataset manifests and pinned checksums
artifacts/           small fitted artifacts (numeric JSON) with terms
results/e1..e6/      metrics, predictions, logs per experiment
examples/offline/    example inputs (zones), JSON results and Markdown reports
docs/                reports, register, API contract, OpenAPI, acceptance
tests/               pytest suite (synthetic vs integration markers)
```

Raw datasets, media, model weights, caches and credentials are never committed.

## License and terms

Project code: private research code. Third-party data and model terms (ESC-50
CC BY-NC 3.0; Open Images CC BY 4.0 / CC BY 2.0 per image; YAMNet/EfficientDet
Apache-2.0 as stated or claimed) are listed in the register. Research use only;
commercial permission is not assumed.
