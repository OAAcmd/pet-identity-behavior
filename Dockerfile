# syntax=docker/dockerfile:1.7
# Production image: builds the PWA (stage "web") and runs the API, workers and
# static app in one process (stage "runtime"). See docs/DEPLOYMENT.md.
#
# * Base images are pinned by digest. REGISTRY defaults to Google's Docker Hub
#   mirror (anonymous Docker Hub pulls are rate limited); use docker.io/library
#   for the same digests.
# * Python dependencies come from requirements.lock.txt (exported from uv.lock,
#   with hashes); JavaScript dependencies from web/package-lock.json (npm ci).
# * Model weights are NOT in the image. They are fetched with checksum
#   verification into the /models volume (`petperception fetch-models`, run by
#   the compose "models" service) or mounted read-only.
# * Persistent state (SQLite database, retained media) lives in the
#   /var/lib/petperception volume. Container files are otherwise disposable.
ARG REGISTRY=mirror.gcr.io/library
ARG NODE_IMAGE=${REGISTRY}/node:22-bookworm-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392
ARG UBUNTU_IMAGE=${REGISTRY}/ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

FROM ${NODE_IMAGE} AS web
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm_config_update_notifier=false
WORKDIR /web
COPY web/package.json web/package-lock.json ./
# Optional extra CA bundle for builds behind a TLS-intercepting proxy:
#   docker build --secret id=extra_ca,src=/path/ca.pem ...
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/extra_ca; fi; \
    npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM ${UBUNTU_IMAGE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv ffmpeg ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock.txt ./
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    python3.12 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir "uv==0.11.32" \
 && UV_NATIVE_TLS=1 /opt/venv/bin/uv pip install --no-cache --python /opt/venv/bin/python --require-hashes -r requirements.lock.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY manifests ./manifests
COPY artifacts ./artifacts
COPY --from=web /web/dist ./web/dist
# Assets whose terms do not permit the target release are not shipped:
# research-only artifacts never; CC BY-NC-derived ones not in commercial builds.
ARG RELEASE_PROFILE=personal_noncommercial
RUN cp configs/production.example.yaml configs/production.yaml \
 && rm -rf artifacts/keypoints \
 && if [ "$RELEASE_PROFILE" = "commercial" ]; then rm -f artifacts/audio_event_thresholds.json artifacts/e1_mfcc_catdogother_logreg.json; fi \
 && if [ "$RELEASE_PROFILE" = "research" ]; then echo "use a source checkout for research deployments" >&2; exit 1; fi
ENV PETPERCEPTION_RELEASE_PROFILE=${RELEASE_PROFILE}

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PETPERCEPTION_HOME=/app \
    PETPERCEPTION_CONFIG=/app/configs/production.yaml \
    PETPERCEPTION_MODELS_ROOT=/models

RUN useradd --create-home --uid 10001 app \
 && mkdir -p /models /var/lib/petperception \
 && chown -R app /models /var/lib/petperception
USER app
VOLUME ["/var/lib/petperception", "/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status==200 else 1)"
ENTRYPOINT ["tini", "--", "python", "-m", "petperception.cli"]
CMD ["serve"]
