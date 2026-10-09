# syntax=docker/dockerfile:1.7
# CPU-only image for the Phase 1 research API/CLI.
# Models are NOT baked in by default: mount them at /models (read-only) or
# build with --build-arg FETCH_MODELS=1 to run the explicit, checksum-verified
# `petperception fetch-models` step. Datasets are never included.
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock.txt ./
# Dependencies come from requirements.lock.txt (exported from uv.lock with hashes)
# and are installed with uv, matching the development environment.
# Optional extra CA bundle for builds behind a TLS-intercepting proxy:
#   docker build --secret id=extra_ca,src=/path/ca.pem ...
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

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PETPERCEPTION_HOME=/app \
    PETPERCEPTION_MODELS_ROOT=/models \
    PETPERCEPTION_DATA_ROOT=/data \
    PETPERCEPTION_JOBS_ROOT=/tmp/petperception-jobs

ARG FETCH_MODELS=0
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    mkdir -p /models /data \
 && if [ "$FETCH_MODELS" = "1" ]; then python -m petperception.cli fetch-models; fi

RUN useradd --create-home --uid 10001 app && chown -R app /models /data
USER app
EXPOSE 8000
# Binding 0.0.0.0 inside the container requires PETPERCEPTION_API_TOKEN (the
# server refuses an unauthenticated non-loopback bind). TLS is a Phase 2 item.
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health').status==200 else 1)"
ENTRYPOINT ["python", "-m", "petperception.cli"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
