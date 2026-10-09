#!/usr/bin/env bash
# Starts the REAL backend (real models from var/models) with a throwaway
# database and media directory, serving the built PWA from web/dist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TMP="$(mktemp -d)"
export PETPERCEPTION_DB="$TMP/e2e.sqlite3"
export PETPERCEPTION_MEDIA_ROOT="$TMP/media"
cd "$ROOT"
mkdir -p "$ROOT/web/test-results"
# Server logs go to a file so the test report stays readable.
exec "$ROOT/.venv/bin/python" -m petperception.cli serve --host 127.0.0.1 --port "${E2E_PORT:-8765}" --workers 1 2>>"$ROOT/web/test-results/e2e-server.log"
