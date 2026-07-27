#!/usr/bin/env bash
set -Eeuo pipefail

DEV_MODE=false
if [[ "${1:-}" == "--dev" ]]; then
    DEV_MODE=true
elif [[ -n "${1:-}" ]]; then
    echo "Usage: bash start.sh [--dev]" >&2
    exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -f .env ]]; then
    echo "Missing .env. Copy .env.example to .env and configure secrets." >&2
    exit 1
fi
if [[ ! -f requirements.lock ]]; then
    echo "Missing requirements.lock. Regenerate it from uv.lock." >&2
    exit 1
fi

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
    [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

command -v python3 >/dev/null || {
    echo "python3 is required." >&2
    exit 1
}
command -v npm >/dev/null || {
    echo "npm is required." >&2
    exit 1
}
command -v docker >/dev/null || {
    echo "Docker with Compose is required to start infrastructure." >&2
    exit 1
}

echo "[1/5] Starting Qdrant, Redis, and PostgreSQL..."
docker compose up -d qdrant redis postgres

echo "[2/5] Installing backend dependencies..."
python3 -m pip install --require-hashes -r requirements.lock

# Read only the values used by this shell. The application, Vite, and Docker
# Compose load the same root .env independently.
eval "$(
    python3 - <<'PY'
import shlex
from dotenv import dotenv_values

values = dotenv_values(".env")
defaults = {
    "API_HOST": "127.0.0.1",
    "API_PORT": "8000",
    "STARTUP_READY_ATTEMPTS": "60",
    "STARTUP_READY_INTERVAL_SECONDS": "1",
}
for key, default in defaults.items():
    value = values.get(key) or default
    print(f"{key}={shlex.quote(value)}")
PY
)"

[[ "$API_PORT" =~ ^[0-9]+$ ]] && ((API_PORT >= 1 && API_PORT <= 65535)) || {
    echo "API_PORT must be an integer between 1 and 65535." >&2
    exit 2
}
[[ "$STARTUP_READY_ATTEMPTS" =~ ^[0-9]+$ ]] \
    && ((STARTUP_READY_ATTEMPTS >= 1)) || {
    echo "STARTUP_READY_ATTEMPTS must be a positive integer." >&2
    exit 2
}
[[ "$STARTUP_READY_INTERVAL_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "STARTUP_READY_INTERVAL_SECONDS must be a positive number." >&2
    exit 2
}

echo "[3/5] Installing frontend dependencies..."
npm --prefix mindforge-web ci

if "$DEV_MODE"; then
    echo "[4/5] Starting Vite development server..."
    npm --prefix mindforge-web run dev &
    FRONTEND_PID=$!
else
    echo "[4/5] Building frontend..."
    npm --prefix mindforge-web run build
fi

echo "[5/5] Starting FastAPI on ${API_HOST}:${API_PORT}..."
UVICORN_ARGS=(
    mindforge.api.server:app
    --app-dir src
    --host "$API_HOST"
    --port "$API_PORT"
)
if "$DEV_MODE"; then
    UVICORN_ARGS+=(--reload)
fi
python3 -m uvicorn "${UVICORN_ARGS[@]}" &
BACKEND_PID=$!

for ((attempt = 1; attempt <= STARTUP_READY_ATTEMPTS; attempt++)); do
    if curl --fail --silent \
        "http://127.0.0.1:${API_PORT}/api/v1/ready" >/dev/null; then
        echo "MindForge is ready: http://127.0.0.1:${API_PORT}"
        wait "$BACKEND_PID"
        exit $?
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        wait "$BACKEND_PID"
        exit $?
    fi
    sleep "$STARTUP_READY_INTERVAL_SECONDS"
done

echo "MindForge failed its readiness check after ${STARTUP_READY_ATTEMPTS} attempts." >&2
exit 1
