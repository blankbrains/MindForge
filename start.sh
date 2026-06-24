#!/bin/bash
# MindForge 一键启动脚本
# Usage: bash start.sh [--dev]

DEV_MODE=false
if [ "$1" = "--dev" ]; then
    DEV_MODE=true
fi

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=== MindForge Startup ==="
echo "[1/5] Stopping old services on port 8000..."
kill $(lsof -ti :8000 2>/dev/null) 2>/dev/null || true

echo "[2/5] Starting Docker infrastructure..."
docker compose up -d 2>&1 || echo "  Docker警告: 部分服务启动失败，将以降级模式继续"

echo "[3/5] Installing Python dependencies..."
pip install -e "." --quiet 2>/dev/null || true

echo "[4/5] Building frontend..."
if [ -d "mindforge-web" ]; then
    cd mindforge-web
    npm install --silent 2>/dev/null || true
    npm run build 2>/dev/null || true
    cd "$PROJECT_DIR"
fi

set -e
echo "[5/5] Starting backend server..."
cd src
if $DEV_MODE; then
    python3 -m uvicorn mindforge.api.server:app --reload --host 0.0.0.0 --port 8000
else
    python3 -m uvicorn mindforge.api.server:app --host 0.0.0.0 --port 8000
fi
