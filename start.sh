#!/usr/bin/env bash
# MindForge 一键启动脚本（Linux/macOS/WSL）
# 用法: bash start.sh

set -e

echo "==================== MindForge 启动中 ===================="

# 1. 基础设施
echo "[1/4] 启动 Qdrant + Redis ..."
docker compose up -d 2>/dev/null || echo "  ⚠️ Docker 未运行，跳过基础设施"

# 2. 前端构建
echo "[2/4] 构建前端 ..."
cd "$(dirname "$0")/mindforge-web"
npm install --silent 2>/dev/null
npm run build 2>/dev/null || echo "  ⚠️ 前端构建失败，将以 API-only 模式运行"
cd ..

# 3. 后端启动
echo "[3/4] 启动后端 ..."
cd src
uvicorn mindforge.api.server:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

sleep 2

# 4. 验证
echo "[4/4] 验证服务 ..."
if curl -s --max-time 3 http://localhost:8000/api/v1/health >/dev/null 2>&1; then
    echo ""
    echo "==================== 启动成功 ===================="
    echo "  前端 + API : http://localhost:8000"
    echo "  API 文档   : http://localhost:8000/docs"
    echo "  按 Ctrl+C 停止所有服务"
    echo "=================================================="
else
    echo ""
    echo "==================== 部分启动 ===================="
    echo "  后端可能仍在初始化中，稍后访问 http://localhost:8000"
    echo "  查看日志: tail -f nohup.out"
    echo "=================================================="
fi

wait $BACKEND_PID
