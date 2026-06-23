#!/usr/bin/env bash
# ============================================================================
# MindForge 一键启动脚本
# 用法: bash start.sh
# 自动完成: 基础设施 → 依赖安装 → 前端构建 → 后端启动 → 健康检查
# ============================================================================

set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $1"; }
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       MindForge 自适应研究助理系统        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# 1. 基础设施: Docker (Qdrant + Redis)
# ============================================================================
log "1/5  启动基础设施 (Docker) …"
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    docker compose up -d 2>/dev/null && \
        ok "Qdrant (6333) + Redis (6380) 已启动" || \
        warn "部分容器启动失败，检查 docker compose 日志"
else
    warn "Docker 未运行，跳过。检索功能不可用"
fi

# ============================================================================
# 2. 后端依赖
# ============================================================================
log "2/5  检查后端依赖 …"
if python -c "import mindforge" 2>/dev/null; then
    ok "mindforge 已安装"
else
    warn "安装后端依赖 …"
    pip install -e ".[dev]" -q 2>/dev/null && ok "依赖安装完成" || fail "依赖安装失败"
fi

# ============================================================================
# 3. 环境变量
# ============================================================================
log "3/5  检查环境变量 …"
if [ -f ".env" ]; then
    ok ".env 文件存在"
else
    cp .env.example .env 2>/dev/null && \
        warn ".env 已从模板创建，请编辑填入 API Key" || \
        warn ".env 和 .env.example 均缺失"
fi

# 检查 API Key 是否配置
source .env 2>/dev/null || true
if [ -z "$LLM_DEEPSEEK_API_KEY" ] && [ -z "$LLM_OPENAI_API_KEY" ]; then
    warn "未检测到 LLM API Key，研究任务将降级为文档检索模式"
    warn "编辑 .env 填入 LLM_DEEPSEEK_API_KEY 以启用 LLM"
fi

# ============================================================================
# 4. 前端构建
# ============================================================================
log "4/5  构建前端 …"
FRONTEND_DIR="$ROOT_DIR/mindforge-web"
if [ -f "$FRONTEND_DIR/package.json" ]; then
    cd "$FRONTEND_DIR"
    if [ ! -d "node_modules" ]; then
        npm install --silent 2>/dev/null && ok "npm 依赖已安装" || warn "npm install 失败"
    fi
    npm run build 2>/dev/null && ok "前端构建完成 → dist/" || warn "前端构建失败，将以 API-only 模式运行"
    cd "$ROOT_DIR"
else
    warn "前端目录不存在，跳过"
fi

# ============================================================================
# 5. 后端服务
# ============================================================================
log "5/5  启动后端服务 …"

# 杀掉已占用的 8000 端口
if lsof -ti:8000 &>/dev/null 2>&1; then
    warn "端口 8000 已被占用，尝试释放 …"
    kill "$(lsof -ti:8000)" 2>/dev/null && sleep 1 || true
fi

cd "$ROOT_DIR/src"
uvicorn mindforge.api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    &

BACKEND_PID=$!
cd "$ROOT_DIR"

# 等待后端就绪
RETRIES=0
MAX_RETRIES=30
while [ $RETRIES -lt $MAX_RETRIES ]; do
    if curl -s --max-time 2 http://localhost:8000/api/v1/health >/dev/null 2>&1; then
        break
    fi
    sleep 1
    RETRIES=$((RETRIES + 1))
done

# ============================================================================
# 结果
# ============================================================================
echo ""
if [ $RETRIES -lt $MAX_RETRIES ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║          🚀 MindForge 启动成功            ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  Web 前端+API  ${CYAN}http://localhost:8000${NC}"
    echo -e "  OpenAPI 文档  ${CYAN}http://localhost:8000/docs${NC}"
    echo -e "  健康检查      ${CYAN}http://localhost:8000/api/v1/health${NC}"
    echo ""
    echo -e "  停止服务: ${YELLOW}kill $BACKEND_PID${NC}"
else
    echo -e "${YELLOW}╔══════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║        后端可能仍在初始化中 …            ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  请稍后访问 ${CYAN}http://localhost:8000${NC}"
fi
echo ""

# 前台等待后端进程，Ctrl+C 停止
wait $BACKEND_PID
