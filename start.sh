#!/usr/bin/env bash
# ============================================================================
# MindForge 一键启动脚本
# 用法: bash start.sh [--dev]
#   --dev  开发模式（uvicorn --reload，代码变更自动重启）
#
# 流程: 关闭旧服务 → 基础设施 → 依赖 → 环境变量 → 前端 → 后端 → 健康检查
# ============================================================================

set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

DEV_MODE=false
[[ "$1" == "--dev" ]] && DEV_MODE=true

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $1"; }
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

cleanup() {
    log "正在关闭 MindForge …"
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null && ok "后端已停止"
    log "再见！"
}
trap cleanup EXIT INT TERM

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       MindForge 自适应研究助理系统        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
$DEV_MODE && echo -e "  ${YELLOW}开发模式（热重载）${NC}" || echo -e "  ${GREEN}生产模式${NC}"
echo ""

# ============================================================================
# 0. 关闭所有旧服务
# ============================================================================
log "0/6  关闭已有服务 …"

# 停止后端
if lsof -ti:8000 &>/dev/null 2>&1; then
    kill "$(lsof -ti:8000)" 2>/dev/null && ok "后端已停止 (port 8000)" || warn "后端停止失败"
    sleep 1
else
    ok "后端未运行"
fi

# 停止 Docker 容器
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    cd "$ROOT_DIR"
    docker compose down 2>/dev/null && ok "Docker 容器已停止" || true
else
    ok "Docker 未运行"
fi

# ============================================================================
# 1. 基础设施: Docker (Qdrant + Redis + PostgreSQL)
# ============================================================================
log "1/6  启动基础设施 (Docker) …"
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    docker compose up -d 2>/dev/null && \
        ok "Qdrant(6333) + Redis(6380) + PostgreSQL(5432) 已启动" || \
        warn "部分容器启动失败，检查 docker compose 日志"
    # 等待 PostgreSQL 就绪
    for i in $(seq 1 15); do
        docker exec mindforge-postgres-1 pg_isready -U mindforge &>/dev/null 2>&1 && break
        sleep 1
    done
else
    warn "Docker 未运行，跳过基础设施。检索/缓存/数据库不可用"
fi

# ============================================================================
# 2. 后端依赖
# ============================================================================
log "2/6  检查后端依赖 …"
if python3 -c "import mindforge" 2>/dev/null; then
    ok "mindforge 已安装"
else
    warn "安装后端依赖 …"
    pip3 install -e ".[dev]" -q 2>/dev/null && ok "依赖安装完成" || fail "依赖安装失败"
fi

# ============================================================================
# 3. 环境变量
# ============================================================================
log "3/6  检查环境变量 …"
if [ -f ".env" ]; then
    ok ".env 文件存在"
else
    cp .env.example .env 2>/dev/null && \
        warn ".env 已从模板创建，请编辑填入 API Key" || \
        warn ".env 和 .env.example 均缺失"
fi

set +e; source .env 2>/dev/null; set -e
if [ -z "$LLM_DEEPSEEK_API_KEY" ] && [ -z "$LLM_OPENAI_API_KEY" ]; then
    warn "未检测到 LLM API Key — 研究任务将降级为文档检索模式"
    warn "编辑 .env 填入 LLM_DEEPSEEK_API_KEY 以启用 LLM Agent"
fi

# ============================================================================
# 4. 前端构建
# ============================================================================
log "4/6  构建前端 …"
FRONTEND_DIR="$ROOT_DIR/mindforge-web"
if [ -f "$FRONTEND_DIR/package.json" ]; then
    cd "$FRONTEND_DIR"
    if [ ! -d "node_modules" ]; then
        npm install --silent 2>/dev/null && ok "npm 依赖已安装" || warn "npm install 失败"
    fi
    npm run build 2>/dev/null && ok "前端构建完成 → dist/" || warn "前端构建失败"
    cd "$ROOT_DIR"
else
    warn "前端目录不存在（mcp-server 分支？），跳过"
fi

# ============================================================================
# 5. 后端服务
# ============================================================================
log "5/6  启动后端服务 …"

if lsof -ti:8000 &>/dev/null 2>&1; then
    warn "端口 8000 已占用，释放中 …"
    kill "$(lsof -ti:8000)" 2>/dev/null && sleep 1 || true
fi

UVICORN_ARGS="--host 0.0.0.0 --port 8000 --log-level info"
$DEV_MODE && UVICORN_ARGS="$UVICORN_ARGS --reload"

cd "$ROOT_DIR"
python3 -m uvicorn mindforge.api.server:app $UVICORN_ARGS &
BACKEND_PID=$!

# 等待后端就绪（MCP 初始化 ~60s）
RETRIES=0; MAX_RETRIES=90
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
    HEALTH=$(curl -s http://localhost:8000/api/v1/health)
    echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║          🚀 MindForge 启动成功            ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  Web 前端+API  ${CYAN}http://localhost:8000${NC}"
    echo -e "  OpenAPI 文档  ${CYAN}http://localhost:8000/docs${NC}"
    echo -e "  健康检查      ${CYAN}$HEALTH${NC}"
    echo ""
    echo -e "  停止服务: ${YELLOW}Ctrl+C${NC} 或 ${YELLOW}kill $BACKEND_PID${NC}"
else
    echo -e "${YELLOW}╔══════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║      后端仍在初始化（MCP 连接中）…        ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  请稍后访问 ${CYAN}http://localhost:8000${NC}"
fi
echo ""

wait $BACKEND_PID
