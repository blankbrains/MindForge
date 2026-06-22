"""FastAPI application entry point for MindForge.

Creates and configures the ASGI application with CORS, router
mounting, lifecycle hooks, and a root info endpoint.
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mindforge.api.routes import router

# ── 统一 UTF-8 日志输出（防止控制台中文乱码）──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MindForge",
    version="0.1.0",
    description="Multi-agent research orchestration platform.",
)

# ------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------

app.include_router(router, prefix="/api/v1")


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Probe Qdrant / Redis / MCP readiness on boot and preload MCP registry."""
    from mindforge.config import get_settings
    from mindforge.api.routes import set_mcp_registry
    settings = get_settings()

    # Qdrant probe
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=settings.vector_store.qdrant_url, timeout=5)
        info = qc.get_collections()
        logger.info("Qdrant connected — collections: %s",
                     [c.name for c in info.collections])
    except Exception as e:
        logger.warning("Qdrant not available: %s", e)

    # Redis probe
    try:
        import redis as r
        rc = r.from_url(settings.cache.redis_url or "redis://localhost:6379",
                        decode_responses=True)
        rc.ping()
        rc.close()
        logger.info("Redis connected — %s", settings.cache.redis_url)
    except Exception as e:
        logger.warning("Redis not available: %s", e)

    # MCP Registry — preload at startup for Agent tool use
    import os
    mcp_cfg = settings.mcp.mcp_config_path or os.path.expanduser("~/.claude/mcp.json")
    mcp_registry_ready = False
    if os.path.exists(mcp_cfg):
        try:
            from mindforge.mcp.registry import MCPRegistry, get_mcp_registry

            reg = get_mcp_registry(config_path=mcp_cfg)
            reg.load_config(mcp_cfg)
            logger.info("MCP config loaded — %d servers configured", len(reg.servers))

            # Start MCP server subprocesses (lazy = they connect on first use)
            await reg.start_all()
            if reg.is_any_running:
                tools = await reg.discover_all_tools()
                logger.info("MCP servers started — %d tools discovered", len(tools))
                mcp_registry_ready = True
            else:
                logger.info("MCP servers configured but not started (will init on demand)")
        except Exception as e:
            logger.debug("MCP registry init skipped: %s", e)

    # Store registry reference for routes
    if mcp_registry_ready:
        try:
            set_mcp_registry(reg)
        except Exception:
            pass

    logger.info("MindForge startup complete")


# ------------------------------------------------------------------
# Root endpoint
# ------------------------------------------------------------------

@app.get("/")
async def root():
    """Return service metadata and a link to the interactive docs."""
    return {
        "service": "MindForge",
        "version": "0.1.0",
        "docs": "/docs",
        "description": "Multi-agent research orchestration platform.",
    }
