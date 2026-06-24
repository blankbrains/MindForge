"""FastAPI application entry point for MindForge.

Creates and configures the ASGI application with CORS, router
mounting, lifecycle hooks, and a root info endpoint.
"""

from __future__ import annotations

import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
    allow_credentials=False,  # must be False when allow_origins=["*"] per CORS spec
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

    # Database init
    try:
        from mindforge.db import init_db
        init_db()
        logger.info("Database initialized (PostgreSQL)")
    except Exception as e:
        logger.warning("Database init skipped: %s", e)

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
            from mindforge.mcp.registry import get_mcp_registry

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
        except Exception as exc:
            logger.warning("Failed to store MCP registry reference: %s", exc)

    logger.info("MindForge startup complete")


# ------------------------------------------------------------------
# Root endpoint
# ------------------------------------------------------------------

@app.get("/")
async def root():
    """Return the SPA or service metadata."""
    return _serve_frontend("index.html")


# ------------------------------------------------------------------
# Static file serving (production frontend)
# ------------------------------------------------------------------

_FRONTEND_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "mindforge-web", "dist")
)


def _serve_frontend(filename: str = "index.html") -> FileResponse:
    """Serve a static file from the frontend build directory.

    Falls back to ``FileResponse`` which FastAPI handles directly.
    """
    return FileResponse(os.path.join(_FRONTEND_DIR, filename))


# Mount static assets (JS / CSS / images) if the build directory exists.
if os.path.isdir(_FRONTEND_DIR):
    _assets_dir = os.path.join(_FRONTEND_DIR, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    # Favicon
    _favicon = os.path.join(_FRONTEND_DIR, "favicon.svg")
    if os.path.isfile(_favicon):

        @app.get("/favicon.svg", include_in_schema=False)
        async def favicon():
            return FileResponse(_favicon)

    # SPA fallback — serve index.html for any unmatched path.
    # Registered last so API routes (/api/v1/*) take priority.
    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        candidate = os.path.join(_FRONTEND_DIR, full_path)
        if os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))
