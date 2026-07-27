"""FastAPI application entry point for MindForge.

Creates and configures the ASGI application with CORS, router
mounting, lifecycle hooks, and a root info endpoint.
"""

from __future__ import annotations

import logging
import os
import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from mindforge.api.routes import router
from mindforge.config import get_project_root, get_settings
from mindforge import __version__

_settings = get_settings()
_background_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await startup()
    try:
        yield
    finally:
        await shutdown()

# ── 统一 UTF-8 日志输出（防止控制台中文乱码）──
logging.basicConfig(
    level=getattr(logging, _settings.app.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MindForge",
    version=__version__,
    description="Multi-agent research orchestration platform.",
    lifespan=lifespan,
)

# ------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=()",
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            (
                "default-src 'self'; "
                "base-uri 'self'; "
                "frame-ancestors 'none'; "
                "object-src 'none'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self' data:; "
                "connect-src 'self'"
            ),
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.api.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------

app.include_router(router, prefix="/api/v1")


# ------------------------------------------------------------------
# Background helpers
# ------------------------------------------------------------------

async def _preload_embedder() -> None:
    """Preload the embedding model in background so first upload is instant."""
    try:
        from mindforge.ingestion.embedder import EmbeddingManager
        from mindforge.config import get_settings
        settings = get_settings()
        model = settings.llm.local_embedding_model or "BAAI/bge-m3"
        logger.info("Preloading embedding model '%s' in background...", model)
        await asyncio.to_thread(
            EmbeddingManager,
            model_name=model,
            provider="sentence-transformers",
        )
        logger.info("Embedding model '%s' ready.", model)
    except Exception as exc:
        logger.warning("Embedding preload skipped (will use hash fallback): %s", exc)


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

async def startup():
    """Probe Qdrant / Redis / MCP readiness on boot and preload MCP registry."""
    from mindforge.api.routes import set_mcp_registry
    settings = get_settings()

    async def _initialize_database() -> bool:
        try:
            from mindforge.db import init_db

            await asyncio.to_thread(init_db)
            return True
        except Exception as exc:
            logger.warning("Database init skipped: %s", exc)
            return False

    from mindforge.api.routes import (
        _probe_qdrant_connection,
        _probe_redis_connection,
    )

    async def _initialize_qdrant() -> bool:
        try:
            from mindforge.retrieval.vector_store import get_vector_store

            store = get_vector_store()
            await asyncio.to_thread(store.ensure_collection)
            return await _probe_qdrant_connection()
        except Exception as exc:
            logger.warning("Qdrant init skipped: %s", exc)
            return False

    database_ok, qdrant_ok, redis_ok = await asyncio.gather(
        _initialize_database(),
        _initialize_qdrant(),
        _probe_redis_connection(),
    )
    logger.info(
        "Core services — PostgreSQL=%s Qdrant=%s Redis=%s",
        "ready" if database_ok else "unavailable",
        "ready" if qdrant_ok else "unavailable",
        "ready" if redis_ok else "unavailable",
    )

    # MCP Registry — preload at startup for Agent tool use
    import os
    mcp_cfg = settings.mcp.mcp_config_path
    if settings.mcp.mcp_servers_json.strip() or os.path.exists(mcp_cfg):
        try:
            from mindforge.mcp.registry import get_mcp_registry

            reg = get_mcp_registry(
                config_path=mcp_cfg,
                auto_load=False,
            )
            if settings.mcp.mcp_servers_json.strip():
                reg.load_config_json(settings.mcp.mcp_servers_json)
            else:
                reg.load_config(mcp_cfg)
            logger.info("MCP config loaded — %d servers configured", len(reg.servers))
            set_mcp_registry(reg)

            # Start MCP server subprocesses (lazy = they connect on first use)
            await reg.start_all()
            if reg.is_any_running:
                tools = await reg.discover_all_tools()
                logger.info("MCP servers started — %d tools discovered", len(tools))
            else:
                logger.info("MCP servers configured but not started (will init on demand)")
        except Exception as e:
            logger.debug("MCP registry init skipped: %s", e)

    # 后台预加载 embedding 模型（BGE-M3 1.7GB，首次加载 ~170s）
    # 避免用户上传文档时等待
    if settings.llm.embedding_provider in ("bge", "sentence-transformers"):
        task = asyncio.create_task(_preload_embedder())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    logger.info("MindForge startup complete")


async def shutdown() -> None:
    """Release process-backed integrations during application shutdown."""
    from mindforge.api.routes import get_mcp_registry

    tasks = list(_background_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _background_tasks.clear()

    registry = get_mcp_registry()
    if registry is not None:
        try:
            await registry.stop_all()
        except Exception:
            logger.exception("Failed to stop MCP servers cleanly.")


# ------------------------------------------------------------------
# Root endpoint
# ------------------------------------------------------------------

@app.get("/")
async def root():
    """Return the SPA or service metadata."""
    if not os.path.isfile(os.path.join(_FRONTEND_DIR, "index.html")):
        return JSONResponse(
            {
                "name": "MindForge",
                "version": __version__,
                "api": "/docs",
                "health": "/api/v1/health",
                "readiness": "/api/v1/ready",
            }
        )
    return _serve_frontend("index.html")


# ------------------------------------------------------------------
# Static file serving (production frontend)
# ------------------------------------------------------------------

_FRONTEND_DIR = os.path.normpath(
    str(get_project_root() / "mindforge-web" / "dist")
)
_FRONTEND_PATH = Path(_FRONTEND_DIR).resolve()


def _safe_frontend_candidate(full_path: str) -> Path | None:
    candidate = (_FRONTEND_PATH / full_path).resolve()
    try:
        candidate.relative_to(_FRONTEND_PATH)
    except ValueError:
        return None
    return candidate


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
        # Prevent path traversal: resolve the canonical path and verify it stays
        # within the frontend build directory.
        candidate = _safe_frontend_candidate(full_path)
        if candidate is None:
            return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))
