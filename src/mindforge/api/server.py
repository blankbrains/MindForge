"""FastAPI application entry point for MindForge.

Creates and configures the ASGI application with CORS, router
mounting, lifecycle hooks, and a root info endpoint.
"""

from __future__ import annotations

import logging
import os
import sys
import asyncio
import hmac
import ipaddress
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
        if request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
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


def _api_binding_is_loopback() -> bool:
    docker_binding = os.getenv("DOCKER_API_BIND_ADDRESS", "").strip()
    binding = docker_binding or _settings.api.host.strip()
    if binding.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(binding).is_loopback
    except ValueError:
        return False


class APIProtectionMiddleware(BaseHTTPMiddleware):
    """Prevent accidental exposure of the single-user administrative API."""

    _PUBLIC_API_PATHS = {
        "/api/v1/health",
        "/api/v1/ready",
    }

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api/") or path in self._PUBLIC_API_PATHS:
            return await call_next(request)

        configured_token = _settings.api.access_token.strip()
        if configured_token:
            scheme, _, supplied_token = request.headers.get(
                "Authorization",
                "",
            ).partition(" ")
            authenticated = (
                scheme.lower() == "bearer"
                and bool(supplied_token)
                and hmac.compare_digest(
                    supplied_token.strip(),
                    configured_token,
                )
            )
            if not authenticated:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "API authentication is required."},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return await call_next(request)

        if _api_binding_is_loopback():
            return await call_next(request)
        client_host = request.client.host if request.client is not None else ""
        try:
            local_client = ipaddress.ip_address(client_host).is_loopback
        except ValueError:
            local_client = client_host.lower() == "localhost"
        if local_client:
            return await call_next(request)
        return JSONResponse(
            status_code=403,
            content={
                "detail": (
                    "Remote API access is disabled until API_ACCESS_TOKEN "
                    "is configured."
                )
            },
        )


app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(APIProtectionMiddleware)
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
        from mindforge.ingestion.embedder import get_embedder

        logger.info("Preloading configured embedding backend.")
        embedder = await asyncio.to_thread(get_embedder)
        logger.info(
            "Embedding backend '%s' ready.",
            embedder.provider,
        )
    except Exception as exc:
        logger.error("Embedding preload failed: %s", exc)


async def _preload_reranker() -> None:
    """Load the configured reranker before it can delay a user query."""
    try:
        from mindforge.retrieval.service import preload_reranker

        logger.info("Preloading configured reranker.")
        loaded = await preload_reranker()
        logger.info(
            "Reranker preload %s.",
            "complete" if loaded else "disabled or unavailable",
        )
    except Exception as exc:
        logger.error("Reranker preload failed: %s", exc)


async def _preload_models(
    *,
    preload_embedder: bool,
    preload_reranker: bool,
) -> None:
    """Load local NLP models sequentially to protect shared HF clients."""
    if preload_embedder:
        await _preload_embedder()
    if preload_reranker:
        await _preload_reranker()


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------


async def startup():
    """Probe core services on boot and preload the embedding model."""
    settings = get_settings()

    async def _initialize_database() -> bool:
        try:
            from mindforge.db import init_db

            await asyncio.to_thread(init_db)
            return True
        except Exception as exc:
            logger.warning("Database init skipped: %s", exc)
            return False

    async def _initialize_qdrant() -> bool:
        try:
            from mindforge.retrieval.vector_store import get_vector_store

            store = get_vector_store()
            await asyncio.to_thread(store.ensure_collection)
            await store.ping()
            return True
        except Exception as exc:
            logger.warning("Qdrant init skipped: %s", exc)
            return False

    database_ok, qdrant_ok = await asyncio.gather(
        _initialize_database(),
        _initialize_qdrant(),
    )
    from mindforge.services.health import get_health_monitor

    snapshot = await get_health_monitor().start()
    logger.info(
        "Core services — PostgreSQL=%s Qdrant=%s Redis=%s",
        "ready" if database_ok and snapshot.postgres_connected else "unavailable",
        "ready" if qdrant_ok and snapshot.qdrant_connected else "unavailable",
        "ready" if snapshot.redis_connected else "unavailable",
    )

    if database_ok and qdrant_ok:
        try:
            from mindforge.services.indexing import (
                reconcile_document_catalog,
            )

            await reconcile_document_catalog()
        except Exception:
            logger.exception("Document catalog reconciliation failed.")

    if database_ok:
        try:
            from mindforge.services.index_jobs import (
                get_index_job_service,
            )

            await get_index_job_service().start()
        except Exception:
            logger.exception("Persistent index-job workers failed to start.")

    preload_embedder = settings.llm.embedding_provider in (
        "bge",
        "sentence-transformers",
    )
    preload_reranker = bool(
        settings.retrieval.reranker_model and settings.retrieval.reranker_preload
    )
    if preload_embedder or preload_reranker:
        task = asyncio.create_task(
            _preload_models(
                preload_embedder=preload_embedder,
                preload_reranker=preload_reranker,
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    logger.info("MindForge startup complete")


async def shutdown() -> None:
    """Release background tasks during application shutdown."""
    from mindforge.observability.tracer import close_tracer
    from mindforge.services.health import get_health_monitor
    from mindforge.services.index_jobs import get_index_job_service

    await get_index_job_service().stop()
    await get_health_monitor().stop()
    tasks = list(_background_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _background_tasks.clear()
    await asyncio.to_thread(close_tracer)


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

_FRONTEND_DIR = os.path.normpath(str(get_project_root() / "mindforge-web" / "dist"))
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
        if full_path == "api" or full_path.startswith("api/"):
            return JSONResponse(
                status_code=404,
                content={"detail": "Not Found"},
            )
        # Prevent path traversal: resolve the canonical path and verify it stays
        # within the frontend build directory.
        candidate = _safe_frontend_candidate(full_path)
        if candidate is None:
            return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))
