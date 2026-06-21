"""FastAPI application entry point for MindForge.

Creates and configures the ASGI application with CORS, router
mounting, lifecycle hooks, and a root info endpoint.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mindforge.api.routes import router

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
    """Ensure required infrastructure is ready on boot."""
    # TODO: wire in real Qdrant / Redis / MCP clients
    #       and run readiness probes here.
    pass


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
