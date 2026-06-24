"""Pydantic schemas for the MindForge REST API.

Defines request / response models used by all API endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Query endpoints
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Payload for submitting a research task."""

    task: str = Field(..., description="Natural-language research task or question.")
    user_id: str | None = Field(None, description="Optional caller identifier.")
    stream: bool = Field(False, description="If true, use SSE streaming response.")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary options forwarded to the orchestrator.",
    )


class QueryResponse(BaseModel):
    """Result returned by the research orchestrator."""

    task_id: str
    report: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    quality_score: float | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    iterations: int = 0


# ------------------------------------------------------------------
# Index endpoints
# ------------------------------------------------------------------

class IndexRequest(BaseModel):
    """Payload for ingesting a document into the knowledge base."""

    file_url: str | None = Field(None, description="Public URL of the document.")
    file_path: str | None = Field(None, description="Local filesystem path.")
    metadata: dict[str, Any] = Field(default_factory=dict)
    strategy: str = Field(
        "auto",
        description="Chunking strategy: 'auto', 'fixed', 'semantic'.",
    )
    use_raptor: bool = Field(False, description="Apply RAPTOR summarisation.")
    use_graphrag: bool = Field(False, description="Apply GraphRAG indexing.")


class IndexResponse(BaseModel):
    """Confirmation of a completed indexing operation."""

    doc_id: str
    filename: str
    chunk_count: int
    status: str = "indexed"


class DocumentItem(BaseModel):
    """A document visible in the knowledge-base listing."""

    doc_id: str
    filename: str
    chunk_count: int
    status: str = "indexed"


DocumentsListResponse = list[DocumentItem]


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

class DocumentContentResponse(BaseModel):
    """Full content of a document (all chunks combined)."""

    doc_id: str
    filename: str
    content: str
    chunk_count: int
    chunks: list[dict[str, Any]] = Field(default_factory=list)


class SettingsResponse(BaseModel):
    """User settings (API keys masked)."""

    llm_provider: str = "deepseek"
    deepseek_api_key: str = ""
    openai_api_key: str = ""
    embedding_provider: str = "openai"


class SettingsUpdateRequest(BaseModel):
    """Payload for updating user settings."""

    llm_provider: str | None = None
    deepseek_api_key: str | None = None
    openai_api_key: str | None = None
    embedding_provider: str | None = None
    retrieval_top_k: int | None = None
    rerank_top_k: int | None = None
    max_iterations: int | None = None
    critic_threshold: float | None = None


class HistoryItem(BaseModel):
    """A single research history entry."""

    id: int
    task: str
    report: str | None = None
    quality_score: float | None = None
    model_used: str | None = None
    created_at: str | None = None


class HistoryListResponse(BaseModel):
    """Paginated history list."""

    entries: list[HistoryItem] = Field(default_factory=list)
    total: int = 0


class HistorySaveRequest(BaseModel):
    """Request body for saving a research history entry."""

    task: str
    report: str = ""
    quality_score: float | None = None
    model_used: str | None = None
    token_usage: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Service health information."""

    status: str = "ok"
    version: str = "0.1.0"
    qdrant_connected: bool = False
    redis_connected: bool = False
    mcp_tools_available: bool = False
