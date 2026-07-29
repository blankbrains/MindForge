"""Pydantic schemas for the MindForge REST API.

Defines request / response models used by all API endpoints.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator


# ------------------------------------------------------------------
# Query endpoints
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Payload for submitting a research task."""

    task: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="Natural-language research task or question.",
    )
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
    strategy: Literal["auto", "fixed", "semantic"] = Field(
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
    index_strategy: Literal["auto", "fixed", "semantic"] = "auto"
    use_raptor: bool = False
    use_graphrag: bool = False


class IndexJobResponse(BaseModel):
    """Persistent state for an asynchronous indexing operation."""

    job_id: str
    doc_id: str | None = None
    filename: str
    status: str
    stage: str
    progress: float = Field(ge=0.0, le=100.0)
    chunk_count: int = 0
    timings: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    cancel_requested: bool = False
    strategy: Literal["auto", "fixed", "semantic"] = "auto"
    use_raptor: bool = False
    use_graphrag: bool = False
    created_at: datetime
    updated_at: datetime


class DocumentItem(BaseModel):
    """A document visible in the knowledge-base listing."""

    doc_id: str
    filename: str
    chunk_count: int
    status: str = "indexed"
    index_strategy: Literal["auto", "fixed", "semantic"] = "auto"
    use_raptor: bool = False
    use_graphrag: bool = False


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


LLMProviderName = Literal[
    "openai",
    "deepseek",
    "openai_compatible",
    "local",
]


class LLMProviderConfig(BaseModel):
    """Public provider configuration with a masked API key."""

    provider: LLMProviderName
    label: str
    base_url: str = ""
    api_key: str = ""
    api_key_required: bool = True
    default_model: str = ""
    planner_model: str = ""
    researcher_model: str = ""
    critic_model: str = ""
    synthesizer_model: str = ""
    supports_tools: bool = True
    supports_json_mode: bool = True
    supports_json_schema: bool = False
    configured: bool = False


class LLMProviderUpdate(BaseModel):
    """Editable fields for one provider."""

    provider: LLMProviderName
    base_url: str | None = Field(None, max_length=2048)
    api_key: str | None = Field(None, max_length=4096)
    api_key_required: bool | None = None
    default_model: str | None = Field(None, max_length=512)
    planner_model: str | None = Field(None, max_length=512)
    researcher_model: str | None = Field(None, max_length=512)
    critic_model: str | None = Field(None, max_length=512)
    synthesizer_model: str | None = Field(None, max_length=512)
    supports_tools: bool | None = None
    supports_json_mode: bool | None = None
    supports_json_schema: bool | None = None

    @field_validator("api_key")
    @classmethod
    def reject_api_key_control_characters(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("API keys must not contain control characters.")
        return value

    @field_validator(
        "default_model",
        "planner_model",
        "researcher_model",
        "critic_model",
        "synthesizer_model",
    )
    @classmethod
    def reject_control_characters(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("LLM configuration must not contain control characters.")
        return value.strip()

    @field_validator("base_url")
    @classmethod
    def validate_base_url(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return ""
        if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
            raise ValueError("Base URL must not contain control characters.")
        try:
            parsed = urlsplit(normalized)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Base URL is invalid.") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Base URL must be an absolute HTTP(S) URL.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Base URL must not contain credentials.")
        if parsed.query or parsed.fragment:
            raise ValueError("Base URL must not contain a query or fragment.")
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("Base URL port is invalid.")
        return normalized.rstrip("/")


class SettingsResponse(BaseModel):
    """User settings (API keys masked)."""

    llm_provider: str = "deepseek"
    llm_configured: bool = False
    llm_providers: list[LLMProviderConfig] = Field(default_factory=list)
    deepseek_api_key: str = ""
    openai_api_key: str = ""
    compatible_api_key: str = ""
    local_api_key: str = ""
    embedding_provider: str = "openai"
    retrieval_top_k: int = 20
    rerank_top_k: int = 6
    max_iterations: int = 3
    max_refine_rounds: int = 1
    critic_threshold: float = 7.0
    subtask_timeout: int = 30
    research_timeout: int = 180


class SettingsUpdateRequest(BaseModel):
    """Payload for updating user settings."""

    llm_provider: LLMProviderName | None = None
    llm_provider_config: LLMProviderUpdate | None = None
    llm_provider_configs: list[LLMProviderUpdate] | None = Field(
        None,
        max_length=4,
    )
    deepseek_api_key: str | None = Field(None, max_length=4096)
    openai_api_key: str | None = Field(None, max_length=4096)
    compatible_api_key: str | None = Field(None, max_length=4096)
    local_api_key: str | None = Field(None, max_length=4096)
    embedding_provider: Literal["openai", "bge"] | None = None
    retrieval_top_k: int | None = Field(None, ge=1, le=100)
    rerank_top_k: int | None = Field(None, ge=1, le=50)
    max_iterations: int | None = Field(None, ge=1, le=20)
    max_refine_rounds: int | None = Field(None, ge=0, le=5)
    critic_threshold: float | None = Field(None, ge=0.0, le=10.0)
    subtask_timeout: int | None = Field(None, ge=10, le=600)
    research_timeout: int | None = Field(None, ge=30, le=3600)

    @field_validator(
        "deepseek_api_key",
        "openai_api_key",
        "compatible_api_key",
        "local_api_key",
    )
    @classmethod
    def reject_api_key_control_characters(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("API keys must not contain control characters.")
        return value

    @model_validator(mode="after")
    def validate_provider_updates(self) -> SettingsUpdateRequest:
        updates = list(self.llm_provider_configs or [])
        if self.llm_provider_config is not None:
            updates.append(self.llm_provider_config)
        providers = [update.provider for update in updates]
        if len(providers) != len(set(providers)):
            raise ValueError("Each LLM provider may only be updated once.")
        return self


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

    task: str = Field(min_length=1, max_length=20_000)
    report: str = Field(default="", max_length=2_000_000)
    quality_score: float | None = None
    model_used: str | None = Field(default=None, max_length=200)
    token_usage: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_token_usage_size(self) -> HistorySaveRequest:
        encoded = json.dumps(
            self.token_usage,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        if len(encoded) > 100_000:
            raise ValueError("token_usage exceeds 100000 bytes.")
        return self


class HealthResponse(BaseModel):
    """Service health information."""

    status: str = "ok"
    version: str = "0.1.0"
    qdrant_connected: bool = False
    redis_connected: bool = False
    postgres_connected: bool = False
