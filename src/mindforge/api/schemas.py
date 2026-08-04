"""Pydantic schemas for the MindForge REST API.

Defines request / response models used by all API endpoints.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator


# ------------------------------------------------------------------
# Query endpoints
# ------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Payload for submitting a research task."""

    request_id: str | None = Field(
        None,
        min_length=8,
        max_length=80,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Client-generated identifier used for explicit cancellation.",
    )
    task: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="Natural-language research task or question.",
    )
    stream: bool = Field(False, description="If true, use SSE streaming response.")


class QueryCancelRequest(BaseModel):
    """Payload for cancelling one active streaming research request."""

    request_id: str = Field(
        ...,
        min_length=8,
        max_length=80,
        pattern=r"^[A-Za-z0-9_-]+$",
    )


class QueryCancelResponse(BaseModel):
    """Cancellation acknowledgement for a streaming research request."""

    request_id: str
    cancelled: bool


class QueryResponse(BaseModel):
    """Result returned by the research orchestrator."""

    task_id: str
    trace_id: str | None = None
    report: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    quality_score: float | None = None
    quality_status: Literal[
        "evaluated",
        "not_evaluated",
        "evaluation_failed",
    ] = "not_evaluated"
    latency_ms: float = 0.0
    cost_usd: float | None = None
    cost_status: str = "usage_unavailable"
    iterations: int = 0
    outcome: Literal["success", "degraded", "retrieval_only"] = "success"
    failure_reason: str | None = None
    retrieval_quality: float | None = None


# ------------------------------------------------------------------
# Index endpoints
# ------------------------------------------------------------------


class IndexRequest(BaseModel):
    """Payload for ingesting a document into the knowledge base."""

    file_path: str = Field(
        ...,
        min_length=1,
        description="Local filesystem path inside MINDFORGE_DATA_DIR.",
    )
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
    enabled: bool = True
    index_strategy: Literal["auto", "fixed", "semantic"] = "auto"
    use_raptor: bool = False
    use_graphrag: bool = False


class DocumentEnabledUpdate(BaseModel):
    """Retrieval availability for one indexed document."""

    enabled: bool


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
    "kimi",
    "glm",
    "openai_compatible",
    "local",
]
NativeWebSearchProtocol = Literal[
    "none",
    "openai_responses",
    "kimi_builtin",
    "glm_web_search",
]


def _normalize_http_base_url(
    value: str | None,
    *,
    allow_empty: bool,
) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        if allow_empty:
            return ""
        raise ValueError("Base URL must not be empty.")
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
    native_web_search_protocol: NativeWebSearchProtocol = "none"
    native_web_search_endpoint: str = ""
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
    native_web_search_protocol: NativeWebSearchProtocol | None = None
    native_web_search_endpoint: str | None = Field(None, max_length=2048)

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

    @field_validator("base_url", "native_web_search_endpoint")
    @classmethod
    def validate_base_url(
        cls,
        value: str | None,
    ) -> str | None:
        return _normalize_http_base_url(value, allow_empty=True)


class LLMModelDiscoveryRequest(BaseModel):
    """Draft provider connection values used to fetch model IDs."""

    provider: LLMProviderName
    base_url: str = Field(max_length=2048)
    api_key: str = Field(default="", max_length=4096)
    api_key_required: bool = True
    use_stored_api_key: bool = False

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = _normalize_http_base_url(
            value,
            allow_empty=False,
        )
        assert isinstance(normalized, str)
        return normalized

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("API keys must not contain control characters.")
        if value.startswith("***"):
            raise ValueError("Masked API keys cannot be submitted.")
        return value


class LLMDiscoveredModel(BaseModel):
    """One model exposed by a provider model-list endpoint."""

    id: str
    owned_by: str = ""


class LLMModelDiscoveryResponse(BaseModel):
    """Bounded provider model catalog."""

    models: list[LLMDiscoveredModel] = Field(default_factory=list)
    count: int = Field(ge=0)
    truncated: bool = False


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
    research_mode: Literal["fast", "balanced", "deep"] = "balanced"
    source_policy: Literal["auto", "knowledge_base", "web"] = "auto"
    fallback_enabled: bool = True
    retrieval_top_k: int = 20
    rerank_top_k: int = 6
    retrieval_min_score: float = 0.60
    keyword_min_coverage: float = 0.60
    max_iterations: int = 3
    max_refine_rounds: int = 1
    critic_threshold: float = 7.0
    subtask_timeout: int = 120
    research_timeout: int = 300
    llm_request_timeout: int = 45
    queue_timeout: int = 30
    native_web_search_timeout_seconds: float = 30.0
    sandbox_timeout: int = 15
    max_subtasks: int = 5
    max_tool_calls_total: int = 12
    max_history_entries: int = 0
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    observability_capture_content: bool = False
    trace_retention_days: int = 0
    tavily_configured: bool = False
    native_web_search_enabled: bool = True
    native_web_search_protocol: NativeWebSearchProtocol = "none"
    native_web_search_supported: bool = False
    duckduckgo_enabled: bool = False
    model_only_fallback_enabled: bool = True
    web_search_available: bool = False
    reranker_configured: bool = False
    reranker_available: bool = False
    reranker_load_failed: bool = False


class SettingsUpdateRequest(BaseModel):
    """Payload for updating user settings."""

    llm_provider: LLMProviderName | None = None
    llm_provider_config: LLMProviderUpdate | None = None
    llm_provider_configs: list[LLMProviderUpdate] | None = Field(
        None,
        max_length=6,
    )
    deepseek_api_key: str | None = Field(None, max_length=4096)
    openai_api_key: str | None = Field(None, max_length=4096)
    compatible_api_key: str | None = Field(None, max_length=4096)
    local_api_key: str | None = Field(None, max_length=4096)
    embedding_provider: Literal["openai", "bge"] | None = None
    research_mode: Literal["fast", "balanced", "deep"] | None = None
    source_policy: Literal["auto", "knowledge_base", "web"] | None = None
    fallback_enabled: bool | None = None
    retrieval_top_k: int | None = Field(None, ge=1, le=100)
    rerank_top_k: int | None = Field(None, ge=1, le=50)
    retrieval_min_score: float | None = Field(None, ge=0.0, le=1.0)
    keyword_min_coverage: float | None = Field(None, ge=0.0, le=1.0)
    max_iterations: int | None = Field(None, ge=1, le=20)
    max_refine_rounds: int | None = Field(None, ge=0, le=5)
    critic_threshold: float | None = Field(None, ge=0.0, le=10.0)
    subtask_timeout: int | None = Field(None, ge=10, le=600)
    research_timeout: int | None = Field(None, ge=30, le=3600)
    llm_request_timeout: int | None = Field(None, ge=5, le=600)
    queue_timeout: int | None = Field(None, ge=1, le=600)
    native_web_search_timeout_seconds: float | None = Field(
        None,
        ge=5.0,
        le=120.0,
    )
    sandbox_timeout: int | None = Field(None, ge=5, le=60)
    max_subtasks: int | None = Field(None, ge=1, le=20)
    max_tool_calls_total: int | None = Field(None, ge=1, le=100)
    max_history_entries: int | None = Field(None, ge=0, le=100_000)
    langfuse_public_key: str | None = Field(None, max_length=4096)
    langfuse_secret_key: str | None = Field(None, max_length=4096)
    langfuse_host: str | None = Field(None, max_length=2048)
    observability_capture_content: bool | None = None
    trace_retention_days: int | None = Field(None, ge=0, le=3650)

    @field_validator(
        "deepseek_api_key",
        "openai_api_key",
        "compatible_api_key",
        "local_api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
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

    @field_validator("langfuse_host")
    @classmethod
    def validate_langfuse_host(cls, value: str | None) -> str | None:
        return _normalize_http_base_url(value, allow_empty=False)

    @model_validator(mode="after")
    def validate_provider_updates(self) -> SettingsUpdateRequest:
        updates = list(self.llm_provider_configs or [])
        if self.llm_provider_config is not None:
            updates.append(self.llm_provider_config)
        providers = [update.provider for update in updates]
        if len(providers) != len(set(providers)):
            raise ValueError("Each LLM provider may only be updated once.")
        return self


class HistoryCitationSource(BaseModel):
    """Compact source metadata persisted with a research report."""

    index: int = Field(ge=1, le=100_000)
    title: str = Field(default="", max_length=1_000)
    url: str = Field(default="", max_length=4_096)
    source: str = Field(default="", max_length=200)
    chunk_id: str | None = Field(default=None, max_length=512)
    doc_id: str | None = Field(default=None, max_length=512)

    @field_validator("url")
    @classmethod
    def restrict_url_scheme(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            return ""
        parsed = urlsplit(candidate)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return ""
        return candidate


class HistoryItem(BaseModel):
    """A single research history entry."""

    id: int
    task: str
    report: str | None = None
    quality_score: float | None = None
    model_used: str | None = None
    token_usage: dict[str, Any] = Field(default_factory=dict)
    sources: list[HistoryCitationSource] = Field(default_factory=list)
    trace_id: str | None = None
    created_at: str | None = None


class HistorySaveRequest(BaseModel):
    """Request body for saving a research history entry."""

    task: str = Field(min_length=1, max_length=20_000)
    report: str = Field(default="", max_length=2_000_000)
    quality_score: float | None = None
    model_used: str | None = Field(default=None, max_length=200)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    sources: list[HistoryCitationSource] = Field(
        default_factory=list,
        max_length=200,
    )
    trace_id: str | None = Field(default=None, max_length=32)

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{32}", normalized):
            raise ValueError("trace_id must be 32 lowercase hexadecimal characters.")
        return normalized

    @model_validator(mode="after")
    def enforce_json_field_sizes(self) -> HistorySaveRequest:
        encoded_usage = json.dumps(
            self.token_usage,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        if len(encoded_usage) > 100_000:
            raise ValueError("token_usage exceeds 100000 bytes.")
        encoded_sources = json.dumps(
            [source.model_dump(exclude_none=True) for source in self.sources],
            ensure_ascii=False,
        ).encode("utf-8")
        if len(encoded_sources) > 200_000:
            raise ValueError("sources exceeds 200000 bytes.")
        return self


class HealthResponse(BaseModel):
    """Service health information."""

    status: str = "ok"
    version: str = "0.1.0"
    qdrant_connected: bool = False
    redis_connected: bool = False
    postgres_connected: bool = False


# ------------------------------------------------------------------
# Observability
# ------------------------------------------------------------------


class ObservabilityStatusResponse(BaseModel):
    """Public observability state without backend credentials."""

    enabled: bool
    local_storage: bool
    remote_configured: bool
    langfuse_host: str | None = None
    capture_content: bool
    retention_days: int


class TraceSummary(BaseModel):
    """Bounded summary of one local research trace."""

    trace_id: str
    name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float = 0.0
    status: Literal[
        "success",
        "warning",
        "degraded",
        "error",
        "cancelled",
    ]
    display_name: str | None = None
    error: str | None = None
    failure_summary: str | None = None
    failure_count: int = 0
    task_preview: str | None = None
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    span_count: int = 0
    generation_count: int = 0
    tool_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    cost_status: str = "usage_unavailable"
    remote_url: str | None = None


class TraceListResponse(BaseModel):
    """Paginated local trace summaries."""

    traces: list[TraceSummary] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    truncated: bool = False


class TraceObservation(BaseModel):
    """One local observation in a trace."""

    span_id: str
    trace_id: str
    name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float = 0.0
    parent_id: str | None = None
    error: str | None = None
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    payloads_omitted: str | None = None


class TraceFailure(BaseModel):
    """Structured failure derived from one failed trace observation."""

    span_id: str
    parent_id: str | None = None
    observation_name: str
    stage: str
    error_code: str
    error_type: str
    message: str
    status: str
    agent: str | None = None
    model: str | None = None
    attempt: int | None = None


class TraceDetailResponse(BaseModel):
    """One trace summary and its bounded observation chain."""

    summary: TraceSummary
    observations: list[TraceObservation] = Field(default_factory=list)
    failures: list[TraceFailure] = Field(default_factory=list)
    observations_truncated: bool = False
