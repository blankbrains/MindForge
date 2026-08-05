# src/mindforge/config.py
"""统一配置管理 — 基于 Pydantic Settings，支持环境变量覆盖"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Literal, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


def _resolve_project_root() -> Path:
    explicit = os.getenv("MINDFORGE_PROJECT_ROOT", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file() and (cwd / "src" / "mindforge").is_dir():
        return cwd

    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return cwd


_PROJECT_ROOT = _resolve_project_root()
_env_override = os.getenv("MINDFORGE_ENV_FILE", "").strip()
_env_path = (
    Path(_env_override).expanduser().resolve()
    if _env_override
    else _PROJECT_ROOT / ".env"
)
_dotenv_loaded = False
if _env_path.is_file():
    try:
        from dotenv import load_dotenv

        load_dotenv(str(_env_path), encoding="utf-8")
        _dotenv_loaded = True
    except Exception:
        pass


def get_project_root() -> Path:
    """Return the verified runtime project root."""
    return _PROJECT_ROOT


def resolve_project_path(
    value: str | Path,
    *,
    root: Path | None = None,
) -> Path:
    """Resolve an absolute or project-relative runtime path."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((root or _PROJECT_ROOT) / path).resolve()


def require_environment_variable(name: str) -> str:
    """Return a required environment variable or fail with setup guidance."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is required. Configure it in the project .env file."
        )
    return value


_CONFIGURABLE_LLM_PREFIXES = {
    "kimi": "kimi",
    "glm": "glm",
    "openai_compatible": "compatible",
    "local": "local",
}


class LLMConfig(BaseSettings):
    """Unified cloud and self-hosted LLM provider configuration."""

    llm_provider: str = Field(
        default="openai",
        description=(
            "openai | deepseek | kimi | glm | openai_compatible | local"
        ),
    )
    embedding_provider: str = Field(default="openai", description="openai | bge")
    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    deepseek_api_key: str = Field(default="")
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    kimi_api_key: str = Field(default="")
    kimi_base_url: str = Field(default="https://api.moonshot.cn/v1")
    kimi_api_key_required: bool = Field(default=True)
    kimi_model: str = Field(default="")
    kimi_planner_model: str = Field(default="")
    kimi_researcher_model: str = Field(default="")
    kimi_critic_model: str = Field(default="")
    kimi_synthesizer_model: str = Field(default="")
    kimi_supports_tools: bool = Field(default=True)
    kimi_supports_json_mode: bool = Field(default=True)
    kimi_supports_json_schema: bool = Field(default=False)
    kimi_supports_stream_usage: bool = Field(default=False)
    kimi_native_web_search_protocol: Literal[
        "none",
        "openai_responses",
        "kimi_builtin",
        "glm_web_search",
    ] = "kimi_builtin"
    kimi_native_web_search_endpoint: str = Field(default="")
    glm_api_key: str = Field(default="")
    glm_base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4"
    )
    glm_api_key_required: bool = Field(default=True)
    glm_model: str = Field(default="")
    glm_planner_model: str = Field(default="")
    glm_researcher_model: str = Field(default="")
    glm_critic_model: str = Field(default="")
    glm_synthesizer_model: str = Field(default="")
    glm_supports_tools: bool = Field(default=True)
    glm_supports_json_mode: bool = Field(default=True)
    glm_supports_json_schema: bool = Field(default=False)
    glm_supports_stream_usage: bool = Field(default=False)
    glm_native_web_search_protocol: Literal[
        "none",
        "openai_responses",
        "kimi_builtin",
        "glm_web_search",
    ] = "glm_web_search"
    glm_native_web_search_endpoint: str = Field(default="")
    model_pricing: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "Per-model USD prices per one million tokens. Keys may be "
            "'provider:model' or plain model names. Values support input, "
            "cached_input, and output rates."
        ),
    )
    compatible_api_key: str = Field(default="")
    compatible_base_url: str = Field(default="")
    compatible_api_key_required: bool = Field(default=True)
    compatible_model: str = Field(default="")
    compatible_planner_model: str = Field(default="")
    compatible_researcher_model: str = Field(default="")
    compatible_critic_model: str = Field(default="")
    compatible_synthesizer_model: str = Field(default="")
    compatible_supports_tools: bool = Field(default=True)
    compatible_supports_json_mode: bool = Field(default=True)
    compatible_supports_json_schema: bool = Field(default=False)
    compatible_supports_stream_usage: bool = Field(default=False)
    compatible_native_web_search_protocol: Literal[
        "none",
        "openai_responses",
        "kimi_builtin",
        "glm_web_search",
    ] = "none"
    compatible_native_web_search_endpoint: str = Field(default="")
    local_api_key: str = Field(default="")
    local_base_url: str = Field(default="http://host.docker.internal:11434/v1")
    local_api_key_required: bool = Field(default=False)
    local_model: str = Field(default="")
    local_planner_model: str = Field(default="")
    local_researcher_model: str = Field(default="")
    local_critic_model: str = Field(default="")
    local_synthesizer_model: str = Field(default="")
    local_supports_tools: bool = Field(default=True)
    local_supports_json_mode: bool = Field(default=True)
    local_supports_json_schema: bool = Field(default=False)
    local_supports_stream_usage: bool = Field(default=False)
    local_native_web_search_protocol: Literal[
        "none",
        "openai_responses",
        "kimi_builtin",
        "glm_web_search",
    ] = "none"
    local_native_web_search_endpoint: str = Field(default="")
    planner_model: str = "gpt-4o"
    researcher_model: str = "gpt-4o-mini"
    critic_model: str = "gpt-4o"
    synthesizer_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    deepseek_planner: str = "deepseek-v4-flash"
    deepseek_researcher: str = "deepseek-v4-flash"
    deepseek_critic: str = "deepseek-v4-flash"
    deepseek_synthesizer: str = "deepseek-v4-flash"
    local_embedding_model: str = "BAAI/bge-m3"
    local_embedding_revision: str = Field(
        default="5617a9f61b028005a4858fdac845db406aefb181",
        pattern=r"^[0-9a-fA-F]{40}$",
        description="Immutable Hugging Face commit SHA for local embeddings.",
    )
    sentence_transformers_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1, le=1024)
    torch_num_threads: int = Field(default=0, ge=0, le=256)
    hf_endpoint: str = "https://hf-mirror.com"
    hf_hub_download_timeout: int = Field(default=30, ge=1)

    def get_model(
        self,
        role: str,
        provider: str | None = None,
    ) -> str:
        provider = (provider or self.llm_provider).lower()
        if provider == "deepseek":
            mapping = {
                "planner": self.deepseek_planner,
                "researcher": self.deepseek_researcher,
                "critic": self.deepseek_critic,
                "synthesizer": self.deepseek_synthesizer,
            }
            return mapping.get(role, self.deepseek_researcher)
        if provider in _CONFIGURABLE_LLM_PREFIXES:
            prefix = _CONFIGURABLE_LLM_PREFIXES[provider]
            default_model = getattr(self, f"{prefix}_model")
            role_model = getattr(self, f"{prefix}_{role}_model", "")
            return role_model or default_model
        return getattr(self, f"{role}_model", self.researcher_model)

    def get_api_key(self, provider: str | None = None) -> str:
        selected = (provider or self.llm_provider).lower()
        mapping = {
            "openai": self.openai_api_key,
            "deepseek": self.deepseek_api_key,
            "kimi": self.kimi_api_key,
            "glm": self.glm_api_key,
            "openai_compatible": self.compatible_api_key,
            "local": self.local_api_key,
        }
        return mapping.get(selected, "")

    def get_base_url(self, provider: str | None = None) -> str | None:
        selected = (provider or self.llm_provider).lower()
        mapping = {
            "openai": self.openai_base_url,
            "deepseek": self.deepseek_base_url,
            "kimi": self.kimi_base_url,
            "glm": self.glm_base_url,
            "openai_compatible": self.compatible_base_url,
            "local": self.local_base_url,
        }
        value = mapping.get(selected)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if selected == "openai":
            return "https://api.openai.com/v1"
        if selected == "deepseek":
            return "https://api.deepseek.com"
        return None

    def requires_api_key(self, provider: str | None = None) -> bool:
        selected = (provider or self.llm_provider).lower()
        if selected in {"openai", "deepseek"}:
            return True
        if selected in _CONFIGURABLE_LLM_PREFIXES:
            return bool(
                getattr(
                    self,
                    (
                        f"{_CONFIGURABLE_LLM_PREFIXES[selected]}"
                        "_api_key_required"
                    ),
                )
            )
        return True

    def supports_tools(self, provider: str | None = None) -> bool:
        selected = (provider or self.llm_provider).lower()
        if selected in _CONFIGURABLE_LLM_PREFIXES:
            return bool(
                getattr(
                    self,
                    (
                        f"{_CONFIGURABLE_LLM_PREFIXES[selected]}"
                        "_supports_tools"
                    ),
                )
            )
        return True

    def supports_json_mode(self, provider: str | None = None) -> bool:
        selected = (provider or self.llm_provider).lower()
        if selected in _CONFIGURABLE_LLM_PREFIXES:
            return bool(
                getattr(
                    self,
                    (
                        f"{_CONFIGURABLE_LLM_PREFIXES[selected]}"
                        "_supports_json_mode"
                    ),
                )
            )
        return True

    def supports_json_schema(self, provider: str | None = None) -> bool:
        selected = (provider or self.llm_provider).lower()
        if selected == "openai":
            return True
        if selected in _CONFIGURABLE_LLM_PREFIXES:
            return bool(
                getattr(
                    self,
                    (
                        f"{_CONFIGURABLE_LLM_PREFIXES[selected]}"
                        "_supports_json_schema"
                    ),
                )
            )
        return False

    def supports_stream_usage(self, provider: str | None = None) -> bool:
        selected = (provider or self.llm_provider).lower()
        if selected in {"openai", "deepseek"}:
            return True
        if selected in _CONFIGURABLE_LLM_PREFIXES:
            return bool(
                getattr(
                    self,
                    (
                        f"{_CONFIGURABLE_LLM_PREFIXES[selected]}"
                        "_supports_stream_usage"
                    ),
                )
            )
        return False

    def get_native_web_search_protocol(
        self,
        provider: str | None = None,
    ) -> str:
        selected = (provider or self.llm_provider).lower()
        if selected in {"openai", "deepseek"}:
            return "openai_responses"
        if selected in _CONFIGURABLE_LLM_PREFIXES:
            return str(
                getattr(
                    self,
                    (
                        f"{_CONFIGURABLE_LLM_PREFIXES[selected]}"
                        "_native_web_search_protocol"
                    ),
                )
            )
        return "none"

    def supports_native_web_search(
        self,
        provider: str | None = None,
    ) -> bool:
        return self.get_native_web_search_protocol(provider) != "none"

    def get_native_web_search_endpoint(
        self,
        provider: str | None = None,
    ) -> str | None:
        selected = (provider or self.llm_provider).lower()
        if selected not in _CONFIGURABLE_LLM_PREFIXES:
            return None
        value = getattr(
            self,
            (
                f"{_CONFIGURABLE_LLM_PREFIXES[selected]}"
                "_native_web_search_endpoint"
            ),
        )
        return value.strip() or None

    def get_model_pricing(
        self,
        model: str,
        provider: str | None = None,
    ) -> dict[str, float] | None:
        selected = (provider or self.llm_provider).strip().lower()
        normalized_model = model.strip().lower()
        if not normalized_model:
            return None
        normalized_pricing = {
            key.strip().lower(): value for key, value in self.model_pricing.items()
        }
        return normalized_pricing.get(
            f"{selected}:{normalized_model}"
        ) or normalized_pricing.get(normalized_model)

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")


class AppConfig(BaseSettings):
    data_dir: str = Field(default="data")
    traces_dir: Optional[str] = Field(default=None)
    semantic_memory_dir: str = Field(default="data/semantic-memory")
    log_level: str = Field(default="INFO")
    model_config = SettingsConfigDict(env_prefix="MINDFORGE_", extra="ignore")


class APIConfig(BaseSettings):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)
    access_token: str = Field(default="", max_length=4096)
    allow_insecure_remote_access: bool = Field(default=False)
    cors_origins: str = Field(default="http://localhost:5173")
    max_upload_mb: int = Field(default=200, ge=1, le=2048)
    max_text_file_mb: int = Field(default=20, ge=1, le=512)
    max_parsed_chars: int = Field(default=5_000_000, ge=10_000)
    max_pdf_pages: int = Field(default=500, ge=1, le=10_000)
    max_docx_uncompressed_mb: int = Field(default=100, ge=1, le=2048)
    max_docx_parts: int = Field(default=5000, ge=10, le=100_000)
    max_chunks_per_document: int = Field(default=2000, ge=1, le=100_000)
    index_batch_size: int = Field(default=128, ge=1, le=1000)
    max_concurrent_index_jobs: int = Field(default=1, ge=1, le=16)
    pdf_parse_executor: Literal["process", "thread"] = "process"
    pdf_parse_workers: int = Field(default=12, ge=1, le=64)
    pdf_parallel_page_threshold: int = Field(default=10, ge=1, le=1000)
    pdf_parse_batch_pages: int = Field(default=8, ge=1, le=128)
    health_refresh_seconds: int = Field(default=15, ge=5, le=300)
    max_history_entries: int = Field(
        default=0,
        ge=0,
        le=100_000,
        description="Maximum research history entries. 0 keeps records indefinitely.",
    )
    allow_local_file_index: bool = Field(default=False)
    model_discovery_timeout_seconds: float = Field(
        default=15.0,
        ge=1.0,
        le=60.0,
    )
    model_discovery_max_response_bytes: int = Field(
        default=2 * 1024 * 1024,
        ge=1024,
        le=20 * 1024 * 1024,
    )
    model_discovery_max_models: int = Field(
        default=1000,
        ge=1,
        le=10_000,
    )
    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    @field_validator("access_token")
    @classmethod
    def validate_access_token(cls, value: str) -> str:
        token = value.strip()
        if token and len(token) < 32:
            raise ValueError("API_ACCESS_TOKEN must contain at least 32 characters.")
        if any(ord(char) < 32 or ord(char) == 127 for char in token):
            raise ValueError("API_ACCESS_TOKEN must not contain control characters.")
        return token

    def get_cors_origins(self) -> list[str]:
        values = [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]
        return values or ["http://localhost:5173"]


class VectorStoreConfig(BaseSettings):
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: Optional[str] = Field(default=None)
    collection_name: str = Field(default="mindforge_docs")
    max_scroll_records: int = Field(default=100_000, ge=100, le=10_000_000)
    embedding_dim: int = Field(
        default=1536,
        description="Must match the embedding model dimension. "
        "OpenAI text-embedding-3-small = 1536, BGE-M3 = 1024.",
    )
    model_config = SettingsConfigDict(env_prefix="VECTOR_", extra="ignore")


class RetrievalConfig(BaseSettings):
    vector_top_k: int = Field(default=20)
    bm25_top_k: int = Field(default=20, ge=1, le=500)
    rerank_top_k: int = Field(default=6)
    max_request_top_k: int = Field(default=50, ge=1, le=500)
    reranker_max_candidates: int = Field(default=100, ge=1, le=1000)
    bm25_max_chunks: int = Field(default=100_000, ge=100)
    reranker_model: Optional[str] = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="CrossEncoder model name. Set to an empty value to disable.",
    )
    reranker_model_revision: str = Field(
        default="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        pattern=r"^[0-9a-fA-F]{40}$",
        description="Immutable Hugging Face revision for the reranker model.",
    )
    reranker_device: str = "cpu"
    reranker_preload: bool = True
    reranker_local_files_only: bool = True
    bm25_index_dir: Optional[str] = Field(
        default=None,
        description="Persistent BM25 corpus directory.",
    )
    min_score: float = Field(default=0.60, ge=0.0, le=1.0)
    keyword_min_coverage: float = Field(default=0.60, ge=0.0, le=1.0)
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", extra="ignore")


class ChunkingConfig(BaseSettings):
    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=64)
    model_config = SettingsConfigDict(env_prefix="CHUNK_", extra="ignore")


class ParserConfig(BaseSettings):
    """Adaptive document parsing controls."""

    mode: Literal["auto", "native", "ocr"] = "auto"
    ocr_enabled: bool = True
    ocr_language: str = Field(default="ch", min_length=1, max_length=32)
    ocr_device: str = Field(default="cpu", min_length=1, max_length=64)
    ocr_model_source: Literal["BOS", "HUGGINGFACE"] = "BOS"
    ocr_enable_mkldnn: bool = False
    ocr_dpi: int = Field(default=200, ge=72, le=400)
    ocr_min_native_text_chars: int = Field(default=30, ge=0, le=10_000)
    ocr_min_printable_ratio: float = Field(default=0.65, ge=0.0, le=1.0)
    ocr_max_pages: int = Field(default=600, ge=1, le=10_000)
    layout_enabled: bool = True
    table_extraction_enabled: bool = True
    table_max_cells: int = Field(default=10_000, ge=1, le=1_000_000)
    image_extraction_enabled: bool = True
    image_max_per_page: int = Field(default=20, ge=0, le=1_000)
    asset_persistence_enabled: bool = True
    source_retention_enabled: bool = True
    asset_storage_dir: str = "data/document-assets"
    asset_dpi: int = Field(default=144, ge=72, le=300)
    asset_max_per_document: int = Field(default=120, ge=0, le=2_000)
    asset_max_total_mb: int = Field(default=512, ge=1, le=10_240)
    asset_render_ocr_pages: bool = True
    ocr_handwriting_confidence: float = Field(default=0.62, ge=0.0, le=1.0)
    pipeline_version: int = Field(default=5, ge=1, le=1000)
    model_config = SettingsConfigDict(env_prefix="PARSER_", extra="ignore")


class VisualRetrievalConfig(BaseSettings):
    """Optional vision captioning before text-vector retrieval."""

    enabled: bool = False
    provider: Literal["openai_compatible"] = "openai_compatible"
    api_key: str = ""
    base_url: Optional[str] = None
    model: str = ""
    detail: Literal["low", "high", "auto"] = "low"
    max_assets_per_document: int = Field(default=24, ge=1, le=500)
    caption_concurrency: int = Field(default=4, ge=1, le=32)
    max_asset_bytes: int = Field(
        default=8 * 1024 * 1024,
        ge=64 * 1024,
        le=64 * 1024 * 1024,
    )
    max_tokens: int = Field(default=500, ge=64, le=4_000)
    request_timeout_seconds: int = Field(default=60, ge=5, le=600)
    prompt_version: str = Field(default="v1", min_length=1, max_length=64)
    model_config = SettingsConfigDict(env_prefix="VISUAL_", extra="ignore")


class RAPTORConfig(BaseSettings):
    raptor_levels: int = Field(default=3, ge=1, le=5)
    raptor_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    summary_model: str = Field(
        default="",
        description=(
            "Optional provider-specific override. Empty uses the active "
            "provider's researcher model."
        ),
    )
    max_nodes: int = Field(default=2000, ge=10, le=100_000)
    summary_concurrency: int = Field(default=4, ge=1, le=32)
    model_config = SettingsConfigDict(env_prefix="RAPTOR_", extra="ignore")


class GraphRAGConfig(BaseSettings):
    graph_enabled: bool = Field(default=True)
    entity_extraction_model: str = Field(
        default="",
        description=(
            "Optional provider-specific override. Empty uses the active "
            "provider's researcher model."
        ),
    )
    community_summary_model: str = Field(
        default="",
        description=(
            "Optional provider-specific override. Empty uses the entity "
            "extraction model."
        ),
    )
    max_entities_per_doc: int = Field(default=20)
    min_community_size: int = Field(default=3)
    extraction_char_budget: int = Field(
        default=12_000,
        ge=1_000,
        le=100_000,
    )
    graph_store_path: Optional[str] = Field(
        default=None,
        description="Persistent GraphRAG JSON file path.",
    )
    max_total_entities: int = Field(default=10_000, ge=100)
    max_communities: int = Field(default=500, ge=1)
    summary_concurrency: int = Field(default=4, ge=1, le=32)
    model_config = SettingsConfigDict(env_prefix="GRAPH_", extra="ignore")


class AgentConfig(BaseSettings):
    research_mode: Literal["fast", "balanced", "deep"] = "balanced"
    source_policy: Literal["auto", "knowledge_base", "web"] = "auto"
    fallback_enabled: bool = True
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Researcher Agent ReAct 最大轮次。设为 3 可显著提速，复杂任务可调高。",
    )
    critic_threshold: float = Field(default=7.0, ge=0.0, le=10.0)
    max_refine_rounds: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Critic 精炼最大轮次。设为 1 可减少一轮评估+重写，显著提速。",
    )
    subtask_timeout: int = Field(default=120, ge=10, description="单个子任务超时（秒）")
    research_timeout: int = Field(
        default=300,
        ge=30,
        description="研究全流程超时（秒）。Set via AGENT_RESEARCH_TIMEOUT env var.",
    )
    llm_request_timeout: int = Field(
        default=45,
        ge=5,
        le=600,
        description="Maximum duration of one non-streaming LLM request.",
    )
    max_subtasks: int = Field(default=5, ge=1, le=20)
    max_tool_calls_per_round: int = Field(default=4, ge=1, le=20)
    max_tool_calls_total: int = Field(default=12, ge=1, le=100)
    max_concurrent_research: int = Field(default=2, ge=1, le=32)
    max_concurrent_subtasks: int = Field(default=4, ge=1, le=64)
    max_concurrent_tool_calls: int = Field(default=8, ge=1, le=128)
    queue_timeout: int = Field(default=30, ge=1, le=600)
    sse_heartbeat_seconds: int = Field(default=10, ge=1, le=60)
    stream_chunk_size: int = Field(default=512, ge=64, le=8192)
    research_context_max_chars: int = Field(
        default=12_000,
        ge=1_000,
        le=100_000,
        description="Maximum shared-memory context passed to one research subtask.",
    )
    synthesis_context_max_chars: int = Field(
        default=60_000,
        ge=2_000,
        le=500_000,
        description="Maximum combined subtask context passed to Synthesizer.",
    )
    critic_source_context_max_chars: int = Field(
        default=12_000,
        ge=1_000,
        le=100_000,
        description="Maximum source evidence context passed to Critic.",
    )
    critic_report_context_max_chars: int = Field(
        default=60_000,
        ge=2_000,
        le=500_000,
        description="Maximum report context passed to Critic.",
    )
    model_config = SettingsConfigDict(env_prefix="AGENT_", extra="ignore")


class WebSearchConfig(BaseSettings):
    native_enabled: bool = Field(default=True)
    duckduckgo_enabled: bool = Field(default=False)
    model_only_fallback: bool = Field(default=True)
    max_results: int = Field(default=5, ge=1, le=20)
    native_max_output_tokens: int = Field(default=512, ge=128, le=8192)
    native_timeout_seconds: float = Field(default=30.0, ge=5.0, le=120.0)
    native_failure_cooldown_seconds: float = Field(
        default=300.0,
        ge=0.0,
        le=3600.0,
    )
    model_config = SettingsConfigDict(
        env_prefix="WEB_SEARCH_",
        extra="ignore",
    )


class CacheConfig(BaseSettings):
    redis_url: str = Field(default="redis://localhost:6377")
    model_config = SettingsConfigDict(env_prefix="CACHE_", extra="ignore")


class MemoryConfig(BaseSettings):
    working_capacity_tokens: int = Field(default=8000, ge=1000)
    chars_per_token: int = Field(default=4, ge=1)
    max_episodes: int = Field(default=200, ge=1)
    episodic_ttl_seconds: int = Field(default=2592000, ge=60)
    max_episode_chars: int = Field(default=100_000, ge=1000)
    max_semantic_facts: int = Field(default=500, ge=1)
    max_semantic_fact_chars: int = Field(default=50_000, ge=1000)
    semantic_search_chars: int = Field(default=8_000, ge=500, le=100_000)
    semantic_recall_fact_chars: int = Field(default=2_000, ge=200, le=20_000)
    semantic_recall_context_chars: int = Field(
        default=4_000,
        ge=500,
        le=50_000,
    )
    semantic_min_query_coverage: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    semantic_min_similarity: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
    )
    semantic_retention_days: int = Field(default=30, ge=1, le=3650)
    semantic_max_file_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1024 * 1024,
    )
    auto_capture_preferences: bool = Field(default=True)
    require_confirmation_for_profile: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="MEMORY_", extra="ignore")


class ContextConfig(BaseSettings):
    enabled: bool = Field(default=True)
    default_mode: str = Field(
        default="auto",
        pattern="^(auto|manual|disabled)$",
    )
    recent_message_limit: int = Field(default=8, ge=1, le=50)
    cross_conversation_enabled: bool = Field(default=False)
    budget_tokens: int = Field(default=4000, ge=500, le=100_000)
    artifact_top_k: int = Field(default=8, ge=1, le=50)
    memory_top_k: int = Field(default=4, ge=1, le=20)
    min_relevance: float = Field(default=0.12, ge=0.0, le=1.0)
    summary_message_threshold: int = Field(default=12, ge=4, le=200)
    summary_max_tokens: int = Field(default=1200, ge=100, le=10_000)
    snapshot_max_item_chars: int = Field(
        default=12_000,
        ge=500,
        le=100_000,
    )
    snapshot_retention_days: int = Field(default=90, ge=1, le=3650)
    model_config = SettingsConfigDict(env_prefix="CONTEXT_", extra="ignore")


class ObservabilityConfig(BaseSettings):
    langfuse_public_key: Optional[str] = Field(default=None)
    langfuse_secret_key: Optional[str] = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")
    enable_tracing: bool = Field(default=True)
    capture_content: bool = Field(default=False)
    max_record_chars: int = Field(default=20_000, ge=1000, le=1_000_000)
    max_trace_file_bytes: int = Field(
        default=5 * 1024 * 1024,
        ge=64 * 1024,
        le=1024 * 1024 * 1024,
    )
    trace_retention_days: int = Field(
        default=0,
        ge=0,
        le=3650,
        description="Local trace retention in days. 0 keeps traces indefinitely.",
    )
    trace_list_scan_limit: int = Field(default=1000, ge=100, le=100_000)
    trace_detail_span_limit: int = Field(default=2000, ge=100, le=100_000)
    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_", extra="ignore")


class SandboxConfig(BaseSettings):
    sandbox_timeout: int = Field(default=15, ge=5, le=60)
    temp_dir: str = Field(default="data/sandbox-tmp")
    max_output_length: int = Field(default=5000)
    max_code_length: int = Field(default=50_000, ge=100, le=1_000_000)
    max_vars_bytes: int = Field(default=100_000, ge=1024, le=10_000_000)
    memory_mb: int = Field(default=512, ge=64, le=4096)
    allowed_modules: list[str] = Field(
        default=[
            "numpy",
            "pandas",
            "scipy",
            "sklearn",
            "math",
            "json",
            "collections",
            "itertools",
            "datetime",
            "typing",
            "re",
        ]
    )
    model_config = SettingsConfigDict(env_prefix="SANDBOX_", extra="ignore")


class QAGenerationConfig(BaseSettings):
    output_dir: str = Field(default="data/qa")
    model: str = Field(
        default="",
        description=(
            "Optional provider-specific override. Empty uses the active "
            "provider's researcher model."
        ),
    )
    batch_size: int = Field(default=15, ge=1, le=100)
    concurrency: int = Field(default=3, ge=1, le=32)
    client_max_retries: int = Field(default=5, ge=0, le=20)
    request_attempts: int = Field(default=3, ge=1, le=10)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=256, le=65536)
    retry_base_seconds: float = Field(default=3.0, ge=0.1, le=60.0)
    model_config = SettingsConfigDict(env_prefix="QA_", extra="ignore")


class Settings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    visual_retrieval: VisualRetrievalConfig = Field(
        default_factory=VisualRetrievalConfig
    )
    raptor: RAPTORConfig = Field(default_factory=RAPTORConfig)
    graphrag: GraphRAGConfig = Field(default_factory=GraphRAGConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    qa_generation: QAGenerationConfig = Field(default_factory=QAGenerationConfig)

    model_config = SettingsConfigDict(extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """清除缓存的 Settings 并重新加载。

    在运行时通过 ``update_settings_api`` 修改 ``os.environ`` 后必须调用，
    否则 ``lru_cache`` 仍返回旧实例，配置切换不生效。
    """
    get_settings.cache_clear()
    return get_settings()
