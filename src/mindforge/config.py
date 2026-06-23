# src/mindforge/config.py
"""统一配置管理 — 基于 Pydantic Settings，支持环境变量覆盖"""

from __future__ import annotations
import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# 显式加载 .env 文件（兼容 Windows 编码问题）
_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
if os.path.exists(_env_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, encoding="utf-8")
    except Exception:
        pass


class LLMConfig(BaseSettings):
    """LLM 配置 — 支持 OpenAI / DeepSeek 一键切换"""
    llm_provider: str = Field(default="openai", description="openai | deepseek")
    embedding_provider: str = Field(default="openai", description="openai | bge")
    openai_api_key: str = Field(default="")
    openai_base_url: Optional[str] = Field(default=None)
    deepseek_api_key: str = Field(default="")
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    planner_model: str = "gpt-4o"
    researcher_model: str = "gpt-4o-mini"
    critic_model: str = "gpt-4o"
    synthesizer_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    deepseek_planner: str = "deepseek-v4-flash"
    deepseek_researcher: str = "deepseek-v4-flash"
    deepseek_critic: str = "deepseek-v4-flash"
    deepseek_synthesizer: str = "deepseek-v4-flash"
    deepseek_embedding: str = "BAAI/bge-m3"
    embedding_dim: int = 1536
    local_embedding_model: str = "BAAI/bge-m3"
    local_embedding_dim: int = 1024

    def get_model(self, role: str) -> str:
        if self.llm_provider == "deepseek":
            mapping = {
                "planner": self.deepseek_planner,
                "researcher": self.deepseek_researcher,
                "critic": self.deepseek_critic,
                "synthesizer": self.deepseek_synthesizer,
            }
            return mapping.get(role, self.deepseek_researcher)
        return getattr(self, f"{role}_model", self.researcher_model)

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")


class VectorStoreConfig(BaseSettings):
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: Optional[str] = Field(default=None)
    collection_name: str = Field(default="mindforge_docs")
    embedding_dim: int = Field(
        default=1536,
        description="Must match the embedding model dimension. "
                    "OpenAI text-embedding-3-small = 1536, all-MiniLM-L6-v2 = 384, BGE-M3 = 1024."
    )
    model_config = SettingsConfigDict(env_prefix="VECTOR_", extra="ignore")


class RetrievalConfig(BaseSettings):
    vector_top_k: int = Field(default=20)
    bm25_top_k: int = Field(default=20)
    rerank_top_k: int = Field(default=6)
    min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", extra="ignore")


class ChunkingConfig(BaseSettings):
    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=64)
    use_semantic_chunking: bool = Field(default=False)
    model_config = SettingsConfigDict(env_prefix="CHUNK_", extra="ignore")


class RAPTORConfig(BaseSettings):
    raptor_levels: int = Field(default=3, ge=1, le=5)
    raptor_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    summary_model: str = Field(default="gpt-4o-mini")
    model_config = SettingsConfigDict(env_prefix="RAPTOR_", extra="ignore")


class GraphRAGConfig(BaseSettings):
    graph_enabled: bool = Field(default=True)
    entity_extraction_model: str = Field(default="gpt-4o-mini")
    community_summary_model: str = Field(default="gpt-4o-mini")
    max_entities_per_doc: int = Field(default=20)
    min_community_size: int = Field(default=3)
    graph_embedding_dim: int = Field(default=1536)
    model_config = SettingsConfigDict(env_prefix="GRAPH_", extra="ignore")


class AgentConfig(BaseSettings):
    max_iterations: int = Field(default=8, ge=1, le=20)
    max_search_steps: int = Field(default=5)
    critic_threshold: float = Field(default=7.0, ge=0.0, le=10.0)
    max_refine_rounds: int = Field(default=2)
    subtask_timeout: int = Field(default=45, ge=10)
    research_timeout: int = Field(default=300, ge=30)
    model_config = SettingsConfigDict(env_prefix="AGENT_", extra="ignore")


class MCPConfig(BaseSettings):
    mcp_config_path: str = Field(
        default=os.path.expanduser("~/.claude/mcp.json"),
    )
    mcp_auto_discover: bool = Field(default=True)
    mcp_tool_timeout: int = Field(default=30, ge=5)
    model_config = SettingsConfigDict(env_prefix="MCP_", extra="ignore")


class CacheConfig(BaseSettings):
    redis_url: str = Field(default="redis://localhost:6379")
    cache_ttl: int = Field(default=3600, ge=60)
    embedding_cache_size: int = Field(default=1000)
    model_config = SettingsConfigDict(env_prefix="CACHE_", extra="ignore")


class ObservabilityConfig(BaseSettings):
    langfuse_public_key: Optional[str] = Field(default=None)
    langfuse_secret_key: Optional[str] = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")
    enable_tracing: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_", extra="ignore")


class SandboxConfig(BaseSettings):
    sandbox_timeout: int = Field(default=15, ge=5, le=60)
    max_output_length: int = Field(default=5000)
    allowed_modules: list[str] = Field(default=[
        "numpy", "pandas", "scipy", "sklearn",
        "math", "json", "collections", "itertools",
        "datetime", "typing", "re",
    ])
    model_config = SettingsConfigDict(env_prefix="SANDBOX_", extra="ignore")


class Settings(BaseSettings):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    raptor: RAPTORConfig = Field(default_factory=RAPTORConfig)
    graphrag: GraphRAGConfig = Field(default_factory=GraphRAGConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
