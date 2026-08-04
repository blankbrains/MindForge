"""模型抽象接口 — 支持多种 LLM 和 Embedding 提供者"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, List, Optional


class LLMConfigurationError(ValueError):
    """Raised when an LLM provider cannot start from the current configuration."""


@dataclass
class ChatMessage:
    role: str
    content: str = ""
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: Optional[List[dict]] = None
    usage: dict = field(default_factory=dict)
    model: str = ""
    def __str__(self): return self.content


@dataclass
class StreamEvent:
    type: str
    content: str = ""
    tool_calls: Optional[List[dict]] = None
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""


@dataclass
class NativeWebSearchResult:
    """Provider-normalized result from a server-side web search."""

    text: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    backend: str = ""
    answer_ready: bool = False


def normalize_token_usage(usage: Any) -> dict[str, int]:
    """Convert provider usage objects into a stable flat numeric mapping."""
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump(exclude_none=True)
    elif isinstance(usage, dict):
        raw = usage
    else:
        return {}
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, int] = {}

    def collect(value: Any, prefix: str = "") -> None:
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            if isinstance(item, bool):
                continue
            if isinstance(item, (int, float)):
                normalized[name] = int(item)
            elif isinstance(item, dict):
                collect(item, name)

    collect(raw)

    cached_tokens = normalized.get(
        "prompt_tokens_details_cached_tokens",
        normalized.get("input_tokens_details_cached_tokens", 0),
    )
    if cached_tokens and "prompt_cache_hit_tokens" not in normalized:
        normalized["prompt_cache_hit_tokens"] = cached_tokens
    return normalized


class BaseLLM(ABC):
    @property
    def supports_native_web_search(self) -> bool:
        return False

    async def search_web(
        self,
        query: str,
        *,
        max_results: int = 5,
        max_output_tokens: int = 1200,
    ) -> NativeWebSearchResult:
        del query, max_results, max_output_tokens
        raise LLMConfigurationError(
            "The selected provider does not support native web search."
        )

    @abstractmethod
    async def chat(self, messages, tools=None, response_format=None, temperature=0.7, stream=False):
        pass
    @abstractmethod
    async def embed(self, texts):
        pass
    @abstractmethod
    async def embed_single(self, text):
        pass


def _load_api_key_from_db(provider: str) -> str:
    """从数据库解密读取 API Key（服务器重启后的 fallback）。"""
    try:
        from mindforge.db import SessionLocal, ApiKey, decrypt_api_key
        db = SessionLocal()
        try:
            row = db.query(ApiKey).filter(
                ApiKey.provider == provider, ApiKey.is_active
            ).first()
            if row and row.key_encrypted:
                return decrypt_api_key(row.key_encrypted)
        finally:
            db.close()
    except Exception:
        pass
    return ""


def is_llm_configured(provider: str | None = None) -> bool:
    """Return whether the selected provider has enough runtime configuration."""
    from mindforge.config import get_settings

    settings = get_settings()
    selected = (provider or settings.llm.llm_provider).lower()
    LLMFactory._ensure_builtin_providers()
    if selected not in LLMFactory.available_providers():
        return False
    if not settings.llm.get_model("researcher", selected).strip():
        return False
    if (
        selected != "openai"
        and settings.llm.get_base_url(selected) is None
    ):
        return False
    if not settings.llm.requires_api_key(selected):
        return True
    configured = settings.llm.get_api_key(selected)
    return bool(configured.strip() or _load_api_key_from_db(selected))


def has_llm_credentials(provider: str | None = None) -> bool:
    """Backward-compatible alias for provider readiness checks."""
    return is_llm_configured(provider)


ProviderBuilder = Callable[[str, dict[str, Any]], BaseLLM]


class LLMFactory:
    """Registry-driven factory for cloud and self-hosted LLM providers."""

    _builtin_names = frozenset(
        {
            "openai",
            "deepseek",
            "kimi",
            "glm",
            "openai_compatible",
            "local",
        }
    )
    _providers: dict[str, ProviderBuilder] = {}
    _builtins_registered = False

    @classmethod
    def register_provider(
        cls,
        name: str,
        builder: ProviderBuilder,
        *,
        replace: bool = False,
    ) -> None:
        cls._ensure_builtin_providers()
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Provider name must not be empty.")
        if normalized in cls._providers and not replace:
            raise ValueError(f"Provider '{normalized}' is already registered.")
        cls._providers[normalized] = builder

    @classmethod
    def unregister_provider(cls, name: str) -> None:
        normalized = name.strip().lower()
        cls._providers.pop(normalized, None)
        if normalized in cls._builtin_names:
            cls._builtins_registered = False

    @classmethod
    def available_providers(cls) -> tuple[str, ...]:
        cls._ensure_builtin_providers()
        return tuple(sorted(cls._providers))

    @classmethod
    def create(cls, provider: str, model: str, **kwargs) -> BaseLLM:
        cls._ensure_builtin_providers()
        normalized = provider.strip().lower()
        builder = cls._providers.get(normalized)
        if builder is None:
            available = ", ".join(cls.available_providers())
            raise LLMConfigurationError(
                f"Unknown LLM provider: '{provider}'. "
                f"Available providers: {available}."
            )
        return builder(model, dict(kwargs))

    @classmethod
    def _ensure_builtin_providers(cls) -> None:
        if cls._builtins_registered:
            return
        cls._providers.setdefault("openai", cls._build_openai)
        cls._providers.setdefault("deepseek", cls._build_deepseek)
        cls._providers.setdefault("kimi", cls._build_kimi)
        cls._providers.setdefault("glm", cls._build_glm)
        cls._providers.setdefault(
            "openai_compatible",
            cls._build_openai_compatible,
        )
        cls._providers.setdefault("local", cls._build_local)
        cls._builtins_registered = True

    @staticmethod
    def _resolve_provider_values(
        provider: str,
        kwargs: dict[str, Any],
    ) -> tuple[Any, str, str | None]:
        from mindforge.config import get_settings

        settings = get_settings()
        explicit_key = kwargs.pop("api_key", None)
        api_key = (
            explicit_key
            if explicit_key is not None
            else (
                settings.llm.get_api_key(provider)
                or _load_api_key_from_db(provider)
            )
        )
        explicit_url = kwargs.pop("base_url", None)
        base_url = (
            explicit_url
            if explicit_url is not None
            else settings.llm.get_base_url(provider)
        )
        return settings, str(api_key or ""), base_url

    @classmethod
    def _build_openai(
        cls,
        model: str,
        kwargs: dict[str, Any],
    ) -> BaseLLM:
        from mindforge.models.openai_adapter import OpenAIAdapter

        settings, api_key, base_url = cls._resolve_provider_values(
            "openai",
            kwargs,
        )
        return OpenAIAdapter(
            model=model,
            api_key=api_key,
            base_url=base_url,
            embed_model=kwargs.pop(
                "embed_model",
                settings.llm.embedding_model,
            ),
            **kwargs,
        )

    @classmethod
    def _build_deepseek(
        cls,
        model: str,
        kwargs: dict[str, Any],
    ) -> BaseLLM:
        from mindforge.models.deepseek_adapter import DeepSeekAdapter

        _, api_key, base_url = cls._resolve_provider_values(
            "deepseek",
            kwargs,
        )
        return DeepSeekAdapter(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com",
            **kwargs,
        )

    @classmethod
    def _build_openai_compatible(
        cls,
        model: str,
        kwargs: dict[str, Any],
    ) -> BaseLLM:
        return cls._build_configurable_compatible(
            "openai_compatible",
            "compatible",
            model,
            kwargs,
        )

    @classmethod
    def _build_kimi(
        cls,
        model: str,
        kwargs: dict[str, Any],
    ) -> BaseLLM:
        return cls._build_configurable_compatible(
            "kimi",
            "kimi",
            model,
            kwargs,
        )

    @classmethod
    def _build_glm(
        cls,
        model: str,
        kwargs: dict[str, Any],
    ) -> BaseLLM:
        return cls._build_configurable_compatible(
            "glm",
            "glm",
            model,
            kwargs,
        )

    @classmethod
    def _build_local(
        cls,
        model: str,
        kwargs: dict[str, Any],
    ) -> BaseLLM:
        return cls._build_configurable_compatible(
            "local",
            "local",
            model,
            kwargs,
        )

    @classmethod
    def _build_configurable_compatible(
        cls,
        provider: str,
        config_prefix: str,
        model: str,
        kwargs: dict[str, Any],
    ) -> BaseLLM:
        from mindforge.models.openai_compatible_adapter import (
            OpenAICompatibleAdapter,
        )

        settings, api_key, base_url = cls._resolve_provider_values(
            provider,
            kwargs,
        )
        require_api_key = kwargs.pop(
            "require_api_key",
            getattr(settings.llm, f"{config_prefix}_api_key_required"),
        )
        supports_tools = kwargs.pop(
            "supports_tools",
            getattr(settings.llm, f"{config_prefix}_supports_tools"),
        )
        supports_json_mode = kwargs.pop(
            "supports_json_mode",
            getattr(settings.llm, f"{config_prefix}_supports_json_mode"),
        )
        supports_json_schema = kwargs.pop(
            "supports_json_schema",
            getattr(settings.llm, f"{config_prefix}_supports_json_schema"),
        )
        supports_stream_usage = kwargs.pop(
            "supports_stream_usage",
            getattr(settings.llm, f"{config_prefix}_supports_stream_usage"),
        )
        native_web_search_protocol = kwargs.pop(
            "native_web_search_protocol",
            getattr(
                settings.llm,
                f"{config_prefix}_native_web_search_protocol",
            ),
        )
        native_web_search_endpoint = kwargs.pop(
            "native_web_search_endpoint",
            getattr(
                settings.llm,
                f"{config_prefix}_native_web_search_endpoint",
            ),
        )
        return OpenAICompatibleAdapter(
            model=model,
            api_key=api_key,
            base_url=base_url,
            provider_name=provider,
            require_api_key=require_api_key,
            supports_tools=supports_tools,
            supports_json_mode=supports_json_mode,
            supports_json_schema=supports_json_schema,
            supports_stream_usage=supports_stream_usage,
            native_web_search_protocol=native_web_search_protocol,
            native_web_search_endpoint=native_web_search_endpoint,
            **kwargs,
        )
