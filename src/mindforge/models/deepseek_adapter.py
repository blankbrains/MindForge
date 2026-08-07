"""DeepSeek 适配器 — 通过 OpenAI 兼容接口调用 DeepSeek 模型。

DeepSeek 的 chat API 与 OpenAI 完全兼容，但价格约为 OpenAI 的 1/10。
DeepSeek 不提供原生 Embedding API，因此使用本地 sentence-transformers 模型
（默认 BAAI/bge-m3，由 LLM_LOCAL_EMBEDDING_MODEL 配置，向量维度统一使用
VECTOR_EMBEDDING_DIM）。
"""

from __future__ import annotations
from typing import Any, List, Optional, AsyncIterator, Literal, Union
import asyncio

import openai

from mindforge.models.base import (
    BaseLLM,
    ChatMessage,
    ChatResult,
    LLMConfigurationError,
    NativeWebSearchResult,
    StreamEvent,
    extract_json_object_from_reasoning,
    has_reasoning_delta,
    normalize_token_usage,
)
from mindforge.models.native_search import create_native_search_adapter


# 延迟加载 embedding 模型（单例，线程安全）
import threading as _threading

_EMBEDDER = None
_EMBEDDER_LOCK = _threading.Lock()


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:  # double-check
                from mindforge.config import get_settings
                from sentence_transformers import SentenceTransformer

                settings = get_settings().llm
                model_name = settings.local_embedding_model or "BAAI/bge-m3"
                _EMBEDDER = SentenceTransformer(
                    model_name,
                    device=settings.sentence_transformers_device,
                    revision=settings.local_embedding_revision,
                )
    return _EMBEDDER


DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekAdapter(BaseLLM):
    """DeepSeek 模型适配器

    特性:
    - Chat 使用 OpenAI 兼容接口 (https://api.deepseek.com)
    - Embedding 使用本地 BGE-m3 模型 (sentence-transformers)
    - 价格约为 OpenAI 的 1/10
    """

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: Optional[str] = None,
        base_url: str = DEEPSEEK_BASE_URL,
        max_retries: int = 0,
        thinking_mode: Literal[
            "default",
            "enabled",
            "disabled",
        ] = "default",
        **kwargs,
    ):
        normalized_model = model.strip()
        if not normalized_model:
            raise LLMConfigurationError("DeepSeek model is not configured.")
        self.model = normalized_model
        self._model = normalized_model
        self.provider_name = "deepseek"
        key = api_key or ""
        if not key or not key.strip():
            raise LLMConfigurationError(
                "DeepSeek API key is not configured. "
                "Set LLM_DEEPSEEK_API_KEY in .env or pass api_key."
            )
        self.client = openai.AsyncOpenAI(
            api_key=key,
            base_url=base_url,
            max_retries=max_retries,
        )
        self._native_search_adapter = create_native_search_adapter(
            "openai_responses",
            client=self.client,
            model=self.model,
            provider=self.provider_name,
            api_key=key,
            include_action_sources=False,
        )
        if thinking_mode not in {"default", "enabled", "disabled"}:
            raise LLMConfigurationError(
                "DeepSeek thinking_mode must be default, enabled, or disabled."
            )
        self._thinking_mode = thinking_mode
        self._extra_kwargs = kwargs

    @property
    def supports_native_web_search(self) -> bool:
        return True

    async def search_web(
        self,
        query: str,
        *,
        max_results: int = 5,
        max_output_tokens: int = 1200,
    ) -> NativeWebSearchResult:
        assert self._native_search_adapter is not None
        return await self._native_search_adapter.search(
            query=query,
            max_results=max_results,
            max_output_tokens=max_output_tokens,
        )

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[dict]] = None,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
        stream: bool = False,
        max_output_tokens: int | None = None,
    ) -> Union[ChatResult, AsyncIterator[StreamEvent]]:
        body = dict(
            model=self.model,
            messages=[self._to_openai_msg(m) for m in messages],
            temperature=temperature,
            **self._extra_kwargs,
        )
        if max_output_tokens is not None:
            body["max_tokens"] = int(max_output_tokens)
        self._apply_thinking_mode(body)
        if tools:
            body["tools"] = tools
        if response_format:
            # DeepSeek 仅支持 json_object，不支持 json_schema
            if response_format.get("type") == "json_object":
                body["response_format"] = {"type": "json_object"}

        if stream:
            return self._stream_chat(body)

        resp = await self.client.chat.completions.create(**body)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        content = msg.content or ""
        if not content:
            content = extract_json_object_from_reasoning(msg)

        return ChatResult(
            content=content,
            tool_calls=tool_calls,
            usage=normalize_token_usage(resp.usage),
            model=self.model,
        )

    async def _stream_chat(self, body: dict) -> AsyncIterator[StreamEvent]:
        stream = await self.client.chat.completions.create(
            **body,
            stream=True,
            stream_options={"include_usage": True},
        )
        # 流式 tool_calls 按 index 增量聚合，流结束后一次性发出完整 tool_calls
        tool_acc: dict[int, dict] = {}
        usage: dict[str, int] = {}
        finish_reason = ""
        reasoning_seen = False
        content_seen = False
        async for chunk in stream:
            chunk_usage = normalize_token_usage(getattr(chunk, "usage", None))
            if chunk_usage:
                usage = chunk_usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            chunk_finish_reason = getattr(choice, "finish_reason", None)
            if chunk_finish_reason:
                finish_reason = str(chunk_finish_reason)
            delta = choice.delta
            has_reasoning = has_reasoning_delta(delta)
            reasoning_seen = reasoning_seen or has_reasoning
            if delta.content:
                content_seen = True
                yield StreamEvent(type="chunk", content=delta.content)
            elif has_reasoning:
                yield StreamEvent(type="heartbeat")
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    slot = tool_acc.setdefault(
                        idx,
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": None, "arguments": ""},
                        },
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            slot["function"]["arguments"] += tc.function.arguments
        if tool_acc:
            yield StreamEvent(
                type="tool_call",
                tool_calls=[tool_acc[k] for k in sorted(tool_acc)],
            )
        yield StreamEvent(
            type="done",
            usage=usage,
            model=self.model,
            finish_reason=finish_reason,
            reasoning_only=reasoning_seen and not content_seen,
        )

    # ------------------------------------------------------------------
    # embed  — 使用本地 BGE-m3
    # ------------------------------------------------------------------
    async def embed(self, texts: List[str]) -> List[List[float]]:
        model = _get_embedder()
        # sentence-transformers 是同步的，使用 run_in_executor 避免阻塞事件循环
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, normalize_embeddings=True).tolist(),
        )
        return embeddings

    async def embed_single(self, text: str) -> List[float]:
        return (await self.embed([text]))[0]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _apply_thinking_mode(self, body: dict[str, Any]) -> None:
        if self._thinking_mode == "default":
            return
        existing = body.get("extra_body")
        if existing is not None and not isinstance(existing, dict):
            raise LLMConfigurationError(
                "DeepSeek extra_body must be a mapping."
            )
        extra_body = dict(existing or {})
        extra_body["thinking"] = {"type": self._thinking_mode}
        body["extra_body"] = extra_body

    @staticmethod
    def _to_openai_msg(m: ChatMessage) -> dict:
        d: dict = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        return d
