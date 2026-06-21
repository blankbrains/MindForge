"""DeepSeek 适配器 — 通过 OpenAI 兼容接口调用 DeepSeek 模型。

DeepSeek 的 chat API 与 OpenAI 完全兼容，但价格约为 OpenAI 的 1/10。
DeepSeek 不提供原生 Embedding API，因此使用 sentence-transformers (BGE-m3) 本地计算嵌入向量。
"""
from __future__ import annotations
from typing import List, Optional, AsyncIterator, Union
import asyncio
import json
import os

import openai
import numpy as np

from mindforge.models.base import BaseLLM, ChatMessage, ChatResult, StreamEvent


# 延迟加载 BGE-m3 模型（单例）
_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(
            "BAAI/bge-m3",
            device=os.getenv("SENTENCE_TRANSFORMERS_DEVICE", "cpu"),
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

    def __init__(self, model: str = "deepseek-chat", api_key: Optional[str] = None,
                 base_url: str = DEEPSEEK_BASE_URL, max_retries: int = 3, **kwargs):
        self.model = model
        self.client = openai.AsyncOpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=base_url,
            max_retries=max_retries,
        )
        self._extra_kwargs = kwargs

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------
    async def chat(self, messages: List[ChatMessage], tools: Optional[List[dict]] = None,
                   response_format: Optional[dict] = None, temperature: float = 0.7,
                   stream: bool = False) -> Union[ChatResult, AsyncIterator[StreamEvent]]:
        body = dict(
            model=self.model,
            messages=[self._to_openai_msg(m) for m in messages],
            temperature=temperature,
            **self._extra_kwargs,
        )
        if tools:
            body["tools"] = tools
        if response_format:
            # DeepSeek 也支持 json_object response_format
            body["response_format"] = {"type": "json_object"} if response_format.get("type") == "json_object" else response_format

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
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in msg.tool_calls
            ]

        return ChatResult(
            content=msg.content or "",
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
            } if resp.usage else {},
            model=self.model,
        )

    async def _stream_chat(self, body: dict) -> AsyncIterator[StreamEvent]:
        stream = await self.client.chat.completions.create(**body, stream=True)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield StreamEvent(type="chunk", content=delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    yield StreamEvent(
                        type="tool_call",
                        tool_calls=[{
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        }] if tc.id else [{
                            "type": "function",
                            "function": {"name": tc.function.name if tc.function else None,
                                         "arguments": tc.function.arguments if tc.function else ""}
                        }],
                    )
        yield StreamEvent(type="done")

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
    @staticmethod
    def _to_openai_msg(m: ChatMessage) -> dict:
        d: dict = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        return d
