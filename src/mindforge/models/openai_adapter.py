"""OpenAI 适配器 — 使用 openai.AsyncOpenAI 调用 GPT 系列模型及 embedding"""
from __future__ import annotations
from typing import List, Optional, AsyncIterator, Union
import os

import openai

from mindforge.models.base import BaseLLM, ChatMessage, ChatResult, StreamEvent


class OpenAIAdapter(BaseLLM):
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, max_retries: int = 3,
                 embed_model: str = "text-embedding-3-small", **kwargs):
        self.model = model
        self.embed_model = embed_model
        self.client = openai.AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
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
            rf_type = response_format.get("type")
            # 仅接受 OpenAI 支持的 response_format 类型，避免空 dict 或非法值导致 400
            if rf_type == "json_object":
                body["response_format"] = {"type": "json_object"}
            elif rf_type in ("json_schema", "text"):
                body["response_format"] = response_format

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
        # 流式 tool_calls 按 index 增量聚合，流结束后一次性发出完整 tool_calls
        tool_acc: dict[int, dict] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield StreamEvent(type="chunk", content=delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    slot = tool_acc.setdefault(idx, {
                        "id": None, "type": "function",
                        "function": {"name": None, "arguments": ""},
                    })
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
        yield StreamEvent(type="done")

    # ------------------------------------------------------------------
    # embed
    # ------------------------------------------------------------------
    async def embed(self, texts: List[str]) -> List[List[float]]:
        resp = await self.client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

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
