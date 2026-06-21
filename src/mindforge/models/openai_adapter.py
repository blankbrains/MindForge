"""OpenAI 适配器 — 使用 openai.AsyncOpenAI 调用 GPT 系列模型及 embedding"""
from __future__ import annotations
from typing import List, Optional, AsyncIterator, Union
import os
import json

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
