"""Generic adapter for OpenAI-compatible chat and embedding APIs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit

import openai

from mindforge.models.base import (
    BaseLLM,
    ChatMessage,
    ChatResult,
    LLMConfigurationError,
    NativeWebSearchResult,
    StreamEvent,
    normalize_token_usage,
)
from mindforge.models.native_search import (
    SUPPORTED_NATIVE_SEARCH_PROTOCOLS,
    create_native_search_adapter,
)


class OpenAICompatibleAdapter(BaseLLM):
    """Use one adapter for cloud and self-hosted OpenAI-compatible APIs."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        base_url: str | None = None,
        provider_name: str = "openai_compatible",
        require_api_key: bool = True,
        supports_tools: bool = True,
        supports_json_mode: bool = True,
        supports_json_schema: bool = False,
        supports_stream_usage: bool = False,
        native_web_search_protocol: str = "none",
        native_web_search_endpoint: str | None = None,
        embed_model: str | None = None,
        max_retries: int = 0,
        default_headers: dict[str, str] | None = None,
        **request_kwargs: Any,
    ) -> None:
        normalized_model = model.strip()
        if not normalized_model:
            raise LLMConfigurationError(
                f"{provider_name} model is not configured."
            )
        normalized_url = (base_url or "").strip() or None
        if provider_name not in {"openai"} and normalized_url is None:
            raise LLMConfigurationError(
                f"{provider_name} base URL is not configured."
            )
        if normalized_url is not None:
            normalized_url = self._validate_base_url(
                normalized_url,
                provider_name,
            )
        normalized_key = api_key.strip()
        if require_api_key and not normalized_key:
            raise LLMConfigurationError(
                f"{provider_name} API key is not configured."
            )

        self.model = normalized_model
        self._model = normalized_model
        self.provider_name = provider_name
        self.embed_model = (embed_model or "").strip() or None
        self.supports_tools = supports_tools
        self.supports_json_mode = supports_json_mode
        self.supports_json_schema = supports_json_schema
        self.supports_stream_usage = supports_stream_usage
        normalized_search_protocol = native_web_search_protocol.strip().lower()
        if normalized_search_protocol not in SUPPORTED_NATIVE_SEARCH_PROTOCOLS:
            raise LLMConfigurationError(
                "Unsupported native web search protocol: "
                f"{native_web_search_protocol}."
            )
        self.native_web_search_protocol = normalized_search_protocol
        self.client = openai.AsyncOpenAI(
            api_key=normalized_key or "not-required",
            base_url=normalized_url,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        endpoint = (native_web_search_endpoint or "").strip() or None
        if endpoint is not None:
            endpoint = self._validate_base_url(
                endpoint,
                f"{provider_name} native web search",
            )
        elif normalized_search_protocol == "glm_web_search" and normalized_url:
            endpoint = f"{normalized_url}/web_search"
        try:
            self._native_search_adapter = create_native_search_adapter(
                normalized_search_protocol,
                client=self.client,
                model=self.model,
                provider=self.provider_name,
                api_key=normalized_key,
                endpoint=endpoint,
                include_action_sources=True,
            )
        except ValueError as exc:
            raise LLMConfigurationError(str(exc)) from exc
        self._request_kwargs = request_kwargs

    @property
    def supports_native_web_search(self) -> bool:
        return self._native_search_adapter is not None

    async def search_web(
        self,
        query: str,
        *,
        max_results: int = 5,
        max_output_tokens: int = 1200,
    ) -> NativeWebSearchResult:
        if not self.supports_native_web_search:
            return await super().search_web(
                query,
                max_results=max_results,
                max_output_tokens=max_output_tokens,
            )
        assert self._native_search_adapter is not None
        return await self._native_search_adapter.search(
            query=query,
            max_results=max_results,
            max_output_tokens=max_output_tokens,
        )

    @staticmethod
    def _validate_base_url(value: str, provider_name: str) -> str:
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as exc:
            raise LLMConfigurationError(
                f"{provider_name} base URL is invalid."
            ) from exc
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise LLMConfigurationError(
                f"{provider_name} base URL must be an absolute HTTP(S) URL."
            )
        if parsed.username is not None or parsed.password is not None:
            raise LLMConfigurationError(
                f"{provider_name} base URL must not contain credentials."
            )
        if parsed.query or parsed.fragment:
            raise LLMConfigurationError(
                f"{provider_name} base URL must not contain a query or fragment."
            )
        if port is not None and not 1 <= port <= 65535:
            raise LLMConfigurationError(
                f"{provider_name} base URL port is invalid."
            )
        return value.rstrip("/")

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> ChatResult | AsyncIterator[StreamEvent]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_openai_msg(message) for message in messages],
            "temperature": temperature,
            **self._request_kwargs,
        }
        if tools and self.supports_tools:
            body["tools"] = tools
        normalized_format = self._normalize_response_format(response_format)
        if normalized_format is not None:
            body["response_format"] = normalized_format
        if stream:
            return self._stream_chat(body)

        response = await self.client.chat.completions.create(**body)
        if not response.choices:
            return ChatResult(model=self.model)
        message = response.choices[0].message
        tool_calls = self._serialize_tool_calls(message.tool_calls)
        usage = response.usage
        return ChatResult(
            content=message.content or "",
            tool_calls=tool_calls,
            usage=normalize_token_usage(usage),
            model=self.model,
        )

    async def _stream_chat(
        self,
        body: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        stream_kwargs: dict[str, Any] = {"stream": True}
        if self.supports_stream_usage:
            stream_kwargs["stream_options"] = {"include_usage": True}
        stream = await self.client.chat.completions.create(
            **body,
            **stream_kwargs,
        )
        tool_accumulator: dict[int, dict[str, Any]] = {}
        usage: dict[str, int] = {}
        async for chunk in stream:
            chunk_usage = normalize_token_usage(
                getattr(chunk, "usage", None)
            )
            if chunk_usage:
                usage = chunk_usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield StreamEvent(type="chunk", content=content)
            for tool_call in getattr(delta, "tool_calls", None) or []:
                index = tool_call.index if tool_call.index is not None else 0
                slot = tool_accumulator.setdefault(
                    index,
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": None, "arguments": ""},
                    },
                )
                if tool_call.id:
                    slot["id"] = tool_call.id
                function = tool_call.function
                if function is None:
                    continue
                if function.name:
                    slot["function"]["name"] = function.name
                if function.arguments:
                    slot["function"]["arguments"] += function.arguments
        if tool_accumulator:
            yield StreamEvent(
                type="tool_call",
                tool_calls=[
                    tool_accumulator[index]
                    for index in sorted(tool_accumulator)
                ],
            )
        yield StreamEvent(
            type="done",
            usage=usage,
            model=self.model,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.embed_model is None:
            raise LLMConfigurationError(
                f"{self.provider_name} embedding model is not configured."
            )
        response = await self.client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_single(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

    def _normalize_response_format(
        self,
        response_format: dict | None,
    ) -> dict | None:
        if not response_format:
            return None
        response_type = response_format.get("type")
        if response_type == "json_object" and self.supports_json_mode:
            return {"type": "json_object"}
        if response_type == "json_schema" and self.supports_json_schema:
            return response_format
        return None

    @staticmethod
    def _serialize_tool_calls(tool_calls: Any) -> list[dict] | None:
        if not tool_calls:
            return None
        return [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in tool_calls
        ]

    @staticmethod
    def _to_openai_msg(message: ChatMessage) -> dict[str, Any]:
        serialized: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_calls:
            serialized["tool_calls"] = message.tool_calls
        if message.tool_call_id:
            serialized["tool_call_id"] = message.tool_call_id
        return serialized
