"""Provider-native web search protocol adapters."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx

from mindforge.models.base import (
    NativeWebSearchResult,
    normalize_token_usage,
)

_MARKDOWN_LINK_RE = re.compile(
    r"\[([^\]]{1,300})\]\((https?://[^)\s]+)\)",
    re.IGNORECASE,
)
_RAW_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.IGNORECASE)
_GENERIC_SOURCE_TITLES = frozenset(
    {
        "",
        "untitled",
        "web",
        "web source",
    }
)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _valid_public_url(value: Any) -> str:
    raw = str(value or "").strip().rstrip(".,;:!?")
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            return ""
    except ValueError:
        return ""
    if parsed.fragment.startswith("ws_call_id="):
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.query,
                "",
            )
        )
    return raw


def _message_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in payload.get("output", []):
        item_dict = _as_dict(item)
        if item_dict.get("type") != "message":
            continue
        for part in item_dict.get("content", []):
            part_dict = _as_dict(part)
            text = part_dict.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n\n".join(chunks)


def _source_candidates(
    payload: dict[str, Any],
    text: str,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    def collect_structured(value: Any) -> None:
        value_dict = _as_dict(value)
        if value_dict:
            raw_url = (
                value_dict.get("url")
                or value_dict.get("uri")
                or value_dict.get("link")
            )
            if raw_url:
                candidates.append(
                    (
                        str(
                            value_dict.get("title")
                            or value_dict.get("media")
                            or "Web source"
                        ),
                        str(raw_url),
                    )
                )
            for nested in value_dict.values():
                collect_structured(nested)
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                collect_structured(nested)

    collect_structured(payload)
    for item in payload.get("output", []):
        item_dict = _as_dict(item)
        if item_dict.get("type") == "web_search_call":
            action = _as_dict(item_dict.get("action"))
            for source in action.get("sources", []):
                source_dict = _as_dict(source)
                candidates.append(
                    (
                        str(source_dict.get("title") or "Web source"),
                        str(source_dict.get("url") or source_dict.get("uri") or ""),
                    )
                )
        if item_dict.get("type") != "message":
            continue
        for part in item_dict.get("content", []):
            part_dict = _as_dict(part)
            for annotation in part_dict.get("annotations", []):
                annotation_dict = _as_dict(annotation)
                candidates.append(
                    (
                        str(annotation_dict.get("title") or "Web source"),
                        str(
                            annotation_dict.get("url")
                            or annotation_dict.get("uri")
                            or ""
                        ),
                    )
                )

    candidates.extend(
        (title.strip() or "Web source", url)
        for title, url in _MARKDOWN_LINK_RE.findall(text)
    )
    candidates.extend(("Web source", url) for url in _RAW_URL_RE.findall(text))
    return candidates


def _source_evidence(text: str) -> dict[str, str]:
    evidence_by_url: dict[str, str] = {}
    for block in re.split(r"\n\s*\n", text):
        normalized_block = block.strip()
        if not normalized_block:
            continue
        raw_urls = [
            url
            for _title, url in _MARKDOWN_LINK_RE.findall(normalized_block)
        ]
        raw_urls.extend(_RAW_URL_RE.findall(normalized_block))
        for raw_url in raw_urls:
            url = _valid_public_url(raw_url)
            if not url:
                continue
            existing = evidence_by_url.get(url)
            if existing == normalized_block:
                continue
            evidence_by_url[url] = (
                f"{existing}\n\n{normalized_block}"[:4000]
                if existing
                else normalized_block[:4000]
            )
    return evidence_by_url


def _is_generic_source_title(value: str) -> bool:
    return value.strip().casefold() in _GENERIC_SOURCE_TITLES


def _derived_source_title(url: str) -> str:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").removeprefix("www.")
    path_parts = [
        unquote(part).strip()
        for part in parsed.path.split("/")
        if part.strip()
    ]
    slug = path_parts[-1] if path_parts else ""
    slug = re.sub(r"\.(?:html?|php|aspx?)$", "", slug, flags=re.IGNORECASE)
    if (
        not slug
        or slug.casefold() in {"article", "detail", "index", "page"}
        or slug.isdigit()
    ):
        return host or "Web source"
    readable = re.sub(r"[-_]+", " ", slug)
    readable = re.sub(r"\s+", " ", readable).strip()
    return f"{host} - {readable}" if host and readable else (host or readable)


def normalize_responses_web_search(
    response: Any,
    *,
    provider: str,
    max_results: int,
) -> NativeWebSearchResult:
    payload = _as_dict(response)
    output_text = getattr(response, "output_text", None)
    text = (
        output_text.strip()
        if isinstance(output_text, str) and output_text.strip()
        else _message_text(payload)
    )
    sources: list[dict[str, Any]] = []
    sources_by_url: dict[str, dict[str, Any]] = {}
    evidence_by_url = _source_evidence(text)
    for title, raw_url in _source_candidates(payload, text):
        url = _valid_public_url(raw_url)
        if not url:
            continue
        normalized_title = title.strip()[:500] or "Web source"
        existing = sources_by_url.get(url)
        if existing is not None:
            if (
                _is_generic_source_title(str(existing.get("title") or ""))
                and not _is_generic_source_title(normalized_title)
            ):
                existing["title"] = normalized_title
            if not existing.get("content") and evidence_by_url.get(url):
                existing["content"] = evidence_by_url[url]
            continue
        if len(sources) >= max_results:
            continue
        source = {
            "index": len(sources) + 1,
            "title": normalized_title,
            "url": url,
            "content": evidence_by_url.get(url, ""),
            "source": "web",
            "backend": f"{provider}:native",
            "verification_mode": "provider_native",
        }
        sources.append(source)
        sources_by_url[url] = source

    for source in sources:
        if _is_generic_source_title(str(source.get("title") or "")):
            source["title"] = _derived_source_title(str(source["url"]))

    return NativeWebSearchResult(
        text=text,
        sources=sources,
        usage=normalize_token_usage(payload.get("usage")),
        model=str(payload.get("model") or ""),
        backend=f"{provider}:native",
        answer_ready=False,
    )


async def responses_web_search(
    client: Any,
    *,
    model: str,
    provider: str,
    query: str,
    max_results: int,
    max_output_tokens: int,
    include_action_sources: bool,
) -> NativeWebSearchResult:
    prompt = (
        "Use web search to gather current, verifiable evidence for the "
        "following query. This is an evidence-collection step, not the final "
        "answer. Respond in the query's language with compact, complete "
        "evidence bullets, each paired with its exact source URL as a "
        "Markdown link. Cover all compared subjects and major decision "
        "dimensions before adding detail. Do not start a long report or leave "
        "an unfinished section. Return no more than "
        f"{max_results} distinct sources.\n\nQuery: {query}"
    )
    request: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "tools": [{"type": "web_search"}],
        "tool_choice": "required",
        "max_output_tokens": max_output_tokens,
    }
    if include_action_sources:
        request["include"] = ["web_search_call.action.sources"]
    response = await client.responses.create(**request)
    return normalize_responses_web_search(
        response,
        provider=provider,
        max_results=max_results,
    )


class NativeSearchAdapter(ABC):
    """Provider-specific native web search capability."""

    protocol: str = "none"

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int,
        max_output_tokens: int,
    ) -> NativeWebSearchResult:
        ...


class ResponsesNativeSearchAdapter(NativeSearchAdapter):
    protocol = "openai_responses"

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        provider: str,
        include_action_sources: bool,
    ) -> None:
        self._client = client
        self._model = model
        self._provider = provider
        self._include_action_sources = include_action_sources

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        max_output_tokens: int,
    ) -> NativeWebSearchResult:
        return await responses_web_search(
            self._client,
            model=self._model,
            provider=self._provider,
            query=query,
            max_results=max_results,
            max_output_tokens=max_output_tokens,
            include_action_sources=self._include_action_sources,
        )


class KimiBuiltinSearchAdapter(NativeSearchAdapter):
    protocol = "kimi_builtin"

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        provider: str,
    ) -> None:
        self._client = client
        self._model = model
        self._provider = provider

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        max_output_tokens: int,
    ) -> NativeWebSearchResult:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Use the built-in web search tool. Answer in the query's "
                    "language and include exact source URLs as Markdown links."
                ),
            },
            {"role": "user", "content": query},
        ]
        tools = [
            {
                "type": "builtin_function",
                "function": {"name": "$web_search"},
            }
        ]
        aggregated_usage: dict[str, int] = {}
        evidence_payload: dict[str, Any] = {"tool_results": []}
        final_text = ""
        for _ in range(4):
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                max_tokens=max_output_tokens,
            )
            usage = normalize_token_usage(getattr(response, "usage", None))
            for key, value in usage.items():
                aggregated_usage[key] = aggregated_usage.get(key, 0) + value
            if not response.choices:
                break
            choice = response.choices[0]
            message = choice.message
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if not tool_calls:
                final_text = str(message.content or "").strip()
                break
            dumped_message = _as_dict(message)
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": (
                            _as_dict(tool_call).get("type")
                            or "builtin_function"
                        ),
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
            if dumped_message.get("reasoning_content"):
                assistant_message["reasoning_content"] = dumped_message[
                    "reasoning_content"
                ]
            messages.append(assistant_message)
            for tool_call in tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except (TypeError, ValueError):
                    arguments = {}
                evidence_payload["tool_results"].append(arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps(arguments, ensure_ascii=False),
                    }
                )

        payload = {
            "model": self._model,
            "usage": aggregated_usage,
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": final_text,
                            "annotations": [],
                        }
                    ],
                }
            ],
            **evidence_payload,
        }
        return normalize_responses_web_search(
            payload,
            provider=self._provider,
            max_results=max_results,
        )


class GLMWebSearchAdapter(NativeSearchAdapter):
    protocol = "glm_web_search"

    def __init__(
        self,
        *,
        api_key: str,
        provider: str,
        endpoint: str,
    ) -> None:
        self._api_key = api_key
        self._provider = provider
        self._endpoint = endpoint.rstrip("/")

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        max_output_tokens: int,
    ) -> NativeWebSearchResult:
        del max_output_tokens
        payload = {
            "search_query": query[:70],
            "search_engine": "search_pro",
            "search_intent": False,
            "count": max_results,
            "search_recency_filter": "noLimit",
            "content_size": "high",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self._endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        results = body.get("search_result", [])
        if not isinstance(results, list):
            results = []
        sources: list[dict[str, Any]] = []
        for item in results[:max_results]:
            if not isinstance(item, dict):
                continue
            url = _valid_public_url(item.get("link"))
            if not url:
                continue
            sources.append(
                {
                    "index": len(sources) + 1,
                    "title": str(
                        item.get("title")
                        or item.get("media")
                        or "Web source"
                    ),
                    "url": url,
                    "content": str(item.get("content") or "")[:4000],
                    "source": "web",
                    "backend": f"{self._provider}:native",
                    "published_at": item.get("publish_date"),
                }
            )
        text = "\n\n".join(
            f"[{source['index']}] {source['title']}\n"
            f"URL: {source['url']}\n{source['content']}"
            for source in sources
        )
        return NativeWebSearchResult(
            text=text,
            sources=sources,
            usage={},
            model="glm-web-search",
            backend=f"{self._provider}:native",
        )


SUPPORTED_NATIVE_SEARCH_PROTOCOLS = frozenset(
    {
        "none",
        "openai_responses",
        "kimi_builtin",
        "glm_web_search",
    }
)


def create_native_search_adapter(
    protocol: str,
    *,
    client: Any,
    model: str,
    provider: str,
    api_key: str,
    endpoint: str | None = None,
    include_action_sources: bool = True,
) -> NativeSearchAdapter | None:
    normalized = protocol.strip().lower()
    if normalized == "none":
        return None
    if normalized not in SUPPORTED_NATIVE_SEARCH_PROTOCOLS:
        raise ValueError(f"Unsupported native web search protocol: {protocol}")
    if normalized == "openai_responses":
        return ResponsesNativeSearchAdapter(
            client=client,
            model=model,
            provider=provider,
            include_action_sources=include_action_sources,
        )
    if normalized == "kimi_builtin":
        return KimiBuiltinSearchAdapter(
            client=client,
            model=model,
            provider=provider,
        )
    if not endpoint:
        raise ValueError(
            "glm_web_search requires a native web search endpoint."
        )
    if not api_key.strip():
        raise ValueError("glm_web_search requires an API key.")
    return GLMWebSearchAdapter(
        api_key=api_key,
        provider=provider,
        endpoint=endpoint,
    )
