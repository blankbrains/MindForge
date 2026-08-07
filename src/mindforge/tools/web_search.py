"""Web search tool with native and optional auxiliary backends."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import time
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from bs4 import BeautifulSoup

from mindforge.tools.base import BaseTool, ToolResult

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # type: ignore[assignment]

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """Search the web for current information.

    Provider-native search is primary. Tavily and DuckDuckGo are optional
    auxiliary backends.
    """

    name = "web_search"
    description = (
        "Search the web for current information. Use this when you need "
        "up-to-date facts, news, or data that is not available in the "
        "internal knowledge base."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to return.",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "description": "Depth of search. Advanced includes more context.",
                "default": "basic",
            },
            "include_answer": {
                "type": "boolean",
                "description": "Include an AI-generated answer summary.",
                "default": False,
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of domains to restrict search to.",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        tavily_client: Optional[Any] = None,
        native_llm: Optional[Any] = None,
        native_enabled: Optional[bool] = None,
        duckduckgo_enabled: Optional[bool] = None,
        native_max_output_tokens: Optional[int] = None,
        native_timeout_seconds: Optional[float] = None,
        native_failure_cooldown_seconds: Optional[float] = None,
        prefer_tavily: Optional[bool] = None,
    ) -> None:
        super().__init__()
        from mindforge.config import get_settings

        settings = get_settings().web_search
        self._tavily_client = tavily_client
        self._tavily_api_key = (
            os.environ.get("TAVILY_API_KEY", "")
            if tavily_api_key is None
            else tavily_api_key
        )
        self._native_llm = native_llm
        self._native_enabled = (
            settings.native_enabled
            if native_enabled is None
            else native_enabled
        )
        self._duckduckgo_enabled = (
            settings.duckduckgo_enabled
            if duckduckgo_enabled is None
            else duckduckgo_enabled
        )
        self._prefer_tavily = (
            settings.prefer_tavily
            if prefer_tavily is None
            else prefer_tavily
        )
        self._native_max_output_tokens = (
            settings.native_max_output_tokens
            if native_max_output_tokens is None
            else native_max_output_tokens
        )
        self._native_timeout_seconds = (
            settings.native_timeout_seconds
            if native_timeout_seconds is None
            else native_timeout_seconds
        )
        self._native_failure_cooldown_seconds = (
            settings.native_failure_cooldown_seconds
            if native_failure_cooldown_seconds is None
            else native_failure_cooldown_seconds
        )
        self._native_disabled_until = 0.0
        self._default_max_results = settings.max_results
        self.parameters_schema = copy.deepcopy(type(self).parameters_schema)
        self.parameters_schema["properties"]["max_results"]["default"] = (
            self._default_max_results
        )

    @property
    def native_available(self) -> bool:
        return bool(
            self._native_configured()
            and time.monotonic() >= self._native_disabled_until
        )

    def _native_configured(self) -> bool:
        return bool(
            self._native_enabled
            and self._native_llm is not None
            and getattr(
                self._native_llm,
                "supports_native_web_search",
                False,
            )
        )

    def _open_native_circuit(self) -> None:
        cooldown = max(0.0, self._native_failure_cooldown_seconds)
        self._native_disabled_until = time.monotonic() + cooldown

    def _native_circuit_failure(self) -> ToolResult | None:
        if (
            not self._native_configured()
            or time.monotonic() >= self._native_disabled_until
        ):
            return None
        return ToolResult(
            success=False,
            output=(
                "Provider-native web search is temporarily skipped after "
                "a recent timeout or failure."
            ),
            error=(
                "Provider-native web search is temporarily unavailable "
                "during its failure cooldown."
            ),
            data={
                "backend": self._native_backend_name(),
                "failure_type": "native_circuit_open",
                "retryable": True,
                "terminal_for_run": True,
                "sources": [],
                "total": 0,
            },
        )

    @property
    def tavily_available(self) -> bool:
        return bool(
            self._tavily_client is not None
            or (TavilyClient is not None and self._tavily_api_key)
        )

    @property
    def duckduckgo_available(self) -> bool:
        return bool(self._duckduckgo_enabled and requests is not None)

    @property
    def available(self) -> bool:
        return (
            self.native_available
            or self.tavily_available
            or self.duckduckgo_available
        )

    def _native_backend_name(self) -> str:
        provider = str(
            getattr(self._native_llm, "provider_name", "")
        ).strip()
        return f"provider:{provider}:native" if provider else "provider:native"

    async def _search_native(
        self,
        query: str,
        max_results: int,
    ) -> ToolResult | None:
        if not self.native_available:
            return None
        result = await asyncio.wait_for(
            self._native_llm.search_web(
                query,
                max_results=max_results,
                max_output_tokens=self._native_max_output_tokens,
            ),
            timeout=self._native_timeout_seconds,
        )
        lines = [
            (
                "Web Search Results "
                f"(backend={result.backend}, query={query!r})"
            ),
            f"Found {len(result.sources)} source(s)",
            "-" * 72,
        ]
        if result.text.strip():
            lines.append(result.text.strip())
        elif result.sources:
            for source in result.sources:
                lines.append(
                    f"\n--- Result {source.get('index', '')} ---"
                    f"\nTitle: {source.get('title', 'Untitled')}"
                    f"\nURL:   {source.get('url', '')}"
                    f"\n{source.get('content', '')}"
                )
        return ToolResult(
            success=bool(result.text.strip() or result.sources),
            output="\n".join(lines),
            error=(
                None
                if result.text.strip() or result.sources
                else "Native web search returned no usable result."
            ),
            data={
                "results": result.sources,
                "sources": result.sources,
                "total": len(result.sources),
                "backend": result.backend,
                "usage": result.usage,
                "model": result.model,
                "answer": result.text,
                "answer_ready": result.answer_ready,
            },
        )

    # --- Primary: Tavily --------------------------------------------------------

    def _search_tavily(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = False,
        include_domains: Optional[list[str]] = None,
    ) -> Optional[ToolResult]:
        """Execute search via Tavily. Returns None if unavailable."""
        client = self._tavily_client
        if client is None:
            if TavilyClient is None or not self._tavily_api_key:
                return None
            client = TavilyClient(api_key=self._tavily_api_key)

        params: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
        }
        if include_domains:
            params["include_domains"] = include_domains

        response = client.search(**params)

        results = response.get("results", [])
        if not results:
            return ToolResult(
                success=True,
                output=f"No Tavily results for: {query}",
                data={"results": [], "total": 0, "backend": "tavily"},
            )

        formatted = self._format_tavily_results(
            results, query, response.get("answer")
        )
        return ToolResult(
            success=True,
            output=formatted,
            data={
                "results": results,
                "sources": self._build_sources(results),
                "total": len(results),
                "backend": "tavily",
            },
        )

    def _format_tavily_results(
        self,
        results: list[dict[str, Any]],
        query: str,
        answer: Optional[str] = None,
    ) -> str:
        lines: list[str] = [
            f"Web Search Results (backend=tavily, query={query!r})",
            f"Found {len(results)} result(s)",
            "-" * 72,
        ]

        if answer:
            lines.append(f"\nSummary: {answer}\n")

        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            content = r.get("content", r.get("snippet", ""))
            score = r.get("score", "")
            score_str = f" [score={score:.2f}]" if isinstance(score, (int, float)) else ""

            lines.append(
                f"\n--- Result {i}{score_str} ---"
                f"\nTitle: {title}"
                f"\nURL:   {url}"
                f"\n{content}"
            )

        return "\n".join(lines)

    # --- Fallback: DuckDuckGo (via requests) -----------------------------------

    def _search_duckduckgo(
        self,
        query: str,
        max_results: int = 5,
    ) -> ToolResult:
        """Fallback search using DuckDuckGo's HTML API (no API key needed)."""
        if requests is None:
            return ToolResult(
                success=False,
                error="Neither Tavily nor requests library is available.",
            )

        url = "https://html.duckduckgo.com/html/"
        params: dict[str, str] = {"q": query}

        try:
            resp = requests.post(
                url,
                data=params,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"DuckDuckGo search failed: {exc}",
            )

        results = self._parse_ddg_html(resp.text, max_results)

        if not results:
            return ToolResult(
                success=True,
                output=f"No DuckDuckGo results for: {query}",
                data={"results": [], "total": 0, "backend": "duckduckgo"},
            )

        formatted = self._format_ddg_results(results, query)
        return ToolResult(
            success=True,
            output=formatted,
            data={
                "results": results,
                "sources": self._build_sources(results),
                "total": len(results),
                "backend": "duckduckgo",
            },
        )

    @staticmethod
    def _build_sources(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "title": result.get("title", "网页来源"),
                "url": result.get("url", ""),
                "content": result.get(
                    "content",
                    result.get("snippet", ""),
                ),
                "source": "web",
            }
            for index, result in enumerate(results, start=1)
            if result.get("url")
        ]

    def _parse_ddg_html(
        self, html: str, max_results: int
    ) -> list[dict[str, str]]:
        """Parse DuckDuckGo HTML results, including redirect URLs."""
        results: list[dict[str, str]] = []
        soup = BeautifulSoup(html, "html.parser")
        for block in soup.select(".result__body"):
            if len(results) >= max_results:
                break

            title_node = block.select_one("a.result__a")
            if title_node is None:
                continue
            url = self._normalise_ddg_url(
                str(title_node.get("href", ""))
            )
            if not url:
                continue
            snippet_node = block.select_one(".result__snippet")
            results.append(
                {
                    "title": title_node.get_text(" ", strip=True),
                    "url": url,
                    "content": (
                        snippet_node.get_text(" ", strip=True)
                        if snippet_node is not None
                        else ""
                    ),
                }
            )

        return results

    @staticmethod
    def _normalise_ddg_url(href: str) -> str:
        if not href:
            return ""
        absolute = urljoin("https://duckduckgo.com", href)
        parsed = urlparse(absolute)
        if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                absolute = unquote(target)
                parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""
        return absolute

    @staticmethod
    def _validate_arguments(
        query: Any,
        max_results: Any,
        search_depth: Any,
        include_answer: Any,
        include_domains: Any,
    ) -> tuple[str, int, str, bool, Optional[list[str]]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or not 1 <= max_results <= 20
        ):
            raise ValueError("max_results must be an integer between 1 and 20.")
        if search_depth not in {"basic", "advanced"}:
            raise ValueError(
                "search_depth must be either 'basic' or 'advanced'."
            )
        if not isinstance(include_answer, bool):
            raise ValueError("include_answer must be a boolean.")
        domains: Optional[list[str]] = None
        if include_domains is not None:
            if not isinstance(include_domains, list) or any(
                not isinstance(domain, str) or not domain.strip()
                for domain in include_domains
            ):
                raise ValueError(
                    "include_domains must be a list of non-empty strings."
                )
            domains = [domain.strip() for domain in include_domains]
        return (
            query.strip(),
            max_results,
            search_depth,
            include_answer,
            domains,
        )

    def _format_ddg_results(
        self, results: list[dict[str, str]], query: str
    ) -> str:
        lines: list[str] = [
            f"Web Search Results (backend=duckduckgo, query={query!r})",
            f"Found {len(results)} result(s)",
            "-" * 72,
        ]
        for i, r in enumerate(results, 1):
            lines.append(
                f"\n--- Result {i} ---"
                f"\nTitle: {r.get('title', 'Untitled')}"
                f"\nURL:   {r.get('url', '')}"
                f"\n{r.get('content', '')}"
            )
        return "\n".join(lines)

    # --- Execute ----------------------------------------------------------------

    def execute(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: str = "basic",
        include_answer: bool = False,
        include_domains: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.execute_async(
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_answer=include_answer,
                    include_domains=include_domains,
                    **kwargs,
                )
            )
        return ToolResult(
            success=False,
            error="WebSearchTool.execute_async() is required in an event loop.",
        )

    async def execute_async(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: str = "basic",
        include_answer: bool = False,
        include_domains: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        start = time.perf_counter()
        if max_results is None:
            max_results = self._default_max_results

        try:
            (
                query,
                max_results,
                search_depth,
                include_answer,
                include_domains,
            ) = self._validate_arguments(
                query,
                max_results,
                search_depth,
                include_answer,
                include_domains,
            )
        except ValueError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        native_result: ToolResult | None = None
        native_failure = self._native_circuit_failure()
        tavily_attempted = False
        if self._prefer_tavily and self.tavily_available:
            tavily_attempted = True
            try:
                preferred_result = await asyncio.to_thread(
                    self._search_tavily,
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_answer=include_answer,
                    include_domains=include_domains,
                )
            except Exception as exc:
                logger.warning(
                    "Preferred Tavily search failed; trying provider-native "
                    "search: %s",
                    type(exc).__name__,
                )
            else:
                preferred_sources = (
                    preferred_result.data.get("sources", [])
                    if preferred_result is not None
                    and isinstance(preferred_result.data, dict)
                    else []
                )
                if preferred_result is not None and preferred_sources:
                    preferred_result.execution_time_ms = (
                        time.perf_counter() - start
                    ) * 1000
                    return preferred_result
                if (
                    preferred_result is not None
                    and not self.native_available
                    and not self.duckduckgo_available
                ):
                    preferred_result.execution_time_ms = (
                        time.perf_counter() - start
                    ) * 1000
                    return preferred_result
        if self.native_available:
            try:
                native_result = await self._search_native(
                    query=query,
                    max_results=max_results,
                )
            except asyncio.TimeoutError:
                message = (
                    "Provider-native web search timed out after "
                    f"{self._native_timeout_seconds:g} seconds."
                )
                native_failure = ToolResult(
                    success=False,
                    output=message,
                    error=message,
                    data={
                        "backend": self._native_backend_name(),
                        "failure_type": "native_timeout",
                        "retryable": True,
                        "terminal_for_run": True,
                        "sources": [],
                        "total": 0,
                    },
                )
                self._open_native_circuit()
                logger.warning(
                    "Provider-native web search timed out; trying auxiliary "
                    "backends."
                )
            except Exception as exc:
                message = (
                    "Provider-native web search failed before returning "
                    "results."
                )
                native_failure = ToolResult(
                    success=False,
                    output=message,
                    error=message,
                    data={
                        "backend": self._native_backend_name(),
                        "failure_type": "native_failed",
                        "retryable": True,
                        "terminal_for_run": True,
                        "error_type": type(exc).__name__,
                        "sources": [],
                        "total": 0,
                    },
                )
                self._open_native_circuit()
                logger.warning(
                    "Provider-native web search failed; trying auxiliary "
                    "backends: %s",
                    type(exc).__name__,
                )
            else:
                if native_result is not None and (
                    native_result.data or {}
                ).get("sources"):
                    self._native_disabled_until = 0.0
                    native_result.execution_time_ms = (
                        time.perf_counter() - start
                    ) * 1000
                    return native_result

        # 2. Try optional Tavily
        try:
            result = (
                await asyncio.to_thread(
                    self._search_tavily,
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_answer=include_answer,
                    include_domains=include_domains,
                )
                if self.tavily_available and not tavily_attempted
                else None
            )
        except Exception as exc:
            logger.warning(
                "Tavily search failed; falling back to DuckDuckGo: %s",
                type(exc).__name__,
            )
            result = None
        if result is not None:
            tavily_sources = (
                result.data.get("sources", [])
                if isinstance(result.data, dict)
                else []
            )
            if tavily_sources or (
                native_result is None and native_failure is None
            ):
                result.execution_time_ms = (
                    time.perf_counter() - start
                ) * 1000
                return result

        # 3. Try opt-in DuckDuckGo HTML search.
        if self.duckduckgo_available:
            result = await asyncio.to_thread(
                self._search_duckduckgo,
                query=query,
                max_results=max_results,
            )
            ddg_sources = (
                result.data.get("sources", [])
                if isinstance(result.data, dict)
                else []
            )
            if ddg_sources or (
                native_result is None and native_failure is None
            ):
                result.execution_time_ms = (
                    time.perf_counter() - start
                ) * 1000
                return result

        if native_result is not None:
            native_result.execution_time_ms = (
                time.perf_counter() - start
            ) * 1000
            return native_result

        if native_failure is not None:
            native_failure.execution_time_ms = (
                time.perf_counter() - start
            ) * 1000
            return native_failure

        return ToolResult(
            success=False,
            error="No web search backend is currently available.",
            data={
                "backend": "unavailable",
                "sources": [],
                "total": 0,
            },
            execution_time_ms=(time.perf_counter() - start) * 1000,
        )
