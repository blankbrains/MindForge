"""Web search tool with Tavily API (primary) and DuckDuckGo (fallback)."""

from __future__ import annotations

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

    Uses Tavily API as the primary search backend with a DuckDuckGo-based
    fallback when Tavily is unavailable or unconfigured.
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
    ) -> None:
        super().__init__()
        self._tavily_client = tavily_client
        self._tavily_api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY", "")

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
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = False,
        include_domains: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        start = time.perf_counter()

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

        # 1. Try Tavily
        try:
            result = self._search_tavily(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
                include_domains=include_domains,
            )
        except Exception as exc:
            logger.warning(
                "Tavily search failed; falling back to DuckDuckGo: %s",
                type(exc).__name__,
            )
            result = None
        if result is not None:
            result.execution_time_ms = (time.perf_counter() - start) * 1000
            return result

        # 2. Fallback to DuckDuckGo
        result = self._search_duckduckgo(query=query, max_results=max_results)
        result.execution_time_ms = (time.perf_counter() - start) * 1000
        return result
