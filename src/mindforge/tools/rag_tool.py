"""RAG tool that queries the MindForge knowledge base via AdaptiveRetriever."""

from __future__ import annotations

import time
from typing import Any, Optional

from mindforge.tools.base import BaseTool, ToolResult

try:
    from mindforge.retrieval.adaptive import AdaptiveRetriever, QueryMode
except ImportError:
    AdaptiveRetriever = None  # type: ignore[assignment]


class RAGTool(BaseTool):
    """Tool that queries the knowledge base using AdaptiveRetriever.

    Supports multiple retrieval modes (semantic, hybrid, keyword) and
    configurable top-k result counts.
    """

    name = "search_knowledge_base"
    description = (
        "Search the internal knowledge base for relevant information. "
        "Use this when you need facts, context, or supporting evidence from "
        "the project's stored documents."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query or question to look up.",
            },
            "mode": {
                "type": "string",
                "enum": ["semantic", "hybrid", "keyword"],
                "description": "Retrieval mode to use.",
                "default": "hybrid",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top results to return.",
                "default": 5,
                "minimum": 1,
                "maximum": 50,
            },
            "threshold": {
                "type": "number",
                "description": "Minimum relevance score (0-1) for results.",
                "default": 0.0,
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        retriever: Optional[Any] = None,
        retriever_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._retriever = retriever
        self._retriever_kwargs = retriever_kwargs or {}

    def _get_retriever(self) -> Any:
        """Lazy-init AdaptiveRetriever if none was provided."""
        if self._retriever is not None:
            return self._retriever
        if AdaptiveRetriever is not None:
            self._retriever = AdaptiveRetriever(**self._retriever_kwargs)
            return self._retriever
        raise RuntimeError(
            "AdaptiveRetriever is not available. "
            "Install mindforge with retrieval extras or provide a retriever instance."
        )

    def execute(self, query: str, mode: str = "hybrid", top_k: int = 5, threshold: float = 0.0, **kwargs: Any) -> ToolResult:
        """Synchronous wrapper — uses thread pool when event loop is running."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop — safe to use asyncio.run()
            return asyncio.run(self.execute_async(query=query, mode=mode, top_k=top_k, threshold=threshold, **kwargs))
        else:
            # Event loop is running — use run_until_complete in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.execute_async(query=query, mode=mode, top_k=top_k, threshold=threshold, **kwargs)
                )
                return future.result()

    async def execute_async(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        threshold: float = 0.0,
        **kwargs: Any,
    ) -> ToolResult:
        start = time.perf_counter()

        if not query or not query.strip():
            return ToolResult(
                success=False,
                error="Query must be a non-empty string.",
            )

        retriever = self._get_retriever()

        # Convert string mode to QueryMode enum with distinct strategy mappings
        mode_map = {
            "semantic": QueryMode.CONCEPTUAL,  # vector-heavy
            "hybrid": QueryMode.FACTUAL,       # balanced
            "keyword": QueryMode.PROCEDURAL,   # keyword-heavy
        }
        qmode = mode_map.get(mode, QueryMode.FACTUAL)

        try:
            result_dict = await retriever.retrieve(
                query=query,
                mode=qmode,
                top_k=top_k,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=f"Retrieval failed: {exc}",
                execution_time_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
        results = result_dict.get("results", []) if isinstance(result_dict, dict) else result_dict

        if not results:
            return ToolResult(
                success=True,
                output=f"No results found for query: {query}",
                data={"results": [], "total": 0, "mode": mode},
                execution_time_ms=elapsed,
            )

        formatted = self._format_results(results, query, mode)
        return ToolResult(
            success=True,
            output=formatted,
            data={
                "results": results,
                "total": len(results),
                "mode": mode,
                "query": query,
            },
            execution_time_ms=elapsed,
        )

    def _format_results(self, results: list[Any], query: str, mode: str) -> str:
        """Format retrieved documents into a readable string."""
        lines: list[str] = [
            f"Knowledge Base Results (mode={mode}, query={query!r})",
            f"Found {len(results)} result(s)",
            "-" * 72,
        ]

        for i, doc in enumerate(results, 1):
            if hasattr(doc, "page_content"):
                content = doc.page_content
            elif isinstance(doc, dict):
                content = doc.get("content", doc.get("text", str(doc)))
            else:
                content = str(doc)

            score = ""
            if hasattr(doc, "score"):
                score = f" [score={doc.score:.3f}]"
            elif isinstance(doc, dict) and "score" in doc:
                score = f" [score={doc['score']:.3f}]"

            source = ""
            if hasattr(doc, "metadata") and doc.metadata:
                source = doc.metadata.get("source", doc.metadata.get("title", ""))
            elif isinstance(doc, dict):
                meta = doc.get("metadata", {}) or {}
                source = meta.get("source", meta.get("title", ""))

            header = f"\n--- Result {i}{score} ---"
            if source:
                header += f"  (source: {source})"

            # Truncate very long documents for display
            content_str = content if isinstance(content, str) else str(content)
            if len(content_str) > 2000:
                content_str = content_str[:2000] + "\n... [truncated]"

            lines.append(header)
            lines.append(content_str)

        return "\n".join(lines)
