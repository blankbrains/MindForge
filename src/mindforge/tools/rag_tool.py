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
        """Lazy-init AdaptiveRetriever with full dependency wiring."""
        if self._retriever is not None:
            return self._retriever
        if AdaptiveRetriever is None:
            raise RuntimeError(
                "AdaptiveRetriever is not available. "
                "Install mindforge with retrieval extras or provide a retriever instance."
            )

        from mindforge.retrieval.vector_store import get_vector_store
        from mindforge.retrieval.hybrid import HybridRetriever
        from mindforge.ingestion.embedder import get_embedder

        store = get_vector_store()
        store.ensure_collection()
        embedder = get_embedder()

        async def _async_embed(text: str):
            return embedder.embed_single(text)

        hybrid = HybridRetriever(
            vector_store=store,
            bm25_retriever=None,
            embedding_fn=_async_embed,
        )
        # Skip reranker to avoid HuggingFace download in offline environments
        self._retriever = AdaptiveRetriever(
            hybrid_retriever=hybrid,
            reranker=None,
            **self._retriever_kwargs,
        )
        return self._retriever

    def execute(self, query: str, mode: str = "hybrid", top_k: int = 5, threshold: float = 0.0, **kwargs: Any) -> ToolResult:
        """Synchronous wrapper — uses thread pool when event loop is running."""
        import asyncio
        try:
            asyncio.get_running_loop()  # probe: raises RuntimeError if no loop
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
            return ToolResult(success=False, error="请输入搜索内容。")

        retriever = self._get_retriever()

        mode_map = {
            "semantic": QueryMode.CONCEPTUAL,
            "hybrid": QueryMode.FACTUAL,
            "keyword": QueryMode.PROCEDURAL,
        }
        qmode = mode_map.get(mode, QueryMode.FACTUAL)

        try:
            result_dict = await retriever.retrieve(query=query, mode=qmode, top_k=top_k)
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"检索失败: {exc}",
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        elapsed = (time.perf_counter() - start) * 1000
        results = result_dict.get("results", []) if isinstance(result_dict, dict) else result_dict

        # ── Score filtering ──
        # Real embeddings (BGE-M3): relevant > 0.3, noise < 0.1
        # Hash fallback: all scores ~0.01, use minimal threshold
        if threshold > 0:
            min_score = threshold
        else:
            try:
                from mindforge.ingestion.embedder import get_embedder
                if get_embedder().provider == "fallback":
                    min_score = 0.005   # hash: accept everything
                else:
                    min_score = 0.15    # real embeddings: filter noise
            except Exception:
                min_score = 0.01
        qualified = []
        for r in results:
            s = 0.0
            if hasattr(r, "score"):
                s = r.score
            elif isinstance(r, dict):
                s = r.get("score", 0.0)
            if s >= min_score:
                qualified.append(r)

        # Quality: average of all result scores, scaled to 0-10.
        # RRF fusion blends rank signal (0.6) with raw cosine similarity (0.4),
        # so relevant results score ~0.3-0.6 and noise scores ~0.01.
        total_score = sum(
            r.get("score", 0.0) if isinstance(r, dict) else getattr(r, "score", 0.0)
            for r in qualified
        )
        avg = total_score / max(len(qualified), 1)
        # Scale: 0.5 avg → 8.0, 0.3 → 5.0, 0.1 → 2.0
        quality = round(min(avg * 16, 10.0), 1)

        if not qualified:
            return ToolResult(
                success=True,
                output=(
                    f"关于「{query}」，当前知识库中暂无高度相关的资料。\n\n"
                    "建议尝试更换关键词，或上传更多相关文档到知识库。"
                ),
                data={"results": [], "total": 0, "quality": 0.0, "filtered_out": len(results)},
                execution_time_ms=elapsed,
            )

        formatted = self._format_results(qualified, query)
        return ToolResult(
            success=True,
            output=formatted,
            data={"results": qualified, "total": len(qualified), "quality": quality},
            execution_time_ms=elapsed,
        )

    def _format_results(self, results: list[Any], query: str) -> str:
        """Format results as clean Markdown ready for ReactMarkdown rendering.

        Preserves headers, bold, lists. Removes garbage artifacts.
        Adds proper spacing between sections for readability.
        """
        import re as _re

        lines: list[str] = [f"## {query}\n"]

        for doc in results:
            if hasattr(doc, "page_content"):
                content = doc.page_content
            elif isinstance(doc, dict):
                content = doc.get("content", doc.get("text", str(doc)))
            else:
                content = str(doc)

            text = str(content).strip()

            # 清理不必要的空白/噪声，保留正常文档内容
            # 注意：不再删除 `|` 字符（避免破坏 Markdown 表格）
            # 注意：不再删除 `class Foo:` 类声明（避免破坏代码/文档内容）
            text = _re.sub(r'_{3,}', '', text)
            # Collapse whitespace but keep paragraph structure
            text = _re.sub(r'\n{4,}', '\n\n\n', text)
            text = text.strip()

            # Truncate at 20000 chars, but prefer sentence boundaries
            if len(text) > 20000:
                truncated = text[:20000]
                last_period = max(truncated.rfind('。'), truncated.rfind('. '), truncated.rfind('\n\n'))
                if last_period > 10000:
                    text = truncated[:last_period+1] + "\n\n…"
                else:
                    text = truncated + "\n\n…"

            if text:
                lines.append(text)
                lines.append("")

        if len(lines) <= 2:
            return f"## {query}\n\n当前知识库中暂无高度相关的资料。\n\n建议更换关键词或上传更多文档到知识库。"

        # Join with double newlines for clear paragraph separation
        return "\n\n".join(lines)
