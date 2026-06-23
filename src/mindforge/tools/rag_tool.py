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
                if get_embedder()._provider == "fallback":
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

        if not qualified:
            return ToolResult(
                success=True,
                output=(
                    f"## {query}\n\n"
                    f"> ⚠️ 当前资料库中暂无与「{query}」高度相关的内容。\n\n"
                    "**建议：**\n"
                    "- 尝试使用不同的关键词重新搜索\n"
                    "- 上传更多相关文档到知识库（支持 PDF/DOCX/Markdown）\n"
                    "- 检索到 0 条高质量结果，请提供更多资料"
                ),
                data={"results": [], "total": 0, "filtered_out": len(results)},
                execution_time_ms=elapsed,
            )

        formatted = self._format_results(qualified, query)
        return ToolResult(
            success=True,
            output=formatted,
            data={"results": qualified, "total": len(qualified)},
            execution_time_ms=elapsed,
        )

    def _format_results(self, results: list[Any], query: str) -> str:
        """Format retrieved documents into a clean, readable report."""
        lines: list[str] = [f"## {query}\n"]

        # Estimate overall relevance
        avg_score = 0.0
        count = 0
        for doc in results:
            s = 0.0
            if hasattr(doc, "score"):
                s = doc.score
            elif isinstance(doc, dict):
                s = doc.get("score", 0.0)
            avg_score += s
            count += 1
        avg_score = avg_score / count if count else 0
        if avg_score < 0.2:
            lines.append(f"> 📄 基于现有资料整理（平均相关度 {avg_score:.3f}，仅供参考）\n")

        for i, doc in enumerate(results, 1):
            # Extract content
            if hasattr(doc, "page_content"):
                content = doc.page_content
            elif isinstance(doc, dict):
                content = doc.get("content", doc.get("text", str(doc)))
            else:
                content = str(doc)

            # Extract source
            source = ""
            if isinstance(doc, dict):
                meta = doc.get("metadata", {}) or {}
                source = meta.get("source", meta.get("title", ""))
            elif hasattr(doc, "metadata") and doc.metadata:
                source = doc.metadata.get("source", "")

            # Clean up content — remove code fences, strip leading/trailing garbage
            content_str = str(content).strip()
            # Remove leading Markdown heading fragments like "中 |" or "中 |\n\n"
            import re
            content_str = re.sub(r'^[^\n]{0,3}\s*\|\s*\n+', '', content_str)
            # Collapse 3+ newlines into 2
            content_str = re.sub(r'\n{3,}', '\n\n', content_str)
            # Truncate
            if len(content_str) > 2000:
                content_str = content_str[:2000] + "\n\n…（内容过长，已截断）"

            # Build result block
            lines.append(f"### 📌 来源 {i}")
            if source:
                lines[-1] += f" — *{source}*"
            lines.append("")
            lines.append(content_str)
            lines.append("")

        lines.append("---")
        lines.append(f"*共检索到 {len(results)} 条相关结果*")
        return "\n".join(lines)
