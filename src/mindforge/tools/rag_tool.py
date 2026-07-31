"""RAG tool that queries the MindForge knowledge base via AdaptiveRetriever."""

from __future__ import annotations

import html
import json
import re
import time
from typing import Any, Optional

from mindforge.tools.base import BaseTool, ToolResult

try:
    from mindforge.retrieval.adaptive import AdaptiveRetriever, QueryMode
except ImportError:
    AdaptiveRetriever = None  # type: ignore[assignment]


class RAGTool(BaseTool):
    """Tool that queries the knowledge base using AdaptiveRetriever.

    Supports adaptive, graph, semantic, hybrid, and keyword retrieval and
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
                "enum": [
                    "auto",
                    "graph",
                    "semantic",
                    "hybrid",
                    "keyword",
                ],
                "description": "Retrieval mode to use.",
                "default": "auto",
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
    _CONVERSATIONAL_QUERIES = frozenset(
        {
            "hi",
            "hello",
            "hey",
            "你好",
            "你好啊",
            "你好呀",
            "您好",
            "嗨",
            "在吗",
            "早上好",
            "上午好",
            "下午好",
            "晚上好",
            "谢谢",
            "多谢",
            "再见",
            "你是谁",
            "你叫什么",
        }
    )
    _QUERY_STOPWORDS = frozenset(
        {
            "and",
            "or",
            "the",
            "is",
            "are",
            "what",
            "how",
            "why",
            "vs",
            "versus",
            "和",
            "与",
            "及",
            "或",
            "的",
            "了",
            "是",
            "什么",
            "哪些",
            "怎么",
            "如何",
            "为什么",
            "有什么",
            "区别",
            "差异",
            "比较",
            "对比",
            "请",
            "一下",
        }
    )

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

        from mindforge.retrieval.service import get_retriever

        self._retriever = get_retriever()
        return self._retriever

    def execute(self, query: str, mode: str = "auto", top_k: int | None = None, threshold: float = 0.0, **kwargs: Any) -> ToolResult:
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
        mode: str = "auto",
        top_k: int | None = None,
        threshold: float = 0.0,
        **kwargs: Any,
    ) -> ToolResult:
        start = time.perf_counter()
        from mindforge.config import get_settings

        settings = get_settings()

        if not query or not query.strip():
            return ToolResult(success=False, error="请输入搜索内容。")
        if len(query) > 20_000:
            return ToolResult(
                success=False,
                error="搜索内容不能超过 20000 个字符。",
            )
        if mode not in {
            "auto",
            "graph",
            "semantic",
            "hybrid",
            "keyword",
        }:
            return ToolResult(success=False, error="不支持的检索模式。")
        if top_k is None:
            top_k = settings.retrieval.rerank_top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            return ToolResult(success=False, error="top_k 必须是整数。")
        if not 1 <= top_k <= settings.retrieval.max_request_top_k:
            return ToolResult(
                success=False,
                error=(
                    "top_k 必须在 1 到 "
                    f"{settings.retrieval.max_request_top_k} 之间。"
                ),
            )
        if isinstance(threshold, bool) or not isinstance(
            threshold,
            (int, float),
        ):
            return ToolResult(success=False, error="threshold 必须是数字。")
        if not 0.0 <= float(threshold) <= 1.0:
            return ToolResult(
                success=False,
                error="threshold 必须在 0 到 1 之间。",
            )
        if self._is_conversational_query(query):
            return ToolResult(
                success=True,
                output=(
                    "你好！当前请求处于知识库检索模式，不会调用大模型进行闲聊。\n\n"
                    "请提出与已上传资料相关的具体问题。"
                ),
                data={
                    "results": [],
                    "sources": [],
                    "total": 0,
                    "retrieval_quality": 0.0,
                    "intent": "conversation",
                },
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        import asyncio

        retriever = await asyncio.to_thread(self._get_retriever)

        mode_map = {
            "graph": QueryMode.GRAPH,
            "semantic": QueryMode.CONCEPTUAL,
            "hybrid": QueryMode.FACTUAL,
            "keyword": QueryMode.PROCEDURAL,
        }
        qmode = mode_map.get(mode)

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

        # RRF scores describe rank consensus, not semantic relevance. Because
        # they are normalized, the first result is always near 1.0 even for an
        # unrelated query. Gate on real reranker/vector scores or positive
        # keyword evidence instead.
        min_score = (
            float(threshold)
            if threshold > 0
            else float(settings.retrieval.min_score)
        )
        keyword_min_coverage = float(
            getattr(settings.retrieval, "keyword_min_coverage", 0.60)
        )
        qualified = [
            result
            for result in results
            if self._is_relevant_result(
                result,
                min_score,
                query=query,
                keyword_min_coverage=keyword_min_coverage,
            )
        ]

        total_score = sum(
            max(
                self._relevance_score(result),
                min_score * self._term_coverage(query, result)
                if self._has_keyword_evidence(result)
                else 0.0,
            )
            for result in qualified
        )
        avg = total_score / max(len(qualified), 1)
        retrieval_quality = round(min(avg * 10, 10.0), 1)

        if not qualified:
            return ToolResult(
                success=True,
                output=(
                    f"关于「{query}」，当前知识库中暂无高度相关的资料。\n\n"
                    "建议尝试更换关键词，或上传更多相关文档到知识库。"
                ),
                data={
                    "results": [],
                    "sources": [],
                    "total": 0,
                    "retrieval_quality": 0.0,
                    "filtered_out": len(results),
                },
                execution_time_ms=elapsed,
            )

        formatted = self._format_results(qualified, query)
        sources = self._build_sources(qualified)
        return ToolResult(
            success=True,
            output=formatted,
            data={
                "results": qualified,
                "sources": sources,
                "total": len(qualified),
                "retrieval_quality": retrieval_quality,
            },
            execution_time_ms=elapsed,
        )

    @staticmethod
    def _build_sources(results: list[Any]) -> list[dict[str, Any]]:
        """Convert retrieval hits into stable citation source records."""
        sources: list[dict[str, Any]] = []
        seen: set[str] = set()
        for result in results:
            if not isinstance(result, dict):
                continue
            identity = str(
                result.get("url")
                or result.get("chunk_id")
                or result.get("id")
                or ""
            )
            if not identity or identity in seen:
                continue
            seen.add(identity)
            sources.append(
                {
                    "index": len(sources) + 1,
                    "id": result.get("id", identity),
                    "chunk_id": result.get(
                        "chunk_id",
                        result.get("id", identity),
                    ),
                    "doc_id": result.get("doc_id"),
                    "title": result.get(
                        "title",
                        result.get(
                            "document_source",
                            result.get("source", "知识库文档"),
                        ),
                    ),
                    "source": result.get(
                        "document_source",
                        result.get("source", "knowledge_base"),
                    ),
                    "url": result.get("url", ""),
                    "content": result.get(
                        "content",
                        result.get("text", ""),
                    ),
                    "score": RAGTool._relevance_score(result),
                    "metadata": dict(result.get("metadata") or {}),
                }
            )
        return sources

    def _format_results(self, results: list[Any], query: str) -> str:
        """Format raw retrieval evidence without pretending it is an answer."""
        lines: list[str] = [
            "## 知识库检索结果",
            "",
            (
                "> 当前未使用大模型。以下内容是知识库中的原始命中片段，"
                "未经总结或改写。"
            ),
        ]

        for index, doc in enumerate(results, start=1):
            if hasattr(doc, "page_content"):
                content = doc.page_content
                metadata: dict[str, Any] = {}
                title = "知识库文档"
            elif isinstance(doc, dict):
                content = doc.get("content", doc.get("text", str(doc)))
                metadata = dict(doc.get("metadata") or {})
                title = str(
                    doc.get("document_source")
                    or doc.get("title")
                    or "知识库文档"
                )
            else:
                content = str(doc)
                metadata = {}
                title = "知识库文档"

            text = str(content).strip()
            text = re.sub(r"\n{4,}", "\n\n\n", text)
            text = text.strip()

            if len(text) > 20000:
                truncated = text[:20000]
                last_period = max(
                    truncated.rfind("。"),
                    truncated.rfind(". "),
                    truncated.rfind("\n\n"),
                )
                if last_period > 10000:
                    text = truncated[:last_period + 1] + "\n\n…"
                else:
                    text = truncated + "\n\n…"

            if text:
                page = metadata.get("page")
                page_label = (
                    f" · 第 {int(page)} 页"
                    if isinstance(page, (int, float)) and int(page) > 0
                    else ""
                )
                safe_title = title.replace("[", r"\[").replace("]", r"\]")
                lines.extend(
                    [
                        "",
                        f"### {index}. {safe_title}{page_label}",
                        "",
                    ]
                )
                if self._is_explicit_fenced_code(text):
                    lines.append(text)
                elif (language := self._detect_code_language(text)) is not None:
                    lines.append(self._fenced_code(text, language))
                else:
                    escaped = html.escape(text, quote=False)
                    lines.extend(
                        f"> {line}" if line else ">"
                        for line in escaped.splitlines()
                    )
                lines.append("")

        if len(lines) <= 3:
            return (
                f"关于「{query}」，当前知识库中暂无高度相关的资料。\n\n"
                "建议更换关键词，或上传更多相关文档到知识库。"
            )

        return "\n".join(lines).strip()

    @classmethod
    def _is_conversational_query(cls, query: str) -> bool:
        normalized = re.sub(r"[\s，。！？!?、,.]+", "", query).casefold()
        return normalized in cls._CONVERSATIONAL_QUERIES

    @staticmethod
    def _has_keyword_evidence(result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        sources = {
            str(source)
            for source in result.get("retrieval_sources", [])
        }
        keyword_score = float(result.get("keyword_score", 0.0) or 0.0)
        return bool(
            sources.intersection({"bm25", "multi_query"})
            and keyword_score > 0.0
        )

    @staticmethod
    def _relevance_score(result: Any) -> float:
        if not isinstance(result, dict):
            return float(getattr(result, "score", 0.0) or 0.0)
        if "rerank_score" in result:
            return float(result.get("rerank_score") or 0.0)
        if "semantic_score" in result:
            return float(result.get("semantic_score") or 0.0)
        if "rrf_score" not in result:
            return float(result.get("score", 0.0) or 0.0)
        return 0.0

    @classmethod
    def _result_text(cls, result: Any) -> str:
        if not isinstance(result, dict):
            return str(
                getattr(result, "page_content", None)
                or getattr(result, "text", None)
                or result
            ).casefold()
        metadata = result.get("metadata") or {}
        values = [
            result.get("text"),
            result.get("content"),
            result.get("page_content"),
            result.get("title"),
            result.get("document_source"),
            result.get("source"),
        ]
        if isinstance(metadata, dict):
            values.extend(
                [
                    metadata.get("title"),
                    metadata.get("filename"),
                    metadata.get("source"),
                ]
            )
        return " ".join(str(value) for value in values if value).casefold()

    @classmethod
    def _explicit_identifiers(cls, query: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_+#.\-]{1,63}", query)
            if token.casefold() not in cls._QUERY_STOPWORDS
        }

    @classmethod
    def _query_terms(cls, query: str) -> set[str]:
        terms = set(cls._explicit_identifiers(query))
        try:
            import jieba

            candidates = jieba.lcut(query, cut_all=False)
        except ImportError:
            candidates = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        for candidate in candidates:
            normalized = re.sub(
                r"[^\w+#.\-]+",
                "",
                str(candidate),
                flags=re.UNICODE,
            ).casefold()
            if (
                len(normalized) >= 2
                and normalized not in cls._QUERY_STOPWORDS
                and not normalized.isdigit()
            ):
                terms.add(normalized)
        return terms

    @classmethod
    def _term_coverage(cls, query: str, result: Any) -> float:
        terms = cls._query_terms(query)
        if not terms:
            return 1.0
        text = cls._result_text(result)
        matched = sum(term in text for term in terms)
        return matched / len(terms)

    @classmethod
    def _covers_explicit_identifiers(cls, query: str, result: Any) -> bool:
        identifiers = cls._explicit_identifiers(query)
        if len(identifiers) < 2:
            return True
        text = cls._result_text(result)
        return all(identifier in text for identifier in identifiers)

    @classmethod
    def _has_minimum_query_evidence(cls, query: str, result: Any) -> bool:
        terms = cls._query_terms(query)
        if len(terms) < 2:
            return True
        return cls._term_coverage(query, result) >= (1.0 / len(terms))

    @classmethod
    def _is_relevant_result(
        cls,
        result: Any,
        min_score: float,
        *,
        query: str = "",
        keyword_min_coverage: float = 0.60,
    ) -> bool:
        if query and not cls._covers_explicit_identifiers(query, result):
            return False
        if cls._relevance_score(result) >= min_score:
            return cls._has_minimum_query_evidence(query, result)
        return bool(
            cls._has_keyword_evidence(result)
            and cls._term_coverage(query, result) >= keyword_min_coverage
        )

    @classmethod
    def _detect_code_language(cls, text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, (dict, list)):
            return "json"
        docker_instructions = re.findall(
            (
                r"(?m)^\s*(?:FROM|RUN|CMD|ENTRYPOINT|COPY|ADD|WORKDIR|"
                r"EXPOSE|ENV|ARG|USER|VOLUME|HEALTHCHECK)\s+"
            ),
            text,
        )
        if len(docker_instructions) >= 2 or re.search(
            (
                r"(?m)^\s*FROM\s+(?:--platform=\S+\s+)?"
                r"[\w./-]+(?::[\w.-]+|@sha256:[a-fA-F0-9]+)"
                r"(?:\s+AS\s+\w+)?\s*$"
            ),
            text,
        ):
            return "dockerfile"
        if re.search(
            (
                r"(?im)(?:^\s*#requires\b|\$env:|\b(?:Get|Set|New|Remove|"
                r"Write|Where|ForEach)-[A-Z][A-Za-z]+\b|^\s*param\s*\()"
            ),
            text,
        ):
            return "powershell"
        if re.search(r"(?i)<\?php\b", text):
            return "php"
        if re.search(
            (
                r"(?i)<!doctype\s+html\b|</?(?:html|head|body|main|section|"
                r"article|div|span|label|input|button|script|style|form)\b"
            ),
            text,
        ):
            return "html"
        if re.search(r"(?i)<\?xml\b", text) or re.search(
            r"(?s)^\s*<[A-Za-z_][\w:.-]*(?:\s[^>]*)?>.*</[A-Za-z_][\w:.-]*>\s*$",
            text,
        ):
            return "xml"
        if re.search(r"(?m)^\s*\$(?:[\w-]+)\s*:", text) or re.search(
            r"(?m)^\s*@(?:mixin|include|extend|function)\b",
            text,
        ):
            return "scss"
        if re.search(
            (
                r"(?is)(?:^|\n)\s*(?:[.#][\w-]+|[a-z][\w-]*|"
                r"@(?:media|supports|keyframes)\b[^{]*)\s*\{[^{}]*"
                r"(?:color|background|display|margin|padding|font|border|"
                r"width|height|position|grid|flex|content|transform|"
                r"animation)\s*:[^{};]+;"
            ),
            text,
        ):
            return "css"
        if re.search(
            (
                r"(?m)^\s*(?:interface|type|enum|namespace)\s+[A-Za-z_$]\w*|"
                r"^\s*import\s+type\b|"
                r"\b(?:const|let|function)\s+[A-Za-z_$]\w*\s*"
                r"(?:\([^)]*\))?\s*:\s*(?:string|number|boolean|unknown|never)\b"
            ),
            text,
        ):
            return "typescript"
        if re.search(
            (
                r"(?m)^\s*(?:from\s+\w[\w.]*\s+import\s+[\w*, ()]+|"
                r"def\s+\w+\s*\(|class\s+\w+(?:\([^)]*\))?\s*:|"
                r"if\s+__name__\s*==|"
                r"[A-Z_][A-Z0-9_]*\s*=\s*[\[{])"
            ),
            text,
        ):
            return "python"
        if re.search(
            (
                r"(?m)^\s*(?:const|let|var|function|export\s+(?:default\s+)?)\b|"
                r"=>|\bconsole\.(?:log|error|warn)\s*\("
            ),
            text,
        ):
            return "javascript"
        if re.search(
            r"(?m)^\s*(?:using\s+System(?:\.[\w.]+)?;|namespace\s+[\w.]+\s*[;{])",
            text,
        ) or "Console.WriteLine(" in text:
            return "csharp"
        if re.search(r"(?m)^\s*import\s+java\.[\w.*]+;", text) or re.search(
            r"\bpublic\s+static\s+void\s+main\s*\(\s*String(?:\[\]|\.\.\.)",
            text,
        ) or "System.out.println(" in text:
            return "java"
        if re.search(r"(?m)^\s*#include\s*[<\"][^>\"]+[>\"]", text) and re.search(
            r"\b(?:std::|cout\s*<<|cin\s*>>|namespace\s+std|template\s*<)",
            text,
        ):
            return "cpp"
        if re.search(r"(?m)^\s*#include\s*[<\"][^>\"]+[>\"]", text) and re.search(
            r"\b(?:printf|scanf|malloc|calloc|free|typedef\s+struct)\s*\(?",
            text,
        ):
            return "c"
        if re.search(r"(?m)^\s*package\s+\w+\s*$", text) and re.search(
            r"(?m)^\s*func\s+(?:\([^)]*\)\s*)?\w+\s*\(",
            text,
        ):
            return "go"
        if re.search(r"(?m)^\s*(?:use\s+std::|fn\s+main\s*\()", text) or re.search(
            r"\b(?:println|format|vec)!\s*\(",
            text,
        ):
            return "rust"
        if re.search(
            r"(?m)^\s*(?:fun\s+main\s*\(|data\s+class\s+\w+|sealed\s+class\s+\w+)",
            text,
        ):
            return "kotlin"
        if re.search(
            r"(?m)^\s*import\s+(?:Foundation|SwiftUI|UIKit)\s*$",
            text,
        ) or re.search(r"(?m)^\s*@main\s+(?:struct|class)\s+\w+", text):
            return "swift"
        if re.search(
            (
                r"(?is)\bSELECT\b.+\bFROM\b|\bINSERT\s+INTO\b|"
                r"\bCREATE\s+(?:OR\s+REPLACE\s+)?TABLE\b|"
                r"\bUPDATE\b.+\bSET\b|\bALTER\s+TABLE\b"
            ),
            text,
        ):
            return "sql"
        if re.search(
            r"(?m)^\s*(?:query|mutation|subscription)\s+\w+.*\{",
            text,
        ) or re.search(r"(?m)^\s*(?:type|input|enum)\s+\w+\s*\{", text):
            return "graphql"
        if re.search(r"(?m)^\s*\[[\w.-]+\]\s*$", text) and re.search(
            r"(?m)^\s*[\w.-]+\s*=\s*(?:[\"'\d[{]|true\b|false\b)",
            text,
        ):
            return "toml"
        yaml_keys = re.findall(r"(?m)^\s*(?:-\s+)?[\w.-]+\s*:\s*(?:.*)$", text)
        if len(yaml_keys) >= 2 and (
            re.search(r"(?m)^\s+-\s+", text)
            or re.search(r"(?m)^\s{2,}[\w.-]+\s*:", text)
            or text.lstrip().startswith("---")
        ):
            return "yaml"
        if re.search(r"(?m)^\s*#!/(?:usr/bin/env\s+)?(?:ba|z|k)?sh\b", text) or re.search(
            (
                r"(?m)^\s*(?:set\s+-[a-zA-Z]+|export\s+\w+=|"
                r"(?:curl|docker|npm|pnpm|yarn|python|pip|git)\s+\S+)"
            ),
            text,
        ):
            return "bash"
        if re.search(r"(?m)^\s*(?:require\s+['\"]|puts\s+|class\s+\w+\s*<)", text):
            return "ruby"
        if re.search(r"(?m)^\s*(?:local\s+)?function\s+\w+\s*\(", text) and re.search(
            r"(?m)^\s*end\s*$",
            text,
        ):
            return "lua"
        if re.search(r"(?m)^\s*\w[\w.]*\s*<-\s*", text) and re.search(
            r"\b(?:library|function|data\.frame|ggplot)\s*\(",
            text,
        ):
            return "r"
        if re.search(r"(?m)^[A-Za-z0-9_.-]+:\s*[^\n]*\n\t+\S", text):
            return "makefile"
        return "text" if cls._looks_like_code(text) else None

    @staticmethod
    def _is_explicit_fenced_code(text: str) -> bool:
        match = re.fullmatch(
            (
                r"\s*(?P<fence>`{3,}|~{3,})[ \t]*"
                r"[A-Za-z0-9_+#.-]+[^\n]*\n"
                r".*\n(?P=fence)[ \t]*\s*"
            ),
            text,
            flags=re.DOTALL,
        )
        return match is not None

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        code_lines = sum(
            1
            for line in lines
            if re.search(
                (
                    r"^\s*(?:if|else|for|while|switch|case|return|class|"
                    r"interface|function|fn|func|def|import|from|include|"
                    r"public|private|protected|const|let|var)\b|"
                    r"[A-Za-z_$]\w*\s*(?:=|:=|=>)\s*\S|"
                    r"[{}();]\s*$"
                ),
                line,
            )
        )
        return code_lines >= 2 or (
            code_lines >= 1
            and sum(text.count(symbol) for symbol in ("{", "}", ";")) >= 3
        )

    @staticmethod
    def _fenced_code(text: str, language: str) -> str:
        fence = "```"
        while fence in text:
            fence += "`"
        return f"{fence}{language}\n{text}\n{fence}"
