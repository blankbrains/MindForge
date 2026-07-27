"""MindForgeMCPServer — exposes Agent capabilities as MCP tools over stdio transport."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time as _time
from typing import Any, Optional

try:
    from mindforge.tools.rag_tool import RAGTool
    from mindforge.tools.citation_verifier import CitationVerifier
    from mindforge.tools.web_search import WebSearchTool
except ImportError:
    RAGTool = None  # type: ignore[assignment]
    CitationVerifier = None  # type: ignore[assignment]
    WebSearchTool = None  # type: ignore[assignment]

try:
    from mindforge.agents.orchestrator import Orchestrator as ResearchAgent
except ImportError:
    ResearchAgent = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JSON_RPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2025-03-26"
DEFAULT_TOOL_TIMEOUT = 120  # seconds
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_knowledge_base",
        "description": "Search the internal knowledge base for documents and information.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["semantic", "hybrid", "keyword"],
                    "description": "Retrieval mode.",
                    "default": "hybrid",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return.",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_research_task",
        "description": "Execute a multi-step research task: search, analyze, and synthesize findings into a report.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The research topic or question.",
                },
                "depth": {
                    "type": "string",
                    "enum": ["quick", "standard", "deep"],
                    "description": "Research depth.",
                    "default": "standard",
                },
                "max_sources": {
                    "type": "integer",
                    "description": "Maximum number of sources to gather.",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "verify_citation",
        "description": "Verify citation markers [N] in a report against the provided source list.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "report_text": {
                    "type": "string",
                    "description": "Report text containing [N] markers.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of source objects with 'index' field.",
                },
            },
            "required": ["report_text", "sources"],
        },
    },
    {
        "name": "system_status",
        "description": "Get system status information including available tools, memory usage, and uptime.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_memory": {
                    "type": "boolean",
                    "description": "Include memory usage info.",
                    "default": False,
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class MindForgeMCPServer:
    """MCP server that exposes MindForge Agent capabilities as MCP tools.

    Supports stdio transport — reads JSON-RPC requests from stdin and writes
    responses to stdout. Designed to be launched as a subprocess by an
    MCP host (Claude, VS Code, etc.).
    """

    def __init__(
        self,
        rag_tool: Optional[Any] = None,
        web_search_tool: Optional[Any] = None,
        citation_verifier: Optional[Any] = None,
        research_agent: Optional[Any] = None,
    ) -> None:
        self._rag_tool = rag_tool
        self._web_search_tool = web_search_tool
        self._citation_verifier = citation_verifier
        self._research_agent = research_agent

        self._initialized = False
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_stdio(self) -> None:
        """Run the server in stdio mode — read from stdin, write to stdout.

        This is the main entry point when the server is launched as a
        subprocess (e.g. in mcp.json).
        """
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        # Write responses to stdout (must be unbuffered)
        writer_transport, writer_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin,  # type: ignore[arg-type]
            sys.stdout,
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, loop)

        # stderr is reserved for logging
        self._start_time = _time.time()

        while True:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=None)
            except (asyncio.CancelledError, EOFError):
                break

            if not line:
                break  # EOF

            raw = line.decode("utf-8").strip()
            if not raw:
                continue

            response = await self._handle_line(raw)
            response_line = json.dumps(response) + "\n"
            writer.write(response_line.encode("utf-8"))
            await writer.drain()

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a single JSON-RPC request and return a response.

        This is the main request handler, callable both from stdio mode
        and from programmatic usage.
        """
        method = request.get("method", "")
        req_id = request.get("id", None)
        params = request.get("params", {})

        try:
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "shutdown":
                result = await self._handle_shutdown()
            elif method == "tools/list":
                result = await self._handle_tools_list()
            elif method == "tools/call":
                result = await self._handle_tools_call(params)
            else:
                return self._error(req_id, -32601, f"Method not found: {method}")

            return self._success(req_id, result)
        except Exception:
            logger.exception("MCP request failed: method=%s", method)
            return self._error(
                req_id, -32603, "Internal MCP service error.",
            )

    # ------------------------------------------------------------------
    # JSON-RPC handlers
    # ------------------------------------------------------------------

    async def _handle_line(self, line: str) -> dict[str, Any]:
        """Deserialize a JSON line and dispatch to handle_request."""
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            return self._error(None, -32700, f"Parse error: {exc}")

        return await self.handle_request(request)

    async def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle the initialize handshake."""
        self._initialized = True
        if self._start_time == 0:
            self._start_time = _time.time()

        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": "mindforge-mcp-server",
                "version": "0.1.0",
            },
        }

    async def _handle_shutdown(self) -> dict[str, Any]:
        """Handle shutdown request."""
        self._initialized = False
        return {"status": "shutting_down"}

    async def _handle_tools_list(self) -> dict[str, Any]:
        """Return the list of available tools."""
        return {"tools": _TOOL_DEFINITIONS}

    async def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call."""
        if not isinstance(params, dict):
            raise ValueError("tools/call params must be an object.")
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object.")

        handlers: dict[str, Any] = {
            "search_knowledge_base": self._exec_search_knowledge_base,
            "run_research_task": self._exec_run_research_task,
            "verify_citation": self._exec_verify_citation,
            "system_status": self._exec_system_status,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            raise ValueError(
                f"Unknown tool: {tool_name}. "
                f"Available: {list(handlers.keys())}"
            )

        result = await handler(**arguments)
        return {"content": [{"type": "text", "text": result}]}

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _exec_search_knowledge_base(
        self, query: str, mode: str = "hybrid", top_k: int = 5, **kwargs: Any
    ) -> str:
        """Search the internal knowledge base."""
        from mindforge.config import get_settings

        max_top_k = get_settings().retrieval.max_request_top_k
        if not isinstance(query, str) or not query.strip():
            return "Search failed: query must be a non-empty string."
        if len(query) > 20_000:
            return "Search failed: query exceeds 20000 characters."
        if mode not in {"semantic", "hybrid", "keyword"}:
            return "Search failed: unsupported retrieval mode."
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= max_top_k
        ):
            return f"Search failed: top_k must be between 1 and {max_top_k}."
        if self._rag_tool is None:
            if RAGTool is not None:
                self._rag_tool = RAGTool()
            else:
                return "RAG tool is not available."

        result = self._rag_tool.safe_execute(
            query=query, mode=mode, top_k=top_k
        )
        if result.success:
            return result.output
        return f"Search failed: {result.error}"

    async def _exec_run_research_task(
        self,
        topic: str,
        depth: str = "standard",
        max_sources: int = 10,
        **kwargs: Any,
    ) -> str:
        """Execute a multi-step research task."""
        if not isinstance(topic, str) or not topic.strip():
            return "Research task failed: topic must be a non-empty string."
        if len(topic) > 20_000:
            return "Research task failed: topic exceeds 20000 characters."
        if depth not in {"quick", "standard", "deep"}:
            return "Research task failed: unsupported depth."
        if (
            isinstance(max_sources, bool)
            or not isinstance(max_sources, int)
            or not 1 <= max_sources <= 50
        ):
            return "Research task failed: max_sources must be between 1 and 50."
        if self._research_agent is None and ResearchAgent is not None:
            try:
                self._research_agent = ResearchAgent()
            except Exception:
                logger.exception("Could not initialize MCP research agent.")
                return "Could not initialize research agent."

        if self._research_agent is None:
            # Fallback: use web search and compile report
            return await self._fallback_research(topic, depth, max_sources)

        try:
            # ResearcherAgent.run 签名: run(task, *, context, max_rounds)
            task_desc = (
                f"Research topic: {topic}"
                + (f" (depth: {depth})" if depth else "")
                + (f" (max sources: {max_sources})" if max_sources else "")
            )
            report = await self._research_agent.run(task_desc)
            if isinstance(report, str):
                return report
            return str(report)
        except Exception:
            logger.exception("MCP research task failed.")
            return "Research task failed."

    async def _fallback_research(
        self, topic: str, depth: str, max_sources: int
    ) -> str:
        """Minimal research fallback when ResearchAgent is unavailable."""
        if self._web_search_tool is None:
            if WebSearchTool is not None:
                self._web_search_tool = WebSearchTool()
            else:
                return "Neither research agent nor web search is available."

        max_results = max_sources if depth == "deep" else (5 if depth == "standard" else 3)
        result = self._web_search_tool.safe_execute(
            query=topic, max_results=max_results
        )
        if result.success:
            return result.output
        return f"Research fallback failed: {result.error}"

    async def _exec_verify_citation(
        self, report_text: str, sources: list[dict[str, Any]], **kwargs: Any
    ) -> str:
        """Verify citation markers against a source list."""
        if not isinstance(report_text, str) or len(report_text) > 2_000_000:
            return "Verification failed: invalid report_text."
        if not isinstance(sources, list) or len(sources) > 1000:
            return "Verification failed: invalid sources."
        if self._citation_verifier is None:
            if CitationVerifier is not None:
                self._citation_verifier = CitationVerifier()
            else:
                return "Citation verifier is not available."

        result = self._citation_verifier.safe_execute(
            report_text=report_text, sources=sources
        )
        if result.success:
            return result.output
        return f"Verification failed: {result.error}"

    async def _exec_system_status(self, include_memory: bool = False) -> str:
        """Return system status information."""
        import time as time_module

        uptime = time_module.time() - self._start_time if self._start_time else 0
        uptime_str = (
            f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m "
            f"{int(uptime % 60)}s"
        )

        lines: list[str] = [
            "MindForge MCP Server Status",
            "=" * 50,
            f"Initialized:     {self._initialized}",
            f"Uptime:          {uptime_str}",
            f"Available tools: {len(_TOOL_DEFINITIONS)}",
        ]

        for tdef in _TOOL_DEFINITIONS:
            lines.append(f"  - {tdef['name']}: {tdef['description'][:60]}")

        if include_memory:
            try:
                import psutil  # optional

                process = psutil.Process(os.getpid())
                mem = process.memory_info()
                lines.append("")
                lines.append(f"Memory RSS:      {mem.rss / 1024 / 1024:.1f} MB")
                lines.append(f"Memory VMS:      {mem.vms / 1024 / 1024:.1f} MB")
            except ImportError:
                lines.append("")
                lines.append("Memory info: psutil not installed.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON-RPC response builders
    # ------------------------------------------------------------------

    def _success(self, req_id: Any, result: Any) -> dict[str, Any]:
        return {
            "jsonrpc": JSON_RPC_VERSION,
            "id": req_id,
            "result": result,
        }

    def _error(
        self,
        req_id: Any,
        code: int,
        message: str,
        data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {
            "jsonrpc": JSON_RPC_VERSION,
            "id": req_id,
            "error": err,
        }

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    @staticmethod
    def run() -> None:
        """CLI entry point: launch the MCP server in stdio mode."""
        server = MindForgeMCPServer()
        asyncio.run(server.run_stdio())


# ---------------------------------------------------------------------------
# CLI entry point (python -m mindforge.mcp.server)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MindForgeMCPServer.run()


def main() -> None:
    """Console-script entry point for ``mindforge-mcp-server``."""
    MindForgeMCPServer.run()
