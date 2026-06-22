"""Adapter that wraps MCPClient to expose external MCP tools as BaseTool instances."""

from __future__ import annotations

import time
from typing import Any, Optional

from mindforge.tools.base import BaseTool, ToolResult

try:
    from mindforge.mcp.client import MCPClient
except ImportError:
    MCPClient = None  # type: ignore[assignment]


class MCPToolAdapter(BaseTool):
    """Adapter that wraps MCPClient and exposes MCP server tools as BaseTool.

    Dynamically discovers tools from registered MCP servers and converts them
    to OpenAI function-calling format. Each call is dispatched to the
    appropriate external MCP server via async execution.
    """

    name = "mcp_tool"
    description = (
        "Execute a tool exposed by an external MCP (Model Context Protocol) "
        "server. Use this to access external capabilities such as database "
        "queries, API calls, file operations, or custom services."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "The name of the MCP tool to call (e.g., 'read_file', 'query_database').",
            },
            "params": {
                "type": "object",
                "description": "Parameters to pass to the MCP tool.",
                "default": {},
            },
            "server_name": {
                "type": "string",
                "description": "Optional: specific MCP server to route the call to.",
            },
        },
        "required": ["tool_name"],
    }

    def __init__(self, mcp_client: Optional[Any] = None) -> None:
        super().__init__()
        self._mcp_client = mcp_client
        self._auto_init = mcp_client is None

    async def _ensure_client(self) -> Any:
        """Lazy-initialize MCPClient if not provided."""
        if self._mcp_client is not None:
            return self._mcp_client
        if MCPClient is not None:
            self._mcp_client = MCPClient()
            await self._mcp_client.initialize()
            return self._mcp_client
        raise RuntimeError(
            "MCPClient is not available. Install mindforge with MCP support "
            "or provide an MCPClient instance."
        )

    async def execute_async(self, **kwargs: Any) -> ToolResult:
        """Async execution against the MCP server."""
        start = time.perf_counter()

        tool_name = kwargs.pop("tool_name", None)
        if not tool_name:
            return ToolResult(
                success=False,
                error="tool_name is required.",
            )

        params = kwargs.pop("params", {}) or {}
        server_name = kwargs.pop("server_name", None)

        client = await self._ensure_client()

        try:
            if server_name:
                result_data = await client.call_tool(server_name, tool_name, params)
            else:
                result_data = await client.call_by_function_name(tool_name, params)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=f"MCP call failed: {exc}",
                execution_time_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000

        output_str = self._format_result(result_data)
        return ToolResult(
            success=True,
            output=output_str,
            data={"raw_result": result_data},
            execution_time_ms=elapsed,
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """Synchronous bridge — uses thread pool when event loop is running.

        Delegates to execute_async() via appropriate async driver.
        """
        import asyncio
        try:
            asyncio.get_running_loop()  # probe: raises RuntimeError if no loop
        except RuntimeError:
            return asyncio.run(self.execute_async(**kwargs))
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.execute_async(**kwargs)
                )
                return future.result()

    def _format_result(self, result: Any) -> str:
        """Format MCP tool result for display."""
        if isinstance(result, str):
            return result
        if isinstance(result, (list, tuple)):
            parts: list[str] = []
            for i, item in enumerate(result):
                if isinstance(item, dict):
                    parts.append(f"--- Result {i + 1} ---")
                    for k, v in item.items():
                        parts.append(f"  {k}: {v}")
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        if isinstance(result, dict):
            lines: list[str] = []
            for k, v in result.items():
                lines.append(f"{k}: {v}")
            return "\n".join(lines)
        return str(result)

    async def discover_openai_tools(self) -> list[dict[str, Any]]:
        """Discover all MCP tools and return them in OpenAI function format."""
        client = await self._ensure_client()
        return client.get_openai_tools()

    async def get_tool_descriptions(self) -> str:
        """Get a human-readable description of all available MCP tools."""
        client = await self._ensure_client()
        return client.get_tool_descriptions()
