"""MCPClient — high-level wrapper around MCPRegistry for agent use."""

from __future__ import annotations

from typing import Any, Optional

from mindforge.mcp.registry import (
    MCPRegistry,
    MCPToolNotFoundError,
    get_mcp_registry,
)


class MCPClient:
    """High-level client that wraps MCPRegistry for use by agents and tools.

    Provides convenience methods for initialization, tool calling by name
    or by OpenAI function name, and tool discovery in both raw and
    OpenAI-compatible formats.
    """

    def __init__(
        self,
        registry: Optional[MCPRegistry] = None,
        config_path: str = "mcp.json",
        auto_initialize: bool = False,
    ) -> None:
        self._registry = registry
        self._config_path = config_path
        self._initialized = False

        if auto_initialize:
            import asyncio

            try:
                asyncio.run(self.initialize())
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Start all MCP servers and discover their tools."""
        if self._initialized:
            return

        if self._registry is None:
            self._registry = get_mcp_registry(self._config_path)

        if not self._registry.servers:
            # Config was loaded by get_mcp_registry; if no servers, load again
            try:
                self._registry.load_config(self._config_path)
            except FileNotFoundError:
                pass  # No MCP config — client works in degraded mode

        await self._registry.start_all()
        await self._registry.discover_all_tools()
        self._initialized = True

    async def shutdown(self) -> None:
        """Stop all MCP servers."""
        if self._registry is not None:
            await self._registry.stop_all()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # Tool calling
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Call an MCP tool on a specific server by name."""
        if not self._initialized:
            raise RuntimeError("MCPClient not initialized. Call initialize() first.")

        if self._registry is None:
            raise RuntimeError("Registry not available.")

        result = await self._registry.call_tool_on_server(
            server_name, tool_name, arguments or {}
        )
        return result

    async def call_by_function_name(
        self,
        function_name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Call a tool matching an OpenAI function-calling name.

        This routes to the correct server automatically by looking up the
        function name in the global tool index.
        """
        if not self._initialized:
            raise RuntimeError("MCPClient not initialized. Call initialize() first.")

        if self._registry is None:
            raise RuntimeError("Registry not available.")

        try:
            return await self._registry.call_tool(function_name, arguments or {})
        except MCPToolNotFoundError:
            # Try with hyphens replaced by underscores (OpenAI names often
            # use underscores while MCP names may use hyphens, or vice versa)
            alt_name = function_name.replace("_", "-")
            try:
                return await self._registry.call_tool(alt_name, arguments or {})
            except MCPToolNotFoundError:
                alt_name2 = function_name.replace("-", "_")
                return await self._registry.call_tool(alt_name2, arguments or {})

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Return discovered tools in OpenAI function-calling format."""
        if not self._initialized:
            return []
        if self._registry is None:
            return []
        return self._registry.get_openai_tools()

    def get_tool_descriptions(self) -> str:
        """Return a human-readable summary of all available MCP tools."""
        if not self._initialized or self._registry is None:
            return "No MCP tools available (client not initialized)."

        tools = self._registry.tool_definitions
        if not tools:
            return "No MCP tools discovered."

        lines: list[str] = [
            "Available MCP Tools",
            "=" * 60,
        ]
        for tool in tools:
            lines.append(f"\n  {tool.name}")
            lines.append(f"    Server:     {tool.server_name}")
            lines.append(f"    Description: {tool.description or '(no description)'}")
            params = tool.input_schema.get("properties", {})
            required = tool.input_schema.get("required", [])
            if params:
                lines.append("    Parameters:")
                for p_name, p_schema in params.items():
                    req = " (required)" if p_name in required else ""
                    p_type = p_schema.get("type", "any")
                    p_desc = p_schema.get("description", "")
                    lines.append(f"      - {p_name}: {p_type}{req}")
                    if p_desc:
                        lines.append(f"        {p_desc}")

        return "\n".join(lines)

    def get_tool_summaries(self) -> list[dict[str, str]]:
        """Return concise tool summaries for fast agent lookup."""
        if not self._initialized or self._registry is None:
            return []

        return [
            {
                "name": t.name,
                "server": t.server_name,
                "description": t.description[:100] if t.description else "",
            }
            for t in self._registry.tool_definitions
        ]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_tools = 0
        if self._initialized and self._registry is not None:
            n_tools = len(self._registry.tool_definitions)
        return f"<MCPClient initialized={self._initialized} tools={n_tools}>"
