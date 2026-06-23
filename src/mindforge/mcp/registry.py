"""MCP Registry — manages MCP server lifecycle and tool discovery via JSON-RPC over stdio."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MCP_CONFIG_PATH = "mcp.json"
MCP_PROTOCOL_VERSION = "2025-03-26"
JSON_RPC_VERSION = "2.0"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MCPError(Exception):
    """Base exception for MCP-related errors."""


class MCPConnectionError(MCPError):
    """Raised when connecting to an MCP server fails."""


class MCPToolNotFoundError(MCPError):
    """Raised when a requested tool is not found on any server."""


class MCPTimeoutError(MCPError):
    """Raised when an MCP request times out."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server from mcp.json."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"
    enabled: bool = True


@dataclass
class MCPToolDefinition:
    """A tool exposed by an MCP server."""

    server_name: str
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JSON-RPC message helpers
# ---------------------------------------------------------------------------


def _make_request(method: str, params: Optional[dict[str, Any]] = None) -> str:
    """Build a JSON-RPC request string."""
    request = {
        "jsonrpc": JSON_RPC_VERSION,
        "id": str(id(params) if params else 0),
        "method": method,
        "params": params or {},
    }
    return json.dumps(request) + "\n"


def _parse_response(line: str) -> dict[str, Any]:
    """Parse a JSON-RPC response line."""
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError as exc:
        raise MCPError(f"Invalid JSON-RPC response: {exc}") from exc


# ---------------------------------------------------------------------------
# Server process wrapper
# ---------------------------------------------------------------------------


class MCPServerProcess:
    """Manages a single MCP server subprocess over stdio."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._tools: list[MCPToolDefinition] = []
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        """Launch the subprocess."""
        if self.is_running:
            return

        env = os.environ.copy()
        env.update(self.config.env)

        try:
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError:
            raise MCPConnectionError(
                f"MCP server command not found: {self.config.command}"
            )

        await self._initialize()

    async def _initialize(self) -> None:
        """Send initialize request and await the response."""
        init_params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "mindforge-mcp-registry",
                "version": "0.1.0",
            },
        }
        resp = await self._send_request("initialize", init_params)
        _ = resp  # server capabilities, not stored currently

    async def discover_tools(self) -> list[MCPToolDefinition]:
        """Send tools/list and return discovered tool definitions."""
        resp = await self._send_request("tools/list", {})

        raw_tools = resp.get("tools", resp.get("result", {}).get("tools", []))
        self._tools = [
            MCPToolDefinition(
                server_name=self.config.name,
                name=t.get("name", "unknown"),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", t.get("parameters", {})),
            )
            for t in raw_tools
        ]
        return self._tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool on this server."""
        return await self._send_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )

    async def _send_request(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        if not self.is_running:
            raise MCPConnectionError(
                f"MCP server '{self.config.name}' is not running."
            )

        async with self._lock:
            request_line = _make_request(method, params)

            if self._process is None or self._process.stdin is None:
                raise MCPConnectionError("stdin is None")

            self._process.stdin.write(request_line.encode("utf-8"))
            await self._process.stdin.drain()

            if self._process.stdout is None:
                raise MCPConnectionError("stdout is None")

            # Read lines in a loop: skip non-JSON and notifications,
            # return the first valid JSON-RPC response (with "id").
            deadline = asyncio.get_event_loop().time() + 30.0
            last_error: str | None = None

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    if last_error:
                        raise MCPError(
                            f"Timeout waiting for JSON-RPC response from "
                            f"'{self.config.name}' (last error: {last_error})"
                        )
                    raise MCPError(
                        f"Timeout waiting for JSON-RPC response from "
                        f"'{self.config.name}'"
                    )

                try:
                    raw_line = await asyncio.wait_for(
                        self._process.stdout.readline(), timeout=min(remaining, 10.0)
                    )
                except asyncio.TimeoutError:
                    continue  # keep trying until the full deadline

                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                # Try to parse as JSON
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # Non-JSON line (e.g. npx startup output) — skip
                    continue

                # Notifications have no "id"; skip them
                if "id" not in msg:
                    continue

                if "error" in msg:
                    err = msg["error"]
                    raise MCPError(
                        f"JSON-RPC error ({err.get('code', -1)}): "
                        f"{err.get('message', 'Unknown')}"
                    )

                return msg.get("result", msg)

    async def stop(self) -> None:
        """Gracefully shut down the subprocess."""
        if self._process is None:
            return

        try:
            shutdown_req = _make_request("shutdown", {})
            if self._process.stdin:
                self._process.stdin.write(shutdown_req.encode("utf-8"))
                await self._process.stdin.drain()
        except Exception:
            pass

        if self._process.returncode is None:
            try:
                self._process.kill()
                await self._process.wait()
            except ProcessLookupError:
                pass

        self._process = None

    async def __aenter__(self) -> "MCPServerProcess":
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class MCPRegistry:
    """Manages multiple MCP server subprocesses and tool discovery.

    Reads configuration from an mcp.json file and provides a unified
    interface for tool discovery and invocation across all servers.
    """

    def __init__(self, config_path: str = DEFAULT_MCP_CONFIG_PATH) -> None:
        self._config_path = config_path
        self._servers: dict[str, MCPServerProcess] = {}
        self._tool_index: dict[str, MCPToolDefinition] = {}

    # ---- Config loading -------------------------------------------------------

    def load_config(self, config_path: Optional[str] = None) -> None:
        """Read and parse the mcp.json configuration file."""
        path = config_path or self._config_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"MCP config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        servers_raw = raw.get("mcpServers", raw.get("servers", {}))
        for name, cfg in servers_raw.items():
            server_config = MCPServerConfig(
                name=name,
                command=cfg.get("command", ""),
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                transport=cfg.get("transport", "stdio"),
                enabled=cfg.get("enabled", True),
            )
            if server_config.enabled:
                self._servers[name] = MCPServerProcess(server_config)

    # ---- Lifecycle ------------------------------------------------------------

    async def start_all(self) -> None:
        """Start all enabled MCP server subprocesses."""
        for server in self._servers.values():
            try:
                await server.start()
            except Exception as exc:
                print(f"Warning: Failed to start MCP server '{server.config.name}': {exc}",
                      file=sys.stderr)

    async def stop_all(self) -> None:
        """Stop all MCP server subprocesses."""
        for server in self._servers.values():
            await server.stop()

    async def discover_all_tools(self) -> list[MCPToolDefinition]:
        """Discover tools from all running servers and build the index."""
        all_tools: list[MCPToolDefinition] = []
        for server in self._servers.values():
            if server.is_running:
                try:
                    tools = await server.discover_tools()
                    all_tools.extend(tools)
                except Exception as exc:
                    print(
                        f"Warning: Failed to discover tools from "
                        f"'{server.config.name}': {exc}",
                        file=sys.stderr,
                    )

        # Build name index (last server wins on name collision)
        self._tool_index = {}
        for tool in all_tools:
            self._tool_index[tool.name] = tool

        return all_tools

    # ---- Tool calling ---------------------------------------------------------

    async def call_tool(
        self, tool_name: str, arguments: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Call a tool by name, routing to the correct server."""
        definition = self._tool_index.get(tool_name)
        if definition is None:
            raise MCPToolNotFoundError(
                f"Tool '{tool_name}' not found in any MCP server. "
                f"Available: {list(self._tool_index.keys())}"
            )

        server = self._servers.get(definition.server_name)
        if server is None:
            raise MCPConnectionError(
                f"Server '{definition.server_name}' for tool '{tool_name}' is not running."
            )

        return await server.call_tool(tool_name, arguments or {})

    async def call_tool_on_server(
        self, server_name: str, tool_name: str, arguments: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Call a tool on a specific server by server name."""
        server = self._servers.get(server_name)
        if server is None:
            raise MCPConnectionError(f"Server '{server_name}' not found or not running.")

        return await server.call_tool(tool_name, arguments or {})

    # ---- OpenAI format --------------------------------------------------------

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Return all discovered tools in OpenAI function-calling format."""
        openai_tools: list[dict[str, Any]] = []
        for tool in self._tool_index.values():
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        return openai_tools

    # ---- Properties -----------------------------------------------------------

    @property
    def servers(self) -> dict[str, MCPServerProcess]:
        return dict(self._servers)

    @property
    def tool_definitions(self) -> list[MCPToolDefinition]:
        return list(self._tool_index.values())

    @property
    def is_any_running(self) -> bool:
        return any(s.is_running for s in self._servers.values())

    def __repr__(self) -> str:
        return (
            f"<MCPRegistry servers={len(self._servers)} "
            f"tools={len(self._tool_index)} running={self.is_any_running}>"
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_registry_instance: Optional[MCPRegistry] = None


def get_mcp_registry(
    config_path: str = DEFAULT_MCP_CONFIG_PATH,
    auto_load: bool = True,
) -> MCPRegistry:
    """Get or create the singleton MCPRegistry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = MCPRegistry(config_path=config_path)
        if auto_load and os.path.exists(config_path):
            _registry_instance.load_config(config_path)
    return _registry_instance
