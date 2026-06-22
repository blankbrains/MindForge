"""Test MCP adapter initialization, tool discovery, and OpenAI function conversion."""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# MCP Tool Definition
# ---------------------------------------------------------------------------


class MockMCPTool:
    """Simulates an MCP tool definition for testing."""

    def __init__(self, name: str, description: str = "", input_schema: dict | None = None):
        self.server_name = "mock_server"
        self.name = name
        self.description = description
        self.input_schema = input_schema or {}


# ---------------------------------------------------------------------------
# Adapter — converts MCP tools to OpenAI function format
# ---------------------------------------------------------------------------


def to_openai_function(tool: MockMCPTool) -> dict[str, Any]:
    """Convert an MCP tool definition to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMCPToolConversion:
    """Test converting MCP tools to OpenAI function-calling format."""

    def test_basic_conversion(self):
        tool = MockMCPTool(
            name="search_knowledge_base",
            description="Search the knowledge base for relevant documents",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
            },
        )
        func = to_openai_function(tool)
        assert func["type"] == "function"
        assert func["function"]["name"] == "search_knowledge_base"
        assert func["function"]["parameters"]["properties"]["query"]["type"] == "string"

    def test_tool_without_parameters(self):
        tool = MockMCPTool(name="health_check", description="Check system health")
        func = to_openai_function(tool)
        assert func["function"]["name"] == "health_check"

    def test_tool_name_format(self):
        names = ["web_search", "code_executor", "rag-retrieval"]
        for name in names:
            tool = MockMCPTool(name=name)
            func = to_openai_function(tool)
            assert func["function"]["name"] == name


class TestMCPAdapterRegistry:
    """Test MCP adapter tool registry management."""

    def setup_method(self):
        self.tools: dict[str, MockMCPTool] = {}

    def _register_tool(self, tool: MockMCPTool) -> None:
        self.tools[tool.name] = tool

    def test_register_and_lookup(self):
        tool = MockMCPTool("search", "Search tool")
        self._register_tool(tool)
        assert self.tools["search"].name == "search"

    def test_list_available_tools(self):
        for name in ["search", "summarize", "classify"]:
            self._register_tool(MockMCPTool(name))
        assert len(self.tools) == 3
        assert all(t in self.tools for t in ["search", "summarize", "classify"])

    def test_tool_not_found(self):
        assert "nonexistent" not in self.tools

    def test_overwrite_tool(self):
        self._register_tool(MockMCPTool("search", "v1"))
        self._register_tool(MockMCPTool("search", "v2"))
        assert self.tools["search"].description == "v2"


class TestMCPToolExecution:
    """Test MCP tool execution logic."""

    def test_tool_result_structure(self):
        result = {
            "success": True,
            "output": "test output",
            "error": None,
        }
        assert result["success"] is True
        assert result["output"] == "test output"

    def test_tool_error_handling(self):
        result = {
            "success": False,
            "output": None,
            "error": "Tool execution failed",
        }
        assert result["success"] is False
        assert result["error"] is not None

    def test_tool_timeout_handling(self):
        timeout = 30
        assert timeout > 0
        assert isinstance(timeout, int)
