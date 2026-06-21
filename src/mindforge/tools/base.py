"""Base tool abstractions for MindForge."""

from __future__ import annotations

import asyncio
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolResult:
    """Standard result wrapper for all tool executions."""

    success: bool
    output: str = ""
    error: Optional[str] = None
    data: Any = None
    execution_time_ms: float = 0.0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "data": self.data,
            "execution_time_ms": self.execution_time_ms,
            "truncated": self.truncated,
        }

    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"Error: {self.error or 'Unknown error'}"


class BaseTool(ABC):
    """Abstract base class that all MindForge tools must implement."""

    name: str = ""
    description: str = ""
    parameters_schema: dict[str, Any] = field(default_factory=dict)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            cls.name = cls.__name__.lower()

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters. Must be implemented by subclasses.

        For async tools, override execute_async instead and keep this as a
        synchronous bridge that calls asyncio.run(execute_async(...)).
        """
        ...

    async def execute_async(self, **kwargs: Any) -> ToolResult:
        """Async execution — fall back to execute() for synchronous tools."""
        return await asyncio.to_thread(self.execute, **kwargs)

    def to_openai_function(self) -> dict[str, Any]:
        """Convert this tool to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def safe_execute(self, **kwargs: Any) -> ToolResult:
        """Execute with top-level exception handling.

        Never raises; always returns a ToolResult.
        """
        try:
            return self.execute(**kwargs)
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                data={"traceback": traceback.format_exc()},
            )

    def get_schema_summary(self) -> dict[str, str]:
        """Brief summary for logging / UI display."""
        required = self.parameters_schema.get("required", [])
        return {
            "name": self.name,
            "description": self.description[:120],
            "parameters": ", ".join(
                f"{p}: {self.parameters_schema.get('properties', {}).get(p, {}).get('type', 'any')}"
                for p in required
            ),
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
