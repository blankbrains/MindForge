"""Base agent abstractions for MindForge."""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from mindforge.models.base import BaseLLM, ChatMessage, ChatResult, LLMFactory
from mindforge.config import get_settings
from mindforge.tools.base import BaseTool, ToolResult


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class AgentMessage:
    """A message within an agent's conversation context.

    Mirrors ChatMessage but is owned by the agent layer so we can add
    agent-specific fields without coupling to the LLM layer.
    """

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class AgentResult:
    """Standard result wrapper for all agent execution."""

    agent_name: str = ""
    success: bool = True
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    def __str__(self) -> str:
        return self.output


# Per-model cost per 1K tokens (input / output in USD)
_MODEL_COST_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o":              (0.00250, 0.01000),
    "gpt-4o-mini":         (0.00015, 0.00060),
    "gpt-4o-2024-08-06":   (0.00250, 0.01000),
    "gpt-4o-mini-2024-07-18": (0.00015, 0.00060),
    "deepseek-chat":       (0.00027, 0.00110),
    "deepseek-reasoner":   (0.00055, 0.00219),
    "claude-3-5-sonnet-20241022": (0.00300, 0.01500),
    "claude-3-5-haiku-20241022":  (0.00080, 0.00400),
}


def _estimate_cost(model: str, usage: dict) -> float:
    """Estimate USD cost from token usage and model name."""
    if not usage:
        return 0.0
    rates = _MODEL_COST_PER_1K.get(model, (0.001, 0.002))
    prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
    return (prompt_tokens / 1000) * rates[0] + (completion_tokens / 1000) * rates[1]


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base for all MindForge agents.

    Provides a shared tool-calling loop (function-calling / ReAct), LLM chat
    with retry, and standard result formatting.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        tools: Optional[list[BaseTool]] = None,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.9,
    ) -> None:
        settings = get_settings()

        # Resolve model provider
        if llm is not None:
            self._llm = llm
        else:
            _provider = provider or settings.llm.llm_provider
            _model = model or settings.llm.get_model("researcher")
            self._llm = LLMFactory.create(_provider, _model)

        self._tools: list[BaseTool] = tools or []
        self._tool_dict: dict[str, BaseTool] = {t.name: t for t in self._tools}
        self._temperature = temperature
        self._settings = settings
        self._model_name: str = getattr(self._llm, "_model", model or "unknown")

    # -- Properties ---------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name (e.g. 'planner', 'researcher')."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt that defines the agent's persona and instructions."""

    # -- LLM chat with retry ------------------------------------------------

    async def _chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[dict]] = None,
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        _llm_override: Any = None,
    ) -> ChatResult:
        """Call the LLM with 3-attempt retry and exponential backoff."""
        temp = temperature if temperature is not None else self._temperature
        llm = _llm_override if _llm_override is not None else self._llm
        last_exc: Optional[Exception] = None

        for attempt in range(3):
            try:
                return await llm.chat(
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                    temperature=temp,
                )
            except Exception as exc:
                last_exc = exc
                # 401/400/403 等客户端错误不重试；仅 429/5xx/超时等可恢复错误重试
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= status < 500:
                    raise
                if attempt < 2:
                    wait = 2.0 ** attempt * 1.0
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"LLM chat failed after 3 attempts: {last_exc}"
        ) from last_exc

    # -- Tool helpers -------------------------------------------------------

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Convert internal tools to OpenAI function-calling schema list."""
        return [t.to_openai_function() for t in self._tools]

    async def _execute_tool(self, tool_call: dict) -> dict[str, Any]:
        """Execute a single tool call and return a result dict.

        Returns
        -------
        dict with keys: ``tool_call_id``, ``output``, ``success``, ``error``.
        Output is truncated at 10 000 characters.
        """
        tc_id = tool_call.get("id", "")
        func = tool_call.get("function", {})
        tool_name = func.get("name", "")
        raw_args = func.get("arguments", "{}")

        if not tool_name:
            return {
                "tool_call_id": tc_id,
                "output": "Tool call missing function name.",
                "success": False,
                "error": "Missing function name",
            }

        tool = self._tool_dict.get(tool_name)
        if tool is None:
            return {
                "tool_call_id": tc_id,
                "output": f"Unknown tool: {tool_name}. Available: {list(self._tool_dict)}",
                "success": False,
                "error": f"Unknown tool: {tool_name}",
            }

        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as exc:
            return {
                "tool_call_id": tc_id,
                "output": f"Failed to parse arguments for {tool_name}: {exc}",
                "success": False,
                "error": str(exc),
            }

        try:
            result: ToolResult = await tool.execute_async(**args)
        except Exception as exc:
            return {
                "tool_call_id": tc_id,
                "output": f"Tool {tool_name} raised: {type(exc).__name__}: {exc}",
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        output = result.output or (result.error or "")
        if len(output) > 10_000:
            output = output[:10_000] + "\n... [truncated at 10 000 chars]"

        return {
            "tool_call_id": tc_id,
            "output": output,
            "success": result.success,
            "error": result.error,
            "data": result.data if result.data else None,
        }

    # -- Tool-calling loop --------------------------------------------------

    async def _run_tool_loop(
        self,
        task: str,
        *,
        context: Optional[str] = None,
        max_rounds: Optional[int] = None,
        messages: Optional[list[ChatMessage]] = None,
        _llm_override: Any = None,
    ) -> AgentResult:
        """Run the LLM tool-calling loop (ReAct / function calling).

        Parameters
        ----------
        task : str
            The user task / query.
        context : str, optional
            Additional context to prepend (e.g. retrieved documents).
        max_rounds : int, optional
            Maximum number of tool-calling rounds (default: config or 8).
        messages : list[ChatMessage], optional
            Pre-existing conversation to continue from.

        Returns
        -------
        AgentResult with the final assistant output and aggregated metadata.
        """
        max_rounds = max_rounds or self._settings.agent.max_iterations
        if max_rounds < 1:
            max_rounds = 1  # 防御非正配置
        start_time = time.perf_counter()

        # --- Build message list ---
        conv: list[ChatMessage]
        if messages:
            conv = list(messages)
        else:
            conv = [ChatMessage(role="system", content=self.system_prompt)]

        # Only add the task message if the caller didn't supply messages
        if not messages:
            user_content = task
            if context:
                user_content = f"## Task\n\n{task}\n\n## Context\n\n{context}"
            conv.append(ChatMessage(role="user", content=user_content))

        tool_schemas = self._get_tool_schemas()
        use_tools = bool(tool_schemas)

        aggregated_usage: dict[str, int] = {}
        final_content = ""
        tool_calls_made = 0
        collected_sources: list[dict[str, Any]] = []  # aggregate source metadata from tool calls

        for round_idx in range(max_rounds):
            result = await self._chat(
                conv,
                tools=tool_schemas if use_tools else None,
                _llm_override=_llm_override,
            )

            # Accumulate token usage
            if result.usage:
                for k, v in result.usage.items():
                    aggregated_usage[k] = aggregated_usage.get(k, 0) + (v or 0)

            # --- No tool calls → final answer ---
            if not result.tool_calls:
                final_content = result.content or ""
                # Append the final assistant message
                conv.append(ChatMessage(role="assistant", content=final_content))
                break

            # --- Has tool calls → execute in parallel ---
            tool_calls_made += len(result.tool_calls)

            # 1. Add assistant message with tool_calls to conversation
            assistant_content = result.content or ""
            conv.append(
                ChatMessage(
                    role="assistant",
                    content=assistant_content,
                    tool_calls=result.tool_calls,
                )
            )

            # 2. Execute all tools concurrently
            tool_results = await asyncio.gather(
                *[self._execute_tool(tc) for tc in result.tool_calls],
                return_exceptions=True,
            )

            # 3. Feed tool results back (pair with original tool_call for id)
            for tc, exec_result in zip(result.tool_calls, tool_results):
                tc_id = tc.get("id", "")
                if isinstance(exec_result, BaseException):
                    conv.append(
                        ChatMessage(
                            role="tool",
                            content=f"Tool execution error: {exec_result}",
                            tool_call_id=tc_id,
                        )
                    )
                else:
                    conv.append(
                        ChatMessage(
                            role="tool",
                            content=exec_result["output"],
                            tool_call_id=exec_result["tool_call_id"],
                        )
                    )
                    # Collect source metadata from tool results (e.g. RAGTool returns sources in data)
                    tool_data = exec_result.get("data") if isinstance(exec_result, dict) else None
                    if isinstance(tool_data, dict) and "sources" in tool_data:
                        for src in tool_data["sources"]:
                            if isinstance(src, dict):
                                collected_sources.append(src)

        # --- Determine final output ---
        # If we exited because of max rounds, force one final non-tool call
        # so the LLM can produce a closing answer from accumulated tool results.
        if not final_content and use_tools and tool_calls_made > 0:
            try:
                final_result = await self._chat(
                    conv,
                    tools=None,  # no tools allowed – force text answer
                    _llm_override=_llm_override,
                )
                final_content = final_result.content or ""
                if final_content:
                    conv.append(
                        ChatMessage(role="assistant", content=final_content)
                    )
                if final_result.usage:
                    for k, v in final_result.usage.items():
                        aggregated_usage[k] = aggregated_usage.get(k, 0) + (v or 0)
            except Exception:
                pass  # best-effort; fall through to backward scan

        # Fallback: scan backwards for the last assistant message with content
        if not final_content:
            for msg in reversed(conv):
                if msg.role == "assistant" and msg.content:
                    final_content = msg.content
                    break

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        model_used = getattr(_llm_override, "_model", self._model_name) if _llm_override else self._model_name
        cost = _estimate_cost(model_used, aggregated_usage)

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=final_content,
            data={
                "rounds": min(round_idx + 1, max_rounds),
                "tool_calls": tool_calls_made,
                "messages": len(conv),
                "sources": collected_sources,
            },
            metadata={
                "model": self._model_name,
            },
            token_usage=aggregated_usage,
            latency_ms=elapsed_ms,
            cost_usd=cost,
        )
