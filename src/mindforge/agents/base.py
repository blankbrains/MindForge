"""Base agent abstractions for MindForge."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from mindforge.models.base import (
    BaseLLM,
    ChatMessage,
    ChatResult,
    LLMFactory,
    contains_textual_tool_protocol,
    extract_textual_tool_calls,
)
from mindforge.config import get_settings
from mindforge.tools.base import BaseTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


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
    cost_usd: float | None = None
    cost_status: str = "usage_unavailable"
    trace_id: str | None = None

    def __str__(self) -> str:
        return self.output

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "output": self.output,
            "data": self.data,
            "metadata": self.metadata,
            "token_usage": self.token_usage,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "cost_status": self.cost_status,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AgentResult:
        """Restore a result from trusted application cache data."""
        return cls(
            agent_name=str(value.get("agent_name") or ""),
            success=bool(value.get("success", True)),
            output=str(value.get("output") or ""),
            data=(
                dict(value.get("data") or {})
                if isinstance(value.get("data"), dict)
                else {}
            ),
            metadata=(
                dict(value.get("metadata") or {})
                if isinstance(value.get("metadata"), dict)
                else {}
            ),
            token_usage={
                str(key): int(amount)
                for key, amount in dict(value.get("token_usage") or {}).items()
                if isinstance(amount, (int, float))
            },
            latency_ms=float(value.get("latency_ms") or 0.0),
            cost_usd=(
                float(value["cost_usd"])
                if isinstance(value.get("cost_usd"), (int, float))
                else None
            ),
            cost_status=str(value.get("cost_status") or "usage_unavailable"),
            trace_id=(
                str(value["trace_id"])
                if isinstance(value.get("trace_id"), str)
                else None
            ),
        )


@dataclass(frozen=True)
class CostEstimate:
    amount_usd: float | None
    status: str


def _estimate_cost_details(
    model: str,
    usage: dict,
    provider: str | None = None,
) -> CostEstimate:
    """Estimate API cost without treating missing information as free usage."""
    normalized_provider = (provider or "").strip().lower()
    if normalized_provider == "local":
        return CostEstimate(None, "not_applicable")

    prompt_tokens = int(
        usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
    )
    completion_tokens = int(
        usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
    )
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return CostEstimate(None, "usage_unavailable")

    pricing = get_settings().llm.get_model_pricing(
        model,
        normalized_provider,
    )
    if not pricing:
        return CostEstimate(None, "pricing_unconfigured")

    input_rate = pricing.get("input")
    output_rate = pricing.get("output")
    if input_rate is None or output_rate is None:
        return CostEstimate(None, "pricing_unconfigured")

    cached_tokens = int(
        usage.get("prompt_cache_hit_tokens", 0)
        or usage.get("cached_input_tokens", 0)
        or 0
    )
    cached_tokens = max(0, min(cached_tokens, prompt_tokens))
    uncached_tokens = max(0, prompt_tokens - cached_tokens)
    cached_rate = pricing.get("cached_input", input_rate)
    amount = (
        (uncached_tokens / 1_000_000) * input_rate
        + (cached_tokens / 1_000_000) * cached_rate
        + (completion_tokens / 1_000_000) * output_rate
    )
    return CostEstimate(round(amount, 10), "estimated")


def _estimate_cost(
    model: str,
    usage: dict,
    provider: str | None = None,
) -> float | None:
    """Backward-compatible amount-only cost estimator."""
    return _estimate_cost_details(model, usage, provider).amount_usd


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base for all MindForge agents.

    Provides a shared tool-calling loop (function-calling / ReAct), LLM chat
    with retry, and standard result formatting.
    """

    model_role = "researcher"
    output_token_role: str | None = None
    deepseek_thinking_mode = "default"

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        tools: Optional[list[BaseTool]] = None,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.9,
        tool_semaphore: asyncio.Semaphore | None = None,
        tool_queue_timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        selected_provider = provider or settings.llm.llm_provider

        # Resolve model provider
        if llm is not None:
            self._llm = llm
        else:
            _provider = selected_provider
            _model = model or settings.llm.get_model(
                self.model_role,
                _provider,
            )
            token_role = self.output_token_role or self.model_role
            max_output_tokens = getattr(
                settings.agent,
                f"{token_role}_max_output_tokens",
                None,
            )
            factory_kwargs = (
                {"max_tokens": int(max_output_tokens)}
                if isinstance(max_output_tokens, int)
                and not isinstance(max_output_tokens, bool)
                else {}
            )
            if str(_provider).strip().lower() == "deepseek":
                thinking_mode = self.deepseek_thinking_mode
                if thinking_mode in {"enabled", "disabled"}:
                    factory_kwargs["thinking_mode"] = thinking_mode
            self._llm = LLMFactory.create(
                _provider,
                _model,
                **factory_kwargs,
            )

        self._tools: list[BaseTool] = tools or []
        self._tool_dict: dict[str, BaseTool] = {t.name: t for t in self._tools}
        self._temperature = temperature
        self._settings = settings
        self._model_name: str = getattr(self._llm, "_model", model or "unknown")
        self._provider_name: str = getattr(
            self._llm,
            "provider_name",
            selected_provider,
        )
        self._tracer: Any = None
        self._tool_semaphore = tool_semaphore
        self._tool_queue_timeout = (
            tool_queue_timeout
            if tool_queue_timeout is not None
            else settings.agent.queue_timeout
        )

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
        max_attempts: int = 3,
        deadline: float | None = None,
        max_output_tokens: int | None = None,
        _llm_override: Any = None,
    ) -> ChatResult:
        """Call the LLM with 3-attempt retry and exponential backoff."""
        temp = temperature if temperature is not None else self._temperature
        llm = _llm_override if _llm_override is not None else self._llm
        last_exc: Optional[Exception] = None
        request_timeout = float(
            getattr(self._settings.agent, "llm_request_timeout", 45)
        )

        def attempt_timeout() -> float:
            if deadline is None:
                return request_timeout
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise asyncio.TimeoutError(
                    "LLM request deadline was exhausted."
                )
            return min(request_timeout, remaining)

        async def call_llm(timeout_seconds: float) -> ChatResult:
            return await asyncio.wait_for(
                llm.chat(
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                    temperature=temp,
                    max_output_tokens=max_output_tokens,
                ),
                timeout=timeout_seconds,
            )

        attempts = max(1, min(int(max_attempts), 3))
        attempts_made = 0
        for attempt in range(attempts):
            attempts_made = attempt + 1
            timeout_seconds = attempt_timeout()
            try:
                tracer = self._get_tracer()
                if tracer is None:
                    return await call_llm(timeout_seconds)
                model_name = getattr(
                    llm,
                    "_model",
                    getattr(llm, "model", self._model_name),
                )
                with tracer.span(
                    "llm.chat",
                    metadata={
                        "agent": self.name,
                        "model": model_name,
                        "provider": self._provider_name,
                        "attempt": attempt + 1,
                        "stage": "llm_request",
                        "timeout_seconds": timeout_seconds,
                        "max_output_tokens": max_output_tokens,
                    },
                ) as span:
                    span.input = {
                        "messages": [
                            {
                                "role": message.role,
                                "content": message.content[:4000],
                            }
                            for message in messages
                        ],
                        "tools": [
                            tool.get("function", {}).get("name", "")
                            for tool in (tools or [])
                        ],
                    }
                    try:
                        result = await call_llm(timeout_seconds)
                    except asyncio.TimeoutError:
                        span.metadata.update(
                            {
                                "status": "error",
                                "error_code": "llm_request_timeout",
                                "error_type": "TimeoutError",
                            }
                        )
                        span.error = (
                            "LLM request timed out after "
                            f"{timeout_seconds:g} seconds."
                        )
                        raise
                    except asyncio.CancelledError:
                        span.metadata.update(
                            {
                                "status": "cancelled",
                                "error_code": "llm_request_cancelled",
                                "error_type": "CancelledError",
                            }
                        )
                        span.error = "LLM request was cancelled before completion."
                        raise
                    except Exception as exc:
                        status_code = getattr(exc, "status_code", None)
                        span.metadata.update(
                            {
                                "status": "error",
                                "error_code": "llm_request_failed",
                                "error_type": type(exc).__name__,
                            }
                        )
                        if isinstance(status_code, int):
                            span.metadata["http_status"] = status_code
                        span.error = (
                            "LLM request failed: "
                            f"{tracer.describe_exception(exc)}"
                        )
                        raise
                    span.output = {
                        "content": result.content[:4000],
                        "tool_call_count": len(result.tool_calls or []),
                        "usage": result.usage,
                    }
                    return result
            except Exception as exc:
                last_exc = exc
                # 401/400/403 等客户端错误不重试；仅 429/5xx/超时等可恢复错误重试
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= status < 500:
                    raise
                if attempt < attempts - 1:
                    wait = 2.0**attempt * 1.0
                    if deadline is not None:
                        remaining_after_wait = (
                            deadline - time.perf_counter() - wait
                        )
                        if (
                            isinstance(exc, asyncio.TimeoutError)
                            and remaining_after_wait < request_timeout
                        ):
                            break
                        if remaining_after_wait <= 0:
                            break
                    await asyncio.sleep(wait)

        detail = (
            str(last_exc).strip()
            if last_exc is not None
            else ""
        ) or (
            type(last_exc).__name__
            if last_exc is not None
            else "unknown error"
        )
        raise RuntimeError(
            f"LLM chat failed after {attempts_made} attempt(s): {detail}"
        ) from last_exc

    async def _chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_attempts: int = 3,
        max_output_tokens: int | None = None,
        _llm_override: Any = None,
    ) -> AsyncIterator[Any]:
        """Stream one LLM response with the same timeout, retry, and trace policy."""
        temp = temperature if temperature is not None else self._temperature
        llm = _llm_override if _llm_override is not None else self._llm
        request_timeout = float(
            getattr(self._settings.agent, "llm_request_timeout", 45)
        )
        attempts = max(1, min(int(max_attempts), 3))
        last_exc: Exception | None = None

        for attempt in range(attempts):
            emitted_event = False
            tracer = self._get_tracer()
            model_name = getattr(
                llm,
                "_model",
                getattr(llm, "model", self._model_name),
            )
            span_context = (
                tracer.span(
                    "llm.chat",
                    metadata={
                        "agent": self.name,
                        "model": model_name,
                        "provider": self._provider_name,
                        "attempt": attempt + 1,
                        "stage": "llm_stream",
                        "timeout_seconds": request_timeout,
                        "max_output_tokens": max_output_tokens,
                    },
                )
                if tracer is not None
                else nullcontext(None)
            )
            span = None
            stream = None
            try:
                with span_context as span:
                    if span is not None:
                        span.input = {
                            "messages": [
                                {
                                    "role": message.role,
                                    "content": message.content[:4000],
                                }
                                for message in messages
                            ],
                            "stream": True,
                        }
                    stream = await asyncio.wait_for(
                        llm.chat(
                            messages=messages,
                            temperature=temp,
                            stream=True,
                            max_output_tokens=max_output_tokens,
                        ),
                        timeout=request_timeout,
                    )
                    iterator = stream.__aiter__()
                    event_count = 0
                    content_event_count = 0
                    heartbeat_event_count = 0
                    finish_reason = ""
                    reasoning_only = False
                    while True:
                        try:
                            event = await asyncio.wait_for(
                                anext(iterator),
                                timeout=request_timeout,
                            )
                        except StopAsyncIteration:
                            break
                        event_count += 1
                        if event.type == "chunk" and event.content:
                            content_event_count += 1
                        elif event.type == "heartbeat":
                            heartbeat_event_count += 1
                        elif event.type == "done":
                            finish_reason = str(
                                getattr(event, "finish_reason", "") or ""
                            )
                            reasoning_only = bool(
                                getattr(event, "reasoning_only", False)
                            )
                        emitted_event = True
                        yield event
                    if span is not None:
                        span.output = {
                            "event_count": event_count,
                            "content_event_count": content_event_count,
                            "heartbeat_event_count": heartbeat_event_count,
                            "finish_reason": finish_reason,
                            "reasoning_only": reasoning_only,
                        }
                    return
            except asyncio.CancelledError:
                if span is not None:
                    span.metadata.update(
                        {
                            "status": "cancelled",
                            "error_code": "llm_request_cancelled",
                            "error_type": "CancelledError",
                        }
                    )
                    span.error = "LLM stream was cancelled before completion."
                raise
            except Exception as exc:
                last_exc = exc
                is_timeout = isinstance(exc, asyncio.TimeoutError)
                if span is not None:
                    span.metadata.update(
                        {
                            "status": "error",
                            "error_code": (
                                "llm_request_timeout"
                                if is_timeout
                                else "llm_request_failed"
                            ),
                            "error_type": type(exc).__name__,
                        }
                    )
                    span.error = (
                        f"LLM stream timed out after {request_timeout:g} seconds."
                        if is_timeout
                        else (
                            "LLM stream failed: "
                            f"{tracer.describe_exception(exc)}"
                        )
                    )
                if is_timeout:
                    raise RuntimeError(
                        "LLM stream timed out after "
                        f"{request_timeout:g} seconds."
                    ) from exc
                status = getattr(exc, "status_code", None)
                if (
                    emitted_event
                    or (status is not None and 400 <= status < 500)
                    or attempt >= attempts - 1
                ):
                    raise
                await asyncio.sleep(2.0**attempt)
            finally:
                close_stream = getattr(stream, "aclose", None)
                if callable(close_stream):
                    try:
                        await close_stream()
                    except Exception:
                        logger.warning(
                            "Failed to close LLM stream for %s.",
                            self.name,
                            exc_info=True,
                        )

        detail = (
            str(last_exc).strip()
            if last_exc is not None
            else ""
        ) or (
            type(last_exc).__name__
            if last_exc is not None
            else "unknown error"
        )
        raise RuntimeError(
            f"LLM stream failed after {attempts} attempt(s): {detail}"
        ) from last_exc

    def _get_tracer(self) -> Any:
        if not self._settings.observability.enable_tracing:
            return None
        if self._tracer is None:
            try:
                from mindforge.observability.tracer import get_tracer

                self._tracer = get_tracer()
            except Exception:
                return None
        return self._tracer

    # -- Tool helpers -------------------------------------------------------

    def _get_tool_schemas(
        self,
        active_tool_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert internal tools to OpenAI function-calling schema list."""
        return [
            tool.to_openai_function()
            for tool in self._tools
            if active_tool_names is None or tool.name in active_tool_names
        ]

    async def _execute_tool(
        self,
        tool_call: dict,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
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
        if not isinstance(args, dict):
            return {
                "tool_call_id": tc_id,
                "output": f"Arguments for {tool_name} must be a JSON object.",
                "success": False,
                "error": "Tool arguments must be an object",
            }

        started = time.perf_counter()
        acquired_tool_slot = False

        def timeout_result(message: str) -> dict[str, Any]:
            return {
                "tool_call_id": tc_id,
                "output": message,
                "success": False,
                "error": message,
                "data": {
                    "backend": tool_name,
                    "failure_type": "tool_timeout",
                    "retryable": True,
                    "terminal_for_run": True,
                },
                "execution_time_ms": (
                    time.perf_counter() - started
                ) * 1000,
            }

        if timeout_seconds is not None and timeout_seconds <= 0:
            return timeout_result(
                f"Tool {tool_name} was skipped because the subtask time "
                "budget was exhausted."
            )

        try:
            if self._tool_semaphore is not None:
                queue_timeout = self._tool_queue_timeout
                budget_limited_queue = False
                if timeout_seconds is not None:
                    budget_limited_queue = timeout_seconds <= queue_timeout
                    queue_timeout = min(queue_timeout, timeout_seconds)
                try:
                    await asyncio.wait_for(
                        self._tool_semaphore.acquire(),
                        timeout=queue_timeout,
                    )
                    acquired_tool_slot = True
                except asyncio.TimeoutError:
                    if budget_limited_queue:
                        return timeout_result(
                            f"Tool {tool_name} could not start before the "
                            "subtask time budget expired."
                        )
                    return {
                        "tool_call_id": tc_id,
                        "output": (
                            "Tool execution queue is full. Please retry "
                            "after current research tasks finish."
                        ),
                        "success": False,
                        "error": "Tool execution queue timeout",
                        "execution_time_ms": (time.perf_counter() - started) * 1000,
                    }

            async def invoke_tool() -> Any:
                tracer = self._get_tracer()
                if tracer is None:
                    return await tool.execute_async(**args)
                with tracer.span(
                    "tool.execute",
                    metadata={
                        "agent": self.name,
                        "tool": tool_name,
                        "timeout_seconds": timeout_seconds,
                    },
                ) as span:
                    span.input = args
                    tool_result = await tool.execute_async(**args)
                    span.output = {
                        "success": tool_result.success,
                        "output": (tool_result.output or "")[:4000],
                        "error": tool_result.error,
                    }
                    if not tool_result.success:
                        span.metadata["status"] = "error"
                        span.error = (
                            tool_result.error
                            or f"Tool {tool_name} returned no usable result."
                        )
                    return tool_result

            execution_timeout = timeout_seconds
            if execution_timeout is not None:
                execution_timeout -= time.perf_counter() - started
                if execution_timeout <= 0:
                    return timeout_result(
                        f"Tool {tool_name} could not start before the "
                        "subtask time budget expired."
                    )
                result = await asyncio.wait_for(
                    invoke_tool(),
                    timeout=execution_timeout,
                )
            else:
                result = await invoke_tool()
        except asyncio.TimeoutError:
            timeout_label = (
                f"{timeout_seconds:g} seconds"
                if timeout_seconds is not None
                else "its configured limit"
            )
            return timeout_result(
                f"Tool {tool_name} timed out after {timeout_label}."
            )
        except Exception as exc:
            logger.exception("Tool execution failed: %s", tool_name)
            return {
                "tool_call_id": tc_id,
                "output": f"Tool {tool_name} failed internally.",
                "success": False,
                "error": f"{type(exc).__name__}",
                "execution_time_ms": (time.perf_counter() - started) * 1000,
            }
        finally:
            if acquired_tool_slot and self._tool_semaphore is not None:
                self._tool_semaphore.release()

        output = result.output or (result.error or "")
        if len(output) > 10_000:
            output = output[:10_000] + "\n... [truncated at 10 000 chars]"

        return {
            "tool_call_id": tc_id,
            "output": output,
            "success": result.success,
            "error": result.error,
            "data": result.data if result.data else None,
            "execution_time_ms": result.execution_time_ms
            or (time.perf_counter() - started) * 1000,
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
        require_sources: bool = False,
        source_requirement: str = "required",
        deadline: float | None = None,
        final_answer_instruction: str | None = None,
        max_output_tokens: int | None = None,
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

        active_tool_names = set(self._tool_dict)
        use_tools = bool(active_tool_names)

        aggregated_usage: dict[str, int] = {}
        final_content = ""
        tool_calls_made = 0
        tool_calls_rejected = 0
        automatic_source_search_attempted = False
        citation_rewrite_requested = False
        force_model_only_generation = False
        force_text_generation = False
        native_answer_candidate = ""
        collected_sources: list[
            dict[str, Any]
        ] = []  # aggregate source metadata from tool calls
        source_indices: dict[str, int] = {}
        tool_call_details: list[dict[str, Any]] = []
        source_failures: list[dict[str, Any]] = []
        source_tool_name_set = {
            "search_knowledge_base",
            "web_search",
        }
        model_only_fallback = bool(
            getattr(
                getattr(self._settings, "web_search", None),
                "model_only_fallback",
                True,
            )
        )
        citation_rewrite_instruction = (
            "这份草稿没有使用来源编号。请基于已经提供的真实来源重写答案，"
            "并在对应事实后加入全局 [N] 引用标记；不要编造来源。"
        )
        source_tool_names = self._ordered_source_tools(task)
        strict_sources = (
            require_sources and source_requirement == "required"
        )
        bounded_final_instruction = (
            str(final_answer_instruction or "").strip()[:2_000]
            or "请直接基于现有信息生成完整最终答案。"
        )
        llm_timeout = float(
            getattr(self._settings.agent, "llm_request_timeout", 45)
        )
        total_deadline_budget = (
            max(0.0, deadline - start_time)
            if deadline is not None
            else None
        )
        final_answer_reserve = (
            min(
                llm_timeout,
                max(5.0, total_deadline_budget * 0.35),
            )
            if total_deadline_budget is not None
            else 0.0
        )

        def tool_timeout_budget() -> float | None:
            if deadline is None:
                return None
            remaining = deadline - time.perf_counter()
            return max(0.0, remaining - final_answer_reserve)

        async def execute_tool_call(
            tool_call: dict[str, Any],
        ) -> dict[str, Any]:
            tool_name = str(
                tool_call.get("function", {}).get("name", "")
            )
            if tool_name not in active_tool_names:
                return {
                    "tool_call_id": tool_call.get("id", ""),
                    "output": (
                        f"Tool {tool_name} is unavailable for the remainder "
                        "of this subtask because an earlier call failed."
                    ),
                    "success": False,
                    "error": "Tool disabled after earlier failure",
                    "data": {
                        "backend": tool_name,
                        "failure_type": "tool_disabled",
                        "retryable": False,
                        "terminal_for_run": False,
                    },
                }
            return await self._execute_tool(
                tool_call,
                timeout_seconds=tool_timeout_budget(),
            )

        async def chat_with_deadline(
            *,
            tools: list[dict[str, Any]] | None,
        ) -> ChatResult:
            if deadline is not None:
                remaining = deadline - time.perf_counter() - 0.25
                if remaining <= 0:
                    raise asyncio.TimeoutError(
                        "Subtask deadline exhausted before LLM generation."
                    )
                return await asyncio.wait_for(
                    self._chat(
                        conv,
                        tools=tools,
                        deadline=deadline,
                        max_output_tokens=max_output_tokens,
                        _llm_override=_llm_override,
                    ),
                    timeout=remaining,
                )
            return await self._chat(
                conv,
                tools=tools,
                max_output_tokens=max_output_tokens,
                _llm_override=_llm_override,
            )

        def register_tool_result(
            tool_call: dict[str, Any],
            exec_result: dict[str, Any],
        ) -> str:
            nonlocal native_answer_candidate
            tool_output = str(exec_result["output"])
            tool_data = exec_result.get("data")
            tool_name = str(
                tool_call.get("function", {}).get("name", "")
            )
            if isinstance(tool_data, dict):
                tool_usage = tool_data.get("usage")
                if isinstance(tool_usage, dict):
                    for key, value in tool_usage.items():
                        if isinstance(value, (int, float)) and not isinstance(
                            value,
                            bool,
                        ):
                            normalized_key = str(key)
                            aggregated_usage[normalized_key] = (
                                aggregated_usage.get(normalized_key, 0)
                                + int(value)
                            )
                raw_sources = tool_data.get("sources")
                if isinstance(raw_sources, list):
                    mapping_lines: list[str] = []
                    for position, source in enumerate(raw_sources, 1):
                        if not isinstance(source, dict):
                            continue
                        identity = self._source_identity(source)
                        if not identity:
                            continue
                        global_index = source_indices.get(identity)
                        if global_index is None:
                            global_index = len(collected_sources) + 1
                            source_indices[identity] = global_index
                            collected_sources.append(
                                {**source, "index": global_index}
                            )
                        local_index = source.get("index", position)
                        mapping_lines.append(
                            f"- Local result/source {local_index} "
                            f"must be cited as [{global_index}]"
                        )
                    if mapping_lines:
                        tool_output = (
                            "Global citation mapping for this tool result:\n"
                            + "\n".join(mapping_lines)
                            + "\nUse these global [N] numbers in the final "
                            "answer; do not reuse local result numbers.\n\n"
                            + tool_output
                        )
                    if (
                        tool_data.get("answer_ready") is True
                        and isinstance(tool_data.get("answer"), str)
                    ):
                        native_answer_candidate = (
                            self._ground_native_answer(
                                tool_data["answer"],
                                raw_sources,
                                source_indices,
                            )
                        )
                if (
                    tool_name in source_tool_name_set
                    and not exec_result.get("success", False)
                ):
                    failure_type = str(
                        tool_data.get("failure_type") or "tool_failed"
                    )
                    if failure_type != "tool_disabled":
                        source_failures.append(
                            {
                                "tool": tool_name,
                                "backend": str(
                                    tool_data.get("backend") or tool_name
                                ),
                                "failure_type": failure_type,
                                "retryable": bool(
                                    tool_data.get("retryable", False)
                                ),
                                "message": str(
                                    exec_result.get("error")
                                    or exec_result.get("output")
                                    or "Source tool failed."
                                ),
                            }
                        )
                    if tool_data.get("terminal_for_run") is True:
                        active_tool_names.discard(tool_name)
            tool_call_details.append(
                {
                    "tool": tool_name,
                    "success": exec_result.get("success", False),
                    "latency_ms": exec_result.get(
                        "execution_time_ms",
                        0.0,
                    ),
                    "failure_type": (
                        tool_data.get("failure_type")
                        if isinstance(tool_data, dict)
                        else None
                    ),
                    "backend": (
                        tool_data.get("backend")
                        if isinstance(tool_data, dict)
                        else None
                    ),
                }
            )
            return tool_output

        async def collect_required_sources() -> None:
            nonlocal automatic_source_search_attempted, tool_calls_made
            if (
                automatic_source_search_attempted
                or not source_tool_names
                or tool_calls_made
                >= self._settings.agent.max_tool_calls_total
            ):
                return
            automatic_source_search_attempted = True
            for source_tool_name in source_tool_names:
                if (
                    tool_calls_made
                    >= self._settings.agent.max_tool_calls_total
                ):
                    break
                if source_tool_name not in active_tool_names:
                    continue
                tool_call = {
                    "id": f"automatic-source-{tool_calls_made + 1}",
                    "type": "function",
                    "function": {
                        "name": source_tool_name,
                        "arguments": "",
                    },
                }
                arguments: dict[str, Any] = {"query": task}
                if source_tool_name == "search_knowledge_base":
                    tool = self._tool_dict.get(source_tool_name)
                    if tool is not None:
                        function_schema = tool.to_openai_function()
                        properties = (
                            function_schema.get("function", {})
                            .get("parameters", {})
                            .get("properties", {})
                        )
                        if "mode" in properties:
                            arguments["mode"] = "hybrid"
                tool_call["function"]["arguments"] = json.dumps(
                    arguments,
                    ensure_ascii=False,
                )
                tool_calls_made += 1
                exec_result = await execute_tool_call(tool_call)
                tool_output = register_tool_result(
                    tool_call,
                    exec_result,
                )
                conv.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "## 已检索证据\n\n"
                            f"{tool_output}\n\n"
                            "请仅基于以上证据回答；有来源时使用对应 [N] "
                            "引用，不要编造来源。\n"
                            f"{bounded_final_instruction}"
                        ),
                    )
                )
                if collected_sources:
                    break

        # Factual research needs evidence before generation. This avoids paying
        # for an ungrounded draft that would immediately be discarded.
        if require_sources:
            await collect_required_sources()
            active_tool_names.discard("verify_citation")
            if collected_sources:
                active_tool_names.difference_update(source_tool_name_set)
                force_text_generation = True
                conv.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "证据收集已经完成。不要再调用任何工具；"
                            "请直接基于现有证据完成分配产物。\n"
                            f"{bounded_final_instruction}"
                        ),
                    )
                )
            elif model_only_fallback:
                force_model_only_generation = True
                active_tool_names.difference_update(source_tool_name_set)
                conv.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "未能获得可核验来源。请基于模型已有知识给出有用答案，"
                            "严格遵守用户指定的篇幅、句数和输出格式；"
                            "不要在正文中追加来源免责声明，系统会通过结构化状态"
                            "单独提示来源情况。不要编造引用或声称已经联网检索成功。"
                            f"\n{bounded_final_instruction}"
                        ),
                    )
                )

        rounds_completed = 0
        if native_answer_candidate and collected_sources:
            final_content = native_answer_candidate
            conv.append(ChatMessage(role="assistant", content=final_content))

        for round_idx in (
            range(0) if final_content else range(max_rounds)
        ):
            rounds_completed = round_idx + 1
            round_tool_schemas = (
                []
                if force_model_only_generation or force_text_generation
                else self._get_tool_schemas(active_tool_names)
            )
            exposed_tool_names = {
                str(schema.get("function", {}).get("name", ""))
                for schema in round_tool_schemas
                if isinstance(schema, dict)
            }
            result = await chat_with_deadline(
                tools=(
                    round_tool_schemas
                    if use_tools
                    and round_tool_schemas
                    and tool_calls_made < self._settings.agent.max_tool_calls_total
                    else None
                ),
            )
            protocol_detected = contains_textual_tool_protocol(
                result.content
            )
            candidate_tool_calls = list(result.tool_calls or [])
            if protocol_detected:
                cleaned_content, textual_calls = (
                    extract_textual_tool_calls(result.content)
                )
                result.content = cleaned_content
                if not candidate_tool_calls:
                    candidate_tool_calls = textual_calls
            allowed_tool_calls = [
                call
                for call in candidate_tool_calls
                if str(call.get("function", {}).get("name", ""))
                in exposed_tool_names
            ]
            boundary_rejected_calls = (
                len(candidate_tool_calls) - len(allowed_tool_calls)
            )
            tool_calls_rejected += boundary_rejected_calls
            result.tool_calls = allowed_tool_calls or None

            # Accumulate token usage
            if result.usage:
                for k, v in result.usage.items():
                    aggregated_usage[k] = aggregated_usage.get(k, 0) + (v or 0)

            if (
                protocol_detected or boundary_rejected_calls
            ) and not result.tool_calls:
                conv.append(
                    ChatMessage(
                        role="assistant",
                        content=result.content,
                    )
                )
                conv.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "上一条输出包含无法解析的内部工具协议。不要展示或"
                            "重复任何 DSML、XML、tool_calls 或 invoke 标签；"
                            "请直接基于已有上下文和工具结果生成最终答案。"
                        ),
                    )
                )
                force_text_generation = True
                continue

            # --- No tool calls → final answer ---
            if not result.tool_calls:
                candidate_content = result.content or ""
                if (
                    require_sources
                    and not collected_sources
                    and not automatic_source_search_attempted
                    and source_tool_names
                    and tool_calls_made
                    < self._settings.agent.max_tool_calls_total
                ):
                    await collect_required_sources()
                    continue
                if (
                    require_sources
                    and collected_sources
                    and not self._contains_citation_marker(candidate_content)
                    and not citation_rewrite_requested
                ):
                    citation_rewrite_requested = True
                    conv.append(
                        ChatMessage(
                            role="assistant",
                            content=candidate_content,
                        )
                    )
                    conv.append(
                        ChatMessage(
                            role="user",
                            content=citation_rewrite_instruction,
                        )
                    )
                    continue
                final_content = candidate_content
                # Append the final assistant message
                conv.append(ChatMessage(role="assistant", content=final_content))
                break

            # --- Has tool calls → execute in parallel ---
            remaining = max(
                0,
                self._settings.agent.max_tool_calls_total - tool_calls_made,
            )
            allowed_count = min(
                len(result.tool_calls),
                self._settings.agent.max_tool_calls_per_round,
                remaining,
            )
            selected_calls = result.tool_calls[:allowed_count]
            rejected_calls = result.tool_calls[allowed_count:]
            tool_calls_made += len(selected_calls)
            tool_calls_rejected += len(rejected_calls)

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
                *[execute_tool_call(tc) for tc in selected_calls],
                return_exceptions=True,
            )
            tool_results.extend(
                {
                    "tool_call_id": tc.get("id", ""),
                    "output": (
                        "Tool call rejected because the configured per-round "
                        "or total tool-call limit was reached."
                    ),
                    "success": False,
                    "error": "Tool-call limit reached",
                }
                for tc in rejected_calls
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
                    tool_output = register_tool_result(tc, exec_result)
                    conv.append(
                        ChatMessage(
                            role="tool",
                            content=tool_output,
                            tool_call_id=exec_result["tool_call_id"],
                        )
                    )

        # --- Determine final output ---
        # If we exited because of max rounds, force one final non-tool call
        # so the LLM can produce a closing answer from accumulated tool results.
        if not final_content and use_tools and tool_calls_made > 0:
            try:
                final_result = await chat_with_deadline(
                    tools=None,  # no tools allowed – force text answer
                )
                candidate = final_result.content or ""
                final_content = (
                    ""
                    if final_result.tool_calls
                    or contains_textual_tool_protocol(candidate)
                    else candidate
                )
                if final_content:
                    conv.append(ChatMessage(role="assistant", content=final_content))
                if final_result.usage:
                    for k, v in final_result.usage.items():
                        aggregated_usage[k] = aggregated_usage.get(k, 0) + (v or 0)
            except Exception:
                pass  # best-effort; fall through to backward scan

        if (
            final_content
            and require_sources
            and collected_sources
            and not self._contains_citation_marker(final_content)
            and not citation_rewrite_requested
        ):
            citation_rewrite_requested = True
            conv.append(ChatMessage(role="assistant", content=final_content))
            conv.append(
                ChatMessage(
                    role="user",
                    content=citation_rewrite_instruction,
                )
            )
            try:
                citation_result = await chat_with_deadline(
                    tools=None,
                )
                if (
                    not citation_result.tool_calls
                    and not contains_textual_tool_protocol(
                        citation_result.content
                    )
                    and citation_result.content.strip()
                ):
                    final_content = citation_result.content
                    conv.append(
                        ChatMessage(
                            role="assistant",
                            content=final_content,
                        )
                    )
                if citation_result.usage:
                    for key, value in citation_result.usage.items():
                        aggregated_usage[key] = (
                            aggregated_usage.get(key, 0) + (value or 0)
                        )
            except Exception:
                logger.warning(
                    "Citation rewrite failed for %s.",
                    self.name,
                    exc_info=True,
                )

        # Only a non-tool assistant message is a valid final answer. Returning
        # tool-call preambles here can leak protocol text such as XML-like tags.
        if not final_content:
            for msg in reversed(conv):
                if (
                    msg.role == "assistant"
                    and msg.content
                    and not msg.tool_calls
                    and not contains_textual_tool_protocol(msg.content)
                ):
                    final_content = msg.content
                    break

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        model_used = (
            getattr(_llm_override, "_model", self._model_name)
            if _llm_override
            else self._model_name
        )
        cost_estimate = _estimate_cost_details(
            model_used,
            aggregated_usage,
            self._provider_name,
        )
        has_citations = self._contains_citation_marker(final_content)
        citation_status = (
            "available"
            if collected_sources and has_citations
            else (
                "missing_markers"
                if collected_sources
                else ("unavailable" if require_sources else "not_required")
            )
        )
        failure_reason: str | None = None
        source_warning: str | None = None
        terminal_failure = False
        outcome = "success"
        grounding_status = (
            "grounded"
            if require_sources and collected_sources and has_citations
            else ("not_required" if not require_sources else "unavailable")
        )
        if require_sources and not collected_sources:
            primary_source_failure = (
                source_failures[-1] if source_failures else None
            )
            source_warning = (
                f"{primary_source_failure['tool']}:"
                f"{primary_source_failure['failure_type']}"
                if primary_source_failure is not None
                else "sources_unavailable"
            )
            if model_only_fallback and final_content.strip():
                outcome = "degraded" if strict_sources else "success"
                grounding_status = "model_only"
                failure_reason = source_warning if strict_sources else None
            else:
                failure_reason = source_warning
                terminal_failure = True
                final_content = str(
                    primary_source_failure["message"]
                    if primary_source_failure is not None
                    else (
                        "未能获取与当前问题高度相关且可核验的来源，"
                        "因此无法可靠完成本次研究。"
                    )
                )
        elif require_sources and not has_citations:
            failure_reason = "citation_markers_missing"
            terminal_failure = True
            final_content = (
                "已获取研究资料，但模型未能生成可核验的引用标记，"
                "因此本次结果未被视为有效研究报告。"
            )

        success = bool(final_content.strip()) and not terminal_failure
        if not success:
            final_content = final_content.strip()

        return AgentResult(
            agent_name=self.name,
            success=success,
            output=final_content,
            data={
                "rounds": rounds_completed,
                "tool_calls": tool_calls_made,
                "tool_calls_rejected": tool_calls_rejected,
                "messages": len(conv),
                "sources": collected_sources,
                "citation_status": citation_status,
                "grounding_status": grounding_status,
                "outcome": outcome if success else "error",
                "tool_call_details": tool_call_details,
                "source_failures": source_failures,
                "source_warning": source_warning,
                "failure_reason": (
                    failure_reason
                    if failure_reason is not None
                    else (None if success else "empty_llm_response")
                ),
            },
            metadata={
                "model": model_used,
                "citation_status": citation_status,
                "grounding_status": grounding_status,
                "outcome": outcome if success else "error",
                "failure_reason": failure_reason,
                "source_warning": source_warning,
            },
            token_usage=aggregated_usage,
            latency_ms=elapsed_ms,
            cost_usd=cost_estimate.amount_usd,
            cost_status=cost_estimate.status,
        )

    @staticmethod
    def _source_identity(source: dict[str, Any]) -> str:
        return str(
            source.get("url")
            or source.get("chunk_id")
            or source.get("id")
            or (
                f"{source.get('title', source.get('source', ''))}:"
                f"{str(source.get('content', source.get('text', '')))[:200]}"
            )
        ).strip()

    @staticmethod
    def _contains_citation_marker(text: str) -> bool:
        return bool(re.search(r"\[[1-9]\d*\]", text))

    def _ordered_source_tools(self, task: str) -> list[str]:
        available = {
            name
            for name in ("search_knowledge_base", "web_search")
            if name in self._tool_dict
        }
        if not available:
            return []
        normalized = re.sub(r"\s+", " ", task).strip().casefold()
        web_only_markers = (
            "只使用联网",
            "仅使用联网",
            "仅联网",
            "只查官网",
            "仅查官网",
            "web only",
            "online sources only",
        )
        web_priority_markers = (
            "联网",
            "官网",
            "官方网站",
            "最新",
            "目前",
            "现在的",
            "当前版本",
            "发布日期",
            "实时",
            "today",
            "latest",
            "current",
            "official website",
        )
        if "web_search" in available and any(
            marker in normalized for marker in web_only_markers
        ):
            return ["web_search"]
        preferred = (
            ("web_search", "search_knowledge_base")
            if "web_search" in available
            and any(marker in normalized for marker in web_priority_markers)
            else ("search_knowledge_base", "web_search")
        )
        return [name for name in preferred if name in available]

    @classmethod
    def _ground_native_answer(
        cls,
        answer: str,
        sources: list[Any],
        source_indices: dict[str, int],
    ) -> str:
        rendered = answer.strip()
        heading_match = re.search(r"#{1,6}\s+\S", rendered)
        if heading_match is not None:
            prefix = rendered[:heading_match.start()]
            progress_markers = (
                "我来搜索",
                "让我搜索",
                "让我打开",
                "我将搜索",
                "接下来搜索",
                "再验证",
                "正在搜索",
            )
            if any(marker in prefix for marker in progress_markers):
                rendered = rendered[heading_match.start():].lstrip()
        available_indices: list[int] = []
        url_indices: dict[str, int] = {}
        for source in sources:
            if not isinstance(source, dict):
                continue
            global_index = source_indices.get(cls._source_identity(source))
            if global_index is None:
                continue
            available_indices.append(global_index)
            url = str(source.get("url") or "").strip()
            if not url:
                continue
            url_indices[cls._canonical_citation_url(url)] = global_index

        markdown_link_pattern = re.compile(
            r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
            re.IGNORECASE,
        )
        raw_url_pattern = re.compile(
            r"https?://[^\s<>()\[\]{}\"']+",
            re.IGNORECASE,
        )
        source_line_pattern = re.compile(
            r"^\s*(?:[-*+]\s*)?"
            r"(?:来源|参考来源|参考资料|sources?)\s*[:：]\s*(.+?)\s*$",
            re.IGNORECASE,
        )

        def indices_in_text(value: str) -> list[int]:
            indices: list[int] = []
            for match in markdown_link_pattern.finditer(value):
                index = url_indices.get(
                    cls._canonical_citation_url(match.group(2))
                )
                if index is not None:
                    indices.append(index)
            for match in raw_url_pattern.finditer(value):
                index = url_indices.get(
                    cls._canonical_citation_url(match.group(0))
                )
                if index is not None:
                    indices.append(index)
            indices.extend(
                int(value)
                for value in re.findall(r"\[([1-9]\d*)\]", value)
                if int(value) in available_indices
            )
            return list(dict.fromkeys(indices))

        def append_markers(line: str, indices: list[int]) -> str:
            missing = [
                index
                for index in indices
                if f"[{index}]" not in line
            ]
            if not missing:
                return line
            markers = "".join(f"[{index}]" for index in missing)
            return f"{line.rstrip()} {markers}"

        def replace_markdown_link(match: re.Match[str]) -> str:
            index = url_indices.get(
                cls._canonical_citation_url(match.group(2))
            )
            if index is None:
                return match.group(0)
            return f"{match.group(1)} [{index}]"

        def replace_raw_url(match: re.Match[str]) -> str:
            index = url_indices.get(
                cls._canonical_citation_url(match.group(0))
            )
            return f"[{index}]" if index is not None else match.group(0)

        normalized_lines: list[str] = []
        for line in rendered.splitlines():
            source_match = source_line_pattern.match(line)
            if source_match is not None:
                indices = indices_in_text(source_match.group(1))
                if indices:
                    cursor = len(normalized_lines) - 1
                    while cursor >= 0 and not normalized_lines[cursor].strip():
                        cursor -= 1
                    bullet_indices: list[int] = []
                    while (
                        cursor >= 0
                        and re.match(
                            r"^\s*[-*+]\s+",
                            normalized_lines[cursor],
                        )
                    ):
                        bullet_indices.append(cursor)
                        cursor -= 1
                    targets = (
                        list(reversed(bullet_indices))
                        if bullet_indices
                        else (
                            [cursor]
                            if cursor >= 0
                            and not normalized_lines[cursor].lstrip().startswith("#")
                            else []
                        )
                    )
                    for target in targets:
                        normalized_lines[target] = append_markers(
                            normalized_lines[target],
                            indices,
                        )
                    continue

            normalized = markdown_link_pattern.sub(
                replace_markdown_link,
                line,
            )
            normalized = raw_url_pattern.sub(
                replace_raw_url,
                normalized,
            )
            normalized_lines.append(normalized)

        rendered = "\n".join(normalized_lines).strip()
        if (
            rendered
            and available_indices
            and not cls._contains_citation_marker(rendered)
        ):
            indices = list(dict.fromkeys(available_indices))
            lines = rendered.splitlines()
            cursor = len(lines) - 1
            while cursor >= 0 and (
                not lines[cursor].strip()
                or lines[cursor].lstrip().startswith("#")
            ):
                cursor -= 1
            if cursor >= 0:
                lines[cursor] = append_markers(lines[cursor], indices)
                rendered = "\n".join(lines)
        return rendered

    @staticmethod
    def _canonical_citation_url(url: str) -> str:
        value = url.strip().rstrip(".,;:!?")
        try:
            parsed = urlsplit(value)
        except ValueError:
            return value
        fragment = (
            ""
            if parsed.fragment.startswith("ws_call_id=")
            else parsed.fragment
        )
        return urlunsplit(
            (
                parsed.scheme.casefold(),
                parsed.netloc.casefold(),
                parsed.path,
                parsed.query,
                fragment,
            )
        )
