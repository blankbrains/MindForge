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

from mindforge.models.base import BaseLLM, ChatMessage, ChatResult, LLMFactory
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
            self._llm = LLMFactory.create(_provider, _model)

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
        _llm_override: Any = None,
    ) -> ChatResult:
        """Call the LLM with 3-attempt retry and exponential backoff."""
        temp = temperature if temperature is not None else self._temperature
        llm = _llm_override if _llm_override is not None else self._llm
        last_exc: Optional[Exception] = None
        request_timeout = float(
            getattr(self._settings.agent, "llm_request_timeout", 45)
        )

        async def call_llm() -> ChatResult:
            return await asyncio.wait_for(
                llm.chat(
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                    temperature=temp,
                ),
                timeout=request_timeout,
            )

        attempts = max(1, min(int(max_attempts), 3))
        for attempt in range(attempts):
            try:
                tracer = self._get_tracer()
                if tracer is None:
                    return await call_llm()
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
                        "timeout_seconds": request_timeout,
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
                        result = await call_llm()
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
                            f"{request_timeout:g} seconds."
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
            f"LLM chat failed after {attempts} attempt(s): {detail}"
        ) from last_exc

    async def _chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_attempts: int = 3,
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
                    deadline = asyncio.get_running_loop().time() + request_timeout
                    stream = await asyncio.wait_for(
                        llm.chat(
                            messages=messages,
                            temperature=temp,
                            stream=True,
                        ),
                        timeout=request_timeout,
                    )
                    iterator = stream.__aiter__()
                    event_count = 0
                    while True:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise asyncio.TimeoutError
                        try:
                            event = await asyncio.wait_for(
                                anext(iterator),
                                timeout=remaining,
                            )
                        except StopAsyncIteration:
                            break
                        event_count += 1
                        emitted_event = True
                        yield event
                    if span is not None:
                        span.output = {"event_count": event_count}
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
                if span is not None:
                    is_timeout = isinstance(exc, asyncio.TimeoutError)
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
        if not isinstance(args, dict):
            return {
                "tool_call_id": tc_id,
                "output": f"Arguments for {tool_name} must be a JSON object.",
                "success": False,
                "error": "Tool arguments must be an object",
            }

        started = time.perf_counter()
        acquired_tool_slot = False
        try:
            if self._tool_semaphore is not None:
                try:
                    await asyncio.wait_for(
                        self._tool_semaphore.acquire(),
                        timeout=self._tool_queue_timeout,
                    )
                    acquired_tool_slot = True
                except asyncio.TimeoutError:
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
            tracer = self._get_tracer()
            if tracer is None:
                result = await tool.execute_async(**args)
            else:
                with tracer.span(
                    "tool.execute",
                    metadata={
                        "agent": self.name,
                        "tool": tool_name,
                    },
                ) as span:
                    span.input = args
                    result = await tool.execute_async(**args)
                    span.output = {
                        "success": result.success,
                        "output": (result.output or "")[:4000],
                        "error": result.error,
                    }
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
        tool_calls_rejected = 0
        automatic_source_search_attempted = False
        citation_rewrite_requested = False
        collected_sources: list[
            dict[str, Any]
        ] = []  # aggregate source metadata from tool calls
        source_indices: dict[str, int] = {}
        tool_call_details: list[dict[str, Any]] = []
        citation_rewrite_instruction = (
            "这份草稿没有使用来源编号。请基于已经提供的真实来源重写答案，"
            "并在对应事实后加入全局 [N] 引用标记；不要编造来源。"
        )

        def register_tool_result(
            tool_call: dict[str, Any],
            exec_result: dict[str, Any],
        ) -> str:
            tool_output = str(exec_result["output"])
            tool_data = exec_result.get("data")
            if isinstance(tool_data, dict):
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
            tool_call_details.append(
                {
                    "tool": tool_call.get("function", {}).get("name", ""),
                    "success": exec_result.get("success", False),
                    "latency_ms": exec_result.get(
                        "execution_time_ms",
                        0.0,
                    ),
                }
            )
            return tool_output

        for round_idx in range(max_rounds):
            result = await self._chat(
                conv,
                tools=(
                    tool_schemas
                    if use_tools
                    and tool_calls_made < self._settings.agent.max_tool_calls_total
                    else None
                ),
                _llm_override=_llm_override,
            )

            # Accumulate token usage
            if result.usage:
                for k, v in result.usage.items():
                    aggregated_usage[k] = aggregated_usage.get(k, 0) + (v or 0)

            # --- No tool calls → final answer ---
            if not result.tool_calls:
                candidate_content = result.content or ""
                source_tool_names = [
                    name
                    for name in ("search_knowledge_base", "web_search")
                    if name in self._tool_dict
                ]
                if (
                    require_sources
                    and not collected_sources
                    and not automatic_source_search_attempted
                    and source_tool_names
                    and tool_calls_made
                    < self._settings.agent.max_tool_calls_total
                ):
                    automatic_source_search_attempted = True
                    for source_tool_name in source_tool_names:
                        if (
                            tool_calls_made
                            >= self._settings.agent.max_tool_calls_total
                        ):
                            break
                        tool_call = {
                            "id": (
                                f"automatic-source-{tool_calls_made + 1}"
                            ),
                            "function": {
                                "name": source_tool_name,
                                "arguments": json.dumps(
                                    {"query": task},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                        tool_calls_made += 1
                        conv.append(
                            ChatMessage(
                                role="assistant",
                                content="",
                                tool_calls=[tool_call],
                            )
                        )
                        exec_result = await self._execute_tool(tool_call)
                        tool_output = register_tool_result(
                            tool_call,
                            exec_result,
                        )
                        conv.append(
                            ChatMessage(
                                role="tool",
                                content=tool_output,
                                tool_call_id=exec_result["tool_call_id"],
                            )
                        )
                        if collected_sources:
                            break
                    if tool_calls_made > 0:
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
                *[self._execute_tool(tc) for tc in selected_calls],
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
                final_result = await self._chat(
                    conv,
                    tools=None,  # no tools allowed – force text answer
                    _llm_override=_llm_override,
                )
                final_content = final_result.content or ""
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
                citation_result = await self._chat(
                    conv,
                    tools=None,
                    _llm_override=_llm_override,
                )
                if citation_result.content.strip():
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
        success = bool(final_content.strip())
        if not success:
            final_content = ""

        return AgentResult(
            agent_name=self.name,
            success=success,
            output=final_content,
            data={
                "rounds": min(round_idx + 1, max_rounds),
                "tool_calls": tool_calls_made,
                "tool_calls_rejected": tool_calls_rejected,
                "messages": len(conv),
                "sources": collected_sources,
                "citation_status": (
                    "available"
                    if collected_sources
                    and self._contains_citation_marker(final_content)
                    else (
                        "missing_markers"
                        if collected_sources
                        else (
                            "unavailable"
                            if require_sources
                            else "not_required"
                        )
                    )
                ),
                "tool_call_details": tool_call_details,
                "failure_reason": (None if success else "empty_llm_response"),
            },
            metadata={
                "model": model_used,
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
