"""Researcher agent — executes a ReAct loop with tool access."""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Optional

from mindforge.agents.base import AgentResult, BaseAgent
from mindforge.models.base import ChatMessage
from mindforge.config import get_settings

# ---------------------------------------------------------------------------
# ResearcherAgent
# ---------------------------------------------------------------------------

_RESEARCHER_SYSTEM_PROMPT = """You are an expert research assistant. You have access to several tools that let you gather information and verify facts.

Available tools:
- **search_knowledge_base** — Query the internal knowledge base for relevant documents and context.
- **web_search** — Search the web for current, up-to-date information.
- **code_executor** — Execute Python code in a sandbox for calculations, data analysis, or prototyping.
- **verify_citation** — Verify that citation markers [N] in a report match their source definitions.
- **mcp_tool** — Access external MCP (Model Context Protocol) tools for database queries, API calls, etc.

Guidelines:
1. Start by searching the knowledge base first if the topic might be covered internally.
2. Use web search for recent or external information.
3. When you find relevant information, cite your sources with [N] markers.
4. Use code_executor for data processing, statistics, or running algorithms.
5. Think step by step. Explain your reasoning before using a tool.
6. When you have enough information, provide a comprehensive answer with proper citations.
7. If you hit dead ends, try alternative search queries or approaches."""


class ResearcherAgent(BaseAgent):
    """Executes a single research subtask via the ReAct tool-calling loop.

    Provides both a standard ``run()`` and a streaming ``stream_run()`` that
    yields intermediate events for UI display.
    """

    @property
    def name(self) -> str:
        return "researcher"

    @property
    def system_prompt(self) -> str:
        return _RESEARCHER_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    async def run(
        self,
        task: str,
        *,
        context: Optional[str] = None,
        max_rounds: Optional[int] = None,
    ) -> AgentResult:
        """Execute a research subtask via the ReAct tool-calling loop.

        Parameters
        ----------
        task : str
            The research question or subtask description.
        context : str, optional
            Extra context (e.g., retrieved documents for grounding).
        max_rounds : int, optional
            Maximum tool-calling rounds (default: config or 8).

        Returns
        -------
        AgentResult with the final researched answer.
        """
        settings = get_settings()
        researcher_model = settings.llm.get_model("researcher")
        _old_llm = getattr(self, "_llm", None)
        if _old_llm is not None:
            from mindforge.models.base import LLMFactory
            self._llm = LLMFactory.create(
                settings.llm.llm_provider, researcher_model
            )
        try:
            return await self._run_tool_loop(
                task,
                context=context,
                max_rounds=max_rounds,
            )
        finally:
            if _old_llm is not None:
                self._llm = _old_llm
            task,
            context=context,
            max_rounds=max_rounds,
        )

    # ------------------------------------------------------------------
    async def stream_run(
        self,
        task: str,
        *,
        context: Optional[str] = None,
        max_rounds: Optional[int] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute research and yield intermediate events for streaming UIs.

        Yields dicts with:
        - ``{"type": "thought", "content": str}`` — LLM reasoning (per round).
        - ``{"type": "tool_call", "name": str, "args": dict}`` — tool being invoked.
        - ``{"type": "tool_result", "name": str, "output": str}`` — tool output.
        - ``{"type": "final_answer", "content": str, "result": AgentResult}`` — done.

        Parameters
        ----------
        task : str
            The research question or subtask description.
        context : str, optional
            Extra context to prepend.
        max_rounds : int, optional
            Maximum tool-calling rounds.
        """
        settings = get_settings()
        max_rounds = max_rounds or settings.agent.max_iterations
        start_time = time.perf_counter()

        # Resolve model (save and restore to avoid mutating shared instance)
        researcher_model = settings.llm.get_model("researcher")
        _old_s_llm = getattr(self, "_llm", None)
        if _old_s_llm is not None:
            from mindforge.models.base import LLMFactory
            self._llm = LLMFactory.create(
                settings.llm.llm_provider, researcher_model
            )

        conv: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt),
        ]
        user_content = task
        if context:
            user_content = f"## Task\n\n{task}\n\n## Context\n\n{context}"
        conv.append(ChatMessage(role="user", content=user_content))

        tool_schemas = self._get_tool_schemas()
        use_tools = bool(tool_schemas)
        aggregated_usage: dict[str, int] = {}

        for _round in range(max_rounds):
            result = await self._chat(
                conv,
                tools=tool_schemas if use_tools else None,
            )

            if result.usage:
                for k, v in result.usage.items():
                    aggregated_usage[k] = aggregated_usage.get(k, 0) + (v or 0)

            # --- Emit thought ---
            if result.content:
                yield {"type": "thought", "content": result.content}

            if not result.tool_calls:
                # --- Final answer ---
                final_content = result.content or ""
                conv.append(ChatMessage(role="assistant", content=final_content))

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                from mindforge.agents.base import _estimate_cost
                cost = _estimate_cost(self._model_name, aggregated_usage)

                agent_result = AgentResult(
                    agent_name=self.name,
                    success=True,
                    output=final_content,
                    data={"rounds": _round + 1, "messages": len(conv)},
                    metadata={"model": self._model_name},
                    token_usage=aggregated_usage,
                    latency_ms=elapsed_ms,
                    cost_usd=cost,
                )
                yield {"type": "final_answer", "content": final_content, "result": agent_result}
                return

            # --- Tool calls ---
            assistant_content = result.content or ""
            conv.append(
                ChatMessage(
                    role="assistant",
                    content=assistant_content,
                    tool_calls=result.tool_calls,
                )
            )

            import asyncio

            for tc in result.tool_calls:
                func = tc.get("function", {})
                yield {
                    "type": "tool_call",
                    "name": func.get("name", ""),
                    "args": func.get("arguments", {}),
                }

            tool_results = await asyncio.gather(
                *[self._execute_tool(tc) for tc in result.tool_calls],
                return_exceptions=True,
            )

            for tc, exec_result in zip(result.tool_calls, tool_results):
                func = tc.get("function", {})
                tool_name = func.get("name", "")

                if isinstance(exec_result, BaseException):
                    output = f"Tool execution error: {exec_result}"
                    conv.append(
                        ChatMessage(role="tool", content=output, tool_call_id="")
                    )
                else:
                    output = exec_result["output"]
                    conv.append(
                        ChatMessage(
                            role="tool",
                            content=output,
                            tool_call_id=exec_result["tool_call_id"],
                        )
                    )

                yield {
                    "type": "tool_result",
                    "name": tool_name,
                    "output": output[:500],
                    "success": not isinstance(exec_result, BaseException) and exec_result.get("success", True),
                }

        # --- Max rounds reached without final answer ---
        # Grab last assistant content
        final_content = ""
        for msg in reversed(conv):
            if msg.role == "assistant":
                final_content = msg.content or ""
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        from mindforge.agents.base import _estimate_cost
        cost = _estimate_cost(self._model_name, aggregated_usage)

        agent_result = AgentResult(
            agent_name=self.name,
            success=True,
            output=final_content,
            data={"rounds": max_rounds, "messages": len(conv)},
            metadata={"model": self._model_name},
            token_usage=aggregated_usage,
            latency_ms=elapsed_ms,
            cost_usd=cost,
        )
        yield {"type": "final_answer", "content": final_content, "result": agent_result}

        # Restore original LLM after stream ends
        if _old_s_llm is not None:
            self._llm = _old_s_llm
