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

_RESEARCHER_SYSTEM_PROMPT = """你是一名专业的研究助理。你可以使用多种工具来收集信息和验证事实。

可用工具：
- **search_knowledge_base** — 查询内部知识库中的相关文档。
- **web_search** — 搜索网络获取最新信息。
- **code_executor** — 在沙箱中执行 Python 代码，用于计算、数据分析或原型设计。
- **verify_citation** — 验证报告中的引用标记 [N] 是否与来源匹配。
- **mcp_tool** — 访问外部 MCP（模型上下文协议）工具。

核心原则：
1. **先直接回答**：如果你的知识储备足够回答该问题，直接在 1 轮内给出全面、详细的答案，**不要调用任何工具**。
2. 只有在确实需要外部数据、实时信息或知识库检索时才使用工具。
3. 如果知识库 1-2 次搜索无结果，立即停止搜索，直接用你自己的知识回答。
4. 每次回答要**详尽、深入、结构化**——给出一次性的完整答案，包含具体细节、例证、数据。不要简短敷衍，要写到用户满意为止。复杂问题的回答应达到 800-2000 字。
5. **输出语言必须是中文**（专业术语可保留英文）。
6. 引用来源时使用 [N] 标记。

记住：你是一个能力强大的模型，拥有广博的知识。优先用你的知识回答，工具只是辅助手段。"""


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
        from mindforge.models.base import LLMFactory
        _llm_override = LLMFactory.create(
            settings.llm.llm_provider, researcher_model
        )
        return await self._run_tool_loop(
            task,
            context=context,
            max_rounds=max_rounds,
            _llm_override=_llm_override,
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

        # Resolve model override (no self._llm mutation → generator-safe)
        researcher_model = settings.llm.get_model("researcher")
        from mindforge.models.base import LLMFactory
        _llm_override = LLMFactory.create(
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
                _llm_override=_llm_override,
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
