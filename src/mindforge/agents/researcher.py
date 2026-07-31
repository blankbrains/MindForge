"""Researcher agent — executes a ReAct loop with tool access."""

from __future__ import annotations

from typing import Optional

from mindforge.agents.base import AgentResult, BaseAgent

# ---------------------------------------------------------------------------
# ResearcherAgent
# ---------------------------------------------------------------------------

_RESEARCHER_SYSTEM_PROMPT = """你是一名专业的研究助理。你可以使用多种工具来收集信息和验证事实。

可用工具：
- **search_knowledge_base** — 查询内部知识库中的相关文档。
- **web_search** — 搜索网络获取最新信息。
- **code_executor** — 在沙箱中执行 Python 代码，用于计算、数据分析或原型设计。
- **verify_citation** — 验证报告中的引用标记 [N] 是否与来源匹配。

核心原则：
1. **先直接回答**：如果你的知识储备足够回答该问题，直接在 1 轮内给出全面、详细的答案，**不要调用任何工具**。
2. 只有在确实需要外部数据、实时信息或知识库检索时才使用工具。
3. 如果知识库 1-2 次搜索无结果，立即停止搜索，直接用你自己的知识回答。
4. 每次回答要**详尽、深入、结构化**——给出一次性的完整答案，包含具体细节、例证、数据。不要简短敷衍，要写到用户满意为止。复杂问题的回答应达到 800-2000 字。
5. **输出语言必须是中文**（专业术语可保留英文）。
6. 引用来源时使用 [N] 标记。
7. 使用标准 Markdown：标题、段落和列表之间保留空行，每段只表达一个主题。
8. 对比项、参数和统计数据等行列信息使用 GFM 表格；代码块必须标注语言。

记住：你是一个能力强大的模型，拥有广博的知识。优先用你的知识回答，工具只是辅助手段。"""


class ResearcherAgent(BaseAgent):
    """Executes a single research subtask via the ReAct tool-calling loop.

    Streaming is owned by the top-level Orchestrator so the application has
    one authoritative execution path for tool limits, costs, and failures.
    """

    model_role = "researcher"

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
        return await self._run_tool_loop(
            task,
            context=context,
            max_rounds=max_rounds,
        )
