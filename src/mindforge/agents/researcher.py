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
1. **事实型问题先检索后回答**：定义、解释、比较、技术分析、研究和时效性问题，
   必须至少调用一次 `search_knowledge_base` 或 `web_search` 获取真实来源。
2. 优先搜索知识库；知识库没有高度相关资料且 `web_search` 可用时，再搜索网页。
3. 只有寒暄、纯创作、翻译、改写或无需外部事实的计算任务可以不调用来源工具。
4. 每次回答要**详尽、深入、结构化**——给出一次性的完整答案，包含具体细节、例证、数据。不要简短敷衍，要写到用户满意为止。复杂问题的回答应达到 800-2000 字。
5. **输出语言必须是中文**（专业术语可保留英文）。
6. 只要工具返回了来源，正文中的对应事实必须使用工具提供的全局 [N] 编号，
   不得省略引用，也不得编造来源。
7. 使用标准 Markdown：标题、段落和列表之间保留空行，每段只表达一个主题。
8. 对比项、参数和统计数据等行列信息使用 GFM 表格；代码块必须标注语言。

记住：来源工具提供可核验的证据；模型知识只能用于解释和组织，不能替代真实引用。"""


class ResearcherAgent(BaseAgent):
    """Executes a single research subtask via the ReAct tool-calling loop.

    Streaming is owned by the top-level Orchestrator so the application has
    one authoritative execution path for tool limits, costs, and failures.
    """

    model_role = "researcher"
    _CONVERSATIONAL_TASKS = frozenset(
        {
            "hi",
            "hello",
            "hey",
            "你好",
            "你好啊",
            "你好呀",
            "您好",
            "嗨",
            "在吗",
            "早上好",
            "上午好",
            "下午好",
            "晚上好",
            "谢谢",
            "多谢",
            "再见",
            "你是谁",
            "你叫什么",
        }
    )

    @property
    def name(self) -> str:
        return "researcher"

    @property
    def system_prompt(self) -> str:
        return _RESEARCHER_SYSTEM_PROMPT

    @classmethod
    def requires_sources(cls, task: str) -> bool:
        normalized = " ".join(task.strip().casefold().split())
        return bool(normalized) and normalized not in cls._CONVERSATIONAL_TASKS

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
            require_sources=self.requires_sources(task),
        )
