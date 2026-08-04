"""Researcher agent — executes a ReAct loop with tool access."""

from __future__ import annotations

import re
from typing import Literal, Optional

from mindforge.agents.base import AgentResult, BaseAgent
from mindforge.agents.response_guidance import (
    build_response_guidance,
    response_profile,
)

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
4. 回答深度必须与问题复杂度匹配：范围集中的问题先给结论，再充分说明理由、
   选择条件、例外和实践建议；复杂研究继续展开细节、例证与数据，避免无关扩写。
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
    _TRANSFORMATION_RE = re.compile(
        r"^(?:(?:请|请你|帮我|麻烦你)\s*)?"
        r"(?:"
        r"(?:翻译|改写|润色|重写|校对|摘要|总结|压缩|扩写|转换格式)"
        r"(?=\s*(?:以下|下面|这|上述|[:：]))"
        r"|(?:把|将)(?:以下|下面|这|上述).{0,80}"
        r"(?:翻译(?:成|为)|改写(?:为|成|得)|润色|重写(?:为|成)|"
        r"校对|摘要|总结|压缩|扩写|转换为)"
        r")",
        re.IGNORECASE,
    )
    _CREATIVE_RE = re.compile(
        r"^(?:(?:请|请你|帮我)\s*)?"
        r"(?:写|创作|生成|续写|起草)"
        r".{0,30}(?:诗|故事|小说|文案|祝福|歌词|对话|标题|口号)",
        re.IGNORECASE,
    )
    _CODE_GENERATION_RE = re.compile(
        r"^(?:(?:请|请你|帮我)\s*)?"
        r"(?:写|实现|生成|编写|补全)"
        r".{0,40}(?:代码|函数|脚本|程序|算法|正则表达式)",
        re.IGNORECASE,
    )
    _CALCULATION_RE = re.compile(
        r"^(?:(?:请|帮我)\s*)?(?:计算|求值|算一下)\s*"
        r"[-+*/%().\d\s×÷^]+[？?]?$",
        re.IGNORECASE,
    )
    _TASK_TYPE_GUIDANCE = {
        "research": "检索并整理可核验事实。",
        "analysis": "基于已有证据进行分析，不重复生成最终综合报告。",
        "code": "优先使用 code_executor 验证实现和边界条件。",
        "verify": "核对事实、来源或引用，明确给出验证结论。",
    }
    _REQUIRED_SOURCE_MARKERS = (
        "联网",
        "网页",
        "官网",
        "官方网站",
        "最新",
        "当前版本",
        "实时",
        "今天",
        "近期",
        "新闻",
        "发布日期",
        "价格",
        "行情",
        "引用",
        "来源",
        "出处",
        "核对",
        "核验",
        "查证",
        "验证事实",
        "知识库",
        "上传的文档",
        "根据文档",
        "文献综述",
        "研究报告",
        "web only",
        "online source",
        "official website",
        "latest",
        "current version",
        "citation",
        "verify",
        "fact check",
    )

    @property
    def name(self) -> str:
        return "researcher"

    @property
    def system_prompt(self) -> str:
        return _RESEARCHER_SYSTEM_PROMPT

    @classmethod
    def requires_sources(
        cls,
        task: str,
        *,
        task_type: str = "research",
    ) -> bool:
        normalized = " ".join(task.strip().casefold().split())
        if not normalized or normalized in cls._CONVERSATIONAL_TASKS:
            return False
        if task_type == "code":
            return False
        return not any(
            pattern.search(normalized)
            for pattern in (
                cls._TRANSFORMATION_RE,
                cls._CREATIVE_RE,
                cls._CODE_GENERATION_RE,
                cls._CALCULATION_RE,
            )
        )

    @classmethod
    def response_length_guidance(
        cls,
        task: str,
        *,
        task_type: str = "research",
        subtask_count: int = 1,
        total_subtasks: int = 1,
    ) -> str:
        profile = response_profile(
            task,
            task_type=task_type,
            subtask_count=subtask_count,
        )
        if total_subtasks > 1 and profile.depth not in {"concise", "code"}:
            budget = (
                "1200-2200"
                if profile.depth == "deep"
                else "800-1600"
            )
            return (
                "这是最终综合报告的一个证据子任务。只输出可供 Synthesizer "
                "使用的高密度事实、依据、适用条件、限制和必要例外，不写执行摘要、"
                "总括结论或重复背景；使用与证据结构匹配的小节或列表。"
                f"正文通常控制在 {budget} 个中文字符，信息完整后立即停止。"
            )
        return build_response_guidance(
            task,
            task_type=task_type,
            subtask_count=subtask_count,
        )

    @classmethod
    def source_requirement(
        cls,
        task: str,
        *,
        task_type: str = "research",
        source_policy: str = "auto",
    ) -> Literal["not_required", "preferred", "required"]:
        if not cls.requires_sources(task, task_type=task_type):
            return "not_required"
        normalized_policy = source_policy.strip().lower()
        if normalized_policy in {"web", "knowledge_base"}:
            return "required"
        if task_type == "verify":
            return "required"
        normalized = " ".join(task.strip().casefold().split())
        if any(
            marker in normalized
            for marker in cls._REQUIRED_SOURCE_MARKERS
        ):
            return "required"
        return "preferred"

    # ------------------------------------------------------------------
    async def run(
        self,
        task: str,
        *,
        context: Optional[str] = None,
        max_rounds: Optional[int] = None,
        task_type: str = "research",
        subtopics: Optional[list[str]] = None,
        total_subtasks: int = 1,
        deadline: float | None = None,
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
        task_type : str
            Planner-provided execution type.
        subtopics : list[str], optional
            Planner-provided search directions or verification angles.
        deadline : float, optional
            Absolute ``time.perf_counter()`` deadline for this subtask.

        Returns
        -------
        AgentResult with the final researched answer.
        """
        guidance = self._TASK_TYPE_GUIDANCE.get(
            task_type,
            self._TASK_TYPE_GUIDANCE["research"],
        )
        normalized_subtopics = [
            subtopic.strip()
            for subtopic in (subtopics or [])
            if isinstance(subtopic, str) and subtopic.strip()
        ]
        depth_guidance = self.response_length_guidance(
            task,
            task_type=task_type,
            subtask_count=1,
            total_subtasks=total_subtasks,
        )
        execution_context = [
            f"## 子任务类型\n\n{task_type}: {guidance}",
            f"## 回答深度\n\n{depth_guidance}",
        ]
        if normalized_subtopics:
            execution_context.append(
                "## 研究方向\n\n"
                + "\n".join(f"- {subtopic}" for subtopic in normalized_subtopics)
            )
        if context:
            execution_context.append(context)

        requirement = self.source_requirement(
            task,
            task_type=task_type,
            source_policy=str(
                getattr(
                    getattr(
                        getattr(self, "_settings", None),
                        "agent",
                        None,
                    ),
                    "source_policy",
                    "auto",
                )
            ),
        )
        return await self._run_tool_loop(
            task,
            context="\n\n".join(execution_context),
            max_rounds=max_rounds,
            require_sources=requirement != "not_required",
            source_requirement=requirement,
            deadline=deadline,
        )
