"""Critic agent — evaluates research quality using LLM-as-Judge."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from mindforge.agents.base import BaseAgent
from mindforge.models.base import ChatMessage
from mindforge.config import get_settings


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CriticScore:
    """Multi-dimensional quality assessment of a research draft."""

    completeness: float = 0.0  # 0-10
    accuracy: float = 0.0      # 0-10
    depth: float = 0.0         # 0-10
    clarity: float = 0.0       # 0-10
    citations: float = 0.0     # 0-10
    overall: float = 0.0       # 0-10
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    should_refine: bool = False
    token_usage: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict, *, threshold: float | None = None) -> CriticScore:
        """从 dict 构造 CriticScore。

        ``should_refine`` 由外部 ``evaluate()`` 根据 ``threshold`` 统一判定，
        ``from_dict`` 不参与阈值逻辑 — 这里仅保留 LLM JSON 中显式声明的值。
        """
        scores = data.get("scores", data)
        issues = data.get("issues", data.get("weaknesses", []))
        suggestions = data.get("suggestions", data.get("improvements", []))

        overall = float(scores.get("overall", 0))

        return cls(
            completeness=float(scores.get("completeness", 0)),
            accuracy=float(scores.get("accuracy", 0)),
            depth=float(scores.get("depth", 0)),
            clarity=float(scores.get("clarity", 0)),
            citations=float(scores.get("citations", 0)),
            overall=overall,
            issues=issues if isinstance(issues, list) else [],
            suggestions=suggestions if isinstance(suggestions, list) else [],
            should_refine=bool(data.get("should_refine", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "depth": self.depth,
            "clarity": self.clarity,
            "citations": self.citations,
            "overall": self.overall,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "should_refine": self.should_refine,
        }


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM_PROMPT = """你是一名专业的研究评审员。你的任务是对研究报告的质量进行批判性评估。

从以下 5 个维度对报告进行评分（每项 0-10 分）：

1. **completeness（完整性）** — 是否完全回答了原始问题？所有方面都覆盖了吗？
2. **accuracy（准确性）** — 事实和主张是否正确且有充分支撑？
3. **depth（深度）** — 分析是否超出表面层面？是否有有意义的洞察？
4. **clarity（清晰性）** — 报告结构是否良好、可读且易于理解？
5. **citations（引用质量）** — 主张是否正确使用 [N] 标记进行了引用？

对每个维度提供：
- 数值评分（0-10，10 为满分）。
- 你发现的具体问题或空白。
- 可操作的改进建议。

最后提供：
- **overall** 总分（0-10）。
- 布尔值 **should_refine**——如果总分 < 7.0 或存在严重问题则为 True。
- 具体的、有序的问题列表。
- 可操作的改进建议列表。

**只返回合法的 JSON——不要加 markdown、代码块或注释。issues 和 suggestions 的内容必须用中文写。**

输出格式：
{
  "scores": {
    "completeness": 7,
    "accuracy": 8,
    "depth": 6,
    "clarity": 9,
    "citations": 5,
    "overall": 7.0
  },
  "issues": ["问题 1：...", "问题 2：..."],
  "suggestions": ["建议 1：...", "建议 2：..."],
  "should_refine": true
}"""


class CriticAgent(BaseAgent):
    """LLM-as-Judge evaluator. Scores a research draft across 5 dimensions."""

    @property
    def name(self) -> str:
        return "critic"

    @property
    def system_prompt(self) -> str:
        return _CRITIC_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    async def evaluate(
        self,
        task: str,
        draft: str,
        sources: Optional[list[dict[str, Any]]] = None,
        *,
        threshold: Optional[float] = None,
    ) -> CriticScore:
        """Evaluate a research draft against the original task.

        Parameters
        ----------
        task : str
            The original task or question the report was supposed to answer.
        draft : str
            The research report / draft to evaluate.
        sources : list[dict], optional
            List of source definitions used in the report, for citation validation.
        threshold : float, optional
            The cut-off for ``should_refine`` (default: from config, 7.0).

        Returns
        -------
        CriticScore with dimension scores, issues, and suggestions.
        """
        settings = get_settings()
        threshold = threshold if threshold is not None else settings.agent.critic_threshold

        # Use the critic-specific model from config (via _llm_override 保证协程安全)
        critic_model = settings.llm.get_model("critic")
        from mindforge.models.base import LLMFactory
        _llm_override = LLMFactory.create(
            settings.llm.llm_provider, critic_model
        )

        # Build the evaluation prompt
        src_text = ""
        if sources:
            src_lines = ["Available sources:"]
            for i, s in enumerate(sources, 1):
                title = s.get("title", s.get("source", f"Source {i}"))
                src_lines.append(f"  [{i}] {title}")
            src_text = "\n".join(src_lines)

        user_prompt = (
            f"## 原始任务\n\n{task}\n\n"
            f"## 研究报告草稿\n\n{draft}\n\n"
            f"{src_text}\n\n"
            "请使用 5 个维度对草稿进行评估。返回包含 scores、issues 和 suggestions 的 JSON。"
            "**issues 和 suggestions 必须用中文撰写。**"
        )

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        try:
            result = await self._chat(
                messages,
                response_format={"type": "json_object"},
                temperature=0.2,
                _llm_override=_llm_override,
            )

            raw = result.content.strip()
            score_dict = json.loads(raw)
            score = CriticScore.from_dict(score_dict)
            score.token_usage = result.usage or {}

            # Apply threshold
            if threshold is not None:
                score.should_refine = score.overall < threshold

            return score

        except Exception as exc:
            # 评估失败时返回中性分数，should_refine=False
            # 避免 critic 自身故障触发无意义的精炼循环
            return CriticScore(
                completeness=5.0,
                accuracy=5.0,
                depth=5.0,
                clarity=5.0,
                citations=5.0,
                overall=5.0,
                issues=[f"Critic evaluation failed: {exc}"],
                suggestions=["Manual review recommended."],
                should_refine=False,
            )
