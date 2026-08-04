"""Critic agent — evaluates research quality using LLM-as-Judge."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from mindforge.agents.base import BaseAgent, _estimate_cost_details
from mindforge.models.base import ChatMessage, ChatResult
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
    cost_usd: float | None = None
    cost_status: str = "usage_unavailable"
    evaluation_status: Literal["evaluated", "failed"] = "evaluated"
    evaluation_error: str | None = None

    @classmethod
    def from_dict(cls, data: dict, *, threshold: float | None = None) -> CriticScore:
        """从 dict 构造 CriticScore。

        ``should_refine`` 由外部 ``evaluate()`` 根据 ``threshold`` 统一判定，
        ``from_dict`` 不参与阈值逻辑 — 这里仅保留 LLM JSON 中显式声明的值。
        """
        scores = data.get("scores", data)
        issues = data.get("issues", data.get("weaknesses", []))
        suggestions = data.get("suggestions", data.get("improvements", []))

        if not isinstance(scores, dict):
            scores = {}
        overall = _bounded_score(scores.get("overall", 0))

        return cls(
            completeness=_bounded_score(scores.get("completeness", 0)),
            accuracy=_bounded_score(scores.get("accuracy", 0)),
            depth=_bounded_score(scores.get("depth", 0)),
            clarity=_bounded_score(scores.get("clarity", 0)),
            citations=_bounded_score(scores.get("citations", 0)),
            overall=overall,
            issues=_bounded_text_list(issues),
            suggestions=_bounded_text_list(suggestions),
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
            "evaluation_status": self.evaluation_status,
            "evaluation_error": self.evaluation_error,
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

    model_role = "critic"

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
        report_context_max_chars = int(
            getattr(
                settings.agent,
                "critic_report_context_max_chars",
                60_000,
            )
        )
        bounded_draft = draft[:report_context_max_chars]
        if len(draft) > report_context_max_chars:
            bounded_draft = (
                bounded_draft.rstrip()
                + "\n\n[报告正文因评审上下文预算已截断]"
            )

        # Build the evaluation prompt
        src_text = ""
        if sources:
            remaining = int(
                getattr(
                    settings.agent,
                    "critic_source_context_max_chars",
                    12_000,
                )
            )
            src_lines = ["可核验来源证据："]
            for i, s in enumerate(sources, 1):
                if remaining <= 0:
                    break
                index = s.get("index", i)
                title = s.get("title", s.get("source", f"Source {i}"))
                url = str(s.get("url") or "").strip()
                evidence = str(
                    s.get("content")
                    or s.get("text")
                    or s.get("snippet")
                    or ""
                ).strip()
                source_text = f"[{index}] {title}"
                if url:
                    source_text += f"\nURL: {url}"
                if evidence:
                    source_text += f"\n证据: {evidence}"
                bounded = source_text[:remaining]
                src_lines.append(bounded)
                remaining -= len(bounded)
            src_text = "\n".join(src_lines)

        user_prompt = (
            f"## 原始任务\n\n{task}\n\n"
            f"## 研究报告草稿\n\n{bounded_draft}\n\n"
            f"{src_text}\n\n"
            "请使用 5 个维度对草稿进行评估。返回包含 scores、issues 和 suggestions 的 JSON。"
            "**issues 和 suggestions 必须用中文撰写。**"
        )

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        result: ChatResult | None = None
        try:
            result = await self._chat(
                messages,
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            raw = result.content.strip()
            score_dict = json.loads(raw)
            _validate_score_payload(score_dict)
            score = CriticScore.from_dict(score_dict)
            score.token_usage = result.usage or {}
            cost_estimate = _estimate_cost_details(
                result.model or self._model_name,
                score.token_usage,
                self._provider_name,
            )
            score.cost_usd = cost_estimate.amount_usd
            score.cost_status = cost_estimate.status

            # Apply threshold
            if threshold is not None:
                score.should_refine = (
                    score.should_refine or score.overall < threshold
                )

            return score

        except Exception as exc:
            usage = result.usage if result is not None else {}
            model_used = (
                result.model
                if result is not None and result.model
                else self._model_name
            )
            cost_estimate = _estimate_cost_details(
                model_used,
                usage,
                self._provider_name,
            )
            error_message = str(exc).strip() or type(exc).__name__
            # 评估失败不伪造分数，同时避免 Critic 自身故障触发精炼循环。
            return CriticScore(
                issues=[f"质量评审失败：{error_message}"],
                suggestions=["请检查评审模型配置或稍后重新评审。"],
                should_refine=False,
                token_usage=usage,
                cost_usd=cost_estimate.amount_usd,
                cost_status=cost_estimate.status,
                evaluation_status="failed",
                evaluation_error=error_message[:1000],
            )
def _bounded_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return min(10.0, max(0.0, score))


def _validate_score_payload(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Critic response must be a JSON object.")
    scores = data.get("scores", data)
    if not isinstance(scores, dict):
        raise ValueError("Critic response must contain a scores object.")
    required = (
        "completeness",
        "accuracy",
        "depth",
        "clarity",
        "citations",
        "overall",
    )
    missing = [name for name in required if name not in scores]
    if missing:
        raise ValueError(
            "Critic response is missing required scores: "
            + ", ".join(missing)
        )
    for name in required:
        value = scores[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Critic score '{name}' must be numeric.")
        if not math.isfinite(float(value)):
            raise ValueError(f"Critic score '{name}' must be finite.")


def _bounded_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item[:2000]
        for item in value
        if isinstance(item, str) and item.strip()
    ][:20]
