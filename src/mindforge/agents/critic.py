"""Critic agent — evaluates research quality using LLM-as-Judge."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from mindforge.agents.base import BaseAgent, _estimate_cost_details
from mindforge.agents.contracts import (
    AGENT_CONTRACT_VERSION,
    role_contract,
)
from mindforge.agents.response_guidance import response_profile
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
    contract_violations: list[str] = field(default_factory=list)
    contract_version: str = AGENT_CONTRACT_VERSION

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
            "contract_violations": self.contract_violations,
            "contract_version": self.contract_version,
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
- 最多 3 个具体、有序的问题。
- 最多 3 个可操作的改进建议。

评分必须以原始问题实际需要的回答深度为准。范围集中的问题不应因为没有写成长篇
研究报告而降低 completeness 或 depth；只要直接回答、关键依据、必要条件和限制已经
覆盖，就应视为深度匹配。不得为了偏好更长文本而惩罚简洁、完整的答案。

评审的第一步必须核对报告主题与原始任务是否一致。如果报告没有回答原始任务的核心
对象或核心概念，即使报告自身结构完整、内容丰富，也必须将 completeness 评为 0-2、
overall 评为 0-3，并将 should_refine 设为 true。不得把“内部写得好”误判为“回答正确”。

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
}""" + "\n\n" + role_contract("critic")


class CriticAgent(BaseAgent):
    """LLM-as-Judge evaluator. Scores a research draft across 5 dimensions."""

    model_role = "critic"
    deepseek_thinking_mode = "disabled"

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
        profile = response_profile(task)
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
                elif s.get("verification_mode") == "provider_native":
                    source_text += (
                        "\n证据类型: 模型供应商原生搜索返回的来源 URL 映射，"
                        "未附网页正文摘录；不要将其视为重复正文或完整页面证据。"
                    )
                bounded = source_text[:remaining]
                src_lines.append(bounded)
                remaining -= len(bounded)
            src_text = "\n".join(src_lines)

        direct_answer_source_guidance = (
            ""
            if sources
            else (
                "本次是无需外部检索的稳定知识直接回答。引用维度应评估是否"
                "诚实地没有伪造来源，不得仅因没有 [N] 引用而扣分。\n\n"
            )
        )
        user_prompt = (
            f"## 原始任务\n\n{task}\n\n"
            f"## 预期回答深度\n\n{profile.depth}: {profile.guidance()}\n\n"
            f"## 研究报告草稿\n\n{bounded_draft}\n\n"
            f"{src_text}\n\n"
            f"{direct_answer_source_guidance}"
            "请按上述预期深度使用 5 个维度评估草稿。返回包含 scores、issues 和 "
            "suggestions 的 JSON；issues 和 suggestions 各自最多 3 项。"
            "**issues 和 suggestions 必须用中文撰写。**"
        )

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        result: ChatResult | None = None
        total_usage: dict[str, int] = {}
        last_parse_error: Exception | None = None
        try:
            score_dict: dict[str, Any] | None = None
            for attempt in range(3):
                result = await self._chat(
                    messages,
                    response_format=(
                        {"type": "json_object"}
                        if attempt != 1
                        else None
                    ),
                    temperature=0.2,
                    max_output_tokens=min(
                        int(
                            getattr(
                                settings.agent,
                                "critic_max_output_tokens",
                                2_400,
                            )
                        ),
                        1_200,
                    ),
                )
                for key, value in (result.usage or {}).items():
                    if isinstance(value, (int, float)) and not isinstance(
                        value,
                        bool,
                    ):
                        total_usage[key] = total_usage.get(key, 0) + int(value)
                raw = result.content.strip()
                try:
                    score_dict = _parse_score_payload(raw)
                    break
                except ValueError as exc:
                    last_parse_error = exc
                    if attempt >= 2:
                        raise
                    if raw:
                        messages.append(
                            ChatMessage(role="assistant", content=raw)
                        )
                    messages.append(
                        ChatMessage(
                            role="user",
                            content=(
                                "上一份评审结果为空或不是符合约定结构的合法 JSON。"
                                "请重新评审，并且只返回包含完整 scores、issues、"
                                "suggestions 和 should_refine 的 JSON 对象。"
                            ),
                        )
                    )
            if score_dict is None:
                raise last_parse_error or ValueError(
                    "Critic did not return a valid score payload."
                )
            score = CriticScore.from_dict(score_dict)
            score.token_usage = total_usage
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
            _apply_contract_guardrails(
                score,
                draft=draft,
                sources=sources or [],
                expected_min_chars=profile.min_chars,
                expected_depth=profile.depth,
            )

            return score

        except Exception as exc:
            usage = total_usage or (
                result.usage if result is not None else {}
            )
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


def _apply_contract_guardrails(
    score: CriticScore,
    *,
    draft: str,
    sources: list[dict[str, Any]],
    expected_min_chars: int | None,
    expected_depth: str,
) -> None:
    """Apply deterministic checks that an LLM judge cannot override."""

    def add_violation(
        code: str,
        issue: str,
        suggestion: str,
    ) -> None:
        if code not in score.contract_violations:
            score.contract_violations.append(code)
        score.issues = [
            issue,
            *[item for item in score.issues if item != issue],
        ][:3]
        score.suggestions = [
            suggestion,
            *[
                item
                for item in score.suggestions
                if item != suggestion
            ],
        ][:3]
        score.should_refine = True

    if (
        expected_depth == "deep"
        and expected_min_chars is not None
        and len(draft.strip())
        < max(1_600, int(expected_min_chars * 0.75))
    ):
        score.completeness = min(score.completeness, 5.5)
        score.depth = min(score.depth, 5.0)
        score.overall = min(score.overall, 5.5)
        add_violation(
            "insufficient_depth",
            "回答篇幅和覆盖范围明显低于用户要求的深度。",
            "补充核心维度、关键条件、限制、例证和实践建议，避免重复凑字。",
        )

    markers = {
        int(value)
        for value in re.findall(r"\[([1-9]\d*)\]", draft)
    }
    source_indices = {
        int(source["index"])
        for source in sources
        if isinstance(source.get("index"), int)
        and not isinstance(source.get("index"), bool)
    }
    if sources and not markers:
        score.citations = min(score.citations, 2.0)
        score.overall = min(score.overall, 6.0)
        add_violation(
            "missing_citations",
            "提供了来源，但正文没有使用任何 [N] 引用标记。",
            "在对应事实性主张后补充有效的全局 [N] 引用。",
        )
    invalid_markers = sorted(markers - source_indices)
    if invalid_markers:
        score.accuracy = min(score.accuracy, 5.0)
        score.citations = 0.0
        score.overall = min(score.overall, 5.0)
        rendered = "、".join(f"[{index}]" for index in invalid_markers[:5])
        add_violation(
            "invalid_citations",
            f"正文包含无法映射到来源列表的引用：{rendered}。",
            "删除无效引用，或改用来源列表中实际存在的全局编号。",
        )
    if not sources and markers:
        score.accuracy = min(score.accuracy, 5.0)
        score.citations = 0.0
        score.overall = min(score.overall, 5.0)
        add_violation(
            "unbacked_citations",
            "正文包含引用标记，但本次没有提供任何可核验来源。",
            "删除伪引用；如任务确实需要证据，应先进入研究链路获取来源。",
        )


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


def _parse_score_payload(raw: str) -> dict[str, Any]:
    """Extract and validate one critic JSON object from model text."""
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Critic response was empty.")

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        try:
            _validate_score_payload(value)
        except ValueError:
            continue
        return value
    raise ValueError("Critic response did not contain a valid score payload.")


def _bounded_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item[:2000]
        for item in value
        if isinstance(item, str) and item.strip()
    ][:20]
