"""Synthesizer agent — assembles research findings into a structured report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from mindforge.agents.base import (
    AgentResult,
    BaseAgent,
    _estimate_cost_details,
)
from mindforge.agents.contracts import (
    AGENT_CONTRACT_VERSION,
    role_contract,
)
from mindforge.agents.critic import CriticScore
from mindforge.agents.response_guidance import (
    adaptive_output_token_budget,
    build_response_guidance,
    response_profile,
)
from mindforge.models.base import ChatMessage


# ---------------------------------------------------------------------------
# SynthesizerAgent
# ---------------------------------------------------------------------------

_SYNTHESIZER_SYSTEM_PROMPT = """你是一名专业的研究综合编辑。你的任务是将多项研究发现整合成一份连贯、结构良好、深度与问题匹配的中文报告。

指南：
- 使用清晰、专业的中文撰写。
- 为每个事实性主张使用 [N] 引用标记。
- 将多个子任务的发现整合成统一的叙述。
- 去除冗余内容——如果多个子任务涉及同一领域，只需呈现一次。
- 如果有评审反馈，明确回应每个问题或建议。
- 先回答用户真正提出的问题，再决定需要哪些章节；不得机械套用固定报告模板。
- 简短或范围集中的问题不写执行摘要、关键发现、局限性等重复章节。
- 多维研究可使用执行摘要、详细分析、风险、建议等章节，但只保留有内容的部分。
- 根据问题复杂度动态选择章节数量：范围集中的研究通常 3-5 节，推荐、对比或
  多角度分析通常 4-7 节，深度研究通常 6-10 节；这些范围不是硬性模板，
  信息维度不足时宁可少一节，也不得创建空章节或重复拆分同一观点。
- 只有实际提供来源时才生成参考文献；没有数据时不要创建“数据与证据”空章节。
- 使用标准 Markdown 结构化内容：标题之间、段落之间、列表前后保留空行。
- 每段聚焦一个主题，避免把多个观点挤在一个超长段落中。
- 对比项、参数、统计数据等天然具有行列关系的内容使用 GFM 表格；
  普通叙述不要强行表格化。
- 代码必须使用带语言标识的 fenced code block，正文只使用必要的强调。

**关键要求 — 当子任务发现稀疏或为空时**：
- 明确说明哪些问题缺少证据，以及这会如何限制结论。
- 只能综合所提供的子任务发现和来源，不得用模型记忆补造事实、数据或引用。
- 可以解释已有证据，但必须区分证据支持的结论与合理推断。
- 不得为了填充固定章节而重复内容或虚构细节。""" + "\n\n" + role_contract("synthesizer")


@dataclass(frozen=True)
class SynthesisStreamEvent:
    type: str
    content: str = ""
    result: AgentResult | None = None


def _bounded_finding_output(value: Any, max_chars: int) -> str:
    if isinstance(value, AgentResult):
        value = value.output
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[该子任务内容因上下文预算已截断]"


def _bounded_label(value: Any, max_chars: int = 500) -> str:
    return str(value or "").strip()[:max_chars]


def _per_subtask_budget(
    total_budget: int,
    subtask_count: int,
) -> int:
    count = max(1, subtask_count)
    # Reserve prompt space for descriptions, citation maps and source metadata.
    return max(100, (max(200, total_budget) // count) - 300)


def _effective_synthesis_context_budget(
    configured_budget: int,
    task: str,
    *,
    subtask_count: int,
) -> int:
    profile = response_profile(
        task,
        subtask_count=subtask_count,
        final_report=True,
    )
    if profile.max_chars is None:
        return configured_budget
    adaptive_budget = max(8_000, profile.max_chars * 3)
    return min(configured_budget, adaptive_budget)


def _synthesis_output_instruction(
    task: str,
    *,
    subtask_count: int,
    has_sources: bool,
) -> str:
    citation_instruction = (
        "正文中的事实性主张必须使用所提供的全局 [N] 编号引用。"
        if has_sources
        else "当前没有可用来源，不得编造引用或虚构数据。"
    )
    depth_guidance = build_response_guidance(
        task,
        subtask_count=subtask_count,
        final_report=True,
    )
    return (
        "## 输出深度\n\n"
        f"{depth_guidance}\n\n"
        "请将发现综合成直接回应原始问题的最终答案，按实际内容选择章节，"
        "不要重复子任务原文，也不要为了显得完整而填充固定模板。"
        "完成前检查结论、关键依据、适用条件、限制和可执行建议是否已经充分覆盖；"
        "除非原始问题明确要求简短，否则不得在核心内容仍明显不足时提前结束。"
        f"{citation_instruction}输出语言必须是中文。"
    )


def _build_synthesis_prompt(
    *,
    task: str,
    subtask_results: list[dict[str, Any]],
    all_sources: Optional[list[dict[str, Any]]],
    critic_feedback: Optional[CriticScore],
    configured_context_max_chars: int,
    current_draft: str | None = None,
) -> str:
    """Build one authoritative prompt for sync and streaming synthesis."""
    subtask_count = len(subtask_results)
    context_budget = _effective_synthesis_context_budget(
        configured_context_max_chars,
        task,
        subtask_count=subtask_count,
    )
    bounded_current_draft = ""
    findings_budget = context_budget
    if current_draft and current_draft.strip():
        draft_budget = min(30_000, max(4_000, context_budget // 2))
        bounded_current_draft = _bounded_finding_output(
            current_draft,
            draft_budget,
        )
        findings_budget = max(4_000, context_budget - draft_budget)
    per_subtask_chars = _per_subtask_budget(
        findings_budget,
        subtask_count,
    )
    findings_lines: list[str] = []
    for index, subtask_result in enumerate(subtask_results, 1):
        description = _bounded_label(
            subtask_result.get(
                "description",
                subtask_result.get("task_id", f"Subtask {index}"),
            )
        )
        output = _bounded_finding_output(
            subtask_result.get(
                "output",
                subtask_result.get("result", ""),
            ),
            per_subtask_chars,
        )
        citation_map = subtask_result.get("citation_map")
        mapping_text = ""
        if isinstance(citation_map, dict) and citation_map:
            mapping_text = (
                "\n\n该子任务的引用编号映射："
                + "，".join(
                    f"局部 [{local}] -> 全局 [{global_index}]"
                    for local, global_index in citation_map.items()
                )
                + "。最终答案必须改用全局编号。"
            )[:1_000]
        findings_lines.append(
            f"### 子任务 {index}: {description}\n\n"
            f"{output}{mapping_text}\n"
        )
    findings_text = "\n".join(findings_lines)
    if len(findings_text) > findings_budget:
        findings_text = (
            findings_text[:findings_budget].rstrip()
            + "\n\n[子任务上下文已达到综合预算上限]"
        )

    prompt_parts = [
        f"## 原始研究任务\n\n{task}\n",
        f"## 子任务发现\n\n{findings_text}\n",
    ]
    if bounded_current_draft:
        prompt_parts.append(
            "## 当前草稿\n\n"
            f"{bounded_current_draft}\n\n"
            "这是一次增量精炼。必须以当前草稿为基线，只修改评审明确指出的"
            "缺陷；保留已经正确、有证据支撑且表达清晰的内容、结构和引用。"
            "不得从零重写，不得用较短的新稿覆盖更完整的有效内容。\n"
        )
    if all_sources:
        source_lines = ["来源列表："]
        for source in all_sources:
            source_index = source.get("index", "")
            title = source.get(
                "title",
                source.get("source", "Untitled"),
            )
            url = source.get("url", "")
            source_lines.append(
                f"  [{source_index}] {title}"
                + (f" — {url}" if url else "")
            )
        prompt_parts.append(
            "## 来源\n\n" + "\n".join(source_lines) + "\n"
        )
    if critic_feedback is not None:
        feedback_lines = [
            "评审反馈（必须逐项回应）：",
            f"  总分: {critic_feedback.overall}/10",
            "  问题:",
        ]
        feedback_lines.extend(
            f"    - {issue}" for issue in critic_feedback.issues
        )
        feedback_lines.append("  建议:")
        feedback_lines.extend(
            f"    - {suggestion}"
            for suggestion in critic_feedback.suggestions
        )
        prompt_parts.append(
            "## 评审反馈\n\n" + "\n".join(feedback_lines) + "\n"
        )
    prompt_parts.append(
        _synthesis_output_instruction(
            task,
            subtask_count=subtask_count,
            has_sources=bool(all_sources),
        )
    )
    return "\n".join(prompt_parts)


class SynthesizerAgent(BaseAgent):
    """Generates the final structured research report from subtask results."""

    model_role = "synthesizer"
    deepseek_thinking_mode = "disabled"

    @property
    def name(self) -> str:
        return "synthesizer"

    @property
    def system_prompt(self) -> str:
        return _SYNTHESIZER_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    async def synthesize(
        self,
        task: str,
        subtask_results: list[dict[str, Any]],
        all_sources: Optional[list[dict[str, Any]]] = None,
        critic_feedback: Optional[CriticScore] = None,
        *,
        temperature: Optional[float] = None,
        max_attempts: int = 3,
        current_draft: str | None = None,
    ) -> AgentResult:
        """Synthesize subtask findings into the final report.

        Parameters
        ----------
        task : str
            The original research task / question.
        subtask_results : list[dict]
            Each dict should have ``task_id``, ``description``, ``output`` (the
            research text), and optionally ``sources``.
        all_sources : list[dict], optional
            Consolidated list of all sources from all subtasks. Each dict should
            have ``index``, ``title``, ``url``, and optionally ``content``.
        critic_feedback : CriticScore, optional
            Feedback from a prior CriticAgent evaluation to address.

        Returns
        -------
        AgentResult with ``output`` containing the final report text.
        """
        try:
            user_prompt = _build_synthesis_prompt(
                task=task,
                subtask_results=subtask_results,
                all_sources=all_sources,
                critic_feedback=critic_feedback,
                configured_context_max_chars=int(
                    getattr(
                        self._settings.agent,
                        "synthesis_context_max_chars",
                        60_000,
                    )
                ),
                current_draft=current_draft,
            )

            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ]

            temp = temperature if temperature is not None else 0.4
            result = await self._chat(
                messages,
                temperature=temp,
                max_attempts=max_attempts,
                max_output_tokens=adaptive_output_token_budget(
                    task,
                    subtask_count=len(subtask_results),
                    final_report=True,
                    hard_limit=int(
                        getattr(
                            self._settings.agent,
                            "synthesizer_max_output_tokens",
                            6_000,
                        )
                    ),
                ),
            )
            output = result.content or ""
            success = bool(output.strip())
            if not success:
                output = ""
            usage = result.usage or {}
            model_used = result.model or getattr(
                self._llm,
                "_model",
                self._model_name,
            )
            cost_estimate = _estimate_cost_details(
                model_used,
                usage,
                self._provider_name,
            )

            return AgentResult(
                agent_name=self.name,
                success=success,
                output=output,
                data={
                    "subtask_count": len(subtask_results),
                    "source_count": len(all_sources) if all_sources else 0,
                    "failure_reason": (
                        None if success else "empty_llm_response"
                    ),
                },
                token_usage=usage,
                metadata={
                    "model": model_used,
                    "contract_version": AGENT_CONTRACT_VERSION,
                },
                cost_usd=cost_estimate.amount_usd,
                cost_status=cost_estimate.status,
            )
        except Exception:
            raise

    async def synthesize_stream(
        self,
        task: str,
        subtask_results: list[dict[str, Any]],
        all_sources: Optional[list[dict[str, Any]]] = None,
        critic_feedback: Optional[CriticScore] = None,
        *,
        temperature: Optional[float] = None,
        max_attempts: int = 3,
        current_draft: str | None = None,
    ):
        """Yield report chunks followed by one usage-bearing result event."""
        user_prompt = _build_synthesis_prompt(
            task=task,
            subtask_results=subtask_results,
            all_sources=all_sources,
            critic_feedback=critic_feedback,
            configured_context_max_chars=int(
                getattr(
                    self._settings.agent,
                    "synthesis_context_max_chars",
                    60_000,
                )
            ),
            current_draft=current_draft,
        )

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        temp = temperature if temperature is not None else 0.4

        output_parts: list[str] = []
        usage: dict[str, int] = {}
        model_used = getattr(self._llm, "_model", self._model_name)
        finish_reason = ""
        reasoning_only = False
        async for event in self._chat_stream(
            messages,
            temperature=temp,
            max_attempts=max_attempts,
            max_output_tokens=adaptive_output_token_budget(
                task,
                subtask_count=len(subtask_results),
                final_report=True,
                hard_limit=int(
                    getattr(
                        self._settings.agent,
                        "synthesizer_max_output_tokens",
                        6_000,
                    )
                ),
            ),
        ):
            if event.type == "chunk" and event.content:
                output_parts.append(event.content)
                yield SynthesisStreamEvent(
                    type="chunk",
                    content=event.content,
                )
            elif event.type == "done":
                usage = event.usage or {}
                model_used = event.model or model_used
                finish_reason = event.finish_reason
                reasoning_only = event.reasoning_only

        output = "".join(output_parts)
        success = bool(output.strip())
        failure_reason = None
        if not success:
            failure_reason = (
                "reasoning_budget_exhausted"
                if reasoning_only and finish_reason == "length"
                else "empty_llm_response"
            )
        cost_estimate = _estimate_cost_details(
            model_used,
            usage,
            self._provider_name,
        )
        yield SynthesisStreamEvent(
            type="done",
            result=AgentResult(
                agent_name=self.name,
                success=success,
                output=output if success else "",
                data={
                    "subtask_count": len(subtask_results),
                    "source_count": len(all_sources) if all_sources else 0,
                    "failure_reason": failure_reason,
                    "finish_reason": finish_reason or None,
                    "reasoning_only": reasoning_only,
                },
                metadata={
                    "model": model_used,
                    "contract_version": AGENT_CONTRACT_VERSION,
                    "finish_reason": finish_reason or None,
                    "reasoning_only": reasoning_only,
                },
                token_usage=usage,
                cost_usd=cost_estimate.amount_usd,
                cost_status=cost_estimate.status,
            ),
        )
