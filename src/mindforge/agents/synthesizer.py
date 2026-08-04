"""Synthesizer agent — assembles research findings into a structured report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from mindforge.agents.base import (
    AgentResult,
    BaseAgent,
    _estimate_cost_details,
)
from mindforge.agents.critic import CriticScore
from mindforge.agents.response_guidance import (
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
- 不得为了填充固定章节而重复内容或虚构细节。"""


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


class SynthesizerAgent(BaseAgent):
    """Generates the final structured research report from subtask results."""

    model_role = "synthesizer"

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
        # --- Build the findings block ---
        synthesis_context_max_chars = int(
            getattr(
                self._settings.agent,
                "synthesis_context_max_chars",
                60_000,
            )
        )
        synthesis_context_max_chars = _effective_synthesis_context_budget(
            synthesis_context_max_chars,
            task,
            subtask_count=len(subtask_results),
        )
        per_subtask_chars = _per_subtask_budget(
            synthesis_context_max_chars,
            len(subtask_results),
        )
        findings_lines: list[str] = []
        for i, sr in enumerate(subtask_results, 1):
            desc = _bounded_label(
                sr.get("description", sr.get("task_id", f"Subtask {i}"))
            )
            output = _bounded_finding_output(
                sr.get("output", sr.get("result", "")),
                per_subtask_chars,
            )
            citation_map = sr.get("citation_map")
            mapping_text = ""
            if isinstance(citation_map, dict) and citation_map:
                mapping_text = (
                    "\n\nCitation remapping for this subtask: "
                    + ", ".join(
                        f"local [{local}] -> global [{global_index}]"
                        for local, global_index in citation_map.items()
                    )
                    + ". Rewrite citations to the global numbers."
                )
                mapping_text = mapping_text[:1_000]
            findings_lines.append(
                f"### Subtask {i}: {desc}\n\n{output}{mapping_text}\n"
            )

        findings_text = "\n".join(findings_lines)
        if len(findings_text) > synthesis_context_max_chars:
            findings_text = (
                findings_text[:synthesis_context_max_chars].rstrip()
                + "\n\n[子任务上下文已达到综合预算上限]"
            )

        try:
            # --- Build the sources block ---
            sources_text = ""
            if all_sources:
                src_lines = ["Consolidated source list:"]
                for s in all_sources:
                    idx = s.get("index", "")
                    title = s.get("title", s.get("source", "Untitled"))
                    url = s.get("url", "")
                    if url:
                        src_lines.append(f"  [{idx}] {title} — {url}")
                    else:
                        src_lines.append(f"  [{idx}] {title}")
                sources_text = "\n".join(src_lines)

            # --- Build the feedback block ---
            feedback_text = ""
            if critic_feedback is not None:
                fb_lines = [
                    "Critic feedback to address:",
                    f"  Overall score: {critic_feedback.overall}/10",
                    "  Issues:",
                ]
                for issue in critic_feedback.issues:
                    fb_lines.append(f"    - {issue}")
                fb_lines.append("  Suggestions:")
                for suggestion in critic_feedback.suggestions:
                    fb_lines.append(f"    - {suggestion}")
                feedback_text = "\n".join(fb_lines)

            # --- Assemble the user prompt ---
            user_prompt_parts: list[str] = [
                f"## 原始研究任务\n\n{task}\n",
                f"## 子任务发现\n\n{findings_text}\n",
            ]
            if sources_text:
                user_prompt_parts.append(f"## 来源\n\n{sources_text}\n")
            if feedback_text:
                user_prompt_parts.append(f"## 评审反馈（需回应）\n\n{feedback_text}\n")

            user_prompt_parts.append(
                _synthesis_output_instruction(
                    task,
                    subtask_count=len(subtask_results),
                    has_sources=bool(all_sources),
                )
            )

            user_prompt = "\n".join(user_prompt_parts)

            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ]

            temp = temperature if temperature is not None else 0.4
            result = await self._chat(
                messages,
                temperature=temp,
                max_attempts=max_attempts,
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
    ):
        """Yield report chunks followed by one usage-bearing result event."""
        synthesis_context_max_chars = int(
            getattr(
                self._settings.agent,
                "synthesis_context_max_chars",
                60_000,
            )
        )
        synthesis_context_max_chars = _effective_synthesis_context_budget(
            synthesis_context_max_chars,
            task,
            subtask_count=len(subtask_results),
        )
        per_subtask_chars = _per_subtask_budget(
            synthesis_context_max_chars,
            len(subtask_results),
        )
        findings_lines: list[str] = []
        for i, sr in enumerate(subtask_results, 1):
            desc = _bounded_label(
                sr.get("description", sr.get("task_id", f"Subtask {i}"))
            )
            output = _bounded_finding_output(
                sr.get("output", sr.get("result", "")),
                per_subtask_chars,
            )
            citation_map = sr.get("citation_map")
            mapping_text = ""
            if isinstance(citation_map, dict) and citation_map:
                mapping_text = (
                    "\n\n该子任务的引用编号映射："
                    + "，".join(
                        f"局部 [{local}] -> 全局 [{global_index}]"
                        for local, global_index in citation_map.items()
                    )
                    + "。最终报告必须改用全局编号。"
                )
                mapping_text = mapping_text[:1_000]
            findings_lines.append(
                f"### 子任务 {i}: {desc}\n\n{output}{mapping_text}\n"
            )
        findings_text = "\n".join(findings_lines)
        if len(findings_text) > synthesis_context_max_chars:
            findings_text = (
                findings_text[:synthesis_context_max_chars].rstrip()
                + "\n\n[子任务上下文已达到综合预算上限]"
            )

        sources_text = ""
        if all_sources:
            src_lines = ["来源列表:"]
            for s in all_sources:
                idx = s.get("index", "")
                title = s.get("title", s.get("source", "Untitled"))
                url = s.get("url", "")
                src_lines.append(f"  [{idx}] {title}" + (f" — {url}" if url else ""))
            sources_text = "\n".join(src_lines)

        feedback_text = ""
        if critic_feedback is not None:
            fb_lines = ["评审反馈（需回应）:", f"  总分: {critic_feedback.overall}/10", "  问题:"]
            for issue in critic_feedback.issues:
                fb_lines.append(f"    - {issue}")
            fb_lines.append("  建议:")
            for suggestion in critic_feedback.suggestions:
                fb_lines.append(f"    - {suggestion}")
            feedback_text = "\n".join(fb_lines)

        user_prompt_parts: list[str] = [
            f"## 原始研究任务\n\n{task}\n",
            f"## 子任务发现\n\n{findings_text}\n",
        ]
        if sources_text:
            user_prompt_parts.append(f"## 来源\n\n{sources_text}\n")
        if feedback_text:
            user_prompt_parts.append(f"## 评审反馈（需回应）\n\n{feedback_text}\n")
        user_prompt_parts.append(
            _synthesis_output_instruction(
                task,
                subtask_count=len(subtask_results),
                has_sources=bool(all_sources),
            )
        )
        user_prompt = "\n".join(user_prompt_parts)

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        temp = temperature if temperature is not None else 0.4

        output_parts: list[str] = []
        usage: dict[str, int] = {}
        model_used = getattr(self._llm, "_model", self._model_name)
        async for event in self._chat_stream(
            messages,
            temperature=temp,
            max_attempts=max_attempts,
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

        output = "".join(output_parts)
        success = bool(output.strip())
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
                    "failure_reason": (
                        None if success else "empty_llm_response"
                    ),
                },
                metadata={"model": model_used},
                token_usage=usage,
                cost_usd=cost_estimate.amount_usd,
                cost_status=cost_estimate.status,
            ),
        )
