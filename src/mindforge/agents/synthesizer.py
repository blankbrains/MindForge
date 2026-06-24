"""Synthesizer agent — assembles research findings into a structured report."""

from __future__ import annotations

from typing import Any, Optional

from mindforge.agents.base import AgentResult, BaseAgent
from mindforge.agents.critic import CriticScore
from mindforge.models.base import ChatMessage
from mindforge.config import get_settings


# ---------------------------------------------------------------------------
# SynthesizerAgent
# ---------------------------------------------------------------------------

_SYNTHESIZER_SYSTEM_PROMPT = """你是一名专业的研究综合编辑。你的任务是将多项研究发现整合成一份连贯、结构良好、内容详实的中文报告。

报告必须按以下结构编写（所有标题和内容使用中文）：

1. **执行摘要** — 研究问题和关键结论的简要概述（2-3 段）。
2. **详细分析** — 对研究问题各方面的深入覆盖，按逻辑组织。
3. **关键发现** — 最重要发现或结论的项目符号列表。
4. **数据与证据** — 支持性数据、统计数据、引用和证据，附上正确的 [N] 引用标记。
5. **局限性** — 承认研究中的任何空白、不确定性或局限性。
6. **参考文献** — 报告中以 [N] 形式引用的所有来源的编号列表。

指南：
- 使用清晰、专业的中文撰写。
- 为每个事实性主张使用 [N] 引用标记。
- 将多个子任务的发现整合成统一的叙述。
- 去除冗余内容——如果多个子任务涉及同一领域，只需呈现一次。
- 如果有评审反馈，明确回应每个问题或建议。
- 力求全面覆盖同时保持可读性。
- 使用 Markdown 格式进行结构化（标题、列表、强调）。

**关键要求 — 当子任务发现稀疏或为空时**：
- 不要生成"未找到信息"的简短报告。
- 应利用你自己的广博训练知识，提供详尽、全面的回答。
- 明确标注知识来源："基于通用知识" vs "基于检索文档"。
- 报告应全面、详尽，结构化分析。长度根据问题复杂度自然决定，不设死板下限。
- Critic 仍会评估和精炼你的输出，请确保内容充实。"""


class SynthesizerAgent(BaseAgent):
    """Generates the final structured research report from subtask results."""

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
        settings = get_settings()

        # Use the synthesizer-specific model from config (via _llm_override for co-routine safety)
        synthesizer_model = settings.llm.get_model("synthesizer")
        from mindforge.models.base import LLMFactory
        _llm_override = LLMFactory.create(
            settings.llm.llm_provider, synthesizer_model
        )

        # --- Build the findings block ---
        findings_lines: list[str] = []
        for i, sr in enumerate(subtask_results, 1):
            desc = sr.get("description", sr.get("task_id", f"Subtask {i}"))
            output = sr.get("output", sr.get("result", ""))
            if isinstance(output, AgentResult):
                output = output.output
            findings_lines.append(f"### Subtask {i}: {desc}\n\n{output}\n")

        findings_text = "\n".join(findings_lines)

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
                "请将这些发现综合成一份全面、结构良好的最终报告，遵循要求的章节结构。"
                "为所有事实性主张使用 [N] 引用标记，引用所提供的来源。"
                "**输出语言必须是中文。**"
            )

            user_prompt = "\n".join(user_prompt_parts)

            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ]

            temp = temperature if temperature is not None else 0.4
            result = await self._chat(messages, temperature=temp, _llm_override=_llm_override)

            return AgentResult(
                agent_name=self.name,
                success=True,
                output=result.content or "",
                data={
                    "subtask_count": len(subtask_results),
                    "source_count": len(all_sources) if all_sources else 0,
                },
                token_usage=result.usage or {},
                metadata={"model": self._model_name},
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
    ):
        """Streaming version — yields content chunks (str) as they arrive from LLM."""
        settings = get_settings()
        synthesizer_model = settings.llm.get_model("synthesizer")
        from mindforge.models.base import LLMFactory
        _llm_override = LLMFactory.create(settings.llm.llm_provider, synthesizer_model)

        findings_lines: list[str] = []
        for i, sr in enumerate(subtask_results, 1):
            desc = sr.get("description", sr.get("task_id", f"Subtask {i}"))
            output = sr.get("output", sr.get("result", ""))
            if isinstance(output, AgentResult):
                output = output.output
            findings_lines.append(f"### 子任务 {i}: {desc}\n\n{output}\n")
        findings_text = "\n".join(findings_lines)

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
            "请将这些发现综合成一份全面、结构良好的最终报告，遵循要求的章节结构。"
            "为所有事实性主张使用 [N] 引用标记，引用所提供的来源。"
            "**输出语言必须是中文。**"
        )
        user_prompt = "\n".join(user_prompt_parts)

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        temp = temperature if temperature is not None else 0.4

        async for event in _llm_override.chat(messages, temperature=temp, stream=True):
            if event.type == "chunk" and event.content:
                yield event.content
