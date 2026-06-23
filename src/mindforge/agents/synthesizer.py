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

_SYNTHESIZER_SYSTEM_PROMPT = """You are an expert research synthesizer. Your role is to combine multiple research findings into a coherent, well-structured final report.

Always structure your reports with these sections:

1. **Executive Summary** — Brief overview of the research question and key conclusions (2-3 paragraphs).
2. **Detailed Analysis** — In-depth coverage of each aspect of the research question, organized logically.
3. **Key Findings** — Bulleted list of the most important discoveries or conclusions.
4. **Data & Evidence** — Supporting data, statistics, quotes, and evidence with proper [N] citations.
5. **Limitations** — Acknowledge any gaps, uncertainties, or limitations in the research.
6. **References** — Numbered list of all sources cited as [N] in the report.

Guidelines:
- Write in a clear, professional tone.
- Use [N] citation markers for every factual claim (e.g., "The sky appears blue due to Rayleigh scattering [1]").
- Integrate findings from multiple subtasks into a unified narrative.
- Eliminate redundancy — if multiple subtasks covered the same ground, present it once.
- If the critic provided feedback, address each issue or suggestion explicitly.
- Aim for comprehensive coverage while maintaining readability.
- Use markdown formatting for structure (headings, lists, emphasis)."""


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

        # Use the synthesizer-specific model from config
        synthesizer_model = settings.llm.get_model("synthesizer")
        request_llm = self._llm
        if request_llm is not None:
            from mindforge.models.base import LLMFactory
            request_llm = LLMFactory.create(
                settings.llm.llm_provider, synthesizer_model
            )

        # --- Build the findings block ---
        findings_lines: list[str] = []
        for i, sr in enumerate(subtask_results, 1):
            desc = sr.get("description", sr.get("task_id", f"Subtask {i}"))
            output = sr.get("output", sr.get("result", ""))
            if isinstance(output, AgentResult):
                output = output.output
            elif not isinstance(output, str):
                output = str(output)
            findings_lines.append(f"### Subtask {i}: {desc}\n\n{output}\n")

        findings_text = "\n".join(findings_lines)

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
        user_prompt_parts = [
            f"## Original Research Task\n\n{task}\n",
            f"## Subtask Findings\n\n{findings_text}\n",
        ]
        if sources_text:
            user_prompt_parts.append(f"## Sources\n\n{sources_text}\n")
        if feedback_text:
            user_prompt_parts.append(f"## Critic Feedback to Address\n\n{feedback_text}\n")

        user_prompt_parts.append(
            "Please synthesize these findings into a comprehensive, well-structured "
            "final report following the required sections. Use [N] citation markers "
            "for all factual claims referencing the provided sources."
        )

        user_prompt = "\n".join(user_prompt_parts)

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        temp = temperature if temperature is not None else 0.4
        result = await self._chat(messages, temperature=temp, _llm_override=request_llm)

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
