"""Orchestrator — top-level controller that drives the full research pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from contextlib import aclosing, nullcontext
from dataclasses import replace
from typing import Any, AsyncIterator, Callable, Optional

from mindforge.agents.base import AgentResult
from mindforge.agents.direct_answer import (
    DirectAnswerAgent,
    DirectAnswerDecision,
)
from mindforge.agents.planner import PlannerAgent, ResearchPlan, SubTask
from mindforge.agents.researcher import ResearcherAgent
from mindforge.agents.critic import CriticAgent, CriticScore
from mindforge.agents.response_guidance import response_profile
from mindforge.agents.synthesizer import SynthesizerAgent
from mindforge.tools.citation_verifier import CitationVerifier
from mindforge.tools.code_executor import CodeExecutor
from mindforge.tools.rag_tool import RAGTool
from mindforge.tools.web_search import WebSearchTool
from mindforge.config import get_settings
from mindforge.interaction import (
    ConversationalTurn,
    classify_conversational_turn,
    is_conversational_task,
)
from mindforge.memory import WorkingMemory
from mindforge.models.base import LLMFactory
from mindforge.context.models import ContextBundle, ResearchRequestContext

logger = logging.getLogger(__name__)

_RESEARCH_CACHE_SCHEMA_VERSION = 8

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Main controller for the MindForge multi-agent research system.

    Drives the full pipeline:
      0. Episodic memory check for cached results.
      1. Plan: decompose task into a DAG of subtasks.
      2. Execute: run subtasks in dependency order, parallel where possible.
      3. Synthesize: combine findings into a coherent report.
      4. Critic + Refine: evaluate and improve (max 2 rounds).
      5. Store: persist results to memory.

    Parameters
    ----------
    planner : PlannerAgent, optional
    researcher : ResearcherAgent, optional
    critic : CriticAgent, optional
    synthesizer : SynthesizerAgent, optional
    episodic_memory : EpisodicMemory, optional
    semantic_memory : SemanticMemory, optional
    tracer : Tracer, optional
    """

    def __init__(
        self,
        planner: Optional[PlannerAgent] = None,
        researcher: Optional[ResearcherAgent] = None,
        critic: Optional[CriticAgent] = None,
        synthesizer: Optional[SynthesizerAgent] = None,
        direct_answer: Optional[DirectAnswerAgent] = None,
        episodic_memory: Any = None,
        semantic_memory: Any = None,
        tracer: Any = None,
    ) -> None:
        self._settings = get_settings()
        self._research_semaphore = asyncio.Semaphore(
            self._settings.agent.max_concurrent_research
        )
        self._direct_answer_semaphore = asyncio.Semaphore(
            int(
                getattr(
                    self._settings.agent,
                    "direct_answer_max_concurrent",
                    8,
                )
            )
        )
        self._subtask_semaphore = asyncio.Semaphore(
            self._settings.agent.max_concurrent_subtasks
        )
        self._tool_semaphore = asyncio.Semaphore(
            self._settings.agent.max_concurrent_tool_calls
        )

        self._planner_injected = planner is not None
        self._planner = planner or PlannerAgent()
        core_agent_injected = any(
            agent is not None
            for agent in (planner, researcher, critic, synthesizer)
        )
        self._direct_answer = (
            direct_answer
            if direct_answer is not None
            else (
                None
                if core_agent_injected
                else DirectAnswerAgent()
            )
        )

        if researcher is None:
            provider = self._settings.llm.llm_provider
            researcher_llm = LLMFactory.create(
                provider,
                self._settings.llm.get_model("researcher", provider),
            )
            source_policy = getattr(
                self._settings.agent,
                "source_policy",
                "auto",
            )
            researcher_tools: list = []
            if source_policy in {"auto", "knowledge_base"}:
                researcher_tools.append(RAGTool())
            if source_policy in {"auto", "web"}:
                web_search = WebSearchTool(
                    native_llm=researcher_llm,
                    native_enabled=self._settings.web_search.native_enabled,
                    duckduckgo_enabled=(
                        self._settings.web_search.duckduckgo_enabled
                    ),
                    native_max_output_tokens=(
                        self._settings.web_search.native_max_output_tokens
                    ),
                    native_timeout_seconds=(
                        self._settings.web_search.native_timeout_seconds
                    ),
                    native_failure_cooldown_seconds=(
                        self._settings.web_search
                        .native_failure_cooldown_seconds
                    ),
                    prefer_tavily=(
                        self._settings.web_search.prefer_tavily
                    ),
                )
                if web_search.available:
                    researcher_tools.append(web_search)
            researcher_tools.extend([CodeExecutor(), CitationVerifier()])
            self._researcher = ResearcherAgent(
                llm=researcher_llm,
                tools=researcher_tools,
                tool_semaphore=self._tool_semaphore,
                tool_queue_timeout=self._settings.agent.queue_timeout,
            )
        else:
            self._researcher = researcher
        if researcher is not None and isinstance(researcher, ResearcherAgent):
            researcher._tool_semaphore = self._tool_semaphore
            researcher._tool_queue_timeout = self._settings.agent.queue_timeout
        self._critic = critic or CriticAgent()
        self._synthesizer = synthesizer or SynthesizerAgent()

        self._episodic_memory = episodic_memory
        self._semantic_memory = semantic_memory
        self._tracer = tracer
        if tracer is not None:
            for agent in (
                self._direct_answer,
                self._planner,
                self._researcher,
                self._critic,
                self._synthesizer,
            ):
                if agent is not None and hasattr(agent, "_tracer"):
                    agent._tracer = tracer

    def close(self) -> None:
        """Release resources owned by injected memory implementations."""
        for memory in (
            self._episodic_memory,
            self._semantic_memory,
        ):
            close = getattr(memory, "close", None)
            if callable(close):
                close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        task: str,
        *,
        create_root_trace: bool = True,
        request_context: ResearchRequestContext | None = None,
        context_bundle: ContextBundle | None = None,
    ) -> AgentResult:
        """Execute one research request inside a top-level trace."""
        trace_context = (
            self._research_trace(task, transport="sync")
            if create_root_trace
            else nullcontext(None)
        )
        with trace_context as root_span:
            if root_span is not None:
                root_span.input = {
                    "task": task,
                    "conversation_id": (
                        request_context.conversation_id
                        if request_context is not None
                        else None
                    ),
                    "context_fingerprint": (
                        context_bundle.fingerprint
                        if context_bundle is not None
                        else None
                    ),
                }
            conversational_turn = classify_conversational_turn(task)
            if conversational_turn is not None:
                result = self._build_conversational_result(
                    conversational_turn,
                    context_bundle=context_bundle,
                )
            else:
                execution_task = (
                    context_bundle.standalone_query
                    if context_bundle is not None
                    else task
                )
                direct_result = await self._try_direct_answer(
                    execution_task,
                    consideration_task=task,
                    context_bundle=context_bundle,
                )
                if direct_result is not None:
                    result = direct_result
                else:
                    research_kwargs: dict[str, Any] = {}
                    if request_context is not None:
                        research_kwargs["request_context"] = request_context
                    if context_bundle is not None:
                        research_kwargs["context_bundle"] = context_bundle
                    result = await self._run_with_limit(
                        execution_task,
                        **research_kwargs,
                    )
            self._attach_trace_id(result)
            self._finish_root_span(root_span, result)
            return result

    async def _run_with_limit(
        self,
        task: str,
        *,
        request_context: ResearchRequestContext | None = None,
        context_bundle: ContextBundle | None = None,
    ) -> AgentResult:
        """Execute one research request within the process-wide budget."""
        queue_timeout = getattr(
            self._settings.agent,
            "queue_timeout",
            30,
        )
        try:
            await asyncio.wait_for(
                self._research_semaphore.acquire(),
                timeout=queue_timeout,
            )
        except asyncio.TimeoutError:
            return AgentResult(
                agent_name="orchestrator",
                success=False,
                output=(
                    "研究任务排队超时，当前服务器正在处理其他研究请求，请稍后重试。"
                ),
                data={"error": "research_queue_timeout"},
            )
        try:
            run_kwargs: dict[str, Any] = {}
            if request_context is not None:
                run_kwargs["request_context"] = request_context
            if context_bundle is not None:
                run_kwargs["context_bundle"] = context_bundle
            return await self._run_unlimited(task, **run_kwargs)
        finally:
            self._research_semaphore.release()

    async def _run_unlimited(
        self,
        task: str,
        *,
        request_context: ResearchRequestContext | None = None,
        context_bundle: ContextBundle | None = None,
    ) -> AgentResult:
        """Execute the full research pipeline for *task*.

        Returns an AgentResult with the final report in ``output`` and
        detailed pipeline metadata in ``data``.
        """
        start_time = time.perf_counter()
        total_usage: dict[str, int] = {}
        total_cost = self._new_cost_accumulator()
        pipeline_log: dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Step 0: Check episodic memory for cached results
        # ------------------------------------------------------------------
        cached_result = await self._recall_cached_result(
            self._cache_task_key(task, context_bundle),
            start_time=start_time,
            pipeline_log=pipeline_log,
            fingerprint_task=task,
        )
        if cached_result is not None:
            return cached_result

        # ------------------------------------------------------------------
        # Core pipeline — wrapped in timeout for safety
        # ------------------------------------------------------------------
        timeout_seconds = self._settings.agent.research_timeout
        try:
            pipeline_kwargs: dict[str, Any] = {}
            if request_context is not None:
                pipeline_kwargs["request_context"] = request_context
            if context_bundle is not None:
                pipeline_kwargs["context_bundle"] = context_bundle
            pipeline = self._run_pipeline(
                task,
                total_usage,
                total_cost,
                pipeline_log,
                start_time,
                **pipeline_kwargs,
            )
            core_result = await asyncio.wait_for(
                pipeline,
                timeout=timeout_seconds,
            )
            return core_result
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "Research task timed out after %d s — returning partial result.",
                timeout_seconds,
            )
            failure_reason = (
                f"Research timed out after {timeout_seconds:g} seconds."
            )
            # Return a clear timeout error message
            partial_output = (
                f"# Research Timed Out\n\n"
                f"The task '{task[:100]}...' exceeded the {timeout_seconds}s "
                f"time limit and was terminated.\n\n"
                f"Consider simplifying the query or increasing AGENT_RESEARCH_TIMEOUT."
            )
            return AgentResult(
                agent_name="orchestrator",
                success=False,
                output=partial_output,
                data={
                    "pipeline": pipeline_log,
                    "error": failure_reason,
                    "error_code": "research_timeout",
                    "error_type": "TimeoutError",
                    "error_message": failure_reason,
                    "stage": "research_pipeline",
                    "timeout_seconds": timeout_seconds,
                },
                metadata={
                    "quality": None,
                    "quality_status": "not_evaluated",
                    "cost": None,
                    "cost_status": "usage_unavailable",
                    "subtask_count": 0,
                    "refine_rounds": 0,
                    "model": self._settings.llm.llm_provider,
                    "timeout": True,
                    "outcome": "error",
                    "failure_reason": failure_reason,
                },
                latency_ms=elapsed_ms,
                cost_status="usage_unavailable",
            )

    async def _run_pipeline(
        self,
        task: str,
        total_usage: dict[str, int],
        total_cost: dict[str, Any],
        pipeline_log: dict[str, Any],
        start_time: float,
        *,
        request_context: ResearchRequestContext | None = None,
        context_bundle: ContextBundle | None = None,
    ) -> AgentResult:
        """Core pipeline steps — separated for timeout wrapping."""
        working_memory = await self._create_working_memory(
            task,
            request_context=request_context,
            context_bundle=context_bundle,
        )

        # ------------------------------------------------------------------
        # Step 1: Plan — decompose into DAG
        # ------------------------------------------------------------------
        with self._agent_span("planner") as planner_span:
            if planner_span is not None:
                planner_span.input = {"task": task}
            plan = await self._create_plan(task)
            if planner_span is not None:
                planner_span.output = {
                    "success": plan.planner_status != "fallback",
                    "subtask_count": len(plan.subtasks),
                    "planner_status": plan.planner_status,
                    "planner_error": plan.planner_error,
                }
                if plan.planner_status == "fallback":
                    planner_span.metadata["status"] = "degraded"
                    planner_span.error = plan.planner_error
        if plan.reasoning:
            working_memory.add_thought(plan.reasoning)
        pipeline_log["plan"] = {
            "subtask_count": len(plan.subtasks),
            "reasoning": plan.reasoning[:200],
        }

        # Track usage
        self._accumulate_usage(total_usage, plan, total_cost)

        # ------------------------------------------------------------------
        # Step 2: Execute DAG (parallel where dependencies allow)
        # ------------------------------------------------------------------
        subtask_outputs: list[dict[str, Any]] = []

        while not plan.is_complete():
            ready = plan.get_ready_tasks()
            if not ready:
                for st in self._fail_deadlocked_tasks(plan):
                    subtask_outputs.append(self._subtask_output(st))
                break

            for st in ready:
                st.status = "in_progress"

            completed = await self._execute_ready_tasks(
                ready,
                plan=plan,
                working_memory=working_memory,
                total_usage=total_usage,
                total_cost=total_cost,
            )
            subtask_outputs.extend(self._subtask_output(st) for st in completed)

        pipeline_log["execution"] = {
            "subtasks_completed": sum(
                1 for s in plan.subtasks if s.status == "completed"
            ),
            "subtasks_failed": sum(1 for s in plan.subtasks if s.status == "failed"),
        }

        if not any(output.get("success") for output in subtask_outputs):
            return self._build_failure_result(
                plan=plan,
                subtask_outputs=subtask_outputs,
                total_usage=total_usage,
                start_time=start_time,
                total_cost=total_cost,
                pipeline_log=pipeline_log,
            )

        # ------------------------------------------------------------------
        # Step 3: Synthesize (skip for single-subtask)
        # ------------------------------------------------------------------
        all_sources = self._collect_sources(subtask_outputs)
        self._attach_citation_maps(subtask_outputs, all_sources)
        skip_syn = len(subtask_outputs) == 1 and subtask_outputs[0].get("success")

        if skip_syn:
            logger.info("单子任务，跳过 Synthesizer（直接用 Researcher 输出）")
            current_draft = subtask_outputs[0].get("output", "")
            draft_result = AgentResult(
                agent_name="synthesizer", success=True, output=current_draft
            )
            pipeline_log["synthesize"] = {"status": "skipped_single_subtask"}
        else:
            with self._agent_span(
                "synthesizer",
                metadata={"phase": "initial"},
            ) as synthesizer_span:
                if synthesizer_span is not None:
                    synthesizer_span.input = {
                        "subtask_count": len(subtask_outputs),
                        "source_count": len(all_sources),
                    }
                draft_result = await self._synthesizer.synthesize(
                    task=task,
                    subtask_results=subtask_outputs,
                    all_sources=all_sources,
                )
                if synthesizer_span is not None:
                    synthesizer_span.output = {
                        "success": draft_result.success,
                        "output_chars": len(draft_result.output),
                    }
            self._accumulate_usage(total_usage, draft_result, total_cost)
            pipeline_log["synthesize"] = {"status": "completed"}

        if not draft_result.success or not draft_result.output.strip():
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            cost_usd, cost_status = self._cost_summary(total_cost)
            failure_reason = (
                "Research synthesis failed because the language model "
                "returned an empty response."
            )
            pipeline_log["synthesize"] = {
                "status": "failed",
                "reason": "empty_response",
            }
            return AgentResult(
                agent_name="orchestrator",
                success=False,
                output=failure_reason,
                data={
                    "pipeline": pipeline_log,
                    "plan": plan.to_dict(),
                    "subtask_outputs": subtask_outputs,
                    "sources": all_sources,
                    "critic_score": None,
                    "refine_rounds": 0,
                    "error_code": "synthesis_empty_response",
                    "error_type": "EmptyResponse",
                    "error_message": failure_reason,
                    "stage": "synthesis",
                },
                metadata={
                    "quality": None,
                    "quality_status": "not_evaluated",
                    "cost": cost_usd,
                    "cost_status": cost_status,
                    "subtask_count": len(plan.subtasks),
                    "refine_rounds": 0,
                    "model": self._settings.llm.llm_provider,
                    "outcome": "error",
                    "failure_reason": failure_reason,
                },
                token_usage=total_usage,
                latency_ms=elapsed_ms,
                cost_usd=cost_usd,
                cost_status=cost_status,
            )

        # ------------------------------------------------------------------
        # Step 4: Critic + refine loop
        # ------------------------------------------------------------------
        current_draft = self._strip_unbacked_citations(
            draft_result.output,
            all_sources,
        )
        final_critic: Optional[CriticScore] = None
        refine_count = 0
        refine_attempts = 0
        refinement_rejections = 0
        refinement_failure: str | None = None
        best_draft = current_draft
        best_critic: CriticScore | None = None
        evaluating_candidate = False

        if not self._should_run_critic(task, plan):
            logger.info("快速模式，跳过 Critic 评估")
            pipeline_log["critic"] = {"skipped": True, "reason": "快速模式"}
        else:
            max_refine = self._max_refine_rounds(plan)
            for evaluation_round in range(max_refine + 1):
                with self._agent_span(
                    "critic",
                    metadata={"round": evaluation_round + 1},
                ) as critic_span:
                    critic_score = await self._critic.evaluate(
                        task=task,
                        draft=current_draft,
                        sources=all_sources,
                    )
                    if critic_span is not None:
                        critic_span.input = {
                            "draft_chars": len(current_draft),
                            "source_count": len(all_sources),
                        }
                        critic_span.output = {
                            "overall": critic_score.overall,
                            "should_refine": critic_score.should_refine,
                        }
                self._accumulate_usage(
                    total_usage,
                    critic_score,
                    total_cost,
                )
                if evaluating_candidate and best_critic is not None:
                    baseline_needs_depth = (
                        self._report_needs_depth_refinement(
                            task,
                            best_draft,
                            plan,
                        )
                    )
                    candidate_needs_depth = (
                        self._report_needs_depth_refinement(
                            task,
                            current_draft,
                            plan,
                        )
                    )
                    if not self._is_refinement_improvement(
                        critic_score,
                        best_critic,
                        candidate_draft=current_draft,
                        baseline_draft=best_draft,
                        baseline_needs_depth_refinement=(
                            baseline_needs_depth
                        ),
                        candidate_needs_depth_refinement=(
                            candidate_needs_depth
                        ),
                    ):
                        refinement_rejections += 1
                        current_draft = best_draft
                        final_critic = best_critic
                        pipeline_log["critic"] = {
                            "evaluations": evaluation_round + 1,
                            "refine_attempts": refine_attempts,
                            "refine_rounds": refine_count,
                            "overall_score": best_critic.overall,
                            "refined": refine_count > 0,
                            "refinement_rejected": True,
                            "candidate_score": critic_score.overall,
                        }
                        logger.info(
                            "Report refinement rejected; keeping the "
                            "higher-quality draft."
                        )
                        break
                    best_draft = current_draft
                    best_critic = critic_score
                    final_critic = critic_score
                    refine_count += 1
                    evaluating_candidate = False
                else:
                    best_draft = current_draft
                    best_critic = critic_score
                    final_critic = critic_score
                needs_depth_refinement = (
                    self._report_needs_depth_refinement(
                        task,
                        current_draft,
                        plan,
                    )
                )
                if critic_score.evaluation_status == "failed":
                    pipeline_log["critic"] = {
                        "status": "failed",
                        "reason": critic_score.evaluation_error,
                    }
                    if (
                        not needs_depth_refinement
                        or refine_attempts >= max_refine
                    ):
                        break

                if (
                    (
                        not needs_depth_refinement
                        and critic_score.evaluation_status == "evaluated"
                        and not self._should_refine_report(critic_score)
                    )
                    or refine_attempts >= max_refine
                ):
                    pipeline_log["critic"] = {
                        "evaluations": evaluation_round + 1,
                        "refine_attempts": refine_attempts,
                        "refine_rounds": refine_count,
                        "overall_score": critic_score.overall,
                        "refined": refine_count > 0,
                    }
                    break

                refinement_feedback = (
                    critic_score
                    if critic_score.evaluation_status == "evaluated"
                    else self._depth_refinement_feedback(
                        task,
                        current_draft,
                        critic_score,
                    )
                )
                # Refine: re-synthesize with critic feedback
                try:
                    with self._agent_span(
                        "synthesizer",
                        metadata={
                            "phase": "refine",
                            "round": refine_attempts + 1,
                        },
                    ) as synthesizer_span:
                        refined_result = await self._synthesizer.synthesize(
                            task=task,
                            subtask_results=subtask_outputs,
                            all_sources=all_sources,
                            critic_feedback=refinement_feedback,
                            current_draft=current_draft,
                            max_attempts=1,
                        )
                        if synthesizer_span is not None:
                            synthesizer_span.output = {
                                "success": refined_result.success,
                                "output_chars": len(refined_result.output),
                            }
                except Exception as exc:
                    refinement_failure = self._describe_exception(exc)
                    pipeline_log["critic"] = {
                        "evaluations": evaluation_round + 1,
                        "refine_attempts": refine_attempts,
                        "refine_rounds": refine_count,
                        "overall_score": critic_score.overall,
                        "refined": refine_count > 0,
                        "refinement_failed": True,
                        "reason": refinement_failure,
                    }
                    logger.warning(
                        "Report refinement failed; keeping the last valid draft: %s",
                        refinement_failure,
                    )
                    break
                self._accumulate_usage(
                    total_usage,
                    refined_result,
                    total_cost,
                )
                if not refined_result.success or not refined_result.output.strip():
                    refinement_failure = str(
                        refined_result.data.get("failure_reason")
                        or "报告精炼返回空结果。"
                    )
                    pipeline_log["critic"] = {
                        "evaluations": evaluation_round + 1,
                        "refine_attempts": refine_attempts,
                        "refine_rounds": refine_count,
                        "overall_score": critic_score.overall,
                        "refined": refine_count > 0,
                        "refinement_failed": True,
                        "reason": refinement_failure,
                    }
                    break
                current_draft = self._strip_unbacked_citations(
                    refined_result.output,
                    all_sources,
                )
                refine_attempts += 1
                evaluating_candidate = True

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        total_cost_usd, cost_status = self._cost_summary(total_cost)
        citation_verification = self._verify_final_citations(
            current_draft,
            all_sources,
        )
        result = self._build_success_result(
            output=current_draft,
            plan=plan,
            subtask_outputs=subtask_outputs,
            sources=all_sources,
            final_critic=final_critic,
            refine_count=refine_count,
            total_usage=total_usage,
            elapsed_ms=elapsed_ms,
            total_cost_usd=total_cost_usd,
            cost_status=cost_status,
            pipeline_log=pipeline_log,
            refinement_failure=refinement_failure,
            refinement_rejections=refinement_rejections,
            citation_verification=citation_verification,
        )
        self._attach_context_metadata(result, context_bundle)
        await self._store_memories(
            self._cache_task_key(task, context_bundle),
            result,
            semantic_task=(None if context_bundle is not None else task),
            fingerprint_task=task,
        )
        return result

    # ------------------------------------------------------------------
    # Streaming variant
    # ------------------------------------------------------------------

    async def stream_run(
        self,
        task: str,
        *,
        create_root_trace: bool = True,
        request_context: ResearchRequestContext | None = None,
        context_bundle: ContextBundle | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream one research request inside a top-level trace."""
        final_result: AgentResult | None = None
        trace_error: str | None = None
        trace_context = (
            self._research_trace(task, transport="sse")
            if create_root_trace
            else nullcontext(None)
        )
        with trace_context as root_span:
            if root_span is not None:
                root_span.input = {
                    "task": task,
                    "conversation_id": (
                        request_context.conversation_id
                        if request_context is not None
                        else None
                    ),
                    "context_fingerprint": (
                        context_bundle.fingerprint
                        if context_bundle is not None
                        else None
                    ),
                }
            try:
                conversational_turn = classify_conversational_turn(task)
                if conversational_turn is not None:
                    result = self._build_conversational_result(
                        conversational_turn,
                        context_bundle=context_bundle,
                    )
                    self._attach_trace_id(result)
                    final_result = result
                    trace_id = result.trace_id
                    chunk_event: dict[str, Any] = {
                        "type": "answer_chunk",
                        "content": result.output,
                    }
                    done_event: dict[str, Any] = {
                        "type": "done",
                        "result": result,
                    }
                    if trace_id:
                        chunk_event["trace_id"] = trace_id
                        done_event["trace_id"] = trace_id
                    yield chunk_event
                    yield done_event
                    return
                execution_task = (
                    context_bundle.standalone_query
                    if context_bundle is not None
                    else task
                )
                direct_progress: asyncio.Queue[dict[str, Any] | object] = (
                    asyncio.Queue()
                )
                direct_finished = object()
                direct_result_holder: dict[str, AgentResult | None] = {}

                async def run_direct_answer() -> None:
                    try:
                        direct_result_holder["result"] = (
                            await self._try_direct_answer(
                                execution_task,
                                consideration_task=task,
                                context_bundle=context_bundle,
                                progress_callback=direct_progress.put_nowait,
                            )
                        )
                    finally:
                        direct_progress.put_nowait(direct_finished)

                direct_task = asyncio.create_task(run_direct_answer())
                try:
                    while True:
                        progress_event = await direct_progress.get()
                        if progress_event is direct_finished:
                            break
                        if not isinstance(progress_event, dict):
                            raise RuntimeError(
                                "Direct-answer progress produced an invalid event."
                            )
                        trace_id = self._current_trace_id()
                        if trace_id:
                            progress_event = {
                                **progress_event,
                                "trace_id": trace_id,
                            }
                        yield progress_event
                    await direct_task
                finally:
                    if not direct_task.done():
                        direct_task.cancel()
                        await asyncio.gather(
                            direct_task,
                            return_exceptions=True,
                        )
                direct_result = direct_result_holder.get("result")
                if direct_result is not None:
                    self._attach_trace_id(direct_result)
                    final_result = direct_result
                    trace_id = direct_result.trace_id
                    chunk_size = min(
                        256,
                        int(self._settings.agent.stream_chunk_size),
                    )
                    for offset in range(0, len(direct_result.output), chunk_size):
                        chunk_event = {
                            "type": "answer_chunk",
                            "content": direct_result.output[
                                offset : offset + chunk_size
                            ],
                        }
                        if trace_id:
                            chunk_event["trace_id"] = trace_id
                        yield chunk_event
                        await asyncio.sleep(0)
                    done_event = {
                        "type": "done",
                        "result": direct_result,
                    }
                    if trace_id:
                        done_event["trace_id"] = trace_id
                    yield done_event
                    return
                stream_kwargs: dict[str, Any] = {}
                if request_context is not None:
                    stream_kwargs["request_context"] = request_context
                if context_bundle is not None:
                    stream_kwargs["context_bundle"] = context_bundle
                async with aclosing(
                    self._stream_run_with_limit(
                        execution_task,
                        **stream_kwargs,
                    )
                ) as limited_stream:
                    async for event in limited_stream:
                        trace_id = self._current_trace_id()
                        if trace_id:
                            event = {**event, "trace_id": trace_id}
                        if event.get("type") == "done":
                            result = event.get("result")
                            if isinstance(result, AgentResult):
                                self._attach_trace_id(result)
                                final_result = result
                        elif event.get("type") == "error":
                            trace_error = str(
                                event.get("content") or "Research failed."
                            )
                        yield event
            except asyncio.CancelledError:
                if root_span is not None:
                    root_span.metadata["status"] = "cancelled"
                    root_span.error = "Research stream cancelled."
                raise
            finally:
                if root_span is not None:
                    if trace_error and final_result is None:
                        root_span.error = trace_error
                    self._finish_root_span(root_span, final_result)

    async def _stream_run_with_limit(
        self,
        task: str,
        *,
        request_context: ResearchRequestContext | None = None,
        context_bundle: ContextBundle | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream the research pipeline under the configured overall timeout."""
        queue_timeout = getattr(
            self._settings.agent,
            "queue_timeout",
            30,
        )
        heartbeat_seconds = getattr(
            self._settings.agent,
            "sse_heartbeat_seconds",
            10,
        )
        try:
            await asyncio.wait_for(
                self._research_semaphore.acquire(),
                timeout=queue_timeout,
            )
        except asyncio.TimeoutError:
            yield {
                "type": "error",
                "content": (
                    "研究任务排队超时，当前服务器正在处理其他研究请求，请稍后重试。"
                ),
            }
            return

        pipeline_task: asyncio.Task[None] | None = None
        try:
            event_queue: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue()
            stream_finished = object()

            async def pump_events() -> None:
                pipeline_kwargs: dict[str, Any] = {}
                if request_context is not None:
                    pipeline_kwargs["request_context"] = request_context
                if context_bundle is not None:
                    pipeline_kwargs["context_bundle"] = context_bundle
                pipeline_source = self._stream_pipeline(
                    task,
                    **pipeline_kwargs,
                )
                async with aclosing(
                    pipeline_source
                ) as pipeline:
                    try:
                        async for event in pipeline:
                            event_queue.put_nowait(event)
                    finally:
                        event_queue.put_nowait(stream_finished)

            pipeline_task = asyncio.create_task(pump_events())
            # Let the pipeline enqueue its immediate lifecycle event before the
            # heartbeat timer starts. This await must remain inside the slot's
            # try/finally because clients can disconnect during startup.
            await asyncio.sleep(0)
            deadline = time.monotonic() + self._settings.agent.research_timeout
            overall_timed_out = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    overall_timed_out = True
                    break
                try:
                    event = await asyncio.wait_for(
                        event_queue.get(),
                        timeout=min(remaining, heartbeat_seconds),
                    )
                except asyncio.TimeoutError:
                    yield {
                        "type": "heartbeat",
                        "timestamp": time.time(),
                    }
                    continue
                if event is stream_finished:
                    await pipeline_task
                    return
                if not isinstance(event, dict):
                    raise RuntimeError("Research stream produced an invalid event.")
                yield event
            if overall_timed_out:
                yield {
                    "type": "error",
                    "content": (
                        "研究任务超过 "
                        f"{self._settings.agent.research_timeout} 秒，已终止。"
                    ),
                }
        finally:
            if pipeline_task is not None and not pipeline_task.done():
                pipeline_task.cancel()
                await asyncio.gather(
                    pipeline_task,
                    return_exceptions=True,
                )
            self._research_semaphore.release()

    async def _stream_pipeline(
        self,
        task: str,
        *,
        request_context: ResearchRequestContext | None = None,
        context_bundle: ContextBundle | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute the pipeline and yield events for streaming UIs.

        Yields events:
        - ``{"type": "plan_ready", "plan": ResearchPlan}``
        - ``{"type": "subtask_start", "task_id": str, "description": str}``
        - ``{"type": "subtask_result", "task_id": str, "result": AgentResult}``
        - ``{"type": "synthesizing", "status": "start" | "done"}``
        - ``{"type": "critic_feedback", "score": CriticScore}``
        - ``{"type": "refining", "round": int}``
        - ``{"type": "done", "result": AgentResult}``
        """
        start_time = time.perf_counter()
        total_usage: dict[str, int] = {}
        total_cost = self._new_cost_accumulator()

        yield {"type": "planning", "status": "start"}

        # --- Step 0: Memory check ---
        cached_result = await self._recall_cached_result(
            self._cache_task_key(task, context_bundle),
            start_time=start_time,
            fingerprint_task=task,
        )
        if cached_result is not None:
            yield {"type": "done", "result": cached_result}
            return
        working_memory = await self._create_working_memory(
            task,
            request_context=request_context,
            context_bundle=context_bundle,
        )

        # --- Step 1: Plan ---
        with self._agent_span("planner") as planner_span:
            if planner_span is not None:
                planner_span.input = {"task": task}
            plan = await self._create_plan(task)
            if planner_span is not None:
                planner_span.output = {
                    "success": plan.planner_status != "fallback",
                    "subtask_count": len(plan.subtasks),
                    "planner_status": plan.planner_status,
                    "planner_error": plan.planner_error,
                }
                if plan.planner_status == "fallback":
                    planner_span.metadata["status"] = "degraded"
                    planner_span.error = plan.planner_error
        if plan.reasoning:
            working_memory.add_thought(plan.reasoning)
        self._accumulate_usage(total_usage, plan, total_cost)
        yield {"type": "planning", "status": "done"}
        yield {"type": "plan_ready", "plan": plan}

        # --- Step 2: Execute DAG ---
        subtask_outputs: list[dict[str, Any]] = []

        while not plan.is_complete():
            ready = plan.get_ready_tasks()
            if not ready:
                for st in self._fail_deadlocked_tasks(plan):
                    subtask_outputs.append(self._subtask_output(st))
                    yield {
                        "type": "subtask_result",
                        "task_id": st.task_id,
                        "result": st.result,
                    }
                break

            for st in ready:
                st.status = "in_progress"
                yield {
                    "type": "subtask_start",
                    "task_id": st.task_id,
                    "description": st.description,
                }

            completed = await self._execute_ready_tasks(
                ready,
                plan=plan,
                working_memory=working_memory,
                total_usage=total_usage,
                total_cost=total_cost,
            )
            for st in completed:
                subtask_outputs.append(self._subtask_output(st))
                yield {
                    "type": "subtask_result",
                    "task_id": st.task_id,
                    "result": st.result,
                }

        if not any(output.get("success") for output in subtask_outputs):
            result = self._build_failure_result(
                plan=plan,
                subtask_outputs=subtask_outputs,
                total_usage=total_usage,
                start_time=start_time,
                total_cost=total_cost,
            )
            yield {"type": "done", "result": result}
            return

        # --- Step 3: Synthesize (skip for single-subtask — use Researcher output directly) ---
        all_sources = self._collect_sources(subtask_outputs)
        self._attach_citation_maps(subtask_outputs, all_sources)
        skip_synthesizer = len(subtask_outputs) == 1 and subtask_outputs[0].get(
            "success"
        )

        if skip_synthesizer:
            logger.info("单子任务，跳过 Synthesizer（流式输出 Researcher 结果）")
            researcher_text = subtask_outputs[0].get("output", "")
            chunk_size = self._settings.agent.stream_chunk_size
            for i in range(0, len(researcher_text), chunk_size):
                yield {
                    "type": "answer_chunk",
                    "content": researcher_text[i : i + chunk_size],
                }
            yield {"type": "synthesizing", "status": "done"}
            current_draft = researcher_text
            draft_result = AgentResult(
                agent_name="synthesizer", success=True, output=researcher_text
            )
        else:
            yield {"type": "synthesizing", "status": "start"}
            draft_result: AgentResult | None = None
            with self._agent_span(
                "synthesizer",
                metadata={"phase": "initial_stream"},
            ) as synthesizer_span:
                if synthesizer_span is not None:
                    synthesizer_span.input = {
                        "subtask_count": len(subtask_outputs),
                        "source_count": len(all_sources),
                    }
                async for synthesis_event in self._synthesizer.synthesize_stream(
                    task=task,
                    subtask_results=subtask_outputs,
                    all_sources=all_sources,
                ):
                    if synthesis_event.type == "chunk":
                        yield {
                            "type": "answer_chunk",
                            "content": synthesis_event.content,
                        }
                    elif synthesis_event.type == "done":
                        draft_result = synthesis_event.result
                if synthesizer_span is not None and draft_result is not None:
                    synthesizer_span.output = {
                        "success": draft_result.success,
                        "output_chars": len(draft_result.output),
                    }
            if draft_result is None:
                draft_result = AgentResult(
                    agent_name="synthesizer",
                    success=False,
                    output="",
                )
            self._accumulate_usage(
                total_usage,
                draft_result,
                total_cost,
            )
            yield {"type": "synthesizing", "status": "done"}

        if not draft_result.success or not draft_result.output.strip():
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            cost_usd, cost_status = self._cost_summary(total_cost)
            failure_reason = (
                "Research synthesis failed because the language model "
                "returned an empty response."
            )
            result = AgentResult(
                agent_name="orchestrator",
                success=False,
                output=failure_reason,
                data={
                    "plan": plan.to_dict(),
                    "subtask_outputs": subtask_outputs,
                    "sources": all_sources,
                    "critic_score": None,
                    "refine_rounds": 0,
                    "error_code": "synthesis_empty_response",
                    "error_type": "EmptyResponse",
                    "error_message": failure_reason,
                    "stage": "synthesis",
                },
                metadata={
                    "quality": None,
                    "quality_status": "not_evaluated",
                    "cost": cost_usd,
                    "cost_status": cost_status,
                    "subtask_count": len(plan.subtasks),
                    "refine_rounds": 0,
                    "model": self._settings.llm.llm_provider,
                    "outcome": "error",
                    "failure_reason": failure_reason,
                },
                token_usage=total_usage,
                latency_ms=elapsed_ms,
                cost_usd=cost_usd,
                cost_status=cost_status,
            )
            yield {"type": "done", "result": result}
            return

        # --- Step 4: Critic + refine ---
        current_draft = self._strip_unbacked_citations(
            draft_result.output,
            all_sources,
        )
        final_critic: Optional[CriticScore] = None
        refine_count = 0
        refine_attempts = 0
        refinement_rejections = 0
        refinement_failure: str | None = None
        best_draft = current_draft
        best_critic: CriticScore | None = None
        evaluating_candidate = False

        if not self._should_run_critic(task, plan):
            logger.info("快速模式，跳过 Critic 评估")
        else:
            max_refine = self._max_refine_rounds(plan)
            for evaluation_round in range(max_refine + 1):
                with self._agent_span(
                    "critic",
                    metadata={"round": evaluation_round + 1},
                ) as critic_span:
                    critic_score = await self._critic.evaluate(
                        task=task,
                        draft=current_draft,
                        sources=all_sources,
                    )
                    if critic_span is not None:
                        critic_span.input = {
                            "draft_chars": len(current_draft),
                            "source_count": len(all_sources),
                        }
                        critic_span.output = {
                            "overall": critic_score.overall,
                            "should_refine": critic_score.should_refine,
                        }
                self._accumulate_usage(
                    total_usage,
                    critic_score,
                    total_cost,
                )

                yield {
                    "type": "critic_feedback",
                    "score": critic_score,
                    "round": evaluation_round + 1,
                }
                if evaluating_candidate and best_critic is not None:
                    baseline_needs_depth = (
                        self._report_needs_depth_refinement(
                            task,
                            best_draft,
                            plan,
                        )
                    )
                    candidate_needs_depth = (
                        self._report_needs_depth_refinement(
                            task,
                            current_draft,
                            plan,
                        )
                    )
                    if not self._is_refinement_improvement(
                        critic_score,
                        best_critic,
                        candidate_draft=current_draft,
                        baseline_draft=best_draft,
                        baseline_needs_depth_refinement=(
                            baseline_needs_depth
                        ),
                        candidate_needs_depth_refinement=(
                            candidate_needs_depth
                        ),
                    ):
                        refinement_rejections += 1
                        current_draft = best_draft
                        final_critic = best_critic
                        logger.info(
                            "Report refinement rejected; keeping the "
                            "higher-quality draft."
                        )
                        break
                    best_draft = current_draft
                    best_critic = critic_score
                    final_critic = critic_score
                    refine_count += 1
                    evaluating_candidate = False
                else:
                    best_draft = current_draft
                    best_critic = critic_score
                    final_critic = critic_score
                needs_depth_refinement = (
                    self._report_needs_depth_refinement(
                        task,
                        current_draft,
                        plan,
                    )
                )
                if (
                    critic_score.evaluation_status == "failed"
                    and (
                        not needs_depth_refinement
                        or refine_attempts >= max_refine
                    )
                ):
                    break

                if (
                    (
                        not needs_depth_refinement
                        and critic_score.evaluation_status == "evaluated"
                        and not self._should_refine_report(critic_score)
                    )
                    or refine_attempts >= max_refine
                ):
                    break

                yield {"type": "refining", "round": refine_attempts + 1}

                refinement_feedback = (
                    critic_score
                    if critic_score.evaluation_status == "evaluated"
                    else self._depth_refinement_feedback(
                        task,
                        current_draft,
                        critic_score,
                    )
                )
                try:
                    with self._agent_span(
                        "synthesizer",
                        metadata={
                            "phase": "refine",
                            "round": refine_attempts + 1,
                        },
                    ) as synthesizer_span:
                        refined_result = await self._synthesizer.synthesize(
                            task=task,
                            subtask_results=subtask_outputs,
                            all_sources=all_sources,
                            critic_feedback=refinement_feedback,
                            current_draft=current_draft,
                            max_attempts=1,
                        )
                        if synthesizer_span is not None:
                            synthesizer_span.output = {
                                "success": refined_result.success,
                                "output_chars": len(refined_result.output),
                            }
                except Exception as exc:
                    refinement_failure = self._describe_exception(exc)
                    logger.warning(
                        "Report refinement failed; keeping the last valid draft: %s",
                        refinement_failure,
                    )
                    break
                self._accumulate_usage(
                    total_usage,
                    refined_result,
                    total_cost,
                )
                if not refined_result.success or not refined_result.output.strip():
                    refinement_failure = str(
                        refined_result.data.get("failure_reason")
                        or "报告精炼返回空结果。"
                    )
                    break
                current_draft = self._strip_unbacked_citations(
                    refined_result.output,
                    all_sources,
                )
                refine_attempts += 1
                evaluating_candidate = True

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        total_cost_usd, cost_status = self._cost_summary(total_cost)
        citation_verification = self._verify_final_citations(
            current_draft,
            all_sources,
        )
        result = self._build_success_result(
            output=current_draft,
            plan=plan,
            subtask_outputs=subtask_outputs,
            sources=all_sources,
            final_critic=final_critic,
            refine_count=refine_count,
            total_usage=total_usage,
            elapsed_ms=elapsed_ms,
            total_cost_usd=total_cost_usd,
            cost_status=cost_status,
            refinement_failure=refinement_failure,
            refinement_rejections=refinement_rejections,
            citation_verification=citation_verification,
        )
        self._attach_context_metadata(result, context_bundle)
        await self._store_memories(
            self._cache_task_key(task, context_bundle),
            result,
            semantic_task=(None if context_bundle is not None else task),
            fingerprint_task=task,
        )
        yield {"type": "done", "result": result}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _try_direct_answer(
        self,
        task: str,
        *,
        consideration_task: str | None = None,
        context_bundle: ContextBundle | None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentResult | None:
        agent = getattr(self, "_direct_answer", None)
        if agent is None:
            return None
        started = time.perf_counter()
        routing_started = False
        routing_completed = False

        def report_progress(event: dict[str, Any]) -> None:
            if progress_callback is not None:
                progress_callback(event)

        try:
            routing_task = consideration_task or task
            if not agent.should_consider(routing_task):
                return None
            routing_started = True
            report_progress({"type": "routing", "status": "start"})

            async def decide() -> DirectAnswerDecision:
                with self._agent_span("direct_answer") as span:
                    if span is not None:
                        span.input = {
                            "task": task,
                            "consideration_task": routing_task,
                            "context_items": (
                                len(context_bundle.items)
                                if context_bundle is not None
                                else 0
                            ),
                        }
                    decision = await agent.decide(
                        task,
                        context_bundle=context_bundle,
                    )
                    if span is not None:
                        span.output = {
                            "route": decision.route,
                            "confidence": decision.confidence,
                            "model": decision.model,
                            "answer_chars": len(decision.answer),
                        }
                return decision

            semaphore = getattr(self, "_direct_answer_semaphore", None)
            if semaphore is None:
                decision = await decide()
            else:
                async with semaphore:
                    decision = await decide()
            report_progress(
                {
                    "type": "routing",
                    "status": "done",
                    "route": decision.route,
                }
            )
            routing_completed = True
            if decision.route != "direct_answer" or not decision.answer:
                return None
            evaluation_task = routing_task
            current_answer = decision.answer
            plan = self._direct_answer_plan(
                evaluation_task,
                decision,
                status="in_progress",
            )
            report_progress({"type": "planning", "status": "start"})
            report_progress({"type": "planning", "status": "done"})
            report_progress(
                {
                    "type": "plan_ready",
                    "plan": ResearchPlan.from_dict(plan.to_dict()),
                }
            )
            report_progress(
                {
                    "type": "subtask_start",
                    "task_id": "direct-answer",
                    "description": plan.subtasks[0].description,
                }
            )
            critic_history: list[CriticScore] = []
            refinement_results: list[AgentResult] = []
            refinement_failure: str | None = None
            refinement_rejections = 0
            refine_count = 0
            refine_attempts = 0
            best_answer = current_answer
            best_critic: CriticScore | None = None
            evaluating_candidate = False
            max_refine = self._max_direct_refine_rounds(
                evaluation_task,
            )
            for evaluation_round in range(max_refine + 1):
                with self._agent_span(
                    "critic",
                    metadata={
                        "round": evaluation_round + 1,
                        "route": "direct_answer",
                    },
                ) as critic_span:
                    critic_score = await self._critic.evaluate(
                        task=evaluation_task,
                        draft=current_answer,
                        sources=[],
                    )
                    if critic_span is not None:
                        critic_span.input = {
                            "draft_chars": len(current_answer),
                            "source_count": 0,
                        }
                        critic_span.output = {
                            "overall": critic_score.overall,
                            "should_refine": critic_score.should_refine,
                            "evaluation_status": critic_score.evaluation_status,
                        }
                critic_history.append(critic_score)
                report_progress(
                    {
                        "type": "critic_feedback",
                        "score": critic_score.to_dict(),
                        "round": evaluation_round + 1,
                    }
                )
                if evaluating_candidate and best_critic is not None:
                    baseline_needs_depth = (
                        self._direct_answer_needs_depth_refinement(
                            evaluation_task,
                            best_answer,
                        )
                    )
                    candidate_needs_depth = (
                        self._direct_answer_needs_depth_refinement(
                            evaluation_task,
                            current_answer,
                        )
                    )
                    if not self._is_refinement_improvement(
                        critic_score,
                        best_critic,
                        candidate_draft=current_answer,
                        baseline_draft=best_answer,
                        baseline_needs_depth_refinement=(
                            baseline_needs_depth
                        ),
                        candidate_needs_depth_refinement=(
                            candidate_needs_depth
                        ),
                    ):
                        refinement_rejections += 1
                        current_answer = best_answer
                        break
                    best_answer = current_answer
                    best_critic = critic_score
                    refine_count += 1
                    evaluating_candidate = False
                else:
                    best_answer = current_answer
                    best_critic = critic_score
                needs_depth_refinement = (
                    self._direct_answer_needs_depth_refinement(
                        evaluation_task,
                        current_answer,
                    )
                )
                if (
                    critic_score.evaluation_status == "failed"
                    and not needs_depth_refinement
                ):
                    break
                if (
                    refine_attempts >= max_refine
                    or (
                        not needs_depth_refinement
                        and critic_score.evaluation_status == "evaluated"
                        and not self._should_refine_report(critic_score)
                    )
                ):
                    break
                refinement_feedback = (
                    critic_score
                    if critic_score.evaluation_status == "evaluated"
                    else self._depth_refinement_feedback(
                        evaluation_task,
                        current_answer,
                        critic_score,
                    )
                )
                report_progress(
                    {
                        "type": "refining",
                        "round": refine_attempts + 1,
                    }
                )
                try:
                    with self._agent_span(
                        "synthesizer",
                        metadata={
                            "phase": "direct_answer_refine",
                            "round": evaluation_round + 1,
                        },
                    ) as synthesizer_span:
                        refined_result = await self._synthesizer.synthesize(
                            task=evaluation_task,
                            subtask_results=[
                                {
                                    "task_id": "direct-answer",
                                    "description": (
                                        "无需外部检索的稳定知识回答草稿"
                                    ),
                                    "output": current_answer,
                                    "success": True,
                                }
                            ],
                            all_sources=[],
                            critic_feedback=refinement_feedback,
                            current_draft=current_answer,
                            max_attempts=1,
                        )
                        if synthesizer_span is not None:
                            synthesizer_span.input = {
                                "draft_chars": len(current_answer),
                                "requested_depth": response_profile(
                                    evaluation_task
                                ).depth,
                            }
                            synthesizer_span.output = {
                                "success": refined_result.success,
                                "output_chars": len(refined_result.output),
                            }
                except Exception as exc:
                    refinement_failure = self._describe_exception(exc)
                    logger.warning(
                        "Direct-answer refinement failed; keeping the last "
                        "valid draft: %s",
                        refinement_failure,
                    )
                    break
                if not refined_result.success or not refined_result.output.strip():
                    refinement_failure = str(
                        refined_result.data.get("failure_reason")
                        or "直接回答精炼返回空结果。"
                    )
                    break
                current_answer = refined_result.output.strip()
                refinement_results.append(refined_result)
                refine_attempts += 1
                evaluating_candidate = True

            final_decision = replace(decision, answer=current_answer)
            final_critic = best_critic or critic_history[-1]
            direct_agent_result = AgentResult(
                agent_name="direct_answer",
                success=True,
                output=current_answer,
                data={"sources": []},
                metadata={"route": "direct_answer"},
                token_usage=final_decision.token_usage,
                latency_ms=decision.latency_ms,
                cost_usd=decision.cost_usd,
                cost_status=decision.cost_status,
            )
            plan.subtasks[0].status = "completed"
            plan.subtasks[0].result = direct_agent_result
            result = self._build_direct_answer_result(
                final_decision,
                critic_score=final_critic,
                critic_history=critic_history,
                refinement_results=refinement_results,
                refine_count=refine_count,
                refinement_rejections=refinement_rejections,
                refinement_failure=refinement_failure,
                plan=plan,
                latency_ms=(time.perf_counter() - started) * 1000,
                context_bundle=context_bundle,
            )
            report_progress(
                {
                    "type": "subtask_result",
                    "task_id": "direct-answer",
                    "result": direct_agent_result,
                }
            )
            return result
        except asyncio.CancelledError:
            raise
        except Exception:
            if routing_started and not routing_completed:
                report_progress(
                    {
                        "type": "routing",
                        "status": "done",
                        "route": "research",
                    }
                )
            logger.warning(
                "Direct-answer preflight failed; continuing with research.",
                exc_info=True,
            )
            return None

    @staticmethod
    def _direct_answer_plan(
        task: str,
        decision: DirectAnswerDecision,
        *,
        status: str,
    ) -> ResearchPlan:
        direct_agent_result = AgentResult(
            agent_name="direct_answer",
            success=True,
            output=decision.answer,
            data={"sources": []},
            metadata={"route": "direct_answer"},
            token_usage=decision.token_usage,
            latency_ms=decision.latency_ms,
            cost_usd=decision.cost_usd,
            cost_status=decision.cost_status,
        )
        return ResearchPlan(
            plan_id=f"direct-answer-{int(time.time() * 1000):x}"[-20:],
            original_task=task,
            subtasks=[
                SubTask(
                    task_id="direct-answer",
                    description=(
                        "基于稳定通用知识和当前可见上下文直接回答，"
                        "无需外部检索。"
                    ),
                    task_type="analysis",
                    dependencies=[],
                    status=status,
                    priority=1,
                    result=direct_agent_result if status == "completed" else None,
                    subtopics=["直接回答", "质量评审"],
                )
            ],
            reasoning=(
                "模型判断该问题不需要外部检索；保留单任务计划和质量评审，"
                "并在深度或质量不足时执行一次受控精炼。"
            ),
            planner_status="direct",
        )

    def _build_direct_answer_result(
        self,
        decision: DirectAnswerDecision,
        *,
        critic_score: CriticScore,
        critic_history: list[CriticScore] | None = None,
        refinement_results: list[AgentResult] | None = None,
        refine_count: int | None = None,
        refinement_rejections: int = 0,
        refinement_failure: str | None = None,
        plan: ResearchPlan,
        latency_ms: float,
        context_bundle: ContextBundle | None,
    ) -> AgentResult:
        total_usage: dict[str, int] = {}
        total_cost = self._new_cost_accumulator()
        self._accumulate_usage(total_usage, decision, total_cost)
        evaluated_scores = critic_history or [critic_score]
        for score in evaluated_scores:
            self._accumulate_usage(total_usage, score, total_cost)
        for refined_result in refinement_results or []:
            self._accumulate_usage(total_usage, refined_result, total_cost)
        cost_usd, cost_status = self._cost_summary(total_cost)
        quality_status = (
            "evaluation_failed"
            if critic_score.evaluation_status == "failed"
            else "evaluated"
        )
        quality = (
            critic_score.overall
            if critic_score.evaluation_status == "evaluated"
            else None
        )
        threshold = float(self._settings.agent.critic_threshold)
        below_threshold = quality is not None and quality < threshold
        depth_incomplete = self._direct_answer_needs_depth_refinement(
            plan.original_task,
            decision.answer,
        )
        outcome = (
            "degraded"
            if (
                below_threshold
                or quality_status == "evaluation_failed"
                or depth_incomplete
                or refinement_failure is not None
            )
            else "success"
        )
        failure_reason = (
            (
                f"直接回答质量评分 {quality:.1f}/10 未达到验收阈值。"
                if below_threshold and quality is not None
                else None
            )
            or (
                f"直接回答评审失败：{critic_score.evaluation_error}"
                if quality_status == "evaluation_failed"
                else None
            )
            or (
                "直接回答在精炼后仍未达到用户要求的详细程度。"
                if depth_incomplete
                else None
            )
            or (
                f"直接回答精炼失败：{refinement_failure}"
                if refinement_failure
                else None
            )
        )
        refine_rounds = (
            len(refinement_results or [])
            if refine_count is None
            else refine_count
        )
        result = AgentResult(
            agent_name="orchestrator",
            success=True,
            output=decision.answer,
            data={
                "sources": [],
                "intent": "direct_answer",
                "route_reason": decision.reason,
                "route_confidence": decision.confidence,
                "critic_score": critic_score.to_dict(),
                "critic_history": [
                    score.to_dict() for score in evaluated_scores
                ],
                "refine_rounds": refine_rounds,
                "refinement_rejections": refinement_rejections,
                "plan": plan.to_dict(),
            },
            metadata={
                "quality": quality,
                "quality_status": quality_status,
                "cost": cost_usd,
                "cost_status": cost_status,
                "subtask_count": 1,
                "completed_subtask_count": 1,
                "failed_subtask_count": 0,
                "refine_rounds": refine_rounds,
                "refinement_status": (
                    "failed"
                    if refinement_failure
                    else (
                        "rejected"
                        if refinement_rejections and refine_rounds == 0
                        else (
                            "completed"
                            if refine_rounds > 0
                            else "not_needed"
                        )
                    )
                ),
                "model": decision.model,
                "outcome": outcome,
                "route": "direct_answer",
                "grounding_status": "not_required",
                "citation_status": "not_applicable",
            },
            token_usage=total_usage,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            cost_status=cost_status,
        )
        if failure_reason:
            result.data["partial_failure"] = failure_reason
            result.metadata["failure_reason"] = failure_reason
        Orchestrator._attach_context_metadata(result, context_bundle)
        return result

    @staticmethod
    def _direct_answer_needs_depth_refinement(
        task: str,
        answer: str,
    ) -> bool:
        profile = response_profile(task)
        if profile.depth != "deep" or profile.min_chars is None:
            return False
        minimum_acceptable = max(1_600, int(profile.min_chars * 0.9))
        return len(answer.strip()) < minimum_acceptable

    @staticmethod
    def _report_needs_depth_refinement(
        task: str,
        draft: str,
        plan: ResearchPlan,
    ) -> bool:
        profile = response_profile(
            task,
            subtask_count=len(plan.subtasks),
            final_report=True,
        )
        if profile.depth != "deep" or profile.min_chars is None:
            return False
        minimum_acceptable = max(2_400, int(profile.min_chars * 0.9))
        return len(draft.strip()) < minimum_acceptable

    @staticmethod
    def _depth_refinement_feedback(
        task: str,
        draft: str,
        failed_score: CriticScore,
    ) -> CriticScore:
        profile = response_profile(task, final_report=True)
        target = profile.min_chars or max(2_400, len(draft))
        return CriticScore(
            issues=[
                (
                    "当前草稿未达到用户显式要求的详细程度，核心主题、"
                    "关键条件、例子、限制或实践建议仍有缺口。"
                )
            ],
            suggestions=[
                (
                    "基于现有草稿和可见上下文补齐缺失维度，保持信息密度，"
                    f"正文以不少于约 {target} 个中文字符为目标；"
                    "不得重复结论、虚构事实或编造引用。"
                )
            ],
            should_refine=True,
            evaluation_status="failed",
            evaluation_error=failed_score.evaluation_error,
        )

    @staticmethod
    def _build_conversational_result(
        turn: ConversationalTurn,
        *,
        context_bundle: ContextBundle | None,
    ) -> AgentResult:
        result = AgentResult(
            agent_name="orchestrator",
            success=True,
            output=turn.response,
            data={
                "sources": [],
                "intent": "conversation",
                "conversation_kind": turn.kind,
                "critic_score": None,
                "refine_rounds": 0,
            },
            metadata={
                "quality": None,
                "quality_status": "not_evaluated",
                "cost": None,
                "cost_status": "not_applicable",
                "subtask_count": 0,
                "refine_rounds": 0,
                "model": "deterministic-router",
                "outcome": "success",
                "route": "conversation",
            },
            cost_status="not_applicable",
        )
        Orchestrator._attach_context_metadata(result, context_bundle)
        return result

    _DEEP_RESEARCH_MARKERS = (
        "深入",
        "全面",
        "系统",
        "详细分析",
        "研究报告",
        "技术方案",
        "实施方案",
        "架构设计",
        "文献综述",
        "发展趋势",
        "风险评估",
        "多维度",
        "逐步验证",
        "deep research",
        "comprehensive",
    )
    _COMPARISON_MARKERS = (
        "比较",
        "对比",
        "区别",
        "差异",
        "优缺点",
        "利弊",
        "取舍",
        " versus ",
        " vs ",
    )

    def _research_mode(self) -> str:
        mode = str(
            getattr(self._settings.agent, "research_mode", "balanced")
        ).strip().lower()
        return mode if mode in {"fast", "balanced", "deep"} else "balanced"

    @classmethod
    def _is_simple_task(cls, task: str) -> bool:
        normalized = re.sub(r"\s+", " ", task).strip().casefold()
        if not normalized or len(normalized) > 120:
            return False
        if any(marker in normalized for marker in cls._DEEP_RESEARCH_MARKERS):
            return False
        if PlannerAgent._is_comparison_task(task):
            return False
        sentence_breaks = len(re.findall(r"[。！？!?;\n]", normalized))
        return sentence_breaks <= 2

    @classmethod
    def _can_use_direct_plan(cls, task: str) -> bool:
        normalized = re.sub(r"\s+", " ", task).strip().casefold()
        if not normalized or len(normalized) > 160:
            return False
        if any(marker in normalized for marker in cls._DEEP_RESEARCH_MARKERS):
            return False
        if PlannerAgent._is_comparison_task(task):
            if PlannerAgent._comparison_subjects(task) is None:
                return False
            sentence_breaks = len(re.findall(r"[。！？!?;\n]", normalized))
            return sentence_breaks <= 1
        if PlannerAgent._minimum_subtask_count(task) > 1:
            return False
        sentence_breaks = len(re.findall(r"[。！？!?;\n]", normalized))
        return sentence_breaks <= 2

    async def _create_plan(self, task: str) -> ResearchPlan:
        if self._planner_injected:
            return await self._planner.run(task)
        mode = self._research_mode()
        max_subtasks = getattr(self._settings.agent, "max_subtasks", 5)
        if mode == "fast" or (
            mode == "balanced"
            and (
                max_subtasks == 1
                or self._can_use_direct_plan(task)
            )
        ):
            return ResearchPlan(
                plan_id=f"direct-{int(time.time() * 1000):x}"[-12:],
                original_task=task,
                subtasks=[
                    SubTask(
                        task_id="t1",
                        description=task,
                        task_type="research",
                        dependencies=[],
                        priority=1,
                        subtopics=[task],
                    )
                ],
                reasoning=(
                    "该问题范围集中，可由单个研究任务直接处理，"
                    "跳过额外规划以降低延迟。"
                ),
                planner_status="direct",
            )
        if mode == "balanced" and max_subtasks >= 2:
            comparison_plan = self._create_direct_comparison_plan(task)
            if comparison_plan is not None:
                return comparison_plan
        return await self._planner.run(task)

    @staticmethod
    def _create_direct_comparison_plan(
        task: str,
    ) -> ResearchPlan | None:
        if not PlannerAgent._is_comparison_task(task):
            return None
        subjects = PlannerAgent._comparison_subjects(task)
        if subjects is None:
            return None

        subtasks: list[SubTask] = []
        for index, subject in enumerate(subjects, 1):
            subtasks.append(
                SubTask(
                    task_id=f"t{index}",
                    description=(
                        f"围绕原问题“{task}”评估 {subject}，"
                        "收集其优势、限制、适用条件和关键依据。"
                    ),
                    task_type="research",
                    dependencies=[],
                    priority=1,
                    subtopics=[
                        f"{subject} 的核心优势与限制",
                        f"{subject} 的生态与工程成熟度",
                        f"{subject} 的性能、部署与运维条件",
                        f"{subject} 的典型适用场景",
                    ],
                )
            )

        return ResearchPlan(
            plan_id=f"compare-{int(time.time() * 1000):x}"[-12:],
            original_task=task,
            subtasks=subtasks,
            reasoning=(
                "该问题包含两个明确的比较对象，分别收集证据后由 "
                "Synthesizer 统一完成对比和选型结论，无需增加重复的汇总子任务。"
            ),
            planner_status="direct",
        )

    def _should_run_critic(
        self,
        task: str,
        plan: ResearchPlan,
    ) -> bool:
        del plan
        if is_conversational_task(task):
            return False
        return self._research_mode() != "fast"

    def _max_refine_rounds(self, plan: ResearchPlan) -> int:
        if self._research_mode() == "fast":
            return 0
        profile = response_profile(
            plan.original_task,
            subtask_count=len(plan.subtasks),
            final_report=True,
        )
        if (
            self._research_mode() == "balanced"
            and len(plan.subtasks) <= 1
            and profile.depth != "deep"
        ):
            return 0
        configured = max(0, self._settings.agent.max_refine_rounds)
        if self._research_mode() == "balanced":
            return min(1, configured)
        return configured

    def _max_direct_refine_rounds(self, task: str) -> int:
        del task
        if self._research_mode() == "fast":
            return 0
        configured = max(
            0,
            int(self._settings.agent.max_refine_rounds),
        )
        if self._research_mode() == "balanced":
            return min(1, configured)
        return configured

    def _should_refine_report(self, score: CriticScore) -> bool:
        if self._research_mode() == "balanced":
            threshold = float(
                getattr(self._settings.agent, "critic_threshold", 7.0)
            )
            if score.overall >= threshold:
                return False
            return any(
                value < threshold
                for value in (
                    score.completeness,
                    score.depth,
                    score.clarity,
                )
            )
        return score.should_refine

    @staticmethod
    def _is_refinement_improvement(
        candidate: CriticScore,
        baseline: CriticScore,
        *,
        candidate_draft: str,
        baseline_draft: str,
        baseline_needs_depth_refinement: bool = False,
        candidate_needs_depth_refinement: bool = False,
    ) -> bool:
        """Accept only revisions that improve measured quality without regressions."""
        if candidate.evaluation_status != "evaluated":
            return (
                baseline.evaluation_status != "evaluated"
                and len(candidate_draft.strip())
                >= max(1, int(len(baseline_draft.strip()) * 1.2))
            )
        if baseline.evaluation_status != "evaluated":
            return True

        candidate_violations = len(candidate.contract_violations)
        baseline_violations = len(baseline.contract_violations)
        dimensions = (
            "completeness",
            "accuracy",
            "depth",
            "clarity",
            "citations",
        )
        largest_regression = max(
            getattr(baseline, dimension) - getattr(candidate, dimension)
            for dimension in dimensions
        )
        if candidate_violations > baseline_violations:
            return False
        if (
            baseline_needs_depth_refinement
            and not candidate_needs_depth_refinement
        ):
            return (
                candidate.overall >= baseline.overall - 0.2
                and largest_regression <= 0.5
            )
        if candidate_violations < baseline_violations:
            return (
                candidate.overall >= baseline.overall - 0.2
                and largest_regression <= 0.5
            )
        return (
            candidate.overall >= baseline.overall + 0.2
            and largest_regression <= 0.5
        )

    def _get_tracer(self) -> Any:
        observability = getattr(self._settings, "observability", None)
        if observability is None:
            return getattr(self, "_tracer", None)
        if not observability.enable_tracing:
            return None
        if self._tracer is None:
            try:
                from mindforge.observability.tracer import get_tracer

                self._tracer = get_tracer()
            except Exception:
                logger.exception("Tracer initialization failed.")
                return None
        return self._tracer

    def _research_trace(self, task: str, *, transport: str):
        tracer = self._get_tracer()
        if tracer is None or tracer.current_trace_id is not None:
            return nullcontext(None)
        return tracer.span(
            "orchestrator.research",
            metadata={
                "component": "orchestrator",
                "transport": transport,
                "task_chars": len(task),
                "display_name": task,
            },
        )

    def _agent_span(
        self,
        agent: str,
        *,
        metadata: dict[str, Any] | None = None,
    ):
        tracer = self._get_tracer()
        if tracer is None:
            return nullcontext(None)
        return tracer.span(
            f"agent.{agent}",
            metadata={
                "agent": agent,
                **(metadata or {}),
            },
        )

    def _current_trace_id(self) -> str | None:
        tracer = self._get_tracer()
        return tracer.current_trace_id if tracer is not None else None

    def _attach_trace_id(self, result: AgentResult) -> None:
        trace_id = self._current_trace_id()
        if not trace_id:
            return
        result.trace_id = trace_id
        result.metadata = {
            **result.metadata,
            "trace_id": trace_id,
        }

    @staticmethod
    def _finish_root_span(
        root_span: Any,
        result: AgentResult | None,
    ) -> None:
        if root_span is None:
            return
        if result is None:
            root_span.metadata.setdefault("status", "error")
            return
        root_span.output = {
            "success": result.success,
            "latency_ms": round(result.latency_ms, 3),
            "cost_usd": result.cost_usd,
            "cost_status": result.cost_status,
            "total_tokens": result.token_usage.get("total_tokens"),
            "subtask_count": result.metadata.get("subtask_count", 0),
            "from_cache": bool(result.data.get("from_cache")),
            "report_chars": len(result.output),
            "route": result.metadata.get("route", "research"),
        }
        outcome = str(result.metadata.get("outcome") or "").strip().lower()
        root_span.metadata["status"] = (
            "degraded"
            if outcome == "degraded"
            else ("success" if result.success else "error")
        )
        if not result.success and not root_span.error:
            root_span.error = (
                str(result.data.get("error") or result.output or "Research failed.")
            )[:1000]

    async def _execute_subtask(
        self,
        subtask: SubTask,
        plan: ResearchPlan,
        *,
        working_memory: WorkingMemory,
    ) -> AgentResult:
        """Execute a single subtask with a timeout.

        The timeout is read from ``settings.agent.subtask_timeout`` (default 45 s).
        """
        timeout = self._settings.agent.subtask_timeout
        queue_timeout = getattr(
            self._settings.agent,
            "queue_timeout",
            30,
        )

        try:
            await asyncio.wait_for(
                self._subtask_semaphore.acquire(),
                timeout=queue_timeout,
            )
        except asyncio.TimeoutError:
            return AgentResult(
                agent_name="researcher",
                success=False,
                output=(
                    f"Subtask '{subtask.task_id}' could not start because "
                    "the research worker queue is full."
                ),
                data={
                    "task_id": subtask.task_id,
                    "error": "subtask_queue_timeout",
                },
            )

        try:
            dependency_context = self._build_dependency_context(
                subtask,
                plan,
            )
            max_context_chars = int(
                getattr(
                    self._settings.agent,
                    "research_context_max_chars",
                    12_000,
                )
            )
            dependency_context = dependency_context.strip()[:max_context_chars]
            remaining = max(0, max_context_chars - len(dependency_context))
            context_settings = getattr(self._settings, "context", None)
            context_view = working_memory.get_relevant_context(
                subtask.description,
                max_chars=remaining,
                include_types={"context", "thought"},
                min_relevance=float(
                    getattr(
                        context_settings,
                        "subtask_context_min_relevance",
                        0.08,
                    )
                ),
                max_items=int(
                    getattr(
                        context_settings,
                        "subtask_context_max_items",
                        8,
                    )
                ),
                allowed_producer_ids=set(subtask.dependencies),
            )
            context = "\n\n".join(
                section
                for section in (
                    context_view.text,
                    dependency_context,
                )
                if section
            )
            with self._agent_span(
                "researcher",
                metadata={
                    "subtask_id": subtask.task_id,
                    "task_type": subtask.task_type,
                },
            ) as researcher_span:
                if researcher_span is not None:
                    researcher_span.input = {
                        "description": subtask.description,
                        "context_chars": len(context),
                        **context_view.to_dict(),
                    }
                researcher_kwargs: dict[str, Any] = {
                    "context": context or None,
                }
                if isinstance(self._researcher, ResearcherAgent):
                    configured_rounds = self._settings.agent.max_iterations
                    mode = self._research_mode()
                    researcher_kwargs["task_type"] = subtask.task_type
                    researcher_kwargs["subtopics"] = subtask.subtopics
                    researcher_kwargs["total_subtasks"] = len(plan.subtasks)
                    researcher_kwargs["deadline"] = (
                        time.perf_counter() + timeout
                    )
                    researcher_kwargs["max_rounds"] = (
                        1
                        if mode == "fast"
                        else (
                            min(configured_rounds, 2)
                            if mode == "balanced"
                            and self._is_simple_task(subtask.description)
                            else configured_rounds
                        )
                    )
                try:
                    result = await asyncio.wait_for(
                        self._researcher.run(
                            subtask.description,
                            **researcher_kwargs,
                        ),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    if researcher_span is not None:
                        researcher_span.metadata.update(
                            {
                                "status": "error",
                                "stage": "subtask_execution",
                                "error_code": "subtask_timeout",
                                "error_type": "TimeoutError",
                                "timeout_seconds": timeout,
                            }
                        )
                        researcher_span.error = (
                            f"Subtask '{subtask.task_id}' timed out "
                            f"after {timeout:g} seconds."
                        )
                    raise
                except asyncio.CancelledError:
                    if researcher_span is not None:
                        researcher_span.metadata.update(
                            {
                                "status": "cancelled",
                                "stage": "subtask_execution",
                                "error_code": "subtask_cancelled",
                                "error_type": "CancelledError",
                            }
                        )
                        researcher_span.error = (
                            f"Subtask '{subtask.task_id}' was cancelled."
                        )
                    raise
                except Exception as exc:
                    if researcher_span is not None:
                        researcher_span.metadata.update(
                            {
                                "status": "error",
                                "stage": "subtask_execution",
                                "error_code": "subtask_failed",
                                "error_type": type(exc).__name__,
                            }
                        )
                        researcher_span.error = (
                            f"Subtask '{subtask.task_id}' failed: "
                            f"{self._describe_exception(exc)}"
                        )
                    raise
                if researcher_span is not None:
                    researcher_span.output = {
                        "success": result.success,
                        "output_chars": len(result.output),
                        "source_count": len(result.data.get("sources", [])),
                    }
            result.data = {
                **result.data,
                "context_scope": context_view.to_dict(),
            }
            return result
        except asyncio.TimeoutError:
            message = (
                f"Subtask '{subtask.task_id}' timed out after {timeout:g} seconds."
            )
            return AgentResult(
                agent_name="researcher",
                success=False,
                output=message,
                data={
                    "task_id": subtask.task_id,
                    "error_code": "subtask_timeout",
                    "error_type": "TimeoutError",
                    "error_message": message,
                    "stage": "subtask_execution",
                    "timeout_seconds": timeout,
                },
            )
        except Exception as exc:
            logger.exception(
                "Subtask execution failed: %s",
                subtask.task_id,
            )
            detail = self._describe_exception(exc)
            message = f"Subtask '{subtask.task_id}' failed: {detail}"
            return AgentResult(
                agent_name="researcher",
                success=False,
                output=message,
                data={
                    "task_id": subtask.task_id,
                    "error_code": "subtask_failed",
                    "error_type": type(exc).__name__,
                    "error_message": message,
                    "stage": "subtask_execution",
                },
            )
        finally:
            self._subtask_semaphore.release()

    async def _execute_ready_tasks(
        self,
        ready: list[SubTask],
        *,
        plan: ResearchPlan,
        working_memory: WorkingMemory,
        total_usage: dict[str, int],
        total_cost: dict[str, Any],
    ) -> list[SubTask]:
        results = await asyncio.gather(
            *[
                self._execute_subtask(
                    st,
                    plan,
                    working_memory=working_memory,
                )
                for st in ready
            ],
            return_exceptions=True,
        )
        for st, result in zip(ready, results):
            if isinstance(result, BaseException):
                st.status = "failed"
                detail = self._describe_exception(result)
                st.result = AgentResult(
                    agent_name="researcher",
                    success=False,
                    output=f"Subtask '{st.task_id}' failed: {detail}",
                    data={
                        "task_id": st.task_id,
                        "error_code": "subtask_failed",
                        "error_type": type(result).__name__,
                        "error_message": detail,
                        "stage": "subtask_execution",
                    },
                )
            else:
                st.status = "completed" if result.success else "failed"
                st.result = result
            self._accumulate_usage(total_usage, st.result, total_cost)
            self._record_working_result(
                working_memory,
                st,
                st.result,
            )
        return ready

    @staticmethod
    def _fail_deadlocked_tasks(
        plan: ResearchPlan,
    ) -> list[SubTask]:
        failed: list[SubTask] = []
        for st in plan.subtasks:
            if st.status != "pending":
                continue
            st.status = "failed"
            st.result = AgentResult(
                agent_name="researcher",
                success=False,
                output=(f"Subtask {st.task_id} deadlocked: unmet dependencies."),
            )
            failed.append(st)
        return failed

    @staticmethod
    def _subtask_output(st: SubTask) -> dict[str, Any]:
        result = st.result
        error = None
        if result is not None and not result.success:
            error = {
                key: result.data.get(key)
                for key in (
                    "error_code",
                    "error_type",
                    "error_message",
                    "stage",
                    "timeout_seconds",
                )
                if result.data.get(key) is not None
            }
        return {
            "task_id": st.task_id,
            "description": st.description,
            "task_type": st.task_type,
            "output": result.output if result else "",
            "sources": (
                result.data.get("sources", []) if result and result.data else []
            ),
            "success": result.success if result else False,
            "outcome": (
                str(
                    result.metadata.get(
                        "outcome",
                        result.data.get("outcome", "success"),
                    )
                )
                if result
                else "error"
            ),
            "grounding_status": (
                result.metadata.get(
                    "grounding_status",
                    result.data.get("grounding_status"),
                )
                if result
                else None
            ),
            "citation_status": (
                result.metadata.get(
                    "citation_status",
                    result.data.get("citation_status"),
                )
                if result
                else None
            ),
            "failure_reason": (
                result.metadata.get(
                    "failure_reason",
                    result.data.get("failure_reason"),
                )
                if result
                else None
            ),
            "source_warning": (
                result.metadata.get(
                    "source_warning",
                    result.data.get("source_warning"),
                )
                if result
                else None
            ),
            "source_requirement": (
                result.data.get("source_requirement")
                if result and result.data
                else None
            ),
            "output_mode": (
                result.metadata.get(
                    "output_mode",
                    result.data.get("output_mode"),
                )
                if result
                else None
            ),
            "contract_version": (
                result.metadata.get(
                    "contract_version",
                    result.data.get("contract_version"),
                )
                if result
                else None
            ),
            "context_scope": (
                result.data.get("context_scope")
                if result and result.data
                else None
            ),
            "error": error,
        }

    def _describe_exception(self, exc: BaseException) -> str:
        tracer = self._get_tracer()
        if tracer is not None:
            return tracer.describe_exception(exc)
        message = str(exc).strip()
        return (message or type(exc).__name__)[:1000]

    async def _recall_cached_result(
        self,
        task: str,
        *,
        start_time: float,
        pipeline_log: dict[str, Any] | None = None,
        fingerprint_task: str | None = None,
    ) -> AgentResult | None:
        if self._episodic_memory is None:
            return None
        try:
            cached = await self._episodic_memory.recall(task)
            if not isinstance(cached, dict):
                return None
            result = AgentResult.from_dict(cached)
            if not result.output.strip():
                return None
            if self._cached_result_requires_refresh(result):
                return None
            if ResearcherAgent.requires_sources(task):
                cached_sources = result.data.get("sources")
                if (
                    not isinstance(cached_sources, list)
                    or not cached_sources
                    or not ResearcherAgent._contains_citation_marker(
                        result.output
                    )
                ):
                    return None
            cached_subtasks = result.data.get("subtask_outputs")
            has_failed_subtask = (
                isinstance(cached_subtasks, list)
                and any(
                    isinstance(item, dict) and not item.get("success")
                    for item in cached_subtasks
                )
            )
            if (
                str(result.metadata.get("outcome") or "").strip().lower()
                == "degraded"
                or has_failed_subtask
                or result.metadata.get("quality_status") == "evaluation_failed"
                or result.metadata.get("citation_status") == "invalid"
            ):
                return None
            cached_plan = result.data.get("plan")
            if (
                isinstance(cached_plan, dict)
                and cached_plan.get("planner_status") == "fallback"
            ):
                return None
            if (
                result.metadata.get("research_cache_fingerprint")
                != self._research_cache_fingerprint(fingerprint_task)
            ):
                return None
            cached_generation_usage = dict(result.token_usage)
            cached_generation_cost_usd = result.cost_usd
            cached_generation_cost_status = result.cost_status
            result.data = {
                **result.data,
                "from_cache": True,
            }
            result.metadata = {
                **result.metadata,
                "cache_hit": True,
                "cached_generation_token_usage": cached_generation_usage,
                "cached_generation_cost_usd": cached_generation_cost_usd,
                "cached_generation_cost_status": cached_generation_cost_status,
                "cost": None,
                "cost_status": "not_applicable",
            }
            critic_score = result.data.get("critic_score")
            legacy_unreviewed = (
                result.metadata.get("quality") == 0
                and not isinstance(critic_score, dict)
            )
            if legacy_unreviewed:
                result.metadata["quality"] = None
                result.metadata["quality_status"] = "not_evaluated"
            elif "quality_status" not in result.metadata:
                result.metadata["quality_status"] = (
                    "evaluated"
                    if isinstance(result.metadata.get("quality"), (int, float))
                    else "not_evaluated"
                )
            if pipeline_log is not None:
                result.data["pipeline"] = pipeline_log
            result.token_usage = {}
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            result.cost_usd = None
            result.cost_status = "not_applicable"
            return result
        except Exception as exc:
            logger.warning("Episodic memory recall failed: %s", exc)
            return None

    @staticmethod
    def _cached_result_requires_refresh(result: AgentResult) -> bool:
        if "](" in result.output and "]([" in result.output:
            return True
        if "ws_call_id=" in result.output:
            return True
        sources = result.data.get("sources")
        if not isinstance(sources, list):
            return False
        return any(
            isinstance(source, dict)
            and "ws_call_id=" in str(source.get("url") or "")
            for source in sources
        )

    def _research_cache_fingerprint(self, task: str | None = None) -> str:
        """Identify the execution strategy that produced a cached report."""
        settings = self._settings
        llm = getattr(settings, "llm", None)
        agent = getattr(settings, "agent", None)
        web_search = getattr(settings, "web_search", None)
        provider = str(getattr(llm, "llm_provider", "") or "").strip().lower()

        def call_llm(method_name: str, *args: Any) -> Any:
            method = getattr(llm, method_name, None)
            if not callable(method):
                return None
            try:
                return method(*args)
            except (AttributeError, TypeError, ValueError):
                return None

        models = {
            role: (
                call_llm("get_model", role, provider)
                or getattr(llm, f"{role}_model", "")
                or ""
            )
            for role in ("planner", "researcher", "synthesizer", "critic")
        }
        payload = {
            "schema": _RESEARCH_CACHE_SCHEMA_VERSION,
            "llm": {
                "provider": provider,
                "base_url": call_llm("get_base_url", provider),
                "models": models,
                "supports_tools": call_llm("supports_tools", provider),
                "supports_json_mode": call_llm(
                    "supports_json_mode",
                    provider,
                ),
                "supports_json_schema": call_llm(
                    "supports_json_schema",
                    provider,
                ),
                "native_web_search_protocol": call_llm(
                    "get_native_web_search_protocol",
                    provider,
                ),
                "native_web_search_endpoint": call_llm(
                    "get_native_web_search_endpoint",
                    provider,
                ),
            },
            "agent": {
                name: getattr(agent, name, None)
                for name in (
                    "research_mode",
                    "source_policy",
                    "fallback_enabled",
                    "max_iterations",
                    "max_subtasks",
                    "max_refine_rounds",
                    "critic_threshold",
                    "llm_request_timeout",
                    "subtask_timeout",
                    "research_timeout",
                    "research_context_max_chars",
                    "synthesis_context_max_chars",
                    "critic_source_context_max_chars",
                    "critic_report_context_max_chars",
                )
            },
            "web_search": {
                name: getattr(web_search, name, None)
                for name in (
                    "native_enabled",
                    "duckduckgo_enabled",
                    "model_only_fallback",
                    "max_results",
                    "native_max_output_tokens",
                    "native_timeout_seconds",
                    "native_failure_cooldown_seconds",
                )
            },
            "knowledge_index_signature": self._knowledge_index_signature(),
            "freshness_bucket": self._freshness_bucket(task),
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _freshness_bucket(task: str | None) -> str | None:
        if not task:
            return None
        lowered = task.lower()
        volatile = (
            "今天",
            "当前",
            "最新",
            "实时",
            "价格",
            "天气",
            "比分",
            "股价",
            "today",
            "latest",
            "current",
            "live",
        )
        if not any(marker in lowered for marker in volatile):
            return None
        return time.strftime("%Y-%m-%dT%H", time.gmtime())

    @staticmethod
    def _knowledge_index_signature() -> str:
        """Hash enabled indexed-document revisions without making DB mandatory."""
        try:
            from mindforge.db import DocumentCatalog, SessionLocal

            with SessionLocal() as db:
                rows = (
                    db.query(
                        DocumentCatalog.doc_id,
                        DocumentCatalog.index_signature,
                        DocumentCatalog.updated_at,
                    )
                    .filter(
                        DocumentCatalog.enabled.is_(True),
                        DocumentCatalog.status == "indexed",
                    )
                    .order_by(DocumentCatalog.doc_id.asc())
                    .all()
                )
            payload = [
                (
                    str(row.doc_id),
                    str(row.index_signature or ""),
                    row.updated_at.isoformat() if row.updated_at else "",
                )
                for row in rows
            ]
            return hashlib.sha256(
                json.dumps(payload, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        except Exception:
            return "unavailable"

    def _cache_task_key(
        self,
        task: str,
        context_bundle: ContextBundle | None,
    ) -> str:
        if context_bundle is None:
            return task
        payload = {
            "task": task.strip(),
            "context": context_bundle.fingerprint,
            "execution": self._research_cache_fingerprint(task),
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return f"context-cache-v1:{digest}"

    async def _create_working_memory(
        self,
        task: str,
        *,
        request_context: ResearchRequestContext | None = None,
        context_bundle: ContextBundle | None = None,
    ) -> WorkingMemory:
        memory = WorkingMemory()
        if context_bundle is not None:
            memory.add_context(context_bundle.to_working_chunks())
            return memory
        if request_context is not None and (
            request_context.independent
            or request_context.context_mode == "disabled"
        ):
            return memory
        if self._semantic_memory is None:
            return memory
        try:
            facts = await self._semantic_memory.recall(task, top_k=3)
        except Exception as exc:
            logger.warning("Semantic memory recall failed: %s", exc)
            return memory
        memory_config = getattr(self._settings, "memory", None)
        max_total_chars = int(
            getattr(memory_config, "semantic_recall_context_chars", 4000)
        )
        max_fact_chars = int(
            getattr(memory_config, "semantic_recall_fact_chars", 2000)
        )
        chunks: list[dict[str, Any]] = []
        remaining = max_total_chars
        for fact in facts:
            if remaining <= 0:
                break
            recall_content = getattr(self._semantic_memory, "recall_content", None)
            content = (
                recall_content(fact)
                if callable(recall_content)
                else str(fact.content)[:max_fact_chars]
            )
            content = str(content)[: min(max_fact_chars, remaining)]
            if not content.strip():
                continue
            chunks.append(
                {
                    "id": f"semantic:{fact.fact_id}",
                    "content": content,
                    "rerank_score": float(
                        getattr(fact, "match_score", 0.0)
                        or getattr(fact, "confidence", 0.5)
                    ),
                    "sources": fact.sources,
                    "memory_type": "semantic",
                }
            )
            remaining -= len(content)
        memory.add_context(chunks)
        return memory

    @staticmethod
    def _record_working_result(
        memory: WorkingMemory,
        subtask: SubTask,
        result: AgentResult,
    ) -> None:
        if result.output.strip():
            memory.add_tool_result(
                f"subtask:{subtask.task_id}",
                result.output,
                importance=0.9 if result.success else 0.3,
            )
        sources = result.data.get("sources", [])
        if isinstance(sources, list):
            memory.add_context(
                [
                    {
                        **source,
                        "producer_subtask_id": subtask.task_id,
                    }
                    for source in sources
                    if isinstance(source, dict)
                ]
            )

    async def _store_memories(
        self,
        task: str,
        result: AgentResult,
        *,
        semantic_task: str | None = None,
        fingerprint_task: str | None = None,
    ) -> None:
        outcome = str(result.metadata.get("outcome") or "").strip().lower()
        grounding_status = str(
            result.metadata.get("grounding_status") or ""
        ).strip().lower()
        is_complete_success = (
            result.success
            and outcome != "degraded"
            and grounding_status != "model_only"
        )
        if self._episodic_memory is not None and is_complete_success:
            try:
                result.metadata = {
                    **result.metadata,
                    "research_cache_fingerprint": (
                        self._research_cache_fingerprint(fingerprint_task)
                    ),
                }
                await self._episodic_memory.store(task, result)
            except Exception as exc:
                logger.warning("Episodic memory store failed: %s", exc)
        if (
            self._semantic_memory is not None
            and is_complete_success
            and semantic_task is not None
        ):
            quality = result.metadata.get("quality")
            quality_status = result.metadata.get("quality_status")
            confidence = (
                max(0.0, min(1.0, float(quality) / 10.0))
                if isinstance(quality, (int, float)) and quality > 0
                else 0.5
            )
            if (
                quality_status == "evaluated"
                and isinstance(quality, (int, float))
                and quality >= self._settings.agent.critic_threshold
            ):
                try:
                    await self._semantic_memory.store(
                        semantic_task,
                        result.output,
                        sources=result.data.get("sources", []),
                        confidence=confidence,
                    )
                except Exception as exc:
                    logger.warning("Semantic memory store failed: %s", exc)

    @staticmethod
    def _attach_context_metadata(
        result: AgentResult,
        context_bundle: ContextBundle | None,
    ) -> None:
        if context_bundle is None:
            return
        result.data = {
            **result.data,
            "context_snapshot_id": context_bundle.snapshot_id,
            "context_items_used": len(context_bundle.items),
        }
        result.metadata = {
            **result.metadata,
            "context_snapshot_id": context_bundle.snapshot_id,
            "context_token_usage": context_bundle.used_tokens,
            "context_fingerprint": context_bundle.fingerprint,
            "reused_artifact_count": sum(
                item.source_type == "artifact"
                for item in context_bundle.items
            ),
        }

    def _build_success_result(
        self,
        *,
        output: str,
        plan: ResearchPlan,
        subtask_outputs: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        final_critic: CriticScore | None,
        refine_count: int,
        total_usage: dict[str, int],
        elapsed_ms: float,
        total_cost_usd: float | None,
        cost_status: str,
        pipeline_log: dict[str, Any] | None = None,
        refinement_failure: str | None = None,
        refinement_rejections: int = 0,
        citation_verification: dict[str, Any] | None = None,
    ) -> AgentResult:
        completed_subtasks = sum(
            1 for output in subtask_outputs if output.get("success")
        )
        failed_subtasks = len(subtask_outputs) - completed_subtasks
        degradation_reasons: list[str] = []
        if failed_subtasks:
            degradation_reasons.append(
                self._format_partial_failure(subtask_outputs)
            )
        degraded_subtasks = [
            item
            for item in subtask_outputs
            if item.get("success") and item.get("outcome") == "degraded"
        ]
        if degraded_subtasks:
            model_only_count = sum(
                1
                for item in degraded_subtasks
                if item.get("grounding_status") == "model_only"
            )
            degradation_reasons.append(
                (
                    f"{model_only_count} 个子任务未获得可核验来源，"
                    "已保留模型自身回答。"
                )
                if model_only_count
                else f"{len(degraded_subtasks)} 个子任务以降级模式完成。"
            )
        if plan.planner_status == "fallback":
            degradation_reasons.append(
                "Planner 规划失败，当前使用保留原问题语义的单任务回退计划。"
                + (
                    f" 原因：{plan.planner_error}"
                    if plan.planner_error
                    else ""
                )
            )
        if (
            final_critic is not None
            and final_critic.evaluation_status == "failed"
        ):
            degradation_reasons.append(
                "质量评审失败，报告未完成有效评审。"
                + (
                    f" 原因：{final_critic.evaluation_error}"
                    if final_critic.evaluation_error
                    else ""
                )
            )
        if (
            final_critic is not None
            and final_critic.evaluation_status == "evaluated"
            and final_critic.overall
            < float(
                getattr(
                    getattr(self._settings, "agent", None),
                    "critic_threshold",
                    7.0,
                )
            )
        ):
            degradation_reasons.append(
                "报告质量评分 "
                f"{final_critic.overall:.1f}/10 未达到验收阈值。"
            )
        if refinement_failure:
            degradation_reasons.append(
                "报告精炼未完成，当前展示最后一个有效版本。"
                f" 原因：{refinement_failure}"
            )
        if self._report_needs_depth_refinement(
            plan.original_task,
            output,
            plan,
        ):
            degradation_reasons.append(
                "报告在精炼后仍未达到用户要求的详细程度。"
            )
        if (
            citation_verification is not None
            and not citation_verification.get("valid", True)
        ):
            degradation_reasons.append(
                self._format_citation_failure(citation_verification)
            )
        outcome = "degraded" if degradation_reasons else "success"
        failure_reason = (
            " ".join(degradation_reasons) if degradation_reasons else None
        )
        source_warnings = list(
            dict.fromkeys(
                str(item.get("source_warning") or "").strip()
                for item in subtask_outputs
                if str(item.get("source_warning") or "").strip()
            )
        )
        source_warning = (
            "; ".join(source_warnings) if source_warnings else None
        )
        data: dict[str, Any] = {
            "plan": plan.to_dict(),
            "subtask_outputs": subtask_outputs,
            "sources": sources,
            "critic_score": (final_critic.to_dict() if final_critic else None),
            "refine_rounds": refine_count,
            "refinement_rejections": refinement_rejections,
            "citation_verification": citation_verification,
            "grounding_status": (
                "model_only"
                if any(
                    item.get("grounding_status") == "model_only"
                    for item in subtask_outputs
                )
                else ("grounded" if sources else "not_required")
            ),
        }
        if failure_reason:
            data["partial_failure"] = failure_reason
        if source_warning:
            data["source_warning"] = source_warning
        if refinement_failure:
            data["refinement_failure"] = refinement_failure
        if pipeline_log is not None:
            data["pipeline"] = pipeline_log
        quality_status = (
            "not_evaluated"
            if final_critic is None
            else (
                "evaluation_failed"
                if final_critic.evaluation_status == "failed"
                else "evaluated"
            )
        )
        metadata: dict[str, Any] = {
            "quality": (
                final_critic.overall
                if final_critic is not None
                and final_critic.evaluation_status == "evaluated"
                else None
            ),
            "quality_status": quality_status,
            "cost": total_cost_usd,
            "cost_status": cost_status,
            "subtask_count": len(plan.subtasks),
            "completed_subtask_count": completed_subtasks,
            "failed_subtask_count": failed_subtasks,
            "refine_rounds": refine_count,
            "refinement_status": (
                "failed"
                if refinement_failure
                else (
                    "rejected"
                    if refinement_rejections and refine_count == 0
                    else (
                        "completed"
                        if refine_count > 0
                        else "not_needed"
                    )
                )
            ),
            "model": self._settings.llm.llm_provider,
            "outcome": outcome,
            "grounding_status": (
                "model_only"
                if any(
                    item.get("grounding_status") == "model_only"
                    for item in subtask_outputs
                )
                else ("grounded" if sources else "not_required")
            ),
            "citation_status": (
                citation_verification.get("status", "not_applicable")
                if citation_verification is not None
                else "not_applicable"
            ),
        }
        if failure_reason:
            metadata["failure_reason"] = failure_reason
        if source_warning:
            metadata["source_warning"] = source_warning
        return AgentResult(
            agent_name="orchestrator",
            success=True,
            output=output,
            data=data,
            metadata=metadata,
            token_usage=total_usage,
            latency_ms=elapsed_ms,
            cost_usd=total_cost_usd,
            cost_status=cost_status,
        )

    def _build_failure_result(
        self,
        *,
        plan: ResearchPlan,
        subtask_outputs: list[dict[str, Any]],
        total_usage: dict[str, int],
        start_time: float,
        total_cost: dict[str, Any],
        pipeline_log: dict[str, Any] | None = None,
    ) -> AgentResult:
        cost_usd, cost_status = self._cost_summary(total_cost)
        failure_reason = self._format_pipeline_failure(subtask_outputs)
        data: dict[str, Any] = {
            "plan": plan.to_dict(),
            "subtask_outputs": subtask_outputs,
            "sources": [],
            "critic_score": None,
            "refine_rounds": 0,
            "error_code": "all_subtasks_failed",
            "error_type": "SubtaskFailure",
            "error_message": failure_reason,
            "stage": "subtask_execution",
        }
        if pipeline_log is not None:
            data["pipeline"] = pipeline_log
        return AgentResult(
            agent_name="orchestrator",
            success=False,
            output=failure_reason,
            data=data,
            metadata={
                "quality": None,
                "quality_status": "not_evaluated",
                "cost": cost_usd,
                "cost_status": cost_status,
                "subtask_count": len(plan.subtasks),
                "refine_rounds": 0,
                "model": self._settings.llm.llm_provider,
                "outcome": "error",
                "failure_reason": failure_reason,
            },
            token_usage=total_usage,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            cost_usd=cost_usd,
            cost_status=cost_status,
        )

    @staticmethod
    def _format_pipeline_failure(
        subtask_outputs: list[dict[str, Any]],
    ) -> str:
        details = [
            str(output.get("output", "")).strip()
            for output in subtask_outputs
            if str(output.get("output", "")).strip()
        ]
        if not details:
            return "Research failed because no subtask completed successfully."
        return "Research failed because all subtasks failed:\n\n" + "\n".join(
            f"- {detail}" for detail in details
        )

    @staticmethod
    def _format_partial_failure(
        subtask_outputs: list[dict[str, Any]],
    ) -> str:
        failed = [
            output
            for output in subtask_outputs
            if not output.get("success")
        ]
        details = []
        for output in failed:
            task_id = str(output.get("task_id") or "unknown")
            description = str(output.get("description") or "").strip()
            reason = str(output.get("output") or "Subtask failed.").strip()
            label = f"{task_id}（{description}）" if description else task_id
            details.append(f"{label}: {reason[:300]}")
        return (
            f"{len(failed)} 个子任务未完成，当前报告仅基于其余成功结果。"
            + (f" {'; '.join(details)}" if details else "")
        )

    @staticmethod
    def _collect_sources(
        subtask_outputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Aggregate unique sources across all subtask outputs."""
        seen: set[str] = set()
        all_sources: list[dict[str, Any]] = []
        for so in subtask_outputs:
            sources = so.get("sources", [])
            if not isinstance(sources, list):
                continue
            for src in sources:
                if not isinstance(src, dict):
                    continue
                identity = Orchestrator._source_identity(src)
                if not identity or identity in seen:
                    continue
                seen.add(identity)
                all_sources.append({**src, "index": len(all_sources) + 1})
        return all_sources

    @staticmethod
    def _verify_final_citations(
        report: str,
        sources: list[dict[str, Any]],
    ) -> dict[str, Any]:
        markers = [
            int(value)
            for value in re.findall(r"\[([1-9]\d*)\]", report)
        ]
        if not sources:
            valid = not markers
            return {
                "valid": valid,
                "status": "not_applicable" if valid else "invalid",
                "total_markers": len(markers),
                "valid_markers": 0,
                "validity_score": 1.0 if valid else 0.0,
                "has_issues": not valid,
                "issues": (
                    []
                    if valid
                    else [
                        {
                            "marker": f"[{markers[0]}]",
                            "index": markers[0],
                            "type": "missing_source_list",
                            "detail": "报告包含引用标记，但没有可用来源列表。",
                        }
                    ]
                ),
                "unused_sources": [],
                "sources_used": [],
            }
        if not markers:
            return {
                "valid": False,
                "status": "invalid",
                "total_markers": 0,
                "valid_markers": 0,
                "validity_score": 0.0,
                "has_issues": True,
                "issues": [
                    {
                        "marker": "",
                        "index": None,
                        "type": "missing_markers",
                        "detail": "报告使用了来源，但正文没有引用标记。",
                    }
                ],
                "unused_sources": [
                    source.get("index")
                    for source in sources
                    if isinstance(source.get("index"), int)
                ],
                "sources_used": [],
            }

        verification = CitationVerifier().execute(
            report_text=report,
            sources=sources,
            strict_unused=False,
        )
        data = (
            dict(verification.data)
            if isinstance(verification.data, dict)
            else {}
        )
        data["valid"] = verification.success
        data["status"] = "valid" if verification.success else "invalid"
        return data

    @staticmethod
    def _strip_unbacked_citations(
        report: str,
        sources: list[dict[str, Any]],
    ) -> str:
        if sources:
            return report
        return re.sub(
            r"\s*\[([1-9]\d*)\](?!\()",
            "",
            report,
        )

    @staticmethod
    def _format_citation_failure(
        verification: dict[str, Any],
    ) -> str:
        issues = verification.get("issues")
        if isinstance(issues, list) and issues:
            first = issues[0]
            if isinstance(first, dict):
                detail = str(
                    first.get("detail")
                    or first.get("type")
                    or "引用校验未通过"
                ).strip()
                return f"最终报告引用校验未通过：{detail}"
        return "最终报告引用校验未通过。"

    @staticmethod
    def _attach_citation_maps(
        subtask_outputs: list[dict[str, Any]],
        all_sources: list[dict[str, Any]],
    ) -> None:
        global_indices = {
            Orchestrator._source_identity(source): source.get("index")
            for source in all_sources
        }
        for output in subtask_outputs:
            sources = output.get("sources")
            if not isinstance(sources, list):
                continue
            mapping: dict[str, int] = {}
            for position, source in enumerate(sources, 1):
                if not isinstance(source, dict):
                    continue
                global_index = global_indices.get(
                    Orchestrator._source_identity(source)
                )
                if not isinstance(global_index, int):
                    continue
                mapping[str(source.get("index", position))] = global_index
            if mapping:
                output["citation_map"] = mapping

    @staticmethod
    def _source_identity(source: dict[str, Any]) -> str:
        return str(
            source.get("url")
            or source.get("chunk_id")
            or source.get("id")
            or (
                f"{source.get('title', source.get('source', ''))}:"
                f"{str(source.get('content', source.get('text', '')))[:200]}"
            )
        ).strip()

    @staticmethod
    def _build_dependency_context(
        subtask: SubTask,
        plan: ResearchPlan,
    ) -> str:
        """Build grounded context from completed dependency results."""
        if not subtask.dependencies:
            return ""
        by_id = {task.task_id: task for task in plan.subtasks}
        sections: list[str] = []
        for dependency_id in subtask.dependencies:
            dependency = by_id.get(dependency_id)
            if (
                dependency is None
                or dependency.status != "completed"
                or dependency.result is None
                or not dependency.result.output
            ):
                continue
            sections.append(
                f"## 前置子任务 {dependency.task_id}: "
                f"{dependency.description}\n\n{dependency.result.output}"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _new_cost_accumulator() -> dict[str, Any]:
        return {
            "usd": 0.0,
            "estimated_calls": 0,
            "unpriced_calls": 0,
            "missing_usage_calls": 0,
            "not_applicable_calls": 0,
        }

    @staticmethod
    def _cost_summary(
        accumulator: dict[str, Any],
    ) -> tuple[float | None, str]:
        estimated = int(accumulator.get("estimated_calls", 0))
        unavailable = int(accumulator.get("unpriced_calls", 0)) + int(
            accumulator.get("missing_usage_calls", 0)
        )
        if estimated and unavailable:
            return float(accumulator.get("usd", 0.0)), "partial"
        if estimated:
            return float(accumulator.get("usd", 0.0)), "estimated"
        if int(accumulator.get("unpriced_calls", 0)):
            return None, "pricing_unconfigured"
        if int(accumulator.get("missing_usage_calls", 0)):
            return None, "usage_unavailable"
        if int(accumulator.get("not_applicable_calls", 0)):
            return None, "not_applicable"
        return None, "usage_unavailable"

    @staticmethod
    def _accumulate_cost(
        accumulator: dict[str, Any],
        amount: Any,
        status: Any,
    ) -> None:
        normalized_status = status if isinstance(status, str) else "usage_unavailable"
        if normalized_status == "usage_unavailable" and isinstance(
            amount, (int, float)
        ):
            normalized_status = "estimated"
        if normalized_status == "estimated" and isinstance(
            amount,
            (int, float),
        ):
            accumulator["usd"] = float(accumulator.get("usd", 0.0)) + float(amount)
            accumulator["estimated_calls"] = (
                int(accumulator.get("estimated_calls", 0)) + 1
            )
        elif normalized_status == "pricing_unconfigured":
            accumulator["unpriced_calls"] = (
                int(accumulator.get("unpriced_calls", 0)) + 1
            )
        elif normalized_status == "not_applicable":
            accumulator["not_applicable_calls"] = (
                int(accumulator.get("not_applicable_calls", 0)) + 1
            )
        else:
            accumulator["missing_usage_calls"] = (
                int(accumulator.get("missing_usage_calls", 0)) + 1
            )

    @staticmethod
    def _accumulate_usage(
        accumulator: dict[str, int],
        result: Any,
        cost_accumulator: Optional[dict[str, Any]] = None,
    ) -> None:
        """Merge token usage from an AgentResult or other result objects."""
        if result is None:
            return
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, (int, float)) and key != "cost_usd":
                    accumulator[key] = accumulator.get(key, 0) + int(value)
            return
        if isinstance(result, list):
            for item in result:
                Orchestrator._accumulate_usage(
                    accumulator,
                    item,
                    cost_accumulator,
                )
            return
        if cost_accumulator is not None:
            cost = getattr(
                result,
                "cost_usd",
                getattr(result, "planner_cost_usd", None),
            )
            status = getattr(
                result,
                "cost_status",
                getattr(
                    result,
                    "planner_cost_status",
                    "usage_unavailable",
                ),
            )
            Orchestrator._accumulate_cost(
                cost_accumulator,
                cost,
                status,
            )
        usage = getattr(result, "token_usage", None)
        if not usage:
            usage = getattr(result, "planner_usage", None)
        if usage:
            for k, v in usage.items():
                if isinstance(v, (int, float)) and k != "cost_usd":
                    accumulator[k] = accumulator.get(k, 0) + int(v)
