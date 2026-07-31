"""Orchestrator — top-level controller that drives the full research pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from contextlib import nullcontext
from typing import Any, AsyncIterator, Optional

from mindforge.agents.base import AgentResult
from mindforge.agents.planner import PlannerAgent, ResearchPlan, SubTask
from mindforge.agents.researcher import ResearcherAgent
from mindforge.agents.critic import CriticAgent, CriticScore
from mindforge.agents.synthesizer import SynthesizerAgent
from mindforge.tools.citation_verifier import CitationVerifier
from mindforge.tools.code_executor import CodeExecutor
from mindforge.tools.rag_tool import RAGTool
from mindforge.tools.web_search import WebSearchTool
from mindforge.config import get_settings
from mindforge.memory import WorkingMemory

logger = logging.getLogger(__name__)

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
        episodic_memory: Any = None,
        semantic_memory: Any = None,
        tracer: Any = None,
    ) -> None:
        self._settings = get_settings()
        self._research_semaphore = asyncio.Semaphore(
            self._settings.agent.max_concurrent_research
        )
        self._subtask_semaphore = asyncio.Semaphore(
            self._settings.agent.max_concurrent_subtasks
        )
        self._tool_semaphore = asyncio.Semaphore(
            self._settings.agent.max_concurrent_tool_calls
        )

        self._planner_injected = planner is not None
        self._planner = planner or PlannerAgent()

        # Build default tool set for ResearcherAgent
        source_policy = getattr(
            self._settings.agent,
            "source_policy",
            "auto",
        )
        _researcher_tools: list = []
        if source_policy in {"auto", "knowledge_base"}:
            _researcher_tools.append(RAGTool())
        if source_policy in {"auto", "web"} and os.environ.get(
            "TAVILY_API_KEY",
            "",
        ).strip():
            _researcher_tools.append(WebSearchTool())
        _researcher_tools.extend([CodeExecutor(), CitationVerifier()])

        self._researcher = researcher or ResearcherAgent(
            tools=_researcher_tools,
            tool_semaphore=self._tool_semaphore,
            tool_queue_timeout=self._settings.agent.queue_timeout,
        )
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
                self._planner,
                self._researcher,
                self._critic,
                self._synthesizer,
            ):
                if hasattr(agent, "_tracer"):
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

    async def run(self, task: str) -> AgentResult:
        """Execute one research request inside a top-level trace."""
        with self._research_trace(task, transport="sync") as root_span:
            if root_span is not None:
                root_span.input = {"task": task}
            result = await self._run_with_limit(task)
            self._attach_trace_id(result)
            self._finish_root_span(root_span, result)
            return result

    async def _run_with_limit(self, task: str) -> AgentResult:
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
            return await self._run_unlimited(task)
        finally:
            self._research_semaphore.release()

    async def _run_unlimited(self, task: str) -> AgentResult:
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
            task,
            start_time=start_time,
            pipeline_log=pipeline_log,
        )
        if cached_result is not None:
            return cached_result

        # ------------------------------------------------------------------
        # Core pipeline — wrapped in timeout for safety
        # ------------------------------------------------------------------
        timeout_seconds = self._settings.agent.research_timeout
        try:
            core_result = await asyncio.wait_for(
                self._run_pipeline(
                    task,
                    total_usage,
                    total_cost,
                    pipeline_log,
                    start_time,
                ),
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
                    "quality": 0.0,
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
    ) -> AgentResult:
        """Core pipeline steps — separated for timeout wrapping."""
        working_memory = await self._create_working_memory(task)

        # ------------------------------------------------------------------
        # Step 1: Plan — decompose into DAG
        # ------------------------------------------------------------------
        with self._agent_span("planner") as planner_span:
            if planner_span is not None:
                planner_span.input = {"task": task}
            plan = await self._create_plan(task)
            if planner_span is not None:
                planner_span.output = {
                    "success": True,
                    "subtask_count": len(plan.subtasks),
                }
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
                    "quality": 0.0,
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
        # 简单查询（1 个子任务 + 输出较短）跳过 Critic 以提速
        # ------------------------------------------------------------------
        # 用 Researcher 原始输出判断复杂度（Synthesizer 会把简单内容扩写成报告）
        current_draft = draft_result.output
        final_critic: Optional[CriticScore] = None
        refine_count = 0

        if not self._should_run_critic(task, plan):
            logger.info("简单查询，跳过 Critic 评估（提速）")
            pipeline_log["critic"] = {"skipped": True, "reason": "简单查询"}
        else:
            max_refine = self._settings.agent.max_refine_rounds
            for refine_round in range(max_refine):
                with self._agent_span(
                    "critic",
                    metadata={"round": refine_round + 1},
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
                final_critic = critic_score
                self._accumulate_usage(
                    total_usage,
                    critic_score,
                    total_cost,
                )

                if not critic_score.should_refine:
                    pipeline_log["critic"] = {
                        "rounds": refine_round + 1,
                        "overall_score": critic_score.overall,
                        "refined": False,
                    }
                    break

                # Refine: re-synthesize with critic feedback
                with self._agent_span(
                    "synthesizer",
                    metadata={
                        "phase": "refine",
                        "round": refine_round + 1,
                    },
                ) as synthesizer_span:
                    refined_result = await self._synthesizer.synthesize(
                        task=task,
                        subtask_results=subtask_outputs,
                        all_sources=all_sources,
                        critic_feedback=critic_score,
                    )
                    if synthesizer_span is not None:
                        synthesizer_span.output = {
                            "success": refined_result.success,
                            "output_chars": len(refined_result.output),
                        }
                self._accumulate_usage(
                    total_usage,
                    refined_result,
                    total_cost,
                )
                if not refined_result.success or not refined_result.output.strip():
                    pipeline_log["critic"] = {
                        "rounds": refine_round + 1,
                        "overall_score": critic_score.overall,
                        "refined": False,
                        "refinement_failed": True,
                    }
                    break
                current_draft = refined_result.output
                refine_count = refine_round + 1

            # The score shown to users must describe the final refined draft,
            # not the pre-refinement version that triggered the rewrite.
            if refine_count > 0:
                with self._agent_span(
                    "critic",
                    metadata={"phase": "final"},
                ) as critic_span:
                    final_critic = await self._critic.evaluate(
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
                            "overall": final_critic.overall,
                            "should_refine": final_critic.should_refine,
                        }
                self._accumulate_usage(
                    total_usage,
                    final_critic,
                    total_cost,
                )

            if final_critic is not None and refine_count > 0:
                pipeline_log["critic"] = {
                    "rounds": refine_count,
                    "overall_score": final_critic.overall,
                    "refined": True,
                }

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        total_cost_usd, cost_status = self._cost_summary(total_cost)
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
        )
        await self._store_memories(task, result)
        return result

    # ------------------------------------------------------------------
    # Streaming variant
    # ------------------------------------------------------------------

    async def stream_run(self, task: str) -> AsyncIterator[dict[str, Any]]:
        """Stream one research request inside a top-level trace."""
        final_result: AgentResult | None = None
        trace_error: str | None = None
        with self._research_trace(task, transport="sse") as root_span:
            if root_span is not None:
                root_span.input = {"task": task}
            try:
                async for event in self._stream_run_with_limit(task):
                    trace_id = self._current_trace_id()
                    if trace_id:
                        event = {**event, "trace_id": trace_id}
                    if event.get("type") == "done":
                        result = event.get("result")
                        if isinstance(result, AgentResult):
                            self._attach_trace_id(result)
                            final_result = result
                    elif event.get("type") == "error":
                        trace_error = str(event.get("content") or "Research failed.")
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

        event_queue: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue()
        stream_finished = object()

        async def pump_events() -> None:
            try:
                async for event in self._stream_pipeline(task):
                    event_queue.put_nowait(event)
            finally:
                event_queue.put_nowait(stream_finished)

        pipeline_task = asyncio.create_task(pump_events())
        # Let the pipeline enqueue its immediate lifecycle event before the
        # heartbeat timer starts. The generator must remain owned by the same
        # task to preserve tracing ContextVar state across streamed chunks.
        await asyncio.sleep(0)
        deadline = time.monotonic() + self._settings.agent.research_timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
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
        except asyncio.TimeoutError:
            yield {
                "type": "error",
                "content": (
                    f"研究任务超过 {self._settings.agent.research_timeout} 秒，已终止。"
                ),
            }
        finally:
            if not pipeline_task.done():
                pipeline_task.cancel()
                await asyncio.gather(
                    pipeline_task,
                    return_exceptions=True,
                )
            self._research_semaphore.release()

    async def _stream_pipeline(
        self,
        task: str,
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
            task,
            start_time=start_time,
        )
        if cached_result is not None:
            yield {"type": "done", "result": cached_result}
            return
        working_memory = await self._create_working_memory(task)

        # --- Step 1: Plan ---
        with self._agent_span("planner") as planner_span:
            if planner_span is not None:
                planner_span.input = {"task": task}
            plan = await self._create_plan(task)
            if planner_span is not None:
                planner_span.output = {
                    "success": True,
                    "subtask_count": len(plan.subtasks),
                }
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
                    "quality": 0.0,
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
        current_draft = draft_result.output
        final_critic: Optional[CriticScore] = None
        refine_count = 0

        # 用 Researcher 原始输出判断复杂度（Synthesizer 会把简单内容扩写成报告）
        if not self._should_run_critic(task, plan):
            logger.info("简单查询，跳过 Critic 评估（提速）")
        else:
            max_refine = self._settings.agent.max_refine_rounds
            for refine_round in range(max_refine):
                with self._agent_span(
                    "critic",
                    metadata={"round": refine_round + 1},
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
                final_critic = critic_score
                self._accumulate_usage(
                    total_usage,
                    critic_score,
                    total_cost,
                )

                yield {
                    "type": "critic_feedback",
                    "score": critic_score,
                    "round": refine_round + 1,
                }

                if not critic_score.should_refine:
                    break

                yield {"type": "refining", "round": refine_round + 1}

                with self._agent_span(
                    "synthesizer",
                    metadata={
                        "phase": "refine",
                        "round": refine_round + 1,
                    },
                ) as synthesizer_span:
                    refined_result = await self._synthesizer.synthesize(
                        task=task,
                        subtask_results=subtask_outputs,
                        all_sources=all_sources,
                        critic_feedback=critic_score,
                    )
                    if synthesizer_span is not None:
                        synthesizer_span.output = {
                            "success": refined_result.success,
                            "output_chars": len(refined_result.output),
                        }
                self._accumulate_usage(
                    total_usage,
                    refined_result,
                    total_cost,
                )
                if not refined_result.success or not refined_result.output.strip():
                    break
                current_draft = refined_result.output
                refine_count = refine_round + 1

            if refine_count > 0:
                with self._agent_span(
                    "critic",
                    metadata={"phase": "final"},
                ) as critic_span:
                    final_critic = await self._critic.evaluate(
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
                            "overall": final_critic.overall,
                            "should_refine": final_critic.should_refine,
                        }
                self._accumulate_usage(
                    total_usage,
                    final_critic,
                    total_cost,
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        total_cost_usd, cost_status = self._cost_summary(total_cost)
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
        )
        await self._store_memories(task, result)
        yield {"type": "done", "result": result}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        sentence_breaks = len(re.findall(r"[。！？!?;\n]", normalized))
        return sentence_breaks <= 2

    async def _create_plan(self, task: str) -> ResearchPlan:
        if self._planner_injected:
            return await self._planner.run(task)
        mode = self._research_mode()
        if mode == "fast" or (
            mode == "balanced" and self._is_simple_task(task)
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
                    "该问题可由单个研究任务直接回答，跳过额外规划以降低延迟。"
                ),
            )
        return await self._planner.run(task)

    def _should_run_critic(
        self,
        task: str,
        plan: ResearchPlan,
    ) -> bool:
        if self._settings.agent.max_refine_rounds <= 0:
            return False
        mode = self._research_mode()
        if mode == "fast":
            return False
        if (
            mode == "balanced"
            and len(plan.subtasks) == 1
            and self._is_simple_task(task)
        ):
            return False
        return mode == "deep" or len(plan.subtasks) > 1

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
        shared_context: str = "",
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
            context = "\n\n".join(
                section
                for section in (
                    shared_context.strip(),
                    dependency_context.strip(),
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
                    }
                researcher_kwargs: dict[str, Any] = {
                    "context": context or None,
                }
                if isinstance(self._researcher, ResearcherAgent):
                    configured_rounds = self._settings.agent.max_iterations
                    mode = self._research_mode()
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
        shared_context = working_memory.get_context_string()
        results = await asyncio.gather(
            *[
                self._execute_subtask(
                    st,
                    plan,
                    shared_context=shared_context,
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

    async def _create_working_memory(
        self,
        task: str,
    ) -> WorkingMemory:
        memory = WorkingMemory()
        if self._semantic_memory is None:
            return memory
        try:
            facts = await self._semantic_memory.recall(task, top_k=3)
        except Exception as exc:
            logger.warning("Semantic memory recall failed: %s", exc)
            return memory
        memory.add_context(
            [
                {
                    "id": f"semantic:{fact.fact_id}",
                    "content": fact.content,
                    "rerank_score": fact.confidence,
                    "sources": fact.sources,
                    "memory_type": "semantic",
                }
                for fact in facts
            ]
        )
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
                [source for source in sources if isinstance(source, dict)]
            )

    async def _store_memories(
        self,
        task: str,
        result: AgentResult,
    ) -> None:
        outcome = str(result.metadata.get("outcome") or "").strip().lower()
        is_complete_success = result.success and outcome != "degraded"
        if self._episodic_memory is not None and is_complete_success:
            try:
                await self._episodic_memory.store(task, result)
            except Exception as exc:
                logger.warning("Episodic memory store failed: %s", exc)
        if self._semantic_memory is not None and is_complete_success:
            quality = result.metadata.get("quality")
            confidence = (
                max(0.0, min(1.0, float(quality) / 10.0))
                if isinstance(quality, (int, float)) and quality > 0
                else 0.5
            )
            try:
                await self._semantic_memory.store(
                    task,
                    result.output,
                    sources=result.data.get("sources", []),
                    confidence=confidence,
                )
            except Exception as exc:
                logger.warning("Semantic memory store failed: %s", exc)

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
    ) -> AgentResult:
        completed_subtasks = sum(
            1 for output in subtask_outputs if output.get("success")
        )
        failed_subtasks = len(subtask_outputs) - completed_subtasks
        outcome = "degraded" if failed_subtasks else "success"
        failure_reason = (
            self._format_partial_failure(subtask_outputs)
            if failed_subtasks
            else None
        )
        data: dict[str, Any] = {
            "plan": plan.to_dict(),
            "subtask_outputs": subtask_outputs,
            "sources": sources,
            "critic_score": (final_critic.to_dict() if final_critic else None),
            "refine_rounds": refine_count,
        }
        if failure_reason:
            data["partial_failure"] = failure_reason
        if pipeline_log is not None:
            data["pipeline"] = pipeline_log
        metadata: dict[str, Any] = {
            "quality": (final_critic.overall if final_critic else 0.0),
            "cost": total_cost_usd,
            "cost_status": cost_status,
            "subtask_count": len(plan.subtasks),
            "completed_subtask_count": completed_subtasks,
            "failed_subtask_count": failed_subtasks,
            "refine_rounds": refine_count,
            "model": self._settings.llm.llm_provider,
            "outcome": outcome,
        }
        if failure_reason:
            metadata["failure_reason"] = failure_reason
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
                "quality": 0.0,
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
                identity = str(
                    src.get("url")
                    or src.get("chunk_id")
                    or src.get("id")
                    or (
                        f"{src.get('title', src.get('source', ''))}:"
                        f"{src.get('content', src.get('text', ''))[:200]}"
                    )
                )
                if not identity or identity in seen:
                    continue
                seen.add(identity)
                all_sources.append({**src, "index": len(all_sources) + 1})
        return all_sources

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
