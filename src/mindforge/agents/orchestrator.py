"""Orchestrator — top-level controller that drives the full research pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional memory / observability imports (graceful fallback when not yet
# implemented)
# ---------------------------------------------------------------------------

try:
    from mindforge.memory import EpisodicMemory, SemanticMemory
except ImportError:
    EpisodicMemory = None  # type: ignore[assignment,misc]
    SemanticMemory = None  # type: ignore[assignment,misc]

try:
    from mindforge.observability import Tracer
except ImportError:
    Tracer = None  # type: ignore[assignment,misc]


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

        self._planner = planner or PlannerAgent()

        # Build default tool set for ResearcherAgent
        _researcher_tools: list = [
            RAGTool(),
            WebSearchTool(),
            CodeExecutor(),
            CitationVerifier(),
        ]

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
                    "研究任务排队超时，当前服务器正在处理其他研究请求，"
                    "请稍后重试。"
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
        total_cost = {"usd": 0.0}
        pipeline_log: dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Step 0: Check episodic memory for cached results
        # ------------------------------------------------------------------
        if self._episodic_memory is not None:
            try:
                cached = await self._episodic_memory.recall(task)
                cached_output = (
                    str(cached.get("output", "")).strip()
                    if isinstance(cached, dict)
                    else ""
                )
                if cached_output:
                    elapsed = (time.perf_counter() - start_time) * 1000
                    return AgentResult(
                        agent_name="orchestrator",
                        success=True,
                        output=cached_output,
                        data={"from_cache": True, "pipeline": pipeline_log},
                        latency_ms=elapsed,
                    )
            except Exception as exc:
                logger.warning("Episodic memory recall failed: %s", exc)

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
                    "error": f"Timed out after {timeout_seconds}s",
                },
                metadata={
                    "quality": 0.0,
                    "cost": 0.0,
                    "subtask_count": 0,
                    "refine_rounds": 0,
                    "model": self._settings.llm.llm_provider,
                    "timeout": True,
                },
                latency_ms=elapsed_ms,
            )

    async def _run_pipeline(
        self,
        task: str,
        total_usage: dict[str, int],
        total_cost: dict[str, float],
        pipeline_log: dict[str, Any],
        start_time: float,
    ) -> AgentResult:
        """Core pipeline steps — separated for timeout wrapping."""

        # ------------------------------------------------------------------
        # Step 1: Plan — decompose into DAG
        # ------------------------------------------------------------------
        plan = await self._planner.run(task)
        pipeline_log["plan"] = {
            "subtask_count": len(plan.subtasks),
            "reasoning": plan.reasoning[:200],
        }

        # Track usage
        self._accumulate_usage(total_usage, plan.planner_usage, total_cost)

        # ------------------------------------------------------------------
        # Step 2: Execute DAG (parallel where dependencies allow)
        # ------------------------------------------------------------------
        subtask_outputs: list[dict[str, Any]] = []

        while not plan.is_complete():
            ready = plan.get_ready_tasks()
            if not ready:
                # Deadlock or all remaining tasks have unmet deps
                for st in plan.subtasks:
                    if st.status == "pending":
                        st.status = "failed"
                        st.result = AgentResult(
                            agent_name="researcher",
                            success=False,
                            output=(
                                f"Subtask {st.task_id} deadlocked: "
                                "unmet dependencies."
                            ),
                        )
                        subtask_outputs.append(
                            {
                                "task_id": st.task_id,
                                "description": st.description,
                                "task_type": st.task_type,
                                "output": st.result.output,
                                "sources": [],
                                "success": False,
                            }
                        )
                break

            # Mark in-progress
            for st in ready:
                st.status = "in_progress"

            # Execute ready tasks in parallel
            results = await asyncio.gather(
                *[self._execute_subtask(st, plan) for st in ready],
                return_exceptions=True,
            )

            # Collect results
            for st, result in zip(ready, results):
                if isinstance(result, BaseException):
                    st.status = "failed"
                    st.result = AgentResult(
                        agent_name="researcher",
                        success=False,
                        output=f"Subtask failed: {result}",
                    )
                else:
                    st.status = "completed" if result.success else "failed"
                    st.result = result
                    self._accumulate_usage(total_usage, result, total_cost)

                subtask_outputs.append(
                    {
                        "task_id": st.task_id,
                        "description": st.description,
                        "task_type": st.task_type,
                        "output": st.result.output if st.result else "",
                        "sources": (
                            st.result.data.get("sources", [])
                            if st.result and st.result.data
                            else []
                        ),
                        "success": st.result.success if st.result else False,
                    }
                )

        pipeline_log["execution"] = {
            "subtasks_completed": sum(1 for s in plan.subtasks if s.status == "completed"),
            "subtasks_failed": sum(1 for s in plan.subtasks if s.status == "failed"),
        }

        if not any(
            output.get("success")
            for output in subtask_outputs
        ):
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            failure_output = self._format_pipeline_failure(
                subtask_outputs
            )
            return AgentResult(
                agent_name="orchestrator",
                success=False,
                output=failure_output,
                data={
                    "pipeline": pipeline_log,
                    "plan": plan.to_dict(),
                    "subtask_outputs": subtask_outputs,
                    "sources": [],
                    "critic_score": None,
                    "refine_rounds": 0,
                },
                metadata={
                    "quality": 0.0,
                    "cost": total_cost["usd"],
                    "subtask_count": len(plan.subtasks),
                    "refine_rounds": 0,
                    "model": self._settings.llm.llm_provider,
                },
                token_usage=total_usage,
                latency_ms=elapsed_ms,
                cost_usd=total_cost["usd"],
            )

        # ------------------------------------------------------------------
        # Step 3: Synthesize (skip for single-subtask)
        # ------------------------------------------------------------------
        all_sources = self._collect_sources(subtask_outputs)
        skip_syn = len(subtask_outputs) == 1 and subtask_outputs[0].get("success")

        if skip_syn:
            logger.info("单子任务，跳过 Synthesizer（直接用 Researcher 输出）")
            current_draft = subtask_outputs[0].get("output", "")
            draft_result = AgentResult(agent_name="synthesizer", success=True, output=current_draft)
            pipeline_log["synthesize"] = {"status": "skipped_single_subtask"}
        else:
            draft_result = await self._synthesizer.synthesize(
                task=task,
                subtask_results=subtask_outputs,
                all_sources=all_sources,
            )
            self._accumulate_usage(total_usage, draft_result, total_cost)
            pipeline_log["synthesize"] = {"status": "completed"}

        if not draft_result.success or not draft_result.output.strip():
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            pipeline_log["synthesize"] = {
                "status": "failed",
                "reason": "empty_response",
            }
            return AgentResult(
                agent_name="orchestrator",
                success=False,
                output=(
                    "Research synthesis failed because the language model "
                    "returned an empty response."
                ),
                data={
                    "pipeline": pipeline_log,
                    "plan": plan.to_dict(),
                    "subtask_outputs": subtask_outputs,
                    "sources": all_sources,
                    "critic_score": None,
                    "refine_rounds": 0,
                },
                metadata={
                    "quality": 0.0,
                    "cost": total_cost["usd"],
                    "subtask_count": len(plan.subtasks),
                    "refine_rounds": 0,
                    "model": self._settings.llm.llm_provider,
                },
                token_usage=total_usage,
                latency_ms=elapsed_ms,
                cost_usd=total_cost["usd"],
            )

        # ------------------------------------------------------------------
        # Step 4: Critic + refine loop
        # 简单查询（1 个子任务 + 输出较短）跳过 Critic 以提速
        # ------------------------------------------------------------------
        # 用 Researcher 原始输出判断复杂度（Synthesizer 会把简单内容扩写成报告）
        researcher_output = (
            subtask_outputs[0].get("output", "") if subtask_outputs else ""
        )
        is_simple = (
            len(plan.subtasks) == 1
            and len(researcher_output) < 800
        )
        current_draft = draft_result.output
        final_critic: Optional[CriticScore] = None
        refine_count = 0

        if is_simple and self._settings.agent.max_refine_rounds > 0:
            logger.info("简单查询，跳过 Critic 评估（提速）")
            pipeline_log["critic"] = {"skipped": True, "reason": "简单查询"}
        else:
            max_refine = self._settings.agent.max_refine_rounds
            for refine_round in range(max_refine):
                critic_score = await self._critic.evaluate(
                    task=task,
                    draft=current_draft,
                    sources=all_sources,
                )
                final_critic = critic_score
                self._accumulate_usage(
                    total_usage,
                    critic_score.token_usage,
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
                refined_result = await self._synthesizer.synthesize(
                    task=task,
                    subtask_results=subtask_outputs,
                    all_sources=all_sources,
                    critic_feedback=critic_score,
                )
                self._accumulate_usage(
                    total_usage,
                    refined_result,
                    total_cost,
                )
                if (
                    not refined_result.success
                    or not refined_result.output.strip()
                ):
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
                final_critic = await self._critic.evaluate(
                    task=task,
                    draft=current_draft,
                    sources=all_sources,
                )
                self._accumulate_usage(
                    total_usage,
                    final_critic.token_usage,
                    total_cost,
                )

            if final_critic is not None and refine_count > 0:
                pipeline_log["critic"] = {
                    "rounds": refine_count,
                    "overall_score": final_critic.overall,
                    "refined": True,
                }

        # ------------------------------------------------------------------
        # Step 5: Store to memory
        # ------------------------------------------------------------------
        if self._episodic_memory is not None:
            try:
                await self._episodic_memory.store(
                    task=task,
                    result={
                        "output": current_draft,
                        "plan_id": plan.plan_id,
                        "critic_score": (
                            final_critic.to_dict() if final_critic else None
                        ),
                    },
                )
            except Exception as exc:
                logger.warning("Episodic memory store failed: %s", exc)

        if self._semantic_memory is not None:
            try:
                await self._semantic_memory.store(task, current_draft)
            except Exception as exc:
                logger.warning("Semantic memory store failed: %s", exc)

        # ------------------------------------------------------------------
        # Done
        # ------------------------------------------------------------------
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        total_cost_usd = total_cost["usd"]

        return AgentResult(
            agent_name="orchestrator",
            success=True,
            output=current_draft,
            data={
                "pipeline": pipeline_log,
                "plan": plan.to_dict(),
                "subtask_outputs": subtask_outputs,
                "sources": all_sources,
                "critic_score": final_critic.to_dict() if final_critic else None,
                "refine_rounds": refine_count,
            },
            metadata={
                "quality": final_critic.overall if final_critic else 0.0,
                "cost": total_cost_usd,
                "subtask_count": len(plan.subtasks),
                "refine_rounds": refine_count,
                "model": self._settings.llm.llm_provider,
            },
            token_usage=total_usage,
            latency_ms=elapsed_ms,
            cost_usd=total_cost_usd,
        )

    # ------------------------------------------------------------------
    # Streaming variant
    # ------------------------------------------------------------------

    async def stream_run(self, task: str) -> AsyncIterator[dict[str, Any]]:
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
                    "研究任务排队超时，当前服务器正在处理其他研究请求，"
                    "请稍后重试。"
                ),
            }
            return

        iterator = self._stream_pipeline(task).__aiter__()
        deadline = time.monotonic() + self._settings.agent.research_timeout
        pending_event: asyncio.Task | None = None
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                if pending_event is None:
                    pending_event = asyncio.create_task(
                        iterator.__anext__()
                    )
                done, _ = await asyncio.wait(
                    {pending_event},
                    timeout=min(
                        remaining,
                        heartbeat_seconds,
                    ),
                )
                if not done:
                    yield {
                        "type": "heartbeat",
                        "timestamp": time.time(),
                    }
                    continue
                try:
                    event = pending_event.result()
                except StopAsyncIteration:
                    return
                pending_event = None
                yield event
        except asyncio.TimeoutError:
            yield {
                "type": "error",
                "content": (
                    "研究任务超过 "
                    f"{self._settings.agent.research_timeout} 秒，已终止。"
                ),
            }
        finally:
            if pending_event is not None and not pending_event.done():
                pending_event.cancel()
                await asyncio.gather(
                    pending_event,
                    return_exceptions=True,
                )
            await iterator.aclose()
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
        total_cost = {"usd": 0.0}

        yield {"type": "planning", "status": "start"}

        # --- Step 0: Memory check ---
        if self._episodic_memory is not None:
            try:
                cached = await self._episodic_memory.recall(task)
                cached_output = (
                    str(cached.get("output", "")).strip()
                    if isinstance(cached, dict)
                    else ""
                )
                if cached_output:
                    elapsed = (time.perf_counter() - start_time) * 1000
                    result = AgentResult(
                        agent_name="orchestrator",
                        success=True,
                        output=cached_output,
                        data={"from_cache": True},
                        latency_ms=elapsed,
                    )
                    yield {"type": "done", "result": result}
                    return
            except Exception:
                logger.debug("Episodic memory recall failed; continuing with fresh research.")

        # --- Step 1: Plan ---
        plan: ResearchPlan = await self._planner.run(task)
        self._accumulate_usage(total_usage, plan.planner_usage, total_cost)
        yield {"type": "planning", "status": "done"}
        yield {"type": "plan_ready", "plan": plan}

        # --- Step 2: Execute DAG ---
        subtask_outputs: list[dict[str, Any]] = []

        while not plan.is_complete():
            ready = plan.get_ready_tasks()
            if not ready:
                for st in plan.subtasks:
                    if st.status == "pending":
                        st.status = "failed"
                        st.result = AgentResult(
                            agent_name="researcher",
                            success=False,
                            output=(
                                f"Subtask {st.task_id} deadlocked: "
                                "unmet dependencies."
                            ),
                        )
                        subtask_outputs.append(
                            {
                                "task_id": st.task_id,
                                "description": st.description,
                                "task_type": st.task_type,
                                "output": st.result.output,
                                "sources": [],
                                "success": False,
                            }
                        )
                        yield {
                            "type": "subtask_result",
                            "task_id": st.task_id,
                            "result": st.result,
                        }
                break

            for st in ready:
                st.status = "in_progress"
                yield {"type": "subtask_start", "task_id": st.task_id, "description": st.description}

            results = await asyncio.gather(
                *[self._execute_subtask(st, plan) for st in ready],
                return_exceptions=True,
            )

            for st, result in zip(ready, results):
                if isinstance(result, BaseException):
                    st.status = "failed"
                    st.result = AgentResult(
                        agent_name="researcher",
                        success=False,
                        output=f"Subtask failed: {result}",
                    )
                else:
                    st.status = "completed" if result.success else "failed"
                    st.result = result

                self._accumulate_usage(total_usage, st.result, total_cost)

                subtask_outputs.append(
                    {
                        "task_id": st.task_id,
                        "description": st.description,
                        "task_type": st.task_type,
                        "output": st.result.output if st.result else "",
                        "sources": (
                            st.result.data.get("sources", [])
                            if st.result and st.result.data
                            else []
                        ),
                        "success": st.result.success if st.result else False,
                    }
                )

                yield {
                    "type": "subtask_result",
                    "task_id": st.task_id,
                    "result": st.result,
                }

        if not any(
            output.get("success")
            for output in subtask_outputs
        ):
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result = AgentResult(
                agent_name="orchestrator",
                success=False,
                output=self._format_pipeline_failure(
                    subtask_outputs
                ),
                data={
                    "plan": plan.to_dict(),
                    "subtask_outputs": subtask_outputs,
                    "sources": [],
                    "critic_score": None,
                    "refine_rounds": 0,
                },
                metadata={
                    "quality": 0.0,
                    "cost": total_cost["usd"],
                    "subtask_count": len(plan.subtasks),
                    "refine_rounds": 0,
                    "model": self._settings.llm.llm_provider,
                },
                token_usage=total_usage,
                latency_ms=elapsed_ms,
                cost_usd=total_cost["usd"],
            )
            yield {"type": "done", "result": result}
            return

        # --- Step 3: Synthesize (skip for single-subtask — use Researcher output directly) ---
        all_sources = self._collect_sources(subtask_outputs)
        skip_synthesizer = len(subtask_outputs) == 1 and subtask_outputs[0].get("success")

        if skip_synthesizer:
            logger.info("单子任务，跳过 Synthesizer（流式输出 Researcher 结果）")
            researcher_text = subtask_outputs[0].get("output", "")
            chunk_size = self._settings.agent.stream_chunk_size
            for i in range(0, len(researcher_text), chunk_size):
                yield {"type": "answer_chunk", "content": researcher_text[i:i+chunk_size]}
            yield {"type": "synthesizing", "status": "done"}
            current_draft = researcher_text
            draft_result = AgentResult(agent_name="synthesizer", success=True, output=researcher_text)
        else:
            yield {"type": "synthesizing", "status": "start"}
            draft_chunks: list[str] = []
            async for chunk in self._synthesizer.synthesize_stream(
                task=task,
                subtask_results=subtask_outputs,
                all_sources=all_sources,
            ):
                draft_chunks.append(chunk)
                yield {"type": "answer_chunk", "content": chunk}
            draft_result = AgentResult(
                agent_name="synthesizer",
                success=bool("".join(draft_chunks).strip()),
                output="".join(draft_chunks),
            )
            yield {"type": "synthesizing", "status": "done"}

        if not draft_result.success or not draft_result.output.strip():
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result = AgentResult(
                agent_name="orchestrator",
                success=False,
                output=(
                    "Research synthesis failed because the language model "
                    "returned an empty response."
                ),
                data={
                    "plan": plan.to_dict(),
                    "subtask_outputs": subtask_outputs,
                    "sources": all_sources,
                    "critic_score": None,
                    "refine_rounds": 0,
                },
                metadata={
                    "quality": 0.0,
                    "cost": total_cost["usd"],
                    "subtask_count": len(plan.subtasks),
                    "refine_rounds": 0,
                    "model": self._settings.llm.llm_provider,
                },
                token_usage=total_usage,
                latency_ms=elapsed_ms,
                cost_usd=total_cost["usd"],
            )
            yield {"type": "done", "result": result}
            return

        # --- Step 4: Critic + refine ---
        current_draft = draft_result.output
        final_critic: Optional[CriticScore] = None
        refine_count = 0

        # 用 Researcher 原始输出判断复杂度（Synthesizer 会把简单内容扩写成报告）
        researcher_output = (
            subtask_outputs[0].get("output", "") if subtask_outputs else ""
        )
        is_simple = (
            len(plan.subtasks) == 1
            and len(researcher_output) < 800
        )
        if is_simple and self._settings.agent.max_refine_rounds > 0:
            logger.info("简单查询，跳过 Critic 评估（提速）")
        else:
            max_refine = self._settings.agent.max_refine_rounds
            for refine_round in range(max_refine):
                critic_score = await self._critic.evaluate(
                    task=task,
                    draft=current_draft,
                    sources=all_sources,
                )
                final_critic = critic_score
                self._accumulate_usage(
                    total_usage,
                    critic_score.token_usage,
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

                refined_result = await self._synthesizer.synthesize(
                    task=task,
                    subtask_results=subtask_outputs,
                    all_sources=all_sources,
                    critic_feedback=critic_score,
                )
                self._accumulate_usage(
                    total_usage,
                    refined_result,
                    total_cost,
                )
                if (
                    not refined_result.success
                    or not refined_result.output.strip()
                ):
                    break
                current_draft = refined_result.output
                refine_count = refine_round + 1

            if refine_count > 0:
                final_critic = await self._critic.evaluate(
                    task=task,
                    draft=current_draft,
                    sources=all_sources,
                )
                self._accumulate_usage(
                    total_usage,
                    final_critic.token_usage,
                    total_cost,
                )

        # --- Step 5: Memory ---
        if self._episodic_memory is not None:
            try:
                await self._episodic_memory.store(
                    task=task,
                    result={
                        "output": current_draft,
                        "plan_id": plan.plan_id,
                        "critic_score": (
                            final_critic.to_dict() if final_critic else None
                        ),
                    },
                )
            except Exception:
                logger.debug("Episodic memory store skipped in stream_run.")

        if self._semantic_memory is not None:
            try:
                await self._semantic_memory.store(task, current_draft)
            except Exception:
                logger.debug("Semantic memory store skipped in stream_run.")

        # --- Done ---
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        total_cost_usd = total_cost["usd"]

        result = AgentResult(
            agent_name="orchestrator",
            success=True,
            output=current_draft,
            data={
                "plan": plan.to_dict(),
                "subtask_outputs": subtask_outputs,
                "sources": all_sources,
                "critic_score": final_critic.to_dict() if final_critic else None,
                "refine_rounds": refine_count,
            },
            metadata={
                "quality": (
                    final_critic.overall
                    if final_critic
                    else 0.0
                ),
                "cost": total_cost_usd,
                "subtask_count": len(plan.subtasks),
                "refine_rounds": refine_count,
                "model": self._settings.llm.llm_provider,
            },
            token_usage=total_usage,
            latency_ms=elapsed_ms,
            cost_usd=total_cost_usd,
        )
        yield {"type": "done", "result": result}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_subtask(
        self,
        subtask: SubTask,
        plan: ResearchPlan,
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
            context = self._build_dependency_context(subtask, plan)
            result = await asyncio.wait_for(
                self._researcher.run(
                    subtask.description,
                    context=context or None,
                ),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            return AgentResult(
                agent_name="researcher",
                success=False,
                output=f"Subtask '{subtask.task_id}' timed out after {timeout}s.",
                data={"task_id": subtask.task_id},
            )
        except Exception as exc:
            logger.exception(
                "Subtask execution failed: %s",
                subtask.task_id,
            )
            return AgentResult(
                agent_name="researcher",
                success=False,
                output=f"Subtask '{subtask.task_id}' failed internally.",
                data={
                    "task_id": subtask.task_id,
                    "error_type": type(exc).__name__,
                },
            )
        finally:
            self._subtask_semaphore.release()

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
        return (
            "Research failed because all subtasks failed:\n\n"
            + "\n".join(f"- {detail}" for detail in details)
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
    def _accumulate_usage(
        accumulator: dict[str, int],
        result: Any,
        cost_accumulator: Optional[dict[str, float]] = None,
    ) -> None:
        """Merge token usage from an AgentResult or other result objects."""
        if result is None:
            return
        if isinstance(result, dict):
            for key, value in result.items():
                if (
                    isinstance(value, (int, float))
                    and key != "cost_usd"
                ):
                    accumulator[key] = (
                        accumulator.get(key, 0) + int(value)
                    )
            return
        if cost_accumulator is not None:
            cost = getattr(result, "cost_usd", 0.0)
            if isinstance(cost, (int, float)):
                cost_accumulator["usd"] = (
                    cost_accumulator.get("usd", 0.0) + float(cost)
                )
        if hasattr(result, "token_usage") and result.token_usage:
            for k, v in result.token_usage.items():
                if isinstance(v, (int, float)) and k != "cost_usd":
                    accumulator[k] = accumulator.get(k, 0) + int(v)
        # Handle list of subtasks (from planner)
        if isinstance(result, list):
            for item in result:
                if cost_accumulator is not None:
                    cost = getattr(item, "cost_usd", 0.0)
                    if isinstance(cost, (int, float)):
                        cost_accumulator["usd"] = (
                            cost_accumulator.get("usd", 0.0) + float(cost)
                        )
                if hasattr(item, "token_usage") and item.token_usage:
                    for k, v in item.token_usage.items():
                        if isinstance(v, (int, float)) and k != "cost_usd":
                            accumulator[k] = accumulator.get(k, 0) + int(v)
