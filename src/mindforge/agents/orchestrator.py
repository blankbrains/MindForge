"""Orchestrator — top-level controller that drives the full research pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Any, AsyncIterator, Optional

from mindforge.agents.base import AgentResult
from mindforge.agents.planner import PlannerAgent, ResearchPlan, SubTask
from mindforge.agents.researcher import ResearcherAgent
from mindforge.agents.critic import CriticAgent, CriticScore
from mindforge.agents.synthesizer import SynthesizerAgent
from mindforge.tools.rag_tool import RAGTool
from mindforge.tools.web_search import WebSearchTool
from mindforge.tools.mcp_adapter import MCPToolAdapter
from mindforge.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional memory / observability imports (graceful fallback when not yet
# implemented)
# ---------------------------------------------------------------------------

try:
    from mindforge.memory import WorkingMemory, EpisodicMemory, SemanticMemory
except ImportError:
    WorkingMemory = None  # type: ignore[assignment,misc]
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
    working_memory : WorkingMemory, optional
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
        working_memory: Any = None,
        episodic_memory: Any = None,
        semantic_memory: Any = None,
        tracer: Any = None,
    ) -> None:
        self._settings = get_settings()

        self._planner = planner or PlannerAgent()

        # Build default tool set for ResearcherAgent
        _tools: list = [RAGTool(), WebSearchTool()]
        try:
            _tools.append(MCPToolAdapter())
        except Exception:
            pass  # MCP not available — non-fatal
        _researcher_tools = _tools

        self._researcher = researcher or ResearcherAgent(tools=_researcher_tools)
        self._critic = critic or CriticAgent()
        self._synthesizer = synthesizer or SynthesizerAgent()

        self._working_memory = working_memory
        self._episodic_memory = episodic_memory
        self._semantic_memory = semantic_memory
        self._tracer = tracer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, task: str) -> AgentResult:
        """Execute the full research pipeline for *task*.

        Returns an AgentResult with the final report in ``output`` and
        detailed pipeline metadata in ``data``.
        """
        start_time = time.perf_counter()
        total_usage: dict[str, int] = {}
        pipeline_log: dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Step 0: Check episodic memory for cached results
        # ------------------------------------------------------------------
        if self._episodic_memory is not None:
            try:
                cached = await self._episodic_memory.recall(task)
                if cached is not None:
                    elapsed = (time.perf_counter() - start_time) * 1000
                    return AgentResult(
                        agent_name="orchestrator",
                        success=True,
                        output=cached.get("output", ""),
                        data={"from_cache": True, "pipeline": pipeline_log},
                        latency_ms=elapsed,
                    )
            except Exception as exc:
                logger.warning("Episodic memory recall failed: %s", exc)

        # ------------------------------------------------------------------
        # Step 1: Plan — decompose into DAG
        # ------------------------------------------------------------------
        plan: ResearchPlan = await self._planner.run(task)
        pipeline_log["plan"] = {
            "subtask_count": len(plan.subtasks),
            "reasoning": plan.reasoning[:200],
        }

        # Track usage
        self._accumulate_usage(total_usage, plan.subtasks)

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
                break

            # Mark in-progress
            for st in ready:
                st.status = "in_progress"

            # Execute ready tasks in parallel
            results = await asyncio.gather(
                *[self._execute_subtask(st) for st in ready],
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
                    st.status = "completed"
                    st.result = result
                    self._accumulate_usage(total_usage, result)

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

        # ------------------------------------------------------------------
        # Step 3: Synthesize
        # ------------------------------------------------------------------
        all_sources = self._collect_sources(subtask_outputs)

        draft_result = await self._synthesizer.synthesize(
            task=task,
            subtask_results=subtask_outputs,
            all_sources=all_sources,
        )
        self._accumulate_usage(total_usage, draft_result)
        pipeline_log["synthesize"] = {"status": "completed"}

        # ------------------------------------------------------------------
        # Step 4: Critic + refine loop (max 2 rounds)
        # ------------------------------------------------------------------
        max_refine = self._settings.agent.max_refine_rounds
        current_draft = draft_result.output
        final_critic: Optional[CriticScore] = None
        refine_count = 0

        for refine_round in range(max_refine):
            critic_score = await self._critic.evaluate(
                task=task,
                draft=current_draft,
                sources=all_sources,
            )
            final_critic = critic_score
            self._accumulate_usage(total_usage, critic_score)

            if not critic_score.should_refine:
                pipeline_log["critic"] = {
                    "rounds": refine_round + 1,
                    "overall_score": critic_score.overall,
                    "refined": False,
                }
                break

            # Refine: re-synthesize with critic feedback
            current_draft = await self._synthesizer.synthesize(
                task=task,
                subtask_results=subtask_outputs,
                all_sources=all_sources,
                critic_feedback=critic_score,
            )
            self._accumulate_usage(total_usage, current_draft)
            current_draft = current_draft.output
            refine_count = refine_round + 1

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
        total_cost = total_usage.get("cost_usd", 0)

        return AgentResult(
            agent_name="orchestrator",
            success=True,
            output=current_draft,
            data={
                "pipeline": pipeline_log,
                "plan": plan.to_dict(),
                "subtask_outputs": subtask_outputs,
                "critic_score": final_critic.to_dict() if final_critic else None,
                "refine_rounds": refine_count,
            },
            metadata={
                "quality": final_critic.overall if final_critic else 0.0,
                "cost": total_cost,
                "subtask_count": len(plan.subtasks),
                "refine_rounds": refine_count,
                "model": self._settings.llm.llm_provider,
            },
            token_usage=total_usage,
            latency_ms=elapsed_ms,
            cost_usd=total_cost,
        )

    # ------------------------------------------------------------------
    # Streaming variant
    # ------------------------------------------------------------------

    async def stream_run(self, task: str) -> AsyncIterator[dict[str, Any]]:
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

        # --- Step 0: Memory check ---
        if self._episodic_memory is not None:
            try:
                cached = await self._episodic_memory.recall(task)
                if cached is not None:
                    elapsed = (time.perf_counter() - start_time) * 1000
                    result = AgentResult(
                        agent_name="orchestrator",
                        success=True,
                        output=cached.get("output", ""),
                        data={"from_cache": True},
                        latency_ms=elapsed,
                    )
                    yield {"type": "done", "result": result}
                    return
            except Exception:
                logger.debug("Episodic memory recall failed; continuing with fresh research.")

        # --- Step 1: Plan ---
        plan: ResearchPlan = await self._planner.run(task)
        yield {"type": "plan_ready", "plan": plan}

        # --- Step 2: Execute DAG ---
        subtask_outputs: list[dict[str, Any]] = []

        while not plan.is_complete():
            ready = plan.get_ready_tasks()
            if not ready:
                for st in plan.subtasks:
                    if st.status == "pending":
                        st.status = "failed"
                break

            for st in ready:
                st.status = "in_progress"
                yield {"type": "subtask_start", "task_id": st.task_id, "description": st.description}

            results = await asyncio.gather(
                *[self._execute_subtask(st) for st in ready],
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
                    st.status = "completed"
                    st.result = result

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

        # --- Step 3: Synthesize ---
        all_sources = self._collect_sources(subtask_outputs)
        yield {"type": "synthesizing", "status": "start"}

        draft_result = await self._synthesizer.synthesize(
            task=task,
            subtask_results=subtask_outputs,
            all_sources=all_sources,
        )
        yield {"type": "synthesizing", "status": "done"}

        # --- Step 4: Critic + refine ---
        max_refine = self._settings.agent.max_refine_rounds
        current_draft = draft_result.output
        final_critic: Optional[CriticScore] = None
        refine_count = 0

        for refine_round in range(max_refine):
            critic_score = await self._critic.evaluate(
                task=task,
                draft=current_draft,
                sources=all_sources,
            )
            final_critic = critic_score

            yield {
                "type": "critic_feedback",
                "score": critic_score,
                "round": refine_round + 1,
            }

            if not critic_score.should_refine:
                break

            yield {"type": "refining", "round": refine_round + 1}

            current_draft = await self._synthesizer.synthesize(
                task=task,
                subtask_results=subtask_outputs,
                all_sources=all_sources,
                critic_feedback=critic_score,
            )
            current_draft = current_draft.output
            refine_count = refine_round + 1

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
        total_cost = total_usage.get("cost_usd", 0)

        result = AgentResult(
            agent_name="orchestrator",
            success=True,
            output=current_draft,
            data={
                "plan": plan.to_dict(),
                "subtask_outputs": subtask_outputs,
                "critic_score": final_critic.to_dict() if final_critic else None,
                "refine_rounds": refine_count,
            },
            token_usage=total_usage,
            latency_ms=elapsed_ms,
            cost_usd=total_cost,
        )
        yield {"type": "done", "result": result}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_subtask(self, subtask: SubTask) -> AgentResult:
        """Execute a single subtask with a timeout.

        The timeout is read from ``settings.agent.subtask_timeout`` (default 45 s).
        """
        timeout = self._settings.agent.subtask_timeout

        try:
            result = await asyncio.wait_for(
                self._researcher.run(subtask.description),
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
            return AgentResult(
                agent_name="researcher",
                success=False,
                output=f"Subtask '{subtask.task_id}' failed: {type(exc).__name__}: {exc}",
                data={"task_id": subtask.task_id, "traceback": traceback.format_exc()},
            )

    @staticmethod
    def _collect_sources(
        subtask_outputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Aggregate unique sources across all subtask outputs."""
        seen: set[int] = set()
        all_sources: list[dict[str, Any]] = []
        for so in subtask_outputs:
            sources = so.get("sources", [])
            if not isinstance(sources, list):
                continue
            for src in sources:
                idx = src.get("index") if isinstance(src, dict) else None
                if idx is not None and idx not in seen:
                    seen.add(idx)
                    all_sources.append(src)
        return all_sources

    @staticmethod
    def _accumulate_usage(
        accumulator: dict[str, int],
        result: Any,
    ) -> None:
        """Merge token usage from an AgentResult or other result objects."""
        if result is None:
            return
        if hasattr(result, "token_usage") and result.token_usage:
            for k, v in result.token_usage.items():
                if isinstance(v, (int, float)):
                    accumulator[k] = accumulator.get(k, 0) + int(v)
        if hasattr(result, "cost_usd") and result.cost_usd:
            accumulator["cost_usd"] = accumulator.get("cost_usd", 0) + result.cost_usd
        # Handle list of subtasks (from planner)
        if isinstance(result, list):
            for item in result:
                if hasattr(item, "token_usage") and item.token_usage:
                    for k, v in item.token_usage.items():
                        if isinstance(v, (int, float)):
                            accumulator[k] = accumulator.get(k, 0) + int(v)
