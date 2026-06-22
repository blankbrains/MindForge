"""Planner agent — decomposes a complex task into a DAG of subtasks."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from mindforge.agents.base import AgentResult, BaseAgent
from mindforge.models.base import ChatMessage
from mindforge.config import get_settings


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SubTask:
    """A single unit of work within a research plan."""

    task_id: str
    description: str
    task_type: str = "research"  # "research" | "analysis" | "code" | "verify"
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"  # "pending" | "in_progress" | "completed" | "failed"
    priority: int = 5
    result: Optional[AgentResult] = None
    subtopics: list[str] = field(default_factory=list)


@dataclass
class ResearchPlan:
    """A complete DAG-based research plan."""

    plan_id: str
    original_task: str
    subtasks: list[SubTask]
    reasoning: str = ""

    # ------------------------------------------------------------------
    def get_ready_tasks(self) -> list[SubTask]:
        """Return subtasks whose dependencies are all completed or absent."""
        completed_ids = {
            st.task_id
            for st in self.subtasks
            if st.status == "completed"
        }
        ready: list[SubTask] = []
        for st in self.subtasks:
            if st.status != "pending":
                continue
            if all(dep in completed_ids for dep in st.dependencies):
                ready.append(st)
        return ready

    # ------------------------------------------------------------------
    def is_complete(self) -> bool:
        """Return True when every subtask is either completed or failed."""
        return all(st.status in ("completed", "failed") for st in self.subtasks)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "original_task": self.original_task,
            "reasoning": self.reasoning,
            "subtasks": [
                {
                    "task_id": s.task_id,
                    "description": s.description,
                    "task_type": s.task_type,
                    "dependencies": s.dependencies,
                    "status": s.status,
                    "priority": s.priority,
                    "subtopics": s.subtopics,
                }
                for s in self.subtasks
            ],
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> ResearchPlan:
        subtasks = [
            SubTask(
                task_id=s.get("task_id", str(uuid.uuid4())[:8]),
                description=s.get("description", ""),
                task_type=s.get("task_type", "research"),
                dependencies=s.get("dependencies", []),
                status=s.get("status", "pending"),
                priority=s.get("priority", 5),
                subtopics=s.get("subtopics", []),
            )
            for s in data.get("subtasks", [])
        ]
        return cls(
            plan_id=data.get("plan_id", str(uuid.uuid4())[:8]),
            original_task=data.get("original_task", ""),
            subtasks=subtasks,
            reasoning=data.get("reasoning", ""),
        )


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """You are an expert research planner. Your role is to decompose complex tasks into a Directed Acyclic Graph (DAG) of subtasks.

Rules:
1. Break the task into 3-10 concrete, executable subtasks.
2. Each subtask must have a clear description and a type (research / analysis / code / verify).
3. Specify dependencies between subtasks using task_id references.
4. Subtasks with no dependencies can be executed in parallel.
5. Assign priority (1=highest, 10=lowest).
6. Include subtopics as specific search queries or angles for each subtask.
7. Return ONLY valid JSON — no markdown, no code fences, no commentary.

Output JSON schema:
{
  "reasoning": "Brief explanation of the decomposition strategy.",
  "subtasks": [
    {
      "task_id": "t1",
      "description": "Clear description of what this subtask investigates.",
      "task_type": "research | analysis | code | verify",
      "dependencies": [],
      "priority": 1,
      "subtopics": ["specific query 1", "specific query 2"]
    }
  ]
}"""


class PlannerAgent(BaseAgent):
    """Decomposes a user task into a DAG-structured research plan."""

    @property
    def name(self) -> str:
        return "planner"

    @property
    def system_prompt(self) -> str:
        return _PLANNER_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    async def run(self, task: str) -> ResearchPlan:
        """Decompose *task* into a ResearchPlan.

        Falls back to a single-step plan on any parse error.
        """
        settings = get_settings()
        # Use the planner-specific model from config
        planner_model = settings.llm.get_model("planner")
        # Temporarily switch LLM to planner-specific model (restore after)
        _old_llm = getattr(self, "_llm", None)
        if _old_llm is not None:
            from mindforge.models.base import LLMFactory
            self._llm = LLMFactory.create(
                settings.llm.llm_provider, planner_model
            )

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(
                role="user",
                content=(
                    f"Please decompose the following task into a DAG of subtasks:\n\n{task}"
                ),
            ),
        ]

        try:
            result = await self._chat(
                messages,
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            raw = result.content.strip()
            plan_dict = json.loads(raw)
            plan_dict["original_task"] = task
            plan_dict["plan_id"] = uuid.uuid4().hex[:12]

            plan = ResearchPlan.from_dict(plan_dict)

            # Validate at least one subtask
            if not plan.subtasks:
                raise ValueError("Planner returned zero subtasks.")

            return plan

        except Exception as exc:
            # Fallback: create a single-step plan
            return ResearchPlan(
                plan_id=uuid.uuid4().hex[:12],
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
                    f"Fallback single-step plan (planner error: {exc}). "
                    "Could not decompose into multiple subtasks."
                ),
            )
        finally:
            # Restore original LLM to avoid mutating shared instance
            if _old_llm is not None:
                self._llm = _old_llm
