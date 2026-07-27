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
    planner_usage: dict[str, int] = field(default_factory=dict)

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

    def validate(self, max_subtasks: int = 5) -> None:
        """Validate task ids, dependency references, and DAG acyclicity."""
        if not 1 <= len(self.subtasks) <= max_subtasks:
            raise ValueError(
                "Planner returned "
                f"{len(self.subtasks)} subtasks; allowed range is "
                f"1-{max_subtasks}."
            )
        task_ids = [task.task_id for task in self.subtasks]
        if any(not task_id for task_id in task_ids):
            raise ValueError("Planner returned an empty task_id.")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Planner returned duplicate task_id values.")

        known_ids = set(task_ids)
        indegree = {task_id: 0 for task_id in task_ids}
        dependents: dict[str, list[str]] = {
            task_id: [] for task_id in task_ids
        }
        for task in self.subtasks:
            normalized = list(dict.fromkeys(task.dependencies))
            task.dependencies = normalized
            for dependency in normalized:
                if dependency not in known_ids:
                    raise ValueError(
                        f"Task {task.task_id} depends on unknown task "
                        f"{dependency}."
                    )
                if dependency == task.task_id:
                    raise ValueError(
                        f"Task {task.task_id} depends on itself."
                    )
                indegree[task.task_id] += 1
                dependents[dependency].append(task.task_id)

        ready = [
            task_id
            for task_id, degree in indegree.items()
            if degree == 0
        ]
        visited = 0
        while ready:
            current = ready.pop()
            visited += 1
            for dependent in dependents[current]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
        if visited != len(task_ids):
            raise ValueError("Planner returned cyclic task dependencies.")

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

_PLANNER_SYSTEM_PROMPT = """你是一名专业的研究规划师。你的任务是将复杂任务分解为有向无环图（DAG）形式的子任务。

规则：
1. **极简问题不回退**：如果问题极其简单（如"你好""1+1等于几""hello world"），直接创建 1 个 research 类型子任务即可，不要创建 code 类型。
2. 将任务分解为 1-5 个子任务（简单问题 1 个即可，不要过度拆分）。
3. 每个子任务必须有清晰的描述和类型（research / analysis / code / verify）。
4. 使用 task_id 指定子任务之间的依赖关系。
5. 没有依赖的子任务可以并行执行。
6. 分配优先级（1=最高，10=最低）。
7. 为每个子任务提供具体的搜索方向或角度（subtopics）。
8. 只返回合法的 JSON——不要加 markdown、代码块或注释。
9. 所有 description、reasoning 文本必须使用中文。

输出 JSON 格式：
{
  "reasoning": "分解策略的简要说明（中文）。",
  "subtasks": [
    {
      "task_id": "t1",
      "description": "该子任务的具体描述（中文）。",
      "task_type": "research",
      "dependencies": [],
      "priority": 1,
      "subtopics": ["具体搜索关键词1", "具体搜索关键词2"]
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
        # 使用 _llm_override 而非直接改 self._llm，保证协程安全
        from mindforge.models.base import LLMFactory
        _llm_override = LLMFactory.create(
            settings.llm.llm_provider, planner_model
        )

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(
                role="user",
                content=(
                    "请将以下任务分解为 DAG 子任务，子任务总数不得超过 "
                    f"{settings.agent.max_subtasks}：\n\n{task}"
                ),
            ),
        ]

        try:
            result = await self._chat(
                messages,
                response_format={"type": "json_object"},
                temperature=0.3,
                _llm_override=_llm_override,
            )

            raw = result.content.strip()
            plan_dict = json.loads(raw)
            plan_dict["original_task"] = task
            plan_dict["plan_id"] = uuid.uuid4().hex[:12]

            plan = ResearchPlan.from_dict(plan_dict)
            plan.planner_usage = result.usage or {}

            # Validate at least one subtask
            if not plan.subtasks:
                raise ValueError("Planner returned zero subtasks.")
            plan.validate(max_subtasks=settings.agent.max_subtasks)

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
