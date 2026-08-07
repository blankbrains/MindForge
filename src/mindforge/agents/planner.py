"""Planner agent — decomposes a complex task into a DAG of subtasks."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from mindforge.agents.base import (
    AgentResult,
    BaseAgent,
    _estimate_cost_details,
)
from mindforge.agents.contracts import (
    AGENT_CONTRACT_VERSION,
    ResearchTaskType,
    SUPPORTED_RESEARCH_TASK_TYPES,
    is_finalization_only_task,
    role_contract,
)
from mindforge.models.base import ChatMessage, ChatResult
from mindforge.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SubTask:
    """A single unit of work within a research plan."""

    task_id: str
    description: str
    task_type: ResearchTaskType = "research"
    dependencies: list[str] = field(default_factory=list)
    status: Literal[
        "pending",
        "in_progress",
        "completed",
        "failed",
    ] = "pending"
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
    planner_status: Literal["planned", "direct", "fallback"] = "planned"
    planner_error: str | None = None
    planner_usage: dict[str, int] = field(default_factory=dict)
    planner_cost_usd: float | None = None
    planner_cost_status: str = "usage_unavailable"
    contract_version: str = AGENT_CONTRACT_VERSION

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
        return sorted(ready, key=lambda task: (task.priority, task.task_id))

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
        if any(not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids):
            raise ValueError("Planner returned an empty task_id.")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Planner returned duplicate task_id values.")

        known_ids = set(task_ids)
        indegree = {task_id: 0 for task_id in task_ids}
        dependents: dict[str, list[str]] = {
            task_id: [] for task_id in task_ids
        }
        for task in self.subtasks:
            if not isinstance(task.description, str) or not task.description.strip():
                raise ValueError(
                    f"Task {task.task_id} has an empty description."
                )
            if task.task_type not in SUPPORTED_RESEARCH_TASK_TYPES:
                raise ValueError(
                    f"Task {task.task_id} has unsupported task_type "
                    f"{task.task_type!r}."
                )
            if (
                isinstance(task.priority, bool)
                or not isinstance(task.priority, int)
                or not 1 <= task.priority <= 10
            ):
                raise ValueError(
                    f"Task {task.task_id} priority must be an integer from 1 to 10."
                )
            if not isinstance(task.dependencies, list) or any(
                not isinstance(dependency, str) or not dependency.strip()
                for dependency in task.dependencies
            ):
                raise ValueError(
                    f"Task {task.task_id} dependencies must be non-empty strings."
                )
            if not isinstance(task.subtopics, list) or any(
                not isinstance(subtopic, str) or not subtopic.strip()
                for subtopic in task.subtopics
            ):
                raise ValueError(
                    f"Task {task.task_id} subtopics must be non-empty strings."
                )
            normalized = list(dict.fromkeys(task.dependencies))
            task.dependencies = normalized
            if len(self.subtasks) > 1 and is_finalization_only_task(
                task.description,
                dependencies=normalized,
            ):
                raise ValueError(
                    f"Task {task.task_id} duplicates final synthesis."
                )
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
            "planner_status": self.planner_status,
            "planner_error": self.planner_error,
            "contract_version": self.contract_version,
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
        raw_subtasks = data.get("subtasks", [])
        if not isinstance(raw_subtasks, list):
            raw_subtasks = []
        subtasks = [
            SubTask(
                task_id=s.get("task_id", str(uuid.uuid4())[:8]),
                description=s.get("description", ""),
                task_type=s.get("task_type", "research"),
                dependencies=s.get("dependencies", []),
                # Planner output describes work; it never controls runtime state.
                status="pending",
                priority=s.get("priority", 5),
                subtopics=s.get("subtopics", []),
            )
            for s in raw_subtasks
            if isinstance(s, dict)
        ]
        return cls(
            plan_id=data.get("plan_id", str(uuid.uuid4())[:8]),
            original_task=data.get("original_task", ""),
            subtasks=subtasks,
            reasoning=data.get("reasoning", ""),
            planner_status=(
                data.get("planner_status")
                if data.get("planner_status") in {"planned", "direct", "fallback"}
                else "planned"
            ),
            planner_error=(
                str(data["planner_error"])[:1000]
                if data.get("planner_error")
                else None
            ),
            contract_version=str(
                data.get("contract_version") or AGENT_CONTRACT_VERSION
            )[:50],
        )


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """你是一名专业的研究规划师。你的任务是将复杂任务分解为有向无环图（DAG）形式的子任务。

规则：
1. **极简问题不回退**：如果问题极其简单（如"你好""1+1等于几""hello world"），直接创建 1 个 research 类型子任务即可，不要创建 code 类型。
2. 遵守用户消息给出的子任务数量上限（简单定义或单一事实问题 1 个即可）。
3. 根据问题中的独立信息需求进行拆分，而不是套用固定数量：
   - 单一事实或定义问题通常 1 个子任务；
   - 比较、区别、优缺点或选型问题，应分别获取各对象或维度的证据；
   - 多问、诊断、方案设计或需要验证的问题，应覆盖各独立目标，并按真实数据流
     设置依赖关系。
4. 每个子任务必须有清晰的描述和类型（research / analysis / code / verify）。
5. 使用 task_id 指定子任务之间的依赖关系。
6. 没有依赖的子任务可以并行执行。
7. 分配优先级（1=最高，10=最低）。
8. 为每个子任务提供具体的搜索方向或角度（subtopics）。
9. 检查所有子任务合起来是否完整覆盖原问题，既不能遗漏，也不能重复。
10. 只返回合法的 JSON——不要加 markdown、代码块或注释。
11. 所有 description、reasoning 文本必须使用中文。
12. 不要创建“汇总全部结果”“生成最终报告”之类的收尾任务，最终综合由
    Synthesizer 统一负责。

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
}""" + "\n\n" + role_contract("planner")


class PlannerAgent(BaseAgent):
    """Decomposes a user task into a DAG-structured research plan."""

    model_role = "planner"
    deepseek_thinking_mode = "disabled"
    _FINAL_DELIVERABLE_RE = re.compile(
        r"(?:生成|制定|构建|输出|给出).{0,24}"
        r"(?:最终)?(?:决策矩阵|选型矩阵|评分矩阵)",
        re.IGNORECASE,
    )
    _COMPARISON_MARKERS = (
        "对比",
        "区别",
        "差异",
        "优缺点",
        "利弊",
        "取舍",
        "选型",
        "哪个更好",
        "哪一个更好",
        " versus ",
        " vs ",
    )
    _COMPLEX_MARKERS = (
        "全面",
        "深入",
        "系统分析",
        "研究报告",
        "技术方案",
        "实施方案",
        "架构设计",
        "文献综述",
        "根因",
        "诊断并修复",
    )
    _COMPOUND_MARKERS = (
        "以及",
        "同时",
        "分别",
        "并分析",
        "并比较",
        "原因和",
        "原因、",
        "影响和",
        "影响、",
        "现状和",
        "现状、",
        "问题和解决",
        "问题、解决",
    )
    _INTERROGATIVE_MARKERS = (
        "为什么",
        "为何",
        "如何",
        "怎么",
        "哪些",
        "是否",
    )
    _COMPARISON_SUFFIX_RE = re.compile(
        r"(?:之间)?(?:有(?:什么|何))?"
        r"(?:区别|差异|优缺点|利弊|取舍)"
        r"(?:是(?:什么|哪些))?[？?。.！!]*$",
        re.IGNORECASE,
    )
    _COMPARISON_PREFIX_RE = re.compile(
        r"^(?:(?:请|请你|帮我|分析一下|分析|说明一下|说明)\s*)?"
        r"(?:比较|对比)\s*",
        re.IGNORECASE,
    )
    _COMPARISON_CONNECTOR_RE = re.compile(
        r"\s*(?:和|与|跟|及|vs\.?|versus)\s*",
        re.IGNORECASE,
    )
    _DECISION_SUFFIX_RE = re.compile(
        r"(?:应该)?(?:怎么|如何)选(?:择)?"
        r"|(?:应该)?选(?:择)?(?:哪一个|哪个)"
        r"|(?:哪一个|哪个)(?:更|最)?(?:合适|适合|好)"
        r"[？?。.！!]*$",
        re.IGNORECASE,
    )
    _SELECTION_CRITERIA_SUFFIX_RE = re.compile(
        r"\s*(?:(?:在|用于|应用于).+?)?的?"
        r"(?:选型|选择)(?:依据|标准|条件|建议|策略|原则)?"
        r"[？?。.！!]*$",
        re.IGNORECASE,
    )
    @property
    def name(self) -> str:
        return "planner"

    @property
    def system_prompt(self) -> str:
        return _PLANNER_SYSTEM_PROMPT

    @classmethod
    def _is_comparison_task(cls, task: str) -> bool:
        normalized = f" {' '.join(task.casefold().split())} "
        if normalized.strip().startswith(("比较 ", "比较一下 ")):
            return True
        if (
            cls._COMPARISON_CONNECTOR_RE.search(task) is not None
            and cls._DECISION_SUFFIX_RE.search(task.strip()) is not None
        ):
            return True
        if any(
            marker in normalized
            for marker in cls._COMPARISON_MARKERS
            if marker != "比较"
        ):
            return True
        if "比较" not in normalized:
            return False
        return (
            cls._COMPARISON_CONNECTOR_RE.search(task) is not None
            and cls._comparison_subjects(task) is not None
        )

    @classmethod
    def _comparison_subjects(cls, task: str) -> tuple[str, str] | None:
        text = " ".join(task.strip().split())
        if not text:
            return None
        text = cls._COMPARISON_PREFIX_RE.sub("", text)
        text = cls._COMPARISON_SUFFIX_RE.sub("", text).strip()
        decision_task = cls._DECISION_SUFFIX_RE.search(text) is not None
        text = cls._DECISION_SUFFIX_RE.sub("", text).strip()
        text = cls._SELECTION_CRITERIA_SUFFIX_RE.sub("", text).strip()
        parts = cls._COMPARISON_CONNECTOR_RE.split(text, maxsplit=1)
        if len(parts) != 2:
            return None

        left = parts[0].strip(" ，,、:：")
        right = parts[1].strip(" ，,、:：的")
        if decision_task:
            right = re.sub(
                r"\s*(?:在|用于|应用于).+$",
                "",
                right,
            ).strip(" ，,、:：的")
        if not left or not right:
            return None
        shared_suffix = re.search(r"(的[^，,、:：]+)$", parts[1].strip())
        if shared_suffix and "的" not in left:
            left = f"{left}{shared_suffix.group(1)}"
            right = parts[1].strip(" ，,、:：")
        if left.casefold() == right.casefold():
            return None
        return left[:200], right[:200]

    @classmethod
    def _minimum_subtask_count(cls, task: str) -> int:
        normalized = " ".join(task.casefold().split())
        if cls._is_comparison_task(task):
            return 2
        if any(marker in normalized for marker in cls._COMPLEX_MARKERS):
            return 3
        question_count = len(re.findall(r"[？?]", task))
        intent_count = sum(
            1 for marker in cls._INTERROGATIVE_MARKERS if marker in task
        )
        explicit_goal_count = max(question_count, intent_count)
        if explicit_goal_count > 1:
            return min(explicit_goal_count, 3)
        if task.count("、") >= 1 and re.search(r"(?:以及|和|及)", task):
            return 3
        if task.count("、") >= 2:
            return 3
        if any(marker in normalized for marker in cls._COMPOUND_MARKERS):
            return 2
        return 1

    @classmethod
    def _quality_errors(
        cls,
        task: str,
        plan: ResearchPlan,
        *,
        max_subtasks: int | None = None,
    ) -> list[str]:
        errors: list[str] = []
        descriptions = [
            " ".join(subtask.description.casefold().split())
            for subtask in plan.subtasks
        ]
        if len(descriptions) != len(set(descriptions)):
            errors.append("子任务描述重复，未形成独立研究目标。")
        if any(not subtask.subtopics for subtask in plan.subtasks):
            errors.append("至少一个子任务缺少具体的 subtopics。")
        if len(plan.subtasks) > 1 and any(
            is_finalization_only_task(
                subtask.description,
                dependencies=subtask.dependencies,
            )
            for subtask in plan.subtasks
        ):
            errors.append(
                "计划包含由 Synthesizer 负责的最终汇总或报告生成任务。"
            )

        normalized_task = " ".join(task.casefold().split())
        comparison_task = cls._is_comparison_task(task)
        complex_task = any(
            marker in normalized_task for marker in cls._COMPLEX_MARKERS
        )
        question_count = len(re.findall(r"[？?]", task))
        minimum_subtasks = cls._minimum_subtask_count(task)
        if max_subtasks is not None:
            minimum_subtasks = min(minimum_subtasks, max_subtasks)
        requires_decomposition = minimum_subtasks > 1
        if len(plan.subtasks) < minimum_subtasks:
            errors.append(
                f"该问题至少需要 {minimum_subtasks} 个独立子任务，"
                f"当前只有 {len(plan.subtasks)} 个。"
            )
        if len(plan.subtasks) == 1:
            description = descriptions[0] if descriptions else ""
            if (
                description == normalized_task
                and requires_decomposition
                and (comparison_task or complex_task or question_count > 1)
            ):
                errors.append("唯一子任务只是原问题的原样复述。")
            if comparison_task and requires_decomposition:
                errors.append(
                    "比较类问题需要通过独立子任务获取比较对象或维度的证据，"
                    "不能只有一个研究子任务。"
                )
            if complex_task and requires_decomposition:
                errors.append("复杂研究目标没有被拆分为可独立执行的步骤。")

        if comparison_task:
            root_tasks = [
                subtask for subtask in plan.subtasks if not subtask.dependencies
            ]
            if len(plan.subtasks) < minimum_subtasks:
                errors.append(
                    "比较计划缺少独立的证据获取任务。"
                )
            required_root_tasks = min(2, minimum_subtasks)
            if len(root_tasks) < required_root_tasks:
                errors.append("比较计划缺少可并行执行的独立证据任务。")
            subjects = cls._comparison_subjects(task)
            if subjects is not None and required_root_tasks >= 2:
                left, right = subjects
                left_key = left.split("的", 1)[0].strip().casefold()
                right_key = right.split("的", 1)[0].strip().casefold()
                left_ids = {
                    subtask.task_id
                    for subtask in root_tasks
                    if left_key in subtask.description.casefold()
                }
                right_ids = {
                    subtask.task_id
                    for subtask in root_tasks
                    if right_key in subtask.description.casefold()
                }
                if not left_ids or not right_ids:
                    errors.append("比较计划没有分别覆盖两个比较对象。")

        return errors

    @classmethod
    def _duplicates_downstream_stage(
        cls,
        task: SubTask,
    ) -> bool:
        if is_finalization_only_task(
            task.description,
            dependencies=task.dependencies,
        ):
            return True
        if task.dependencies and cls._FINAL_DELIVERABLE_RE.search(
            task.description
        ):
            return True
        return False

    @classmethod
    def _prune_finalization_branches(
        cls,
        plan: ResearchPlan,
    ) -> list[str]:
        """Remove model-generated work already owned by later pipeline stages.

        Planner models sometimes append a final comparison/report task even
        though Synthesizer and Critic already own final composition and review.
        Removing that terminal branch is deterministic and preserves the
        independent evidence tasks. Invalid dependency structures are left
        untouched so normal validation can reject them.
        """
        if len(plan.subtasks) <= 1 or any(
            not isinstance(task.dependencies, list)
            for task in plan.subtasks
        ):
            return []

        removed_ids = {
            task.task_id
            for task in plan.subtasks
            if cls._duplicates_downstream_stage(task)
        }
        if not removed_ids:
            return []

        while True:
            descendants = {
                task.task_id
                for task in plan.subtasks
                if task.task_id not in removed_ids
                and any(
                    dependency in removed_ids
                    for dependency in task.dependencies
                )
            }
            if not descendants:
                break
            removed_ids.update(descendants)

        retained = [
            task for task in plan.subtasks if task.task_id not in removed_ids
        ]
        if not retained:
            return []

        plan.subtasks = retained
        normalized_ids = sorted(str(task_id) for task_id in removed_ids)
        logger.info(
            "Planner normalization removed finalization branch: %s",
            ", ".join(normalized_ids),
        )
        return normalized_ids

    # ------------------------------------------------------------------
    async def run(self, task: str) -> ResearchPlan:
        """Decompose *task* into a ResearchPlan.

        Falls back to a single-step plan on any parse error.
        """
        settings = get_settings()
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

        result: ChatResult | None = None
        total_usage: dict[str, int] = {}
        last_error = ""
        model_used = self._model_name
        for attempt in range(3):
            try:
                result = await self._chat(
                    messages,
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                model_used = result.model or model_used
                for key, value in (result.usage or {}).items():
                    if isinstance(value, (int, float)) and not isinstance(
                        value, bool
                    ):
                        total_usage[key] = total_usage.get(key, 0) + int(value)

                raw = result.content.strip()
                plan_dict = json.loads(raw)
                plan_dict["original_task"] = task
                plan_dict["plan_id"] = uuid.uuid4().hex[:12]

                plan = ResearchPlan.from_dict(plan_dict)
                self._prune_finalization_branches(plan)
                plan.validate(max_subtasks=settings.agent.max_subtasks)
                quality_errors = self._quality_errors(
                    task,
                    plan,
                    max_subtasks=settings.agent.max_subtasks,
                )
                if quality_errors:
                    raise ValueError("；".join(quality_errors))

                plan.planner_status = "planned"
                plan.planner_usage = total_usage
                cost_estimate = _estimate_cost_details(
                    model_used,
                    total_usage,
                    self._provider_name,
                )
                plan.planner_cost_usd = cost_estimate.amount_usd
                plan.planner_cost_status = cost_estimate.status
                return plan
            except Exception as exc:
                last_error = str(exc).strip() or type(exc).__name__
                if attempt < 2:
                    messages.extend(
                        [
                            ChatMessage(
                                role="assistant",
                                content=(
                                    result.content
                                    if result is not None
                                    else ""
                                ),
                            ),
                            ChatMessage(
                                role="user",
                                content=(
                                    "上一个计划未通过质量校验："
                                    f"{last_error}\n"
                                    "请重新理解原问题，修正覆盖范围、任务独立性"
                                    "和依赖关系，只返回完整 JSON。"
                                ),
                            ),
                        ]
                    )

        error_message = last_error or "Planner failed without an error message."
        logger.warning(
            "Planner failed after revision; exposing a degraded plan: %s",
            error_message,
        )
        cost_estimate = _estimate_cost_details(
            model_used,
            total_usage,
            self._provider_name,
        )
        fallback = ResearchPlan(
            plan_id=uuid.uuid4().hex[:12],
            original_task=task,
            subtasks=[
                SubTask(
                    task_id="t1",
                    description=task,
                    task_type="research",
                    priority=1,
                    subtopics=[task],
                )
            ],
            reasoning=(
                "Planner 未能返回通过结构与语义质量校验的计划；"
                "当前退回到保留原始问题语义的最小可执行任务。"
            ),
        )
        fallback.planner_status = "fallback"
        fallback.planner_error = error_message[:1000]
        fallback.planner_usage = total_usage
        fallback.planner_cost_usd = cost_estimate.amount_usd
        fallback.planner_cost_status = cost_estimate.status
        fallback.validate(max_subtasks=settings.agent.max_subtasks)
        return fallback
