"""Shared role contracts for MindForge's four research agents."""

from __future__ import annotations

import re
from typing import Literal


AGENT_CONTRACT_VERSION = "1.0"

AgentRole = Literal["planner", "researcher", "synthesizer", "critic"]
ResearchTaskType = Literal["research", "analysis", "code", "verify"]
ResearchOutputMode = Literal["final_answer", "evidence_brief"]

SUPPORTED_RESEARCH_TASK_TYPES = frozenset(
    {"research", "analysis", "code", "verify"}
)

_ROLE_CONTRACTS: dict[AgentRole, str] = {
    "planner": (
        "你只负责规划，不回答原问题、不调用工具、不生成最终报告，也不修改运行状态。"
        "输出必须是可执行 DAG；每个子任务只描述一个独立目标及其研究方向。"
        "最终整合、用户答复和质量评分分别属于 Synthesizer 与 Critic。"
    ),
    "researcher": (
        "你一次只执行一个已分配子任务，不修改 DAG、不扩展到兄弟子任务，也不替"
        "Synthesizer 完成多任务总报告。必须区分证据、推断和未知；工具失败时明确"
        "降级，不得伪造来源、工具结果或已完成的验证。"
    ),
    "synthesizer": (
        "你只根据提供的子任务发现、来源和评审反馈生成最终答案，不调用工具、不"
        "新增未提供的事实或来源、不改变研究计划。证据不足或相互冲突时必须保留"
        "不确定性，并把缺口明确写入最终答案。"
    ),
    "critic": (
        "你只评估，不重写报告、不补充新事实、不假设未提供的来源。评分必须对应"
        "明确问题；无效引用、缺少必要引用、深度明显不足等确定性缺陷必须进入问题"
        "列表并触发精炼，不能被整体印象分掩盖。"
    ),
}

_FINALIZATION_PATTERNS = (
    re.compile(
        r"(?:最后|最终).{0,12}(?:汇总|综合|整合|合并).{0,24}"
        r"(?:完整)?(?:建议|报告|答案|答复|回复|结果|结论)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:汇总|综合|整合|合并).{0,24}"
        r"(?:全部|所有|各(?:个)?|子任务|研究)(?:结果|发现|证据|结论)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:生成|撰写|编写|输出).{0,16}"
        r"(?:最终)?(?:报告|答案|答复|回复|总结)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:最终|完整)(?:研究)?(?:报告|答案|答复|回复)$",
        re.IGNORECASE,
    ),
)


def role_contract(role: AgentRole) -> str:
    """Return the authoritative prompt boundary for one agent role."""
    return (
        f"## 角色合约 v{AGENT_CONTRACT_VERSION}\n\n"
        f"{_ROLE_CONTRACTS[role]}"
    )


def is_finalization_only_task(
    description: str,
    *,
    dependencies: list[str] | None = None,
) -> bool:
    """Return whether a planned subtask duplicates final synthesis."""
    normalized = " ".join(str(description or "").strip().split())
    if not normalized:
        return False
    if any(pattern.search(normalized) for pattern in _FINALIZATION_PATTERNS):
        return True
    normalized_dependencies = {
        str(dependency).strip().casefold()
        for dependency in dependencies or []
        if str(dependency).strip()
    }
    if normalized_dependencies and re.search(
        r"(?:汇总|综合|整合|合并).{0,20}"
        r"(?:结果|发现|证据|结论)",
        normalized,
        re.IGNORECASE,
    ):
        return True
    if len(normalized_dependencies) >= 2 and re.search(
        r"(?:基于|根据|结合).{0,48}(?:结果|发现|证据|结论)",
        normalized,
        re.IGNORECASE,
    ):
        return True
    return bool(normalized_dependencies) and bool(
        re.search(
            r"(?:全部|所有|各(?:个)?|前述|上述|已有).{0,20}"
            r"(?:子任务)?(?:结果|发现|证据|结论)",
            normalized,
            re.IGNORECASE,
        )
    )


def research_output_mode(total_subtasks: int) -> ResearchOutputMode:
    """Select whether a Researcher writes a final answer or evidence brief."""
    return "final_answer" if total_subtasks <= 1 else "evidence_brief"


def research_output_instruction(mode: ResearchOutputMode) -> str:
    """Return the binding output instruction for a Researcher task."""
    if mode == "evidence_brief":
        return (
            "输出本子任务的高密度证据简报：覆盖事实、依据、适用条件、限制和"
            "必要例外；不要写执行摘要、最终报告、跨子任务总括结论或面向用户的"
            "完整答复。"
        )
    return (
        "输出直接回应当前单一研究问题的完整答案；不要讨论未分配的其他目标，"
        "也不要描述 Agent 分工或执行过程。"
    )
