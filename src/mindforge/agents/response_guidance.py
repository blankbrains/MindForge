"""Shared response-depth guidance for research and synthesis agents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ResponseDepth = Literal["concise", "focused", "standard", "deep", "code"]


@dataclass(frozen=True)
class ResponseProfile:
    """A soft output budget and structure appropriate for one task."""

    depth: ResponseDepth
    min_chars: int | None
    max_chars: int | None
    structure: str

    def guidance(self) -> str:
        if self.min_chars is None or self.max_chars is None:
            budget = "不要按说明文字数凑篇幅，以完整、可运行的结果为准。"
        else:
            budget = (
                f"正文通常控制在 {self.min_chars}-{self.max_chars} 个中文字符。"
                "这是软目标：信息完整后立即停止，不得为了达到下限重复结论、"
                "堆砌背景或增加无关章节；除非用户明确要求简短，否则在核心问题、"
                "证据、关键条件、例外或实践建议尚未覆盖时不得提前结束。"
                "确有必要时可以适度超出上限。"
            )
        return f"{self.structure}{budget}"


_CONVERSATIONAL_TASKS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "你好",
        "你好啊",
        "你好呀",
        "您好",
        "嗨",
        "在吗",
        "早上好",
        "上午好",
        "下午好",
        "晚上好",
        "谢谢",
        "多谢",
        "再见",
        "你是谁",
        "你叫什么",
    }
)
_CONCISE_MARKERS = (
    "一句话",
    "一两句",
    "三句话",
    "简单说",
    "简单回答",
    "简短",
    "简要",
    "简洁",
    "只要结论",
    "直接结论",
    "概括一下",
    "不要展开",
    "brief",
    "briefly",
    "concise",
    "tl;dr",
)
_DEEP_MARKERS = (
    "深入",
    "全面",
    "系统",
    "详细",
    "研究报告",
    "技术方案",
    "实施方案",
    "架构设计",
    "文献综述",
    "风险评估",
    "多维度",
    "逐步验证",
    "完整方案",
    "deep research",
    "comprehensive",
    "in depth",
)
_DECISION_MARKERS = (
    "推荐",
    "选择",
    "选型",
    "比较",
    "对比",
    "区别",
    "差异",
    "优缺点",
    "利弊",
    "取舍",
    "哪个好",
    "哪个更",
    "怎么选",
    "如何选",
    "建议",
    "recommend",
    "compare",
    "versus",
    " vs ",
)
_MULTI_INTENT_MARKERS = (
    "原因、影响和",
    "原因、方案",
    "现状、问题",
    "优点、缺点",
    "同时分析",
    "分别分析",
    "并给出",
    "并说明",
    "以及如何",
    "从多个角度",
    "多个方面",
)
_ENUMERATED_INTENT_RE = re.compile(
    r"(?:^|[\s，,；;])(?:[1-9][.、：:]|[一二三四五六七八九十]+[、：:])"
)


def _normalize(task: str) -> str:
    return " ".join(task.strip().casefold().split())


def is_conversational_task(task: str) -> bool:
    """Return whether a task is a bounded greeting or social exchange."""
    normalized = _normalize(task)
    return bool(normalized) and normalized in _CONVERSATIONAL_TASKS


def classify_response_depth(
    task: str,
    *,
    task_type: str = "research",
    subtask_count: int = 1,
) -> ResponseDepth:
    """Classify the amount of explanation the task actually needs."""

    normalized = _normalize(task)
    if task_type == "code":
        return "code"
    if not normalized or is_conversational_task(task):
        return "concise"
    if any(marker in normalized for marker in _CONCISE_MARKERS):
        return "concise"
    if any(marker in normalized for marker in _DEEP_MARKERS):
        return "deep"

    sentence_breaks = len(re.findall(r"[。！？!?；;\n]", normalized))
    multi_intent_count = sum(
        marker in normalized for marker in _MULTI_INTENT_MARKERS
    )
    is_compound = (
        subtask_count >= 3
        or len(normalized) > 180
        or sentence_breaks >= 3
        or multi_intent_count >= 2
        or len(_ENUMERATED_INTENT_RE.findall(normalized)) >= 2
    )
    if is_compound:
        return "deep"
    if subtask_count >= 2 or any(
        marker in normalized for marker in _DECISION_MARKERS
    ):
        return "standard"
    if len(normalized) <= 100 and sentence_breaks <= 1:
        return "focused"
    return "standard"


def response_profile(
    task: str,
    *,
    task_type: str = "research",
    subtask_count: int = 1,
    final_report: bool = False,
) -> ResponseProfile:
    """Return a soft budget and structure for one answer."""

    depth = classify_response_depth(
        task,
        task_type=task_type,
        subtask_count=subtask_count,
    )
    if depth == "code":
        return ResponseProfile(
            depth=depth,
            min_chars=None,
            max_chars=None,
            structure=(
                "以完整、可运行的代码或可执行步骤为主体；补充必要的依赖、"
                "使用方式、边界条件和错误处理，避免重复解释代码本身。"
            ),
        )
    if depth == "concise":
        return ResponseProfile(
            depth=depth,
            min_chars=100,
            max_chars=500,
            structure=(
                "直接回答，不写执行摘要或固定章节；通常使用 1-3 个短段落，"
                "只有确有必要时才列出少量要点。"
            ),
        )
    if depth == "focused":
        return ResponseProfile(
            depth=depth,
            min_chars=1000,
            max_chars=1800,
            structure=(
                "先给直接结论，再解释核心原理、关键条件、典型应用或例子以及必要限制；"
                "通常使用 3-5 个有实际内容的自然小节，不展开与问题无关的背景。"
                "章节数量是指导范围，不得创建空章节。"
            ),
        )
    if depth == "standard":
        if final_report and subtask_count >= 2:
            min_chars, max_chars = 2400, 4200
        else:
            min_chars, max_chars = 1600, 2800
        return ResponseProfile(
            depth=depth,
            min_chars=min_chars,
            max_chars=max_chars,
            structure=(
                "先给结论，再按判断标准、关键理由、对比或实施步骤展开，"
                "最后说明适用条件、例外和可执行建议；通常使用 4-7 个有实际内容的"
                "章节，章节数量是指导范围，不得为凑数量拆分重复观点。"
                "只有信息天然具有行列关系时才使用表格。"
            ),
        )
    return ResponseProfile(
        depth=depth,
        min_chars=4000 if final_report else 3200,
        max_chars=8000 if final_report else 6000,
        structure=(
            "先给执行摘要，再按主要维度展开证据、分析、取舍、风险和建议；"
            "通常使用 6-10 个有实际内容的章节，章节数量是指导范围；"
            "章节必须服务于问题，不得为填充模板创建空章节或重复章节。"
        ),
    )


def build_response_guidance(
    task: str,
    *,
    task_type: str = "research",
    subtask_count: int = 1,
    final_report: bool = False,
) -> str:
    """Build prompt text shared by Researcher and Synthesizer."""

    return response_profile(
        task,
        task_type=task_type,
        subtask_count=subtask_count,
        final_report=final_report,
    ).guidance()
