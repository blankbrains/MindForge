"""Deterministic, source-bound conversation summaries."""

from __future__ import annotations

import re
from typing import Any

_CONSTRAINT_MARKERS = ("不要", "必须", "需要", "要求", "仅", "只能", "不能")


def build_structured_summary(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None = None,
    chars_per_token: int = 4,
) -> dict[str, Any]:
    """Compress visible messages without generating new factual claims."""
    user_messages = [
        str(message.get("content") or "").strip()
        for message in messages
        if message.get("role") == "user"
    ]
    assistant_messages = [
        str(message.get("content") or "").strip()
        for message in messages
        if message.get("role") == "assistant"
    ]
    goal = user_messages[0][:1000] if user_messages else ""
    constraints = [
        text[:800]
        for text in user_messages
        if any(marker in text for marker in _CONSTRAINT_MARKERS)
    ][:8]
    decisions: list[str] = []
    for text in assistant_messages[-4:]:
        headings = re.findall(r"(?m)^#{1,4}\s+(.{2,120})$", text)
        decisions.extend(headings[:4])
        if not headings and text:
            decisions.append(text[:500])
    open_questions = [
        text[:500]
        for text in user_messages[-3:]
        if text.endswith(("?", "？"))
    ]
    entities = sorted(
        {
            token
            for text in user_messages
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_.+#-]{2,}", text)
        }
    )[:30]
    summary = {
        "goal": goal,
        "constraints": constraints,
        "decisions": decisions[:12],
        "entities": entities,
        "open_questions": open_questions,
        "source_refs": [
            str(message["message_id"])
            for message in messages
            if message.get("message_id")
        ],
        "freshness_notes": [
            "该摘要只压缩历史对话，不替代当前知识库或联网证据。"
        ],
    }
    if max_tokens is not None:
        _bound_visible_fields(
            summary,
            max_chars=max(1, max_tokens * max(1, chars_per_token)),
        )
    return summary


def _bound_visible_fields(summary: dict[str, Any], *, max_chars: int) -> None:
    """Bound prompt-visible summary fields while retaining source lineage."""
    fields = ("goal", "constraints", "decisions", "open_questions")
    values = {
        "goal": [str(summary.get("goal") or "")],
        "constraints": [str(value) for value in summary.get("constraints") or []],
        "decisions": [str(value) for value in summary.get("decisions") or []],
        "open_questions": [
            str(value) for value in summary.get("open_questions") or []
        ],
    }
    bounded: dict[str, list[str]] = {field: [] for field in fields}
    remaining = max_chars
    per_item_cap = max(1, max_chars * 2 // 5)
    max_items = max(len(values[field]) for field in fields)

    for index in range(max_items):
        for field in fields:
            if remaining <= 0 or index >= len(values[field]):
                continue
            value = values[field][index].strip()
            if not value:
                continue
            separator_cost = 1 if any(bounded[name] for name in fields) else 0
            available = remaining - separator_cost
            if available <= 0:
                remaining = 0
                break
            chunk = value[: min(per_item_cap, available)]
            if chunk:
                bounded[field].append(chunk)
                remaining -= len(chunk) + separator_cost

    summary["goal"] = bounded["goal"][0] if bounded["goal"] else ""
    for field in fields[1:]:
        summary[field] = bounded[field]
