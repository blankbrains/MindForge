"""Deterministic, source-bound conversation summaries."""

from __future__ import annotations

import re
from typing import Any

from mindforge.context.ranker import lexical_relevance

_CONSTRAINT_MARKERS = ("不要", "必须", "需要", "要求", "仅", "只能", "不能")
_VISIBLE_FIELDS = ("goal", "constraints", "decisions", "open_questions")


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
        "_compression": {
            "method": "deterministic",
            "source_message_count": len(messages),
            "source_chars": sum(
                len(str(message.get("content") or ""))
                for message in messages
            ),
        },
    }
    if max_tokens is not None:
        _bound_visible_fields(
            summary,
            max_chars=max(1, max_tokens * max(1, chars_per_token)),
        )
    summary["_compression"]["compressed_chars"] = len(
        render_summary_text(summary)
    )
    return summary


def merge_source_bound_model_summary(
    model_summary: dict[str, Any],
    *,
    fallback_summary: dict[str, Any],
    source_text: str,
    min_coverage: float,
    max_tokens: int,
    chars_per_token: int,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Merge only model fields that remain lexically grounded in source text."""
    merged = {
        key: (
            list(value)
            if isinstance(value, list)
            else dict(value)
            if isinstance(value, dict)
            else value
        )
        for key, value in fallback_summary.items()
    }
    accepted: list[str] = []
    rejected: list[str] = []

    goal = model_summary.get("goal")
    if isinstance(goal, str) and _is_source_bound(
        goal,
        source_text,
        min_coverage=min_coverage,
    ):
        merged["goal"] = goal.strip()
        accepted.append("goal")
    elif goal not in (None, ""):
        rejected.append("goal")

    for field in ("constraints", "decisions", "open_questions"):
        values = model_summary.get(field)
        if values is None:
            continue
        if not isinstance(values, list):
            rejected.append(field)
            continue
        valid = [
            value.strip()
            for value in values
            if isinstance(value, str)
            and _is_source_bound(
                value,
                source_text,
                min_coverage=min_coverage,
            )
        ]
        if valid:
            merged[field] = valid
            accepted.append(field)
        if len(valid) != len(values):
            rejected.append(field)

    entities = model_summary.get("entities")
    if isinstance(entities, list):
        valid_entities = [
            value.strip()
            for value in entities
            if isinstance(value, str)
            and value.strip()
            and value.strip().casefold() in source_text.casefold()
        ]
        if valid_entities:
            merged["entities"] = list(dict.fromkeys(valid_entities))[:30]
            accepted.append("entities")
        if len(valid_entities) != len(entities):
            rejected.append("entities")
    elif entities is not None:
        rejected.append("entities")

    merged["source_refs"] = list(fallback_summary.get("source_refs") or [])
    merged["freshness_notes"] = [
        "该摘要只压缩历史对话，不替代当前知识库或联网证据。"
    ]
    _bound_visible_fields(
        merged,
        max_chars=max(1, max_tokens * max(1, chars_per_token)),
    )
    return merged, list(dict.fromkeys(accepted)), list(dict.fromkeys(rejected))


def render_summary_text(summary: dict[str, Any]) -> str:
    """Render only fields that are visible to downstream Agent prompts."""
    sections = [str(summary.get("goal") or "").strip()]
    for field in _VISIBLE_FIELDS[1:]:
        sections.extend(
            str(value).strip()
            for value in summary.get(field) or []
            if str(value).strip()
        )
    return "\n".join(section for section in sections if section)


def _is_source_bound(
    value: str,
    source_text: str,
    *,
    min_coverage: float,
) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    if normalized.casefold() in source_text.casefold():
        return True
    return lexical_relevance(normalized, source_text) >= min_coverage


def _bound_visible_fields(summary: dict[str, Any], *, max_chars: int) -> None:
    """Bound prompt-visible summary fields while retaining source lineage."""
    fields = _VISIBLE_FIELDS
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
