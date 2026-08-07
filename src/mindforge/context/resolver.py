"""Deterministic follow-up reference resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mindforge.context.ranker import lexical_relevance

_FOLLOW_UP_PATTERNS = (
    r"(刚才|之前|上一个|上一轮|前面|继续|接着|再展开|进一步)",
    r"(这个|那个|它|其|该方案|第二个|第一个|上述|前述)",
    r"^(?:有)?(?:什么|哪些)?(?:区别|风险|优缺点)(?:呢)?[？?]?$",
    r"^(?:相比|比较|为什么|怎么做|怎么办)(?:呢)?[？?]?$",
    r"^(那|那么|还有|以及|另外|然后)",
    (
        r"^(?:请)?(?:给我)?(?:再|继续)?"
        r"(?:(?:详细|具体|深入|全面|完整)(?:地)?){0,3}"
        r"(?:说明|描述|介绍|解释|展开(?:讲讲|说说)?|讲讲|说说|说)"
        r"(?:一下|一遍|一些|一点|点)?[。！？!?]*$"
    ),
    (
        r"^(?:再)?(?:(?:详细|具体|深入|全面|完整)(?:地)?){1,3}"
        r"(?:一点|一些|点)?[。！？!?]*$"
    ),
)


@dataclass(frozen=True)
class ReferenceResolution:
    requires_context: bool
    standalone_query: str
    referenced_message_ids: tuple[str, ...]
    confidence: float
    ambiguity: str | None = None


def ordered_visible_messages(
    recent_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return visible conversation messages in stable chronological order."""
    indexed = [
        (index, message)
        for index, message in enumerate(recent_messages)
        if message.get("role") in {"user", "assistant"}
        and str(message.get("content") or "").strip()
    ]

    def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, message = item
        sequence = message.get("sequence")
        if isinstance(sequence, int) and not isinstance(sequence, bool):
            return sequence, index
        return index, index

    return [message for _index, message in sorted(indexed, key=sort_key)]


def latest_conversation_turn(
    recent_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select the latest user message and its following assistant answer."""
    turns = conversation_turns(recent_messages)
    return turns[-1] if turns else []


def conversation_turns(
    recent_messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group chronological messages into user-led conversation turns."""
    visible = ordered_visible_messages(recent_messages)
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in visible:
        if message.get("role") == "user":
            if current:
                turns.append(current)
            current = [message]
            continue
        if current:
            if len(current) == 1:
                current.append(message)
            else:
                current[-1] = message
        else:
            turns.append([message])
    if current:
        turns.append(current)
    return turns


def select_referenced_turn(
    query: str,
    recent_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer the most recent turn among those matching explicit query topics."""
    turns = conversation_turns(recent_messages)
    if not turns:
        return []
    scored = [
        (
            lexical_relevance(
                query,
                "\n".join(
                    str(message.get("content") or "")
                    for message in turn
                ),
            ),
            index,
            turn,
        )
        for index, turn in enumerate(turns)
    ]
    best_relevance, _index, best_turn = max(
        scored,
        key=lambda item: (item[0], item[1]),
    )
    if best_relevance <= 0:
        return turns[-1]
    return best_turn


def resolve_references(
    query: str,
    recent_messages: list[dict[str, Any]],
) -> ReferenceResolution:
    """Resolve common follow-up forms without inventing hidden entities."""
    cleaned = query.strip()
    requires_context = any(
        re.search(pattern, cleaned, re.IGNORECASE)
        for pattern in _FOLLOW_UP_PATTERNS
    )
    if not recent_messages or not requires_context:
        return ReferenceResolution(
            requires_context=False,
            standalone_query=cleaned,
            referenced_message_ids=(),
            confidence=1.0,
        )

    visible = ordered_visible_messages(recent_messages)
    if not visible:
        return ReferenceResolution(
            requires_context=True,
            standalone_query=cleaned,
            referenced_message_ids=(),
            confidence=0.35,
            ambiguity="未找到可用于解析指代的有效历史消息。",
        )

    referenced = select_referenced_turn(cleaned, visible)
    context_lines = [
        f"{'用户' if item.get('role') == 'user' else 'MindForge'}："
        f"{str(item.get('content') or '').strip()[:1600]}"
        for item in referenced
    ]
    standalone = (
        "结合以下明确可见的前序对话回答当前问题。不得把前序结论当作"
        "未经验证的当前事实；涉及时效内容必须重新核验。\n\n"
        + "\n".join(context_lines)
        + f"\n\n当前问题：{cleaned}"
    )
    return ReferenceResolution(
        requires_context=True,
        standalone_query=standalone,
        referenced_message_ids=tuple(
            str(item["message_id"])
            for item in referenced
            if item.get("message_id")
        ),
        confidence=0.82 if len(referenced) == 2 else 0.68,
    )
