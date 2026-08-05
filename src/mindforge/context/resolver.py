"""Deterministic follow-up reference resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_FOLLOW_UP_PATTERNS = (
    r"(刚才|之前|上一个|上一轮|前面|继续|接着|再展开|进一步)",
    r"(这个|那个|它|其|该方案|第二个|第一个|上述|前述)",
    r"(相比|比较|区别|风险|优缺点|为什么|怎么做)[？?]?$",
    r"^(那|那么|还有|以及|另外|然后)",
)


@dataclass(frozen=True)
class ReferenceResolution:
    requires_context: bool
    standalone_query: str
    referenced_message_ids: tuple[str, ...]
    confidence: float
    ambiguity: str | None = None


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

    visible = [
        message
        for message in recent_messages
        if message.get("role") in {"user", "assistant"}
        and str(message.get("content") or "").strip()
    ]
    if not visible:
        return ReferenceResolution(
            requires_context=True,
            standalone_query=cleaned,
            referenced_message_ids=(),
            confidence=0.35,
            ambiguity="未找到可用于解析指代的有效历史消息。",
        )

    referenced = visible[-2:]
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
