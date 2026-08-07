"""Explainable lexical ranking for context candidates."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from mindforge.context.models import ContextCandidate

_LATIN_TOKEN = re.compile(r"[a-zA-Z0-9_+#.-]{2,}")
_CJK_BLOCK = re.compile(r"[\u3400-\u9fff]+")


def lexical_terms(text: str) -> set[str]:
    lowered = text.lower()
    terms = set(_LATIN_TOKEN.findall(lowered))
    for block in _CJK_BLOCK.findall(lowered):
        if len(block) == 1:
            terms.add(block)
        else:
            terms.update(block[index : index + 2] for index in range(len(block) - 1))
    return terms


def lexical_relevance(query: str, content: str) -> float:
    query_terms = lexical_terms(query)
    if not query_terms:
        return 0.0
    content_terms = lexical_terms(content[:20_000])
    if not content_terms:
        return 0.0
    overlap = len(query_terms & content_terms)
    coverage = overlap / len(query_terms)
    precision = overlap / min(max(len(content_terms), 1), max(len(query_terms) * 4, 1))
    return min(1.0, coverage * 0.75 + precision * 0.25)


def rank_candidates(
    query: str,
    candidates: list[ContextCandidate],
    *,
    referenced_message_ids: set[str],
) -> list[ContextCandidate]:
    now = datetime.now(timezone.utc)
    for candidate in candidates:
        relevance = lexical_relevance(query, candidate.content)
        score = relevance
        reasons: list[str] = []
        directly_referenced = (
            candidate.source_type == "message"
            and candidate.source_id in referenced_message_ids
        )
        candidate.metadata["referenced_by_query"] = directly_referenced
        if candidate.explicitly_selected:
            score += 1.0
            reasons.append("用户显式选择")
        if directly_referenced:
            score += 1.25
            reasons.append("当前追问引用最近一轮")
        if candidate.pinned:
            score += 0.65
            reasons.append("用户固定")
        recency_rank = candidate.metadata.get("message_recency_rank")
        if candidate.metadata.get("latest_turn"):
            score += 0.35
            reasons.append("最近一轮")
        elif isinstance(recency_rank, int) and not isinstance(
            recency_rank,
            bool,
        ):
            score += max(0.0, 0.2 - recency_rank * 0.03)
            reasons.append("近期会话")
        if candidate.metadata.get("same_conversation"):
            score += 0.2
            reasons.append("同一会话")
        quality = candidate.metadata.get("quality_score")
        if isinstance(quality, (int, float)):
            score += max(0.0, min(0.2, float(quality) / 50.0))
        if candidate.created_at is not None:
            created = candidate.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - created).total_seconds() / 86400)
            score += max(0.0, 0.15 * (1.0 - age_days / 30.0))
        if candidate.freshness_status == "stale":
            score -= 0.25
        candidate.score = max(0.0, score)
        if reasons:
            candidate.selection_reason = "、".join(reasons)
        elif relevance > 0:
            candidate.selection_reason = "与当前问题相关"
        else:
            candidate.selection_reason = "最近会话上下文"
    def sortable_created_at(candidate: ContextCandidate) -> datetime:
        created = candidate.created_at
        if created is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        return (
            created
            if created.tzinfo is not None
            else created.replace(tzinfo=timezone.utc)
        )

    return sorted(
        candidates,
        key=lambda item: (
            item.explicitly_selected,
            bool(item.metadata.get("referenced_by_query")),
            item.pinned,
            item.score,
            -int(item.metadata.get("message_recency_rank", 1_000_000)),
            sortable_created_at(item),
        ),
        reverse=True,
    )
