"""Context collection, policy filtering, ranking, and allocation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mindforge.config import get_settings
from mindforge.context.budget import allocate_budget
from mindforge.context.models import (
    ContextBundle,
    ContextCandidate,
    ContextSelection,
    ResearchRequestContext,
)
from mindforge.context.ranker import lexical_relevance, rank_candidates
from mindforge.context.resolver import (
    latest_conversation_turn,
    ordered_visible_messages,
    resolve_references,
)
from mindforge.interaction import is_conversational_task


class ContextBuilder:
    """Build observable context from repository-provided, user-visible data."""

    def __init__(
        self,
        *,
        recent_messages: list[dict[str, Any]],
        summary: dict[str, Any] | None,
        artifacts: list[dict[str, Any]],
        memories: list[dict[str, Any]],
    ) -> None:
        self._recent_messages = recent_messages
        self._summary = summary
        self._artifacts = artifacts
        self._memories = memories
        settings = get_settings()
        self._settings = settings.context
        self._chars_per_token = settings.memory.chars_per_token

    def build(
        self,
        query: str,
        request: ResearchRequestContext,
    ) -> ContextBundle:
        mode = request.context_mode
        if is_conversational_task(query):
            return ContextBundle(
                standalone_query=query.strip(),
                requires_context=False,
                items=[],
                excluded=[],
                budget_tokens=self._settings.budget_tokens,
                used_tokens=0,
            )
        disabled = (
            not self._settings.enabled
            or request.independent
            or mode == "disabled"
        )
        resolution = resolve_references(query, self._recent_messages)
        if disabled:
            return ContextBundle(
                standalone_query=query.strip(),
                requires_context=False,
                items=[],
                excluded=[],
                budget_tokens=self._settings.budget_tokens,
                used_tokens=0,
            )

        selected_ids = set(request.selected_context_ids)
        excluded_ids = set(request.excluded_context_ids)
        referenced_message_ids = set(resolution.referenced_message_ids)
        candidates = self._collect_candidates(
            selected_ids,
            conversation_id=request.conversation_id,
        )
        policy_excluded: list[ContextSelection] = []
        eligible: list[ContextCandidate] = []
        now = datetime.now(timezone.utc)

        for candidate in candidates:
            if candidate.context_id in excluded_ids or candidate.source_id in excluded_ids:
                policy_excluded.append(ContextSelection(candidate, "user_excluded"))
                continue
            if mode == "manual" and not candidate.explicitly_selected:
                policy_excluded.append(ContextSelection(candidate, "not_selected"))
                continue
            expires_at = candidate.metadata.get("expires_at")
            if isinstance(expires_at, datetime):
                comparable = (
                    expires_at
                    if expires_at.tzinfo is not None
                    else expires_at.replace(tzinfo=timezone.utc)
                )
                if comparable <= now:
                    policy_excluded.append(ContextSelection(candidate, "expired"))
                    continue
            if (
                candidate.source_type == "artifact"
                and candidate.metadata.get("grounding_status") == "model_only"
            ):
                policy_excluded.append(ContextSelection(candidate, "model_only"))
                continue
            relevance = lexical_relevance(query, candidate.content)
            directly_referenced = (
                candidate.source_type == "message"
                and candidate.source_id in referenced_message_ids
            )
            if directly_referenced:
                candidate.metadata["referenced_by_query"] = True
            if (
                mode == "auto"
                and candidate.source_type
                in {"message", "artifact", "memory"}
                and not candidate.pinned
                and not candidate.explicitly_selected
                and not directly_referenced
                and not candidate.metadata.get("latest_turn")
                and relevance < self._settings.min_relevance
            ):
                policy_excluded.append(ContextSelection(candidate, "low_relevance"))
                continue
            eligible.append(candidate)

        ranked = rank_candidates(
            query,
            eligible,
            referenced_message_ids=referenced_message_ids,
        )
        selected, budget_excluded, used = allocate_budget(
            ranked,
            budget_tokens=self._settings.budget_tokens,
            chars_per_token=self._chars_per_token,
            max_item_chars=self._settings.snapshot_max_item_chars,
        )
        return ContextBundle(
            standalone_query=resolution.standalone_query,
            requires_context=resolution.requires_context,
            items=selected,
            excluded=policy_excluded + budget_excluded,
            budget_tokens=self._settings.budget_tokens,
            used_tokens=used,
        )

    def _collect_candidates(
        self,
        selected_ids: set[str],
        *,
        conversation_id: str | None,
    ) -> list[ContextCandidate]:
        candidates: list[ContextCandidate] = []
        ordered_messages = ordered_visible_messages(self._recent_messages)
        latest_turn_ids = {
            str(message.get("message_id"))
            for message in latest_conversation_turn(ordered_messages)
            if message.get("message_id")
        }
        newest_first = list(reversed(ordered_messages))
        recency_ranks = {
            str(message.get("message_id")): rank
            for rank, message in enumerate(newest_first)
            if message.get("message_id")
        }
        for message in ordered_messages:
            source_id = str(message["message_id"])
            candidates.append(
                ContextCandidate(
                    source_type="message",
                    source_id=source_id,
                    title=(
                        "用户问题"
                        if message.get("role") == "user"
                        else "MindForge 回答"
                    ),
                    content=str(message.get("content") or ""),
                    created_at=message.get("created_at"),
                    pinned=bool(message.get("pinned")),
                    explicitly_selected=(
                        source_id in selected_ids
                        or f"message:{source_id}" in selected_ids
                    ),
                    metadata={
                        "role": message.get("role"),
                        "sequence": message.get("sequence"),
                        "same_conversation": True,
                        "latest_turn": source_id in latest_turn_ids,
                        "message_recency_rank": recency_ranks.get(source_id),
                    },
                )
            )
        if self._summary:
            source_id = str(self._summary["summary_id"])
            summary = self._summary.get("summary") or {}
            compression = dict(summary.get("_compression") or {})
            content = "\n".join(
                [
                    f"目标：{summary.get('goal', '')}",
                    "约束：" + "；".join(summary.get("constraints") or []),
                    "决策：" + "；".join(summary.get("decisions") or []),
                    "待解决：" + "；".join(summary.get("open_questions") or []),
                ]
            ).strip()
            candidates.append(
                ContextCandidate(
                    source_type="summary",
                    source_id=source_id,
                    title="会话摘要",
                    content=content,
                    created_at=self._summary.get("created_at"),
                    explicitly_selected=(
                        source_id in selected_ids
                        or f"summary:{source_id}" in selected_ids
                    ),
                    metadata={"same_conversation": True},
                )
            )
            candidates[-1].metadata.update(
                {
                    "compression_method": compression.get(
                        "method",
                        "deterministic",
                    ),
                    "compression_model": compression.get("model"),
                    "compression_status": compression.get(
                        "model_attempt_status",
                        (
                            "compressed"
                            if compression.get("method") == "model"
                            else "deterministic"
                        ),
                    ),
                    "source_message_count": compression.get(
                        "source_message_count",
                        len(self._summary.get("source_message_ids") or []),
                    ),
                    "compressed_chars": compression.get("compressed_chars"),
                    "from_sequence": self._summary.get("from_sequence"),
                    "to_sequence": self._summary.get("to_sequence"),
                }
            )
        for artifact in self._artifacts:
            source_id = str(artifact["artifact_id"])
            expires_at = artifact.get("expires_at")
            freshness = "current"
            if isinstance(expires_at, datetime):
                now = datetime.now(timezone.utc)
                comparable = (
                    expires_at
                    if expires_at.tzinfo is not None
                    else expires_at.replace(tzinfo=timezone.utc)
                )
                if comparable <= now:
                    freshness = "expired"
            candidates.append(
                ContextCandidate(
                    source_type="artifact",
                    source_id=source_id,
                    title=str(artifact.get("title") or "历史研究产物"),
                    content=str(artifact.get("content") or ""),
                    created_at=artifact.get("created_at"),
                    explicitly_selected=(
                        source_id in selected_ids
                        or f"artifact:{source_id}" in selected_ids
                    ),
                    freshness_status=freshness,
                    metadata={
                        "same_conversation": (
                            conversation_id is not None
                            and artifact.get("conversation_id") == conversation_id
                        ),
                        "quality_score": artifact.get("quality_score"),
                        "grounding_status": artifact.get("grounding_status"),
                        "expires_at": expires_at,
                        "run_id": artifact.get("run_id"),
                    },
                )
            )
        for memory in self._memories:
            source_id = str(memory["memory_id"])
            candidates.append(
                ContextCandidate(
                    source_type="memory",
                    source_id=source_id,
                    title=str(memory.get("category") or "长期记忆"),
                    content=str(memory.get("content") or ""),
                    created_at=memory.get("updated_at"),
                    pinned=bool(memory.get("user_confirmed")),
                    explicitly_selected=(
                        source_id in selected_ids
                        or f"memory:{source_id}" in selected_ids
                    ),
                    metadata={
                        "confidence": memory.get("confidence"),
                        "expires_at": memory.get("expires_at"),
                    },
                )
            )
        return candidates
