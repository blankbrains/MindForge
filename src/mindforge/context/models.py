"""Typed context-domain values shared by services and the orchestrator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

ContextMode = Literal["auto", "manual", "disabled"]
ContextSourceType = Literal["message", "summary", "artifact", "memory", "document"]


@dataclass(frozen=True)
class ResearchRequestContext:
    """User controls and stable identifiers for one research request."""

    run_id: str
    conversation_id: str | None = None
    message_id: str | None = None
    context_mode: ContextMode = "auto"
    selected_context_ids: tuple[str, ...] = ()
    excluded_context_ids: tuple[str, ...] = ()
    independent: bool = False


@dataclass
class ContextCandidate:
    """One eligible candidate before final budget allocation."""

    source_type: ContextSourceType
    source_id: str
    content: str
    title: str
    created_at: datetime | None = None
    score: float = 0.0
    token_count: int = 0
    selection_reason: str = "related"
    pinned: bool = False
    explicitly_selected: bool = False
    freshness_status: str = "current"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_id(self) -> str:
        return f"{self.source_type}:{self.source_id}"

    def to_public_dict(self, *, included: bool, exclusion_reason: str | None = None) -> dict:
        return {
            "context_id": self.context_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "title": self.title,
            "content": self.content,
            "score": round(float(self.score), 6),
            "token_count": self.token_count,
            "selection_reason": self.selection_reason,
            "pinned": self.pinned,
            "explicitly_selected": self.explicitly_selected,
            "freshness_status": self.freshness_status,
            "included": included,
            "exclusion_reason": exclusion_reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ContextSelection:
    """A candidate rejected by policy or budget with an observable reason."""

    candidate: ContextCandidate
    reason: str


@dataclass
class ContextBundle:
    """Final bounded context consumed by one Agent run."""

    standalone_query: str
    requires_context: bool
    items: list[ContextCandidate]
    excluded: list[ContextSelection]
    budget_tokens: int
    used_tokens: int
    policy_version: str = "context-policy-v1"
    embedding_version: str = "lexical-v1"
    snapshot_id: str | None = None

    @property
    def fingerprint(self) -> str:
        payload = {
            "query": self.standalone_query.strip(),
            "policy": self.policy_version,
            "embedding": self.embedding_version,
            "items": [
                {
                    "id": item.context_id,
                    "title": item.title,
                    "content": hashlib.sha256(
                        item.content.encode("utf-8")
                    ).hexdigest(),
                    "selection_reason": item.selection_reason,
                    "freshness": item.freshness_status,
                }
                for item in self.items
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_working_chunks(self) -> list[dict[str, Any]]:
        return [
            {
                "id": item.context_id,
                "content": (
                    f"## {item.title}\n"
                    f"来源类型: {item.source_type}; "
                    f"选择原因: {item.selection_reason}; "
                    f"时效状态: {item.freshness_status}\n"
                    f"{item.content}"
                ),
                "rerank_score": item.score,
                "context_source_type": item.source_type,
                "context_source_id": item.source_id,
                "untrusted_content": True,
            }
            for item in self.items
        ]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "standalone_query": self.standalone_query,
            "requires_context": self.requires_context,
            "budget_tokens": self.budget_tokens,
            "used_tokens": self.used_tokens,
            "context_fingerprint": self.fingerprint,
            "policy_version": self.policy_version,
            "embedding_version": self.embedding_version,
            "items": [
                item.to_public_dict(included=True)
                for item in self.items
            ],
            "excluded": [
                selection.candidate.to_public_dict(
                    included=False,
                    exclusion_reason=selection.reason,
                )
                for selection in self.excluded
            ],
        }
