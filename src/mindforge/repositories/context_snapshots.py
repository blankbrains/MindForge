"""Immutable context snapshot persistence."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from mindforge.context.models import ContextBundle
from mindforge.db import ContextSnapshot, ContextSnapshotItem


def delete_expired(
    db: Session,
    *,
    retention_days: int,
    now: datetime | None = None,
) -> int:
    """Delete immutable snapshots older than the configured retention window."""
    reference = now or datetime.now(timezone.utc)
    cutoff = reference - timedelta(days=max(1, retention_days))
    return (
        db.query(ContextSnapshot)
        .filter(ContextSnapshot.created_at < cutoff)
        .delete(synchronize_session=False)
    )


def create(
    db: Session,
    *,
    run_id: str,
    conversation_id: str | None,
    query_message_id: str | None,
    bundle: ContextBundle,
) -> dict[str, Any]:
    existing = (
        db.query(ContextSnapshot)
        .filter(ContextSnapshot.run_id == run_id)
        .first()
    )
    if existing is not None:
        bundle.snapshot_id = existing.snapshot_id
        return get_by_run(db, run_id=run_id) or {}
    snapshot_id = uuid.uuid4().hex
    snapshot = ContextSnapshot(
        snapshot_id=snapshot_id,
        run_id=run_id,
        conversation_id=conversation_id,
        query_message_id=query_message_id,
        standalone_query=bundle.standalone_query,
        context_fingerprint=bundle.fingerprint,
        budget_tokens=bundle.budget_tokens,
        used_tokens=bundle.used_tokens,
        policy_version=bundle.policy_version,
        embedding_version=bundle.embedding_version,
    )
    db.add(snapshot)
    for rank, item in enumerate(bundle.items, start=1):
        db.add(
            ContextSnapshotItem(
                snapshot_item_id=uuid.uuid4().hex,
                snapshot_id=snapshot_id,
                source_type=item.source_type,
                source_id=item.source_id,
                content_snapshot=item.content,
                rank=rank,
                score=item.score,
                token_count=item.token_count,
                selection_reason=item.selection_reason[:200],
                pinned=item.pinned,
                freshness_status=item.freshness_status,
                metadata_json={
                    **dict(item.metadata),
                    "title": item.title,
                    "explicitly_selected": item.explicitly_selected,
                },
            )
        )
    db.flush()
    bundle.snapshot_id = snapshot_id
    return get_by_run(db, run_id=run_id) or {}


def get_by_run(db: Session, *, run_id: str) -> dict[str, Any] | None:
    snapshot = (
        db.query(ContextSnapshot)
        .filter(ContextSnapshot.run_id == run_id)
        .first()
    )
    if snapshot is None:
        return None
    items = (
        db.query(ContextSnapshotItem)
        .filter(ContextSnapshotItem.snapshot_id == snapshot.snapshot_id)
        .order_by(ContextSnapshotItem.rank.asc())
        .all()
    )
    return {
        "snapshot_id": snapshot.snapshot_id,
        "run_id": snapshot.run_id,
        "conversation_id": snapshot.conversation_id,
        "query_message_id": snapshot.query_message_id,
        "standalone_query": snapshot.standalone_query,
        "context_fingerprint": snapshot.context_fingerprint,
        "budget_tokens": snapshot.budget_tokens,
        "used_tokens": snapshot.used_tokens,
        "policy_version": snapshot.policy_version,
        "embedding_version": snapshot.embedding_version,
        "created_at": snapshot.created_at,
        "items": [
            {
                "context_id": f"{item.source_type}:{item.source_id}",
                "source_type": item.source_type,
                "source_id": item.source_id,
                "title": str(
                    (item.metadata_json or {}).get("title")
                    or item.source_type
                ),
                "content": item.content_snapshot,
                "rank": item.rank,
                "score": item.score,
                "token_count": item.token_count,
                "selection_reason": item.selection_reason,
                "pinned": item.pinned,
                "explicitly_selected": bool(
                    (item.metadata_json or {}).get("explicitly_selected")
                ),
                "freshness_status": item.freshness_status,
                "included": True,
                "exclusion_reason": None,
                "metadata": dict(item.metadata_json or {}),
            }
            for item in items
        ],
    }
