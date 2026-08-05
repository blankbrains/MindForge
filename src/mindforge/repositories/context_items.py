"""Conversation summaries and lineage persistence."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from mindforge.db import ContextLineage, ConversationSummary


def get_active_summary(
    db: Session,
    *,
    conversation_id: str,
) -> dict[str, Any] | None:
    row = (
        db.query(ConversationSummary)
        .filter(
            ConversationSummary.conversation_id == conversation_id,
            ConversationSummary.status == "active",
        )
        .order_by(
            ConversationSummary.to_sequence.desc(),
            ConversationSummary.version.desc(),
        )
        .first()
    )
    if row is None:
        return None
    return {
        "summary_id": row.summary_id,
        "conversation_id": row.conversation_id,
        "from_sequence": row.from_sequence,
        "to_sequence": row.to_sequence,
        "summary": dict(row.summary or {}),
        "source_message_ids": list(row.source_message_ids or []),
        "version": row.version,
        "created_at": row.created_at,
    }


def replace_summary(
    db: Session,
    *,
    conversation_id: str,
    from_sequence: int,
    to_sequence: int,
    summary: dict[str, Any],
    source_message_ids: list[str],
) -> dict[str, Any]:
    db.query(ConversationSummary).filter(
        ConversationSummary.conversation_id == conversation_id,
        ConversationSummary.status == "active",
    ).update({"status": "superseded"}, synchronize_session=False)
    row = ConversationSummary(
        summary_id=uuid.uuid4().hex,
        conversation_id=conversation_id,
        from_sequence=from_sequence,
        to_sequence=to_sequence,
        summary=summary,
        source_message_ids=source_message_ids,
        status="active",
        version=1,
    )
    db.add(row)
    db.flush()
    for message_id in source_message_ids:
        add_lineage(
            db,
            source_type="message",
            source_id=message_id,
            derived_type="summary",
            derived_id=row.summary_id,
            relation="summarized_from",
        )
    return get_active_summary(db, conversation_id=conversation_id) or {}


def invalidate_summaries_for_message(db: Session, *, message_id: str) -> int:
    summary_ids = [
        row.derived_id
        for row in db.query(ContextLineage)
        .filter(
            ContextLineage.source_type == "message",
            ContextLineage.source_id == message_id,
            ContextLineage.derived_type == "summary",
        )
        .all()
    ]
    if not summary_ids:
        return 0
    return db.query(ConversationSummary).filter(
        ConversationSummary.summary_id.in_(summary_ids),
        ConversationSummary.status == "active",
    ).update({"status": "invalidated"}, synchronize_session=False)


def add_lineage(
    db: Session,
    *,
    source_type: str,
    source_id: str,
    derived_type: str,
    derived_id: str,
    relation: str,
) -> None:
    exists = (
        db.query(ContextLineage.lineage_id)
        .filter(
            ContextLineage.source_type == source_type,
            ContextLineage.source_id == source_id,
            ContextLineage.derived_type == derived_type,
            ContextLineage.derived_id == derived_id,
            ContextLineage.relation == relation,
        )
        .first()
    )
    if exists:
        return
    db.add(
        ContextLineage(
            lineage_id=uuid.uuid4().hex,
            source_type=source_type,
            source_id=source_id,
            derived_type=derived_type,
            derived_id=derived_id,
            relation=relation,
            created_at=datetime.now(timezone.utc),
        )
    )
