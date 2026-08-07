"""Transactional context deletion and derivation invalidation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from mindforge.db import (
    ContextLineage,
    ContextSnapshot,
    ContextSnapshotItem,
    ConversationMessage,
    ConversationSummary,
    DeletionJob,
    ResearchArtifact,
    ResearchHistory,
    UserMemory,
)
from mindforge.repositories import context_items, conversations, messages


def invalidate_message_derivatives(
    db: Session,
    *,
    user_id: int,
    conversation_id: str,
    message_id: str,
) -> bool:
    """Invalidate summaries, artifacts, and memories derived from edited text."""
    conversation = conversations.get(
        db,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    if conversation is None:
        return False
    message = messages.get(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    if message is None:
        return False
    context_items.invalidate_summaries_for_message(
        db,
        message_id=message.message_id,
    )
    _invalidate_derived(
        db,
        source_type="message",
        source_id=message.message_id,
        hard=False,
    )
    db.flush()
    return True


def forget_message(
    db: Session,
    *,
    user_id: int,
    conversation_id: str,
    message_id: str,
) -> bool:
    conversation = conversations.get(
        db,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    if conversation is None:
        return False
    message = messages.get(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    if message is None:
        return False
    related = _run_messages(db, message)
    for related_message in related:
        related_message.include_in_context = False
        related_message.pinned = False
        context_items.invalidate_summaries_for_message(
            db,
            message_id=related_message.message_id,
        )
        _invalidate_derived(
            db,
            source_type="message",
            source_id=related_message.message_id,
            hard=False,
        )
    db.flush()
    return True


def delete_message(
    db: Session,
    *,
    user_id: int,
    conversation_id: str,
    message_id: str,
) -> bool:
    conversation = conversations.get(
        db,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    if conversation is None:
        return False
    message = messages.get(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    if message is None:
        return False
    related = _run_messages(db, message)
    related_ids = [item.message_id for item in related]
    deleted_auto_titles = {
        _automatic_conversation_title(item.content)
        for item in related
        if item.role == "user"
    }
    for related_message in related:
        context_items.invalidate_summaries_for_message(
            db,
            message_id=related_message.message_id,
        )
        _invalidate_derived(
            db,
            source_type="message",
            source_id=related_message.message_id,
            hard=True,
        )
    db.query(ContextSnapshotItem).filter(
        ContextSnapshotItem.source_type == "message",
        ContextSnapshotItem.source_id.in_(related_ids),
    ).delete(synchronize_session=False)
    if message.run_id:
        db.query(ContextSnapshot).filter(
            ContextSnapshot.run_id == message.run_id
        ).delete(synchronize_session=False)
        db.query(ResearchArtifact).filter(
            ResearchArtifact.run_id == message.run_id
        ).delete(synchronize_session=False)
        db.query(ResearchHistory).filter(
            ResearchHistory.user_id == user_id,
            ResearchHistory.run_id == message.run_id,
        ).delete(synchronize_session=False)
    db.query(ContextLineage).filter(
        ContextLineage.source_type == "message",
        ContextLineage.source_id.in_(related_ids),
    ).delete(synchronize_session=False)
    for related_message in related:
        db.delete(related_message)
    if conversation.title in deleted_auto_titles:
        replacement = (
            db.query(ConversationMessage)
            .filter(
                ConversationMessage.conversation_id == conversation_id,
                ConversationMessage.message_id.notin_(related_ids),
                ConversationMessage.deleted_at.is_(None),
                ConversationMessage.role == "user",
            )
            .order_by(ConversationMessage.sequence.asc())
            .first()
        )
        conversation.title = (
            _automatic_conversation_title(replacement.content)
            if replacement is not None
            else "新研究"
        )
        conversation.version += 1
        conversation.updated_at = datetime.now(timezone.utc)
    db.flush()
    return True


def delete_conversation(
    db: Session,
    *,
    user_id: int,
    conversation_id: str,
) -> str | None:
    conversation = conversations.soft_delete(
        db,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    if conversation is None:
        return None
    job = DeletionJob(
        deletion_job_id=uuid.uuid4().hex,
        user_id=user_id,
        target_type="conversation",
        target_id=conversation_id,
        status="running",
    )
    db.add(job)
    db.flush()
    message_ids = [
        row[0]
        for row in db.query(ConversationMessage.message_id)
        .filter(ConversationMessage.conversation_id == conversation_id)
        .all()
    ]
    for message_id in message_ids:
        _invalidate_derived(
            db,
            source_type="message",
            source_id=message_id,
            hard=True,
        )
    db.query(ContextSnapshot).filter(
        ContextSnapshot.conversation_id == conversation_id
    ).delete(synchronize_session=False)
    db.query(ConversationSummary).filter(
        ConversationSummary.conversation_id == conversation_id
    ).delete(synchronize_session=False)
    db.query(ResearchArtifact).filter(
        ResearchArtifact.conversation_id == conversation_id
    ).delete(synchronize_session=False)
    db.query(ResearchHistory).filter(
        ResearchHistory.user_id == user_id,
        ResearchHistory.conversation_id == conversation_id,
    ).delete(synchronize_session=False)
    db.query(ConversationMessage).filter(
        ConversationMessage.conversation_id == conversation_id
    ).delete(synchronize_session=False)
    db.query(ContextLineage).filter(
        ContextLineage.source_id.in_(message_ids)
    ).delete(synchronize_session=False)
    db.delete(conversation)
    job.status = "completed"
    job.completed_at = datetime.now(timezone.utc)
    db.flush()
    return job.deletion_job_id


def _run_messages(
    db: Session,
    message: ConversationMessage,
) -> list[ConversationMessage]:
    if not message.run_id:
        return [message]
    return (
        db.query(ConversationMessage)
        .filter(
            ConversationMessage.conversation_id == message.conversation_id,
            ConversationMessage.run_id == message.run_id,
        )
        .all()
    )


def _automatic_conversation_title(content: str) -> str:
    return content.strip().replace("\n", " ")[:80] or "新研究"


def get_deletion_job(
    db: Session,
    *,
    user_id: int,
    deletion_job_id: str,
) -> dict | None:
    row = (
        db.query(DeletionJob)
        .filter(
            DeletionJob.user_id == user_id,
            DeletionJob.deletion_job_id == deletion_job_id,
        )
        .first()
    )
    if row is None:
        return None
    return {
        "deletion_job_id": row.deletion_job_id,
        "target_type": row.target_type,
        "target_id": row.target_id,
        "status": row.status,
        "error": row.error,
        "created_at": row.created_at,
        "completed_at": row.completed_at,
    }


def _invalidate_derived(
    db: Session,
    *,
    source_type: str,
    source_id: str,
    hard: bool,
) -> None:
    edges = (
        db.query(ContextLineage)
        .filter(
            ContextLineage.source_type == source_type,
            ContextLineage.source_id == source_id,
        )
        .all()
    )
    artifact_ids = [
        edge.derived_id for edge in edges if edge.derived_type == "artifact"
    ]
    memory_ids = [
        edge.derived_id for edge in edges if edge.derived_type == "memory"
    ]
    summary_ids = [
        edge.derived_id for edge in edges if edge.derived_type == "summary"
    ]
    if artifact_ids:
        values = {"enabled": False}
        if hard:
            values["deleted_at"] = datetime.now(timezone.utc)
        db.query(ResearchArtifact).filter(
            ResearchArtifact.artifact_id.in_(artifact_ids)
        ).update(values, synchronize_session=False)
    if memory_ids:
        values = {"status": "forgotten"}
        if hard:
            values["deleted_at"] = datetime.now(timezone.utc)
        db.query(UserMemory).filter(UserMemory.memory_id.in_(memory_ids)).update(
            values,
            synchronize_session=False,
        )
    if summary_ids:
        db.query(ConversationSummary).filter(
            ConversationSummary.summary_id.in_(summary_ids)
        ).update({"status": "invalidated"}, synchronize_session=False)
