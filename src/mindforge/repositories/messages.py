"""Conversation message persistence and context visibility rules."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from mindforge.db import Conversation, ConversationMessage


def serialize_message(message: ConversationMessage) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "conversation_id": message.conversation_id,
        "run_id": message.run_id,
        "role": message.role,
        "content": message.content,
        "sequence": message.sequence,
        "include_in_context": message.include_in_context,
        "pinned": message.pinned,
        "context_scope": message.context_scope,
        "metadata": dict(message.metadata_json or {}),
        "created_at": message.created_at,
        "edited_at": message.edited_at,
    }


def add(
    db: Session,
    *,
    conversation: Conversation,
    role: str,
    content: str,
    run_id: str | None = None,
    include_in_context: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    locked = (
        db.query(Conversation)
        .filter(Conversation.conversation_id == conversation.conversation_id)
        .with_for_update()
        .one()
    )
    message = ConversationMessage(
        message_id=uuid.uuid4().hex,
        conversation_id=locked.conversation_id,
        run_id=run_id,
        role=role,
        content=content.strip()[:200_000],
        sequence=locked.next_sequence,
        include_in_context=include_in_context,
        metadata_json=metadata or {},
    )
    locked.next_sequence += 1
    locked.version += 1
    locked.updated_at = datetime.now(timezone.utc)
    db.add(message)
    db.flush()
    return serialize_message(message)


def get(
    db: Session,
    *,
    conversation_id: str,
    message_id: str,
) -> ConversationMessage | None:
    return (
        db.query(ConversationMessage)
        .filter(
            ConversationMessage.message_id == message_id,
            ConversationMessage.conversation_id == conversation_id,
            ConversationMessage.deleted_at.is_(None),
        )
        .first()
    )


def list_visible(
    db: Session,
    *,
    conversation_id: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    query = (
        db.query(ConversationMessage)
        .filter(
            ConversationMessage.conversation_id == conversation_id,
            ConversationMessage.deleted_at.is_(None),
        )
        .order_by(ConversationMessage.sequence.desc())
    )
    if limit is not None:
        query = query.limit(limit)
    rows = list(reversed(query.all()))
    return [serialize_message(row) for row in rows]


def list_context_eligible(
    db: Session,
    *,
    conversation_id: str,
    exclude_message_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    query = db.query(ConversationMessage).filter(
        ConversationMessage.conversation_id == conversation_id,
        ConversationMessage.deleted_at.is_(None),
        ConversationMessage.include_in_context.is_(True),
        ConversationMessage.role.in_(("user", "assistant")),
    )
    if exclude_message_id:
        query = query.filter(ConversationMessage.message_id != exclude_message_id)
    rows = list(
        reversed(
            query.order_by(ConversationMessage.sequence.desc()).limit(limit).all()
        )
    )
    return [serialize_message(row) for row in rows]


def list_selected_context(
    db: Session,
    *,
    conversation_id: str,
    message_ids: list[str],
    exclude_message_id: str | None,
) -> list[dict[str, Any]]:
    if not message_ids:
        return []
    query = db.query(ConversationMessage).filter(
        ConversationMessage.conversation_id == conversation_id,
        ConversationMessage.message_id.in_(message_ids),
        ConversationMessage.deleted_at.is_(None),
        ConversationMessage.include_in_context.is_(True),
        ConversationMessage.role.in_(("user", "assistant")),
    )
    if exclude_message_id:
        query = query.filter(ConversationMessage.message_id != exclude_message_id)
    return [
        serialize_message(row)
        for row in query.order_by(ConversationMessage.sequence.asc()).all()
    ]


def update(
    db: Session,
    *,
    conversation_id: str,
    message_id: str,
    content: str | None = None,
    include_in_context: bool | None = None,
    pinned: bool | None = None,
) -> dict[str, Any] | None:
    message = get(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    if message is None:
        return None
    if content is not None:
        message.content = content.strip()[:200_000]
        message.edited_at = datetime.now(timezone.utc)
    if include_in_context is not None:
        message.include_in_context = include_in_context
    if pinned is not None:
        message.pinned = pinned
    db.flush()
    return serialize_message(message)


def soft_delete(
    db: Session,
    *,
    conversation_id: str,
    message_id: str,
) -> ConversationMessage | None:
    message = get(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    if message is None:
        return None
    message.include_in_context = False
    message.pinned = False
    message.deleted_at = datetime.now(timezone.utc)
    db.flush()
    return message
