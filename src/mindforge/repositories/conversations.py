"""Conversation persistence with user isolation and optimistic versioning."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from mindforge.db import Conversation


def serialize_conversation(conversation: Conversation) -> dict[str, Any]:
    return {
        "conversation_id": conversation.conversation_id,
        "title": conversation.title,
        "status": conversation.status,
        "context_mode": conversation.context_mode,
        "version": conversation.version,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


def create(
    db: Session,
    *,
    user_id: int,
    title: str,
    context_mode: str,
) -> dict[str, Any]:
    conversation = Conversation(
        conversation_id=uuid.uuid4().hex,
        user_id=user_id,
        title=title.strip()[:200] or "新研究",
        context_mode=context_mode,
        status="active",
    )
    db.add(conversation)
    db.flush()
    return serialize_conversation(conversation)


def get(
    db: Session,
    *,
    user_id: int,
    conversation_id: str,
    for_update: bool = False,
) -> Conversation | None:
    query = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id,
        Conversation.user_id == user_id,
        Conversation.deleted_at.is_(None),
    )
    if for_update:
        query = query.with_for_update()
    return query.first()


def list_active(db: Session, *, user_id: int) -> list[dict[str, Any]]:
    rows = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None),
        )
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
        .all()
    )
    return [serialize_conversation(row) for row in rows]


def update(
    db: Session,
    *,
    user_id: int,
    conversation_id: str,
    expected_version: int | None,
    title: str | None = None,
    status: str | None = None,
    context_mode: str | None = None,
) -> dict[str, Any] | None:
    conversation = get(
        db,
        user_id=user_id,
        conversation_id=conversation_id,
        for_update=True,
    )
    if conversation is None:
        return None
    if expected_version is not None and conversation.version != expected_version:
        raise ValueError("conversation_version_conflict")
    if title is not None:
        conversation.title = title.strip()[:200] or conversation.title
    if status is not None:
        conversation.status = status
    if context_mode is not None:
        conversation.context_mode = context_mode
    conversation.version += 1
    conversation.updated_at = datetime.now(timezone.utc)
    db.flush()
    return serialize_conversation(conversation)


def soft_delete(
    db: Session,
    *,
    user_id: int,
    conversation_id: str,
) -> Conversation | None:
    conversation = get(
        db,
        user_id=user_id,
        conversation_id=conversation_id,
        for_update=True,
    )
    if conversation is None:
        return None
    now = datetime.now(timezone.utc)
    conversation.status = "deleted"
    conversation.deleted_at = now
    conversation.updated_at = now
    conversation.version += 1
    db.flush()
    return conversation
