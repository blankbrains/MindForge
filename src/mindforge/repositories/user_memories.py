"""User-managed long-term memory persistence."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from mindforge.context.ranker import lexical_relevance
from mindforge.db import UserMemory


def serialize_memory(row: UserMemory) -> dict[str, Any]:
    return {
        "memory_id": row.memory_id,
        "category": row.category,
        "content": row.content,
        "source_message_ids": list(row.source_message_ids or []),
        "confidence": row.confidence,
        "status": row.status,
        "user_confirmed": row.user_confirmed,
        "valid_from": row.valid_from,
        "expires_at": row.expires_at,
        "metadata": dict(row.metadata_json or {}),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def create(
    db: Session,
    *,
    user_id: int,
    category: str,
    content: str,
    source_message_ids: list[str] | None = None,
    confidence: float = 1.0,
    user_confirmed: bool = True,
    expires_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = content.strip()
    existing = (
        db.query(UserMemory)
        .filter(
            UserMemory.user_id == user_id,
            UserMemory.category == category,
            UserMemory.content == normalized,
            UserMemory.status == "active",
            UserMemory.deleted_at.is_(None),
        )
        .first()
    )
    if existing is not None:
        return serialize_memory(existing)
    now = datetime.now(timezone.utc)
    row = UserMemory(
        memory_id=uuid.uuid4().hex,
        user_id=user_id,
        category=category,
        content=normalized[:20_000],
        source_message_ids=(source_message_ids or [])[:100],
        confidence=max(0.0, min(1.0, confidence)),
        status="active",
        user_confirmed=user_confirmed,
        valid_from=now,
        expires_at=expires_at,
        metadata_json=metadata or {},
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    db.flush()
    return serialize_memory(row)


def list_active(
    db: Session,
    *,
    user_id: int,
    query: str | None = None,
    category: str | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    filters = [
        UserMemory.user_id == user_id,
        UserMemory.status == "active",
        UserMemory.deleted_at.is_(None),
    ]
    if category:
        filters.append(UserMemory.category == category)
    rows = (
        db.query(UserMemory)
        .filter(*filters)
        .order_by(UserMemory.updated_at.desc())
        .limit(250)
        .all()
    )
    rows = [
        row
        for row in rows
        if row.expires_at is None
        or (
            row.expires_at
            if row.expires_at.tzinfo is not None
            else row.expires_at.replace(tzinfo=timezone.utc)
        )
        > now
    ]
    if query:
        rows.sort(
            key=lambda row: lexical_relevance(query, row.content),
            reverse=True,
        )
    if top_k is not None:
        rows = rows[:top_k]
    return [serialize_memory(row) for row in rows]


def get_selected(
    db: Session,
    *,
    user_id: int,
    memory_ids: list[str],
) -> list[dict[str, Any]]:
    if not memory_ids:
        return []
    now = datetime.now(timezone.utc)
    rows = (
        db.query(UserMemory)
        .filter(
            UserMemory.user_id == user_id,
            UserMemory.memory_id.in_(memory_ids),
            UserMemory.status == "active",
            UserMemory.deleted_at.is_(None),
        )
        .all()
    )
    return [
        serialize_memory(row)
        for row in rows
        if row.expires_at is None
        or (
            row.expires_at
            if row.expires_at.tzinfo is not None
            else row.expires_at.replace(tzinfo=timezone.utc)
        )
        > now
    ]


def get(db: Session, *, user_id: int, memory_id: str) -> UserMemory | None:
    return (
        db.query(UserMemory)
        .filter(
            UserMemory.user_id == user_id,
            UserMemory.memory_id == memory_id,
            UserMemory.deleted_at.is_(None),
        )
        .first()
    )


def update(
    db: Session,
    *,
    user_id: int,
    memory_id: str,
    content: str | None = None,
    category: str | None = None,
    status: str | None = None,
    user_confirmed: bool | None = None,
) -> dict[str, Any] | None:
    row = get(db, user_id=user_id, memory_id=memory_id)
    if row is None:
        return None
    if content is not None:
        row.content = content.strip()[:20_000]
    if category is not None:
        row.category = category
    if status is not None:
        row.status = status
    if user_confirmed is not None:
        row.user_confirmed = user_confirmed
    row.updated_at = datetime.now(timezone.utc)
    db.flush()
    return serialize_memory(row)


def soft_delete(db: Session, *, user_id: int, memory_id: str) -> bool:
    row = get(db, user_id=user_id, memory_id=memory_id)
    if row is None:
        return False
    row.status = "forgotten"
    row.deleted_at = datetime.now(timezone.utc)
    row.updated_at = row.deleted_at
    db.flush()
    return True
