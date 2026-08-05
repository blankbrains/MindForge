"""Research artifact persistence and bounded recall."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from mindforge.context.ranker import lexical_relevance
from mindforge.db import ResearchArtifact


def serialize_artifact(row: ResearchArtifact) -> dict[str, Any]:
    return {
        "artifact_id": row.artifact_id,
        "run_id": row.run_id,
        "conversation_id": row.conversation_id,
        "artifact_type": row.artifact_type,
        "title": row.title,
        "content": row.content,
        "source_ids": list(row.source_ids or []),
        "quality_score": row.quality_score,
        "grounding_status": row.grounding_status,
        "freshness_class": row.freshness_class,
        "valid_from": row.valid_from,
        "expires_at": row.expires_at,
        "enabled": row.enabled,
        "metadata": dict(row.metadata_json or {}),
        "created_at": row.created_at,
    }


def create(
    db: Session,
    *,
    user_id: int,
    run_id: str,
    conversation_id: str | None,
    artifact_type: str,
    title: str,
    content: str,
    source_ids: list[str],
    quality_score: float | None,
    grounding_status: str,
    freshness_class: str,
    expires_at: datetime | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = ResearchArtifact(
        artifact_id=uuid.uuid4().hex,
        user_id=user_id,
        run_id=run_id,
        conversation_id=conversation_id,
        artifact_type=artifact_type,
        title=title.strip()[:300] or artifact_type,
        content=content.strip()[:100_000],
        source_ids=source_ids[:100],
        quality_score=quality_score,
        grounding_status=grounding_status,
        freshness_class=freshness_class,
        valid_from=datetime.now(timezone.utc),
        expires_at=expires_at,
        metadata_json=metadata or {},
    )
    db.add(row)
    db.flush()
    return serialize_artifact(row)


def search(
    db: Session,
    *,
    user_id: int,
    conversation_id: str,
    query: str,
    top_k: int,
    allow_cross_conversation: bool,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    filters = [
        ResearchArtifact.user_id == user_id,
        ResearchArtifact.enabled.is_(True),
        ResearchArtifact.deleted_at.is_(None),
        ResearchArtifact.grounding_status != "model_only",
    ]
    if not allow_cross_conversation:
        filters.append(ResearchArtifact.conversation_id == conversation_id)
    rows = (
        db.query(ResearchArtifact)
        .filter(*filters)
        .order_by(ResearchArtifact.created_at.desc())
        .limit(250)
        .all()
    )
    eligible = [
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
    eligible.sort(
        key=lambda row: lexical_relevance(query, f"{row.title}\n{row.content}"),
        reverse=True,
    )
    return [serialize_artifact(row) for row in eligible[:top_k]]


def get_selected(
    db: Session,
    *,
    user_id: int,
    conversation_id: str,
    artifact_ids: list[str],
    allow_cross_conversation: bool,
) -> list[dict[str, Any]]:
    if not artifact_ids:
        return []
    now = datetime.now(timezone.utc)
    filters = [
        ResearchArtifact.user_id == user_id,
        ResearchArtifact.artifact_id.in_(artifact_ids),
        ResearchArtifact.enabled.is_(True),
        ResearchArtifact.deleted_at.is_(None),
        ResearchArtifact.grounding_status != "model_only",
    ]
    if not allow_cross_conversation:
        filters.append(ResearchArtifact.conversation_id == conversation_id)
    rows = db.query(ResearchArtifact).filter(*filters).all()
    return [
        serialize_artifact(row)
        for row in rows
        if row.expires_at is None
        or (
            row.expires_at
            if row.expires_at.tzinfo is not None
            else row.expires_at.replace(tzinfo=timezone.utc)
        )
        > now
    ]


def disable_for_message(
    db: Session,
    *,
    artifact_ids: list[str],
    hard_delete: bool,
) -> int:
    if not artifact_ids:
        return 0
    values: dict[str, Any] = {"enabled": False}
    if hard_delete:
        values["deleted_at"] = datetime.now(timezone.utc)
    return db.query(ResearchArtifact).filter(
        ResearchArtifact.artifact_id.in_(artifact_ids)
    ).update(values, synchronize_session=False)
