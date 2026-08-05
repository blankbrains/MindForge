"""Conversation, observable context, deletion, and long-term-memory APIs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from mindforge.api.schemas import (
    ContextItemUpdateRequest,
    ContextPreviewRequest,
    ConversationCreateRequest,
    ConversationDetail,
    ConversationItem,
    ConversationUpdateRequest,
    DeletionJobResponse,
    MemoryCreateRequest,
    MemoryUpdateRequest,
    MessageUpdateRequest,
)
from mindforge.context import deletion
from mindforge.db import (
    ContextSnapshot,
    ResearchArtifact,
    SessionLocal,
    get_default_user_id,
)
from mindforge.repositories import (
    context_snapshots,
    conversations,
    messages,
    user_memories,
)
from mindforge.services.context_service import ContextService, ContextServiceError
from mindforge.config import get_settings

router = APIRouter()


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=404, detail=detail)


@router.post(
    "/conversations",
    response_model=ConversationItem,
    status_code=201,
)
def create_conversation(body: ConversationCreateRequest):
    with SessionLocal() as db:
        try:
            item = conversations.create(
                db,
                user_id=get_default_user_id(db),
                title=body.title,
                context_mode=(
                    body.context_mode or get_settings().context.default_mode
                ),
            )
            db.commit()
            return item
        except Exception:
            db.rollback()
            raise


@router.get("/conversations", response_model=list[ConversationItem])
def list_conversations():
    with SessionLocal() as db:
        return conversations.list_active(
            db,
            user_id=get_default_user_id(db),
        )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
)
def get_conversation(conversation_id: str):
    with SessionLocal() as db:
        conversation = conversations.get(
            db,
            user_id=get_default_user_id(db),
            conversation_id=conversation_id,
        )
        if conversation is None:
            raise _not_found("Conversation not found.")
        return {
            **conversations.serialize_conversation(conversation),
            "messages": messages.list_visible(
                db,
                conversation_id=conversation_id,
            ),
        }


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationItem,
)
def update_conversation(
    conversation_id: str,
    body: ConversationUpdateRequest,
):
    with SessionLocal() as db:
        try:
            item = conversations.update(
                db,
                user_id=get_default_user_id(db),
                conversation_id=conversation_id,
                expected_version=body.version,
                title=body.title,
                status=body.status,
                context_mode=body.context_mode,
            )
            if item is None:
                raise _not_found("Conversation not found.")
            db.commit()
            return item
        except ValueError as exc:
            db.rollback()
            if str(exc) == "conversation_version_conflict":
                raise HTTPException(
                    status_code=409,
                    detail="Conversation was updated by another request.",
                ) from exc
            raise
        except Exception:
            db.rollback()
            raise


@router.delete(
    "/conversations/{conversation_id}",
    response_model=DeletionJobResponse,
    status_code=202,
)
def delete_conversation(conversation_id: str):
    with SessionLocal() as db:
        try:
            user_id = get_default_user_id(db)
            job_id = deletion.delete_conversation(
                db,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if job_id is None:
                raise _not_found("Conversation not found.")
            db.commit()
            job = deletion.get_deletion_job(
                db,
                user_id=user_id,
                deletion_job_id=job_id,
            )
            if job is None:
                raise RuntimeError("Deletion job was not persisted.")
            return job
        except Exception:
            db.rollback()
            raise


@router.get("/conversations/{conversation_id}/messages")
def list_conversation_messages(conversation_id: str):
    with SessionLocal() as db:
        conversation = conversations.get(
            db,
            user_id=get_default_user_id(db),
            conversation_id=conversation_id,
        )
        if conversation is None:
            raise _not_found("Conversation not found.")
        return messages.list_visible(db, conversation_id=conversation_id)


@router.patch("/conversations/{conversation_id}/messages/{message_id}")
def update_conversation_message(
    conversation_id: str,
    message_id: str,
    body: MessageUpdateRequest,
):
    with SessionLocal() as db:
        try:
            user_id = get_default_user_id(db)
            conversation = conversations.get(
                db,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if conversation is None:
                raise _not_found("Conversation not found.")
            if body.content is not None:
                invalidated = deletion.invalidate_message_derivatives(
                    db,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                )
                if not invalidated:
                    raise _not_found("Message not found.")
            item = messages.update(
                db,
                conversation_id=conversation_id,
                message_id=message_id,
                content=body.content,
                include_in_context=body.include_in_context,
                pinned=body.pinned,
            )
            if item is None:
                raise _not_found("Message not found.")
            db.commit()
            return item
        except Exception:
            db.rollback()
            raise


@router.post("/conversations/{conversation_id}/messages/{message_id}/forget")
def forget_conversation_message(conversation_id: str, message_id: str):
    with SessionLocal() as db:
        try:
            forgotten = deletion.forget_message(
                db,
                user_id=get_default_user_id(db),
                conversation_id=conversation_id,
                message_id=message_id,
            )
            if not forgotten:
                raise _not_found("Message not found.")
            db.commit()
            return {"message_id": message_id, "forgotten": True}
        except Exception:
            db.rollback()
            raise


@router.delete(
    "/conversations/{conversation_id}/messages/{message_id}",
    status_code=204,
)
def delete_conversation_message(conversation_id: str, message_id: str):
    with SessionLocal() as db:
        try:
            removed = deletion.delete_message(
                db,
                user_id=get_default_user_id(db),
                conversation_id=conversation_id,
                message_id=message_id,
            )
            if not removed:
                raise _not_found("Message not found.")
            db.commit()
            return None
        except Exception:
            db.rollback()
            raise


@router.post("/conversations/{conversation_id}/context-preview")
def preview_context(conversation_id: str, body: ContextPreviewRequest):
    try:
        return ContextService().preview(
            conversation_id=conversation_id,
            task=body.task,
            context_mode=body.context_mode,
            selected_context_ids=body.selected_context_ids,
            excluded_context_ids=body.excluded_context_ids,
            independent=body.independent,
        )
    except ContextServiceError as exc:
        raise _not_found(str(exc)) from exc


@router.get("/research-runs/{run_id}/context")
def get_research_context(run_id: str):
    with SessionLocal() as db:
        user_id = get_default_user_id(db)
        snapshot = (
            db.query(ContextSnapshot)
            .filter(ContextSnapshot.run_id == run_id)
            .first()
        )
        if snapshot is None:
            raise _not_found("Context snapshot not found.")
        if snapshot.conversation_id:
            conversation = conversations.get(
                db,
                user_id=user_id,
                conversation_id=snapshot.conversation_id,
            )
            if conversation is None:
                raise _not_found("Context snapshot not found.")
        return context_snapshots.get_by_run(db, run_id=run_id)


@router.patch("/context-items/{source_type}/{source_id}")
def update_context_item(
    source_type: str,
    source_id: str,
    body: ContextItemUpdateRequest,
    conversation_id: str | None = Query(None),
):
    with SessionLocal() as db:
        try:
            user_id = get_default_user_id(db)
            if source_type == "message":
                if not conversation_id:
                    raise HTTPException(
                        status_code=400,
                        detail="conversation_id is required for message context.",
                    )
                conversation = conversations.get(
                    db,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                if conversation is None:
                    raise _not_found("Context item not found.")
                item = messages.update(
                    db,
                    conversation_id=conversation_id,
                    message_id=source_id,
                    include_in_context=body.include_in_context,
                    pinned=body.pinned,
                )
                if item is None:
                    raise _not_found("Context item not found.")
            elif source_type == "memory":
                item = user_memories.update(
                    db,
                    user_id=user_id,
                    memory_id=source_id,
                    status=(
                        "active"
                        if body.enabled is True
                        else ("superseded" if body.enabled is False else None)
                    ),
                    user_confirmed=body.pinned,
                )
                if item is None:
                    raise _not_found("Context item not found.")
            elif source_type == "artifact":
                row = (
                    db.query(ResearchArtifact)
                    .filter(
                        ResearchArtifact.user_id == user_id,
                        ResearchArtifact.artifact_id == source_id,
                        ResearchArtifact.deleted_at.is_(None),
                    )
                    .first()
                )
                if row is None:
                    raise _not_found("Context item not found.")
                if body.enabled is not None:
                    row.enabled = body.enabled
                item = {"artifact_id": row.artifact_id, "enabled": row.enabled}
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported context source type.",
                )
            db.commit()
            return item
        except Exception:
            db.rollback()
            raise


@router.get("/memories")
def list_memories(
    search: str | None = Query(None, max_length=500),
    category: str | None = Query(None, max_length=24),
):
    with SessionLocal() as db:
        return user_memories.list_active(
            db,
            user_id=get_default_user_id(db),
            query=search,
            category=category,
        )


@router.post("/memories", status_code=201)
def create_memory(body: MemoryCreateRequest):
    with SessionLocal() as db:
        try:
            item = user_memories.create(
                db,
                user_id=get_default_user_id(db),
                category=body.category,
                content=body.content,
                confidence=body.confidence,
                user_confirmed=body.user_confirmed,
                expires_at=body.expires_at,
                metadata={"capture_method": "user_created"},
            )
            db.commit()
            return item
        except Exception:
            db.rollback()
            raise


@router.patch("/memories/{memory_id}")
def update_memory(memory_id: str, body: MemoryUpdateRequest):
    with SessionLocal() as db:
        try:
            item = user_memories.update(
                db,
                user_id=get_default_user_id(db),
                memory_id=memory_id,
                category=body.category,
                content=body.content,
                status=body.status,
                user_confirmed=body.user_confirmed,
            )
            if item is None:
                raise _not_found("Memory not found.")
            db.commit()
            return item
        except Exception:
            db.rollback()
            raise


@router.delete("/memories/{memory_id}", status_code=204)
def delete_memory(memory_id: str):
    with SessionLocal() as db:
        try:
            removed = user_memories.soft_delete(
                db,
                user_id=get_default_user_id(db),
                memory_id=memory_id,
            )
            if not removed:
                raise _not_found("Memory not found.")
            db.commit()
            return None
        except Exception:
            db.rollback()
            raise


@router.get(
    "/deletion-jobs/{deletion_job_id}",
    response_model=DeletionJobResponse,
)
def get_deletion_job(deletion_job_id: str):
    with SessionLocal() as db:
        item = deletion.get_deletion_job(
            db,
            user_id=get_default_user_id(db),
            deletion_job_id=deletion_job_id,
        )
        if item is None:
            raise _not_found("Deletion job not found.")
        return item
