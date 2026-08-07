"""Application service for conversations, context snapshots, and memories."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from mindforge.config import get_settings
from mindforge.context.artifacts import extract_artifacts
from mindforge.context.builder import ContextBuilder
from mindforge.context.compression import (
    ConversationSummaryCompressor,
)
from mindforge.context.models import ContextBundle, ResearchRequestContext
from mindforge.context.summaries import build_structured_summary
from mindforge.db import (
    ContextLineage,
    ConversationMessage,
    SessionLocal,
    get_default_user_id,
)
from mindforge.interaction import is_conversational_task
from mindforge.repositories import (
    context_items,
    context_snapshots,
    conversations,
    messages,
    research_artifacts,
    user_memories,
)

_PREFERENCE_PATTERNS = (
    re.compile(r"(?:我偏好|我喜欢|以后请|后续请|回答时请|输出时请)(.{2,300})"),
    re.compile(r"(?:请始终|请默认|不要再)(.{2,300})"),
)
_SENSITIVE_PATTERN = re.compile(
    r"(api[_ -]?key|token|密码|口令|secret|身份证|银行卡|私钥)",
    re.IGNORECASE,
)


class ContextServiceError(RuntimeError):
    """Expected context-service failure with an API-safe code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class PreparedResearchRun:
    request: ResearchRequestContext
    bundle: ContextBundle
    user_id: int
    query_message: dict[str, Any]
    finalized: bool = False


class ContextService:
    """Coordinates repositories in explicit transactions."""

    def prepare_query(
        self,
        *,
        conversation_id: str,
        task: str,
        run_id: str,
        context_mode: str | None,
        selected_context_ids: list[str],
        excluded_context_ids: list[str],
        independent: bool,
    ) -> PreparedResearchRun:
        with SessionLocal() as db:
            try:
                user_id = get_default_user_id(db)
                conversation = conversations.get(
                    db,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    for_update=True,
                )
                if conversation is None:
                    raise ContextServiceError(
                        "conversation_not_found",
                        "Conversation not found.",
                    )
                mode = context_mode or conversation.context_mode
                query_message = messages.add(
                    db,
                    conversation=conversation,
                    role="user",
                    content=task,
                    run_id=run_id,
                    include_in_context=not is_conversational_task(task),
                )
                recent = messages.list_context_eligible(
                    db,
                    conversation_id=conversation_id,
                    exclude_message_id=query_message["message_id"],
                    limit=get_settings().context.recent_message_limit,
                )
                selected = self._selected_source_ids(selected_context_ids)
                recent = self._merge_by_id(
                    recent,
                    messages.list_selected_context(
                        db,
                        conversation_id=conversation_id,
                        message_ids=selected["message"],
                        exclude_message_id=query_message["message_id"],
                    ),
                    "message_id",
                )
                artifact_rows = research_artifacts.search(
                    db,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    query=task,
                    top_k=get_settings().context.artifact_top_k,
                    allow_cross_conversation=(
                        get_settings().context.cross_conversation_enabled
                    ),
                )
                artifact_rows = self._merge_by_id(
                    artifact_rows,
                    research_artifacts.get_selected(
                        db,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        artifact_ids=selected["artifact"],
                        allow_cross_conversation=(
                            get_settings().context.cross_conversation_enabled
                        ),
                    ),
                    "artifact_id",
                )
                memory_rows = user_memories.list_active(
                    db,
                    user_id=user_id,
                    query=task,
                    top_k=get_settings().context.memory_top_k,
                )
                memory_rows = self._merge_by_id(
                    memory_rows,
                    user_memories.get_selected(
                        db,
                        user_id=user_id,
                        memory_ids=selected["memory"],
                    ),
                    "memory_id",
                )
                request = ResearchRequestContext(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    message_id=query_message["message_id"],
                    context_mode=mode,
                    selected_context_ids=tuple(selected_context_ids),
                    excluded_context_ids=tuple(excluded_context_ids),
                    independent=independent,
                )
                bundle = ContextBuilder(
                    recent_messages=recent,
                    summary=context_items.get_active_summary(
                        db,
                        conversation_id=conversation_id,
                    ),
                    artifacts=artifact_rows,
                    memories=memory_rows,
                ).build(task, request)
                context_snapshots.delete_expired(
                    db,
                    retention_days=get_settings().context.snapshot_retention_days,
                )
                context_snapshots.create(
                    db,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    query_message_id=query_message["message_id"],
                    bundle=bundle,
                )
                if (
                    conversation.title == "新研究"
                    or conversation.title.startswith("新研究 ")
                ):
                    conversation.title = task.strip().replace("\n", " ")[:80]
                db.commit()
                return PreparedResearchRun(
                    request=request,
                    bundle=bundle,
                    user_id=user_id,
                    query_message=query_message,
                )
            except Exception:
                db.rollback()
                raise

    def preview(
        self,
        *,
        conversation_id: str,
        task: str,
        context_mode: str | None,
        selected_context_ids: list[str],
        excluded_context_ids: list[str],
        independent: bool,
    ) -> dict[str, Any]:
        with SessionLocal() as db:
            user_id = get_default_user_id(db)
            conversation = conversations.get(
                db,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if conversation is None:
                raise ContextServiceError(
                    "conversation_not_found",
                    "Conversation not found.",
                )
            mode = context_mode or conversation.context_mode
            recent = messages.list_context_eligible(
                db,
                conversation_id=conversation_id,
                exclude_message_id=None,
                limit=get_settings().context.recent_message_limit,
            )
            selected = self._selected_source_ids(selected_context_ids)
            recent = self._merge_by_id(
                recent,
                messages.list_selected_context(
                    db,
                    conversation_id=conversation_id,
                    message_ids=selected["message"],
                    exclude_message_id=None,
                ),
                "message_id",
            )
            request = ResearchRequestContext(
                run_id=f"preview-{uuid.uuid4().hex}",
                conversation_id=conversation_id,
                context_mode=mode,
                selected_context_ids=tuple(selected_context_ids),
                excluded_context_ids=tuple(excluded_context_ids),
                independent=independent,
            )
            bundle = ContextBuilder(
                recent_messages=recent,
                summary=context_items.get_active_summary(
                    db,
                    conversation_id=conversation_id,
                ),
                artifacts=self._merge_by_id(
                    research_artifacts.search(
                        db,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        query=task,
                        top_k=get_settings().context.artifact_top_k,
                        allow_cross_conversation=(
                            get_settings().context.cross_conversation_enabled
                        ),
                    ),
                    research_artifacts.get_selected(
                        db,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        artifact_ids=selected["artifact"],
                        allow_cross_conversation=(
                            get_settings().context.cross_conversation_enabled
                        ),
                    ),
                    "artifact_id",
                ),
                memories=self._merge_by_id(
                    user_memories.list_active(
                        db,
                        user_id=user_id,
                        query=task,
                        top_k=get_settings().context.memory_top_k,
                    ),
                    user_memories.get_selected(
                        db,
                        user_id=user_id,
                        memory_ids=selected["memory"],
                    ),
                    "memory_id",
                ),
            ).build(task, request)
            return bundle.to_public_dict()

    def complete_run(
        self,
        prepared: PreparedResearchRun,
        result: Any,
    ) -> None:
        """Persist only visible output and safe structured derivatives."""
        success = bool(
            result.success if hasattr(result, "success") else result.get("success")
        )
        output = str(
            result.output
            if hasattr(result, "output")
            else result.get("output") or ""
        )
        data = result.data if hasattr(result, "data") else result.setdefault("data", {})
        metadata = (
            result.metadata
            if hasattr(result, "metadata")
            else result.setdefault("metadata", {})
        )
        data["context_snapshot_id"] = prepared.bundle.snapshot_id
        data["context_items_used"] = len(prepared.bundle.items)
        metadata["context_snapshot_id"] = prepared.bundle.snapshot_id
        metadata["context_token_usage"] = prepared.bundle.used_tokens
        metadata["context_fingerprint"] = prepared.bundle.fingerprint
        metadata["run_id"] = prepared.request.run_id
        metadata["conversation_id"] = prepared.request.conversation_id
        metadata["query_message_id"] = prepared.request.message_id
        metadata["reused_artifact_count"] = sum(
            item.source_type == "artifact" for item in prepared.bundle.items
        )
        route = str(metadata.get("route") or "research")
        conversational = route == "conversation"
        artifact_eligible = route not in {
            "conversation",
            "direct_answer",
        }

        with SessionLocal() as db:
            try:
                conversation = conversations.get(
                    db,
                    user_id=prepared.user_id,
                    conversation_id=prepared.request.conversation_id or "",
                    for_update=True,
                )
                if conversation is None:
                    raise ContextServiceError(
                        "conversation_deleted_during_run",
                        "Conversation was deleted before the research run completed.",
                    )
                assistant_message = messages.add(
                    db,
                    conversation=conversation,
                    role="assistant" if success else "system_notice",
                    content=output or "研究任务未生成可见结果。",
                    run_id=prepared.request.run_id,
                    include_in_context=success and not conversational,
                    metadata={
                        "success": success,
                        "context_snapshot_id": prepared.bundle.snapshot_id,
                        "route": metadata.get("route"),
                    },
                )
                artifact_count = (
                    0
                    if not artifact_eligible
                    else self._store_artifacts(
                        db,
                        prepared=prepared,
                        result=result,
                        assistant_message_id=assistant_message["message_id"],
                    )
                )
                metadata["research_artifact_count"] = artifact_count
                if not conversational:
                    self._capture_preference(db, prepared)
                self._refresh_summary(
                    db,
                    conversation_id=conversation.conversation_id,
                )
                db.commit()
                prepared.finalized = True
            except Exception:
                db.rollback()
                raise

    def abandon_run(
        self,
        prepared: PreparedResearchRun,
        *,
        status: str,
        reason: str,
    ) -> bool:
        """Persist one visible terminal notice without reusable derivatives."""
        if prepared.finalized:
            return True
        normalized_status = (
            status if status in {"cancelled", "failed"} else "failed"
        )
        with SessionLocal() as db:
            try:
                conversation = conversations.get(
                    db,
                    user_id=prepared.user_id,
                    conversation_id=prepared.request.conversation_id or "",
                    for_update=True,
                )
                if conversation is None:
                    prepared.finalized = True
                    return False
                existing = (
                    db.query(ConversationMessage)
                    .filter(
                        ConversationMessage.conversation_id
                        == conversation.conversation_id,
                        ConversationMessage.run_id == prepared.request.run_id,
                        ConversationMessage.role != "user",
                    )
                    .first()
                )
                if existing is None:
                    messages.add(
                        db,
                        conversation=conversation,
                        role="system_notice",
                        content=(
                            "研究任务已取消。"
                            if normalized_status == "cancelled"
                            else "研究任务未完成。"
                        ),
                        run_id=prepared.request.run_id,
                        include_in_context=False,
                        metadata={
                            "run_status": normalized_status,
                            "reason": reason.strip()[:500],
                            "context_snapshot_id": prepared.bundle.snapshot_id,
                        },
                    )
                db.commit()
                prepared.finalized = True
                return True
            except Exception:
                db.rollback()
                raise

    async def refine_summary_with_model(
        self,
        *,
        conversation_id: str,
        compressor: ConversationSummaryCompressor | None = None,
    ) -> dict[str, Any]:
        """Replace a deterministic summary only with source-bound model output."""
        with SessionLocal() as db:
            user_id = get_default_user_id(db)
            conversation = conversations.get(
                db,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if conversation is None:
                return {"status": "conversation_not_found"}
            active = context_items.get_active_summary(
                db,
                conversation_id=conversation_id,
            )
            if active is None:
                return {"status": "summary_not_available"}
            compression = dict(
                (active.get("summary") or {}).get("_compression") or {}
            )
            if compression.get("method") == "model":
                return {
                    "status": "already_compressed",
                    "summary_id": active["summary_id"],
                }
            source_ids = list(active.get("source_message_ids") or [])
            rows = (
                db.query(ConversationMessage)
                .filter(
                    ConversationMessage.conversation_id == conversation_id,
                    ConversationMessage.message_id.in_(source_ids),
                    ConversationMessage.deleted_at.is_(None),
                    ConversationMessage.include_in_context.is_(True),
                    ConversationMessage.role.in_(("user", "assistant")),
                )
                .order_by(ConversationMessage.sequence.asc())
                .all()
            )
            serialized = [messages.serialize_message(row) for row in rows]
            expected_summary_id = str(active["summary_id"])
            fallback_summary = dict(active.get("summary") or {})

        outcome = await (
            compressor or ConversationSummaryCompressor()
        ).compress(
            serialized,
            fallback_summary=fallback_summary,
        )

        with SessionLocal() as db:
            current = context_items.get_active_summary(
                db,
                conversation_id=conversation_id,
            )
            if (
                current is None
                or current["summary_id"] != expected_summary_id
            ):
                return {
                    "status": "stale",
                    "attempt_status": outcome.status,
                }
            if outcome.status == "compressed":
                replaced = context_items.replace_summary(
                    db,
                    conversation_id=conversation_id,
                    from_sequence=current["from_sequence"],
                    to_sequence=current["to_sequence"],
                    summary=outcome.summary,
                    source_message_ids=source_ids,
                )
                db.commit()
                return {
                    "status": "compressed",
                    "summary_id": replaced.get("summary_id"),
                    "model": outcome.model,
                }

            compression = dict(
                (current.get("summary") or {}).get("_compression") or {}
            )
            compression.update(
                {
                    "method": "deterministic",
                    "model_attempt_status": outcome.status,
                    "model": outcome.model,
                    "error": outcome.error,
                }
            )
            updated = context_items.update_summary_compression_metadata(
                db,
                summary_id=expected_summary_id,
                compression=compression,
            )
            if updated:
                db.commit()
            return {
                "status": outcome.status,
                "summary_id": expected_summary_id,
                "model": outcome.model,
                "error": outcome.error,
            }

    def _store_artifacts(
        self,
        db: Session,
        *,
        prepared: PreparedResearchRun,
        result: Any,
        assistant_message_id: str,
    ) -> int:
        created = 0
        for artifact in extract_artifacts(
            task=prepared.bundle.standalone_query,
            result=result,
        ):
            row = research_artifacts.create(
                db,
                user_id=prepared.user_id,
                run_id=prepared.request.run_id,
                conversation_id=prepared.request.conversation_id,
                **artifact,
            )
            for message_id in (
                prepared.query_message["message_id"],
                assistant_message_id,
            ):
                context_items.add_lineage(
                    db,
                    source_type="message",
                    source_id=message_id,
                    derived_type="artifact",
                    derived_id=row["artifact_id"],
                    relation="extracted_from",
                )
            created += 1
        return created

    def _capture_preference(
        self,
        db: Session,
        prepared: PreparedResearchRun,
    ) -> None:
        if not get_settings().memory.auto_capture_preferences:
            return
        content = str(prepared.query_message.get("content") or "")
        if _SENSITIVE_PATTERN.search(content):
            return
        for pattern in _PREFERENCE_PATTERNS:
            match = pattern.search(content)
            if match is None:
                continue
            preference = match.group(0).strip()[:500]
            memory = user_memories.create(
                db,
                user_id=prepared.user_id,
                category="preference",
                content=preference,
                source_message_ids=[prepared.query_message["message_id"]],
                confidence=0.85,
                user_confirmed=False,
                metadata={"capture_method": "explicit_preference_rule"},
            )
            context_items.add_lineage(
                db,
                source_type="message",
                source_id=prepared.query_message["message_id"],
                derived_type="memory",
                derived_id=memory["memory_id"],
                relation="extracted_from",
            )
            break

    @staticmethod
    def _refresh_summary(db: Session, *, conversation_id: str) -> None:
        threshold = get_settings().context.summary_message_threshold
        rows = (
            db.query(ConversationMessage)
            .filter(
                ConversationMessage.conversation_id == conversation_id,
                ConversationMessage.deleted_at.is_(None),
                ConversationMessage.include_in_context.is_(True),
                ConversationMessage.role.in_(("user", "assistant")),
            )
            .order_by(ConversationMessage.sequence.asc())
            .all()
        )
        if len(rows) < threshold:
            return
        settings = get_settings()
        keep_recent = settings.context.summary_keep_recent_messages
        source_rows = rows[: max(1, len(rows) - keep_recent)]
        active = context_items.get_active_summary(
            db,
            conversation_id=conversation_id,
        )
        if (
            active is not None
            and int(active["to_sequence"]) >= source_rows[-1].sequence
        ):
            return
        serialized = [messages.serialize_message(row) for row in source_rows]
        summary = build_structured_summary(
            serialized,
            max_tokens=settings.context.summary_max_tokens,
            chars_per_token=settings.memory.chars_per_token,
        )
        context_items.replace_summary(
            db,
            conversation_id=conversation_id,
            from_sequence=source_rows[0].sequence,
            to_sequence=source_rows[-1].sequence,
            summary=summary,
            source_message_ids=[row.message_id for row in source_rows],
        )

    @staticmethod
    def derived_ids(
        db: Session,
        *,
        source_type: str,
        source_id: str,
    ) -> list[tuple[str, str]]:
        return [
            (row.derived_type, row.derived_id)
            for row in db.query(ContextLineage)
            .filter(
                ContextLineage.source_type == source_type,
                ContextLineage.source_id == source_id,
            )
            .all()
        ]

    @staticmethod
    def _selected_source_ids(values: list[str]) -> dict[str, list[str]]:
        selected: dict[str, list[str]] = {
            "message": [],
            "artifact": [],
            "memory": [],
        }
        for value in values:
            source_type, separator, source_id = value.partition(":")
            if separator and source_type in selected and source_id:
                selected[source_type].append(source_id)
        return selected

    @staticmethod
    def _merge_by_id(
        primary: list[dict[str, Any]],
        additional: list[dict[str, Any]],
        key: str,
    ) -> list[dict[str, Any]]:
        merged = {str(item[key]): item for item in primary}
        for item in additional:
            merged[str(item[key])] = item
        return list(merged.values())
