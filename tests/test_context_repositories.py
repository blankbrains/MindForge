from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from mindforge.context import deletion
from mindforge.context.models import ContextBundle, ContextCandidate
from mindforge.db import (
    Base,
    ContextSnapshot,
    Conversation,
    ConversationMessage,
    ResearchArtifact,
    ResearchHistory,
    User,
    UserMemory,
)
from mindforge.services import context_service as context_service_module
from mindforge.services.context_service import (
    ContextService,
    ContextServiceError,
    PreparedResearchRun,
)
from mindforge.repositories import (
    context_items,
    context_snapshots,
    conversations,
    messages,
    research_artifacts,
    user_memories,
)


def _session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return Session(engine)


def _seed_run(db: Session) -> tuple[int, str, str, str]:
    user = User(username="default", password_hash="hash")
    db.add(user)
    db.flush()
    conversation = conversations.create(
        db,
        user_id=user.id,
        title="Context test",
        context_mode="auto",
    )
    conversation_row = conversations.get(
        db,
        user_id=user.id,
        conversation_id=conversation["conversation_id"],
    )
    assert conversation_row is not None
    run_id = "research-delete-propagation"
    user_message = messages.add(
        db,
        conversation=conversation_row,
        role="user",
        content="Question",
        run_id=run_id,
    )
    assistant_message = messages.add(
        db,
        conversation=conversation_row,
        role="assistant",
        content="Answer",
        run_id=run_id,
    )
    bundle = ContextBundle(
        standalone_query="Question",
        requires_context=True,
        items=[
            ContextCandidate(
                source_type="message",
                source_id=user_message["message_id"],
                title="Question",
                content="Question",
                token_count=2,
            )
        ],
        excluded=[],
        budget_tokens=100,
        used_tokens=2,
    )
    context_snapshots.create(
        db,
        run_id=run_id,
        conversation_id=conversation["conversation_id"],
        query_message_id=user_message["message_id"],
        bundle=bundle,
    )
    artifact = research_artifacts.create(
        db,
        user_id=user.id,
        run_id=run_id,
        conversation_id=conversation["conversation_id"],
        artifact_type="subtask_finding",
        title="Finding",
        content="Grounded",
        source_ids=["source"],
        quality_score=8.0,
        grounding_status="grounded",
        freshness_class="stable",
        expires_at=None,
    )
    for source_message_id in (
        user_message["message_id"],
        assistant_message["message_id"],
    ):
        context_items.add_lineage(
            db,
            source_type="message",
            source_id=source_message_id,
            derived_type="artifact",
            derived_id=artifact["artifact_id"],
            relation="extracted_from",
        )
    db.add(
        ResearchHistory(
            user_id=user.id,
            task="Question",
            report="Answer",
            conversation_id=conversation["conversation_id"],
            run_id=run_id,
        )
    )
    db.commit()
    return (
        user.id,
        conversation["conversation_id"],
        user_message["message_id"],
        run_id,
    )


def test_forget_message_excludes_the_entire_run_and_disables_derivatives() -> None:
    with _session() as db:
        user_id, conversation_id, message_id, run_id = _seed_run(db)

        assert deletion.forget_message(
            db,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        db.commit()

        run_messages = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.run_id == run_id)
            .all()
        )
        assert len(run_messages) == 2
        assert all(not message.include_in_context for message in run_messages)
        assert all(not message.pinned for message in run_messages)
        artifact = (
            db.query(ResearchArtifact)
            .filter(ResearchArtifact.run_id == run_id)
            .one()
        )
        assert artifact.enabled is False
        assert db.query(ContextSnapshot).filter(
            ContextSnapshot.run_id == run_id
        ).count() == 1
        assert db.query(ResearchHistory).filter(
            ResearchHistory.run_id == run_id
        ).count() == 1


def test_delete_message_removes_the_entire_run_and_derived_records() -> None:
    with _session() as db:
        user_id, conversation_id, message_id, run_id = _seed_run(db)

        assert deletion.delete_message(
            db,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        db.commit()

        assert db.query(ConversationMessage).filter(
            ConversationMessage.run_id == run_id
        ).count() == 0
        assert db.query(ContextSnapshot).filter(
            ContextSnapshot.run_id == run_id
        ).count() == 0
        assert db.query(ResearchArtifact).filter(
            ResearchArtifact.run_id == run_id
        ).count() == 0
        assert db.query(ResearchHistory).filter(
            ResearchHistory.run_id == run_id
        ).count() == 0


def test_delete_conversation_physically_removes_the_conversation() -> None:
    with _session() as db:
        user_id, conversation_id, _message_id, _run_id = _seed_run(db)

        job_id = deletion.delete_conversation(
            db,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        assert job_id is not None
        db.commit()

        assert db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).count() == 0
        job = deletion.get_deletion_job(
            db,
            user_id=user_id,
            deletion_job_id=job_id,
        )
        assert job is not None
        assert job["status"] == "completed"


def test_snapshot_create_retry_restores_existing_snapshot_id() -> None:
    with _session() as db:
        user_id, conversation_id, message_id, run_id = _seed_run(db)
        del user_id
        existing = context_snapshots.get_by_run(db, run_id=run_id)
        assert existing is not None
        retry_bundle = ContextBundle(
            standalone_query="Question",
            requires_context=False,
            items=[],
            excluded=[],
            budget_tokens=100,
            used_tokens=0,
        )

        returned = context_snapshots.create(
            db,
            run_id=run_id,
            conversation_id=conversation_id,
            query_message_id=message_id,
            bundle=retry_bundle,
        )

        assert retry_bundle.snapshot_id == existing["snapshot_id"]
        assert returned["snapshot_id"] == existing["snapshot_id"]
        assert db.query(ContextSnapshot).filter(
            ContextSnapshot.run_id == run_id
        ).count() == 1


def test_complete_run_reports_deleted_conversation_instead_of_silent_success(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        user = User(username="complete-run", password_hash="hash")
        db.add(user)
        db.flush()
        conversation = conversations.create(
            db,
            user_id=user.id,
            title="Concurrent deletion",
            context_mode="auto",
        )
        row = conversations.get(
            db,
            user_id=user.id,
            conversation_id=conversation["conversation_id"],
        )
        assert row is not None
        query_message = messages.add(
            db,
            conversation=row,
            role="user",
            content="Question",
            run_id="research-concurrent-delete",
        )
        db.commit()
        prepared = PreparedResearchRun(
            request=context_service_module.ResearchRequestContext(
                run_id="research-concurrent-delete",
                conversation_id=conversation["conversation_id"],
                message_id=query_message["message_id"],
            ),
            bundle=ContextBundle(
                standalone_query="Question",
                requires_context=False,
                items=[],
                excluded=[],
                budget_tokens=100,
                used_tokens=0,
                snapshot_id="snapshot",
            ),
            user_id=user.id,
            query_message=query_message,
        )
        assert deletion.delete_conversation(
            db,
            user_id=user.id,
            conversation_id=conversation["conversation_id"],
        )
        db.commit()

    monkeypatch.setattr(context_service_module, "SessionLocal", factory)
    result = {
        "success": True,
        "output": "Answer",
        "data": {},
        "metadata": {},
    }

    with pytest.raises(ContextServiceError, match="deleted"):
        ContextService().complete_run(prepared, result)


def test_conversation_update_rejects_stale_version() -> None:
    with _session() as db:
        user = User(username="version-user", password_hash="hash")
        db.add(user)
        db.flush()
        created = conversations.create(
            db,
            user_id=user.id,
            title="Versioned",
            context_mode="auto",
        )
        original_version = created["version"]
        updated = conversations.update(
            db,
            user_id=user.id,
            conversation_id=created["conversation_id"],
            expected_version=original_version,
            title="Updated",
        )
        assert updated is not None

        with pytest.raises(ValueError, match="conversation_version_conflict"):
            conversations.update(
                db,
                user_id=user.id,
                conversation_id=created["conversation_id"],
                expected_version=original_version,
                title="Stale update",
            )


def test_artifact_and_memory_queries_enforce_user_isolation_and_expiry() -> None:
    with _session() as db:
        first = User(username="first-user", password_hash="hash")
        second = User(username="second-user", password_hash="hash")
        db.add_all([first, second])
        db.flush()
        conversation = conversations.create(
            db,
            user_id=first.id,
            title="First",
            context_mode="auto",
        )
        active_artifact = research_artifacts.create(
            db,
            user_id=first.id,
            run_id="run-active",
            conversation_id=conversation["conversation_id"],
            artifact_type="evidence",
            title="Active",
            content="shared keyword",
            source_ids=["source"],
            quality_score=8.0,
            grounding_status="grounded",
            freshness_class="stable",
            expires_at=None,
        )
        research_artifacts.create(
            db,
            user_id=first.id,
            run_id="run-expired",
            conversation_id=conversation["conversation_id"],
            artifact_type="evidence",
            title="Expired",
            content="shared keyword",
            source_ids=["source"],
            quality_score=8.0,
            grounding_status="grounded",
            freshness_class="volatile",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        active_memory = user_memories.create(
            db,
            user_id=first.id,
            category="preference",
            content="shared keyword",
        )
        user_memories.create(
            db,
            user_id=first.id,
            category="preference",
            content="expired shared keyword",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        db.commit()

        own_artifacts = research_artifacts.search(
            db,
            user_id=first.id,
            conversation_id=conversation["conversation_id"],
            query="shared keyword",
            top_k=10,
            allow_cross_conversation=True,
        )
        other_artifacts = research_artifacts.search(
            db,
            user_id=second.id,
            conversation_id=conversation["conversation_id"],
            query="shared keyword",
            top_k=10,
            allow_cross_conversation=True,
        )
        own_memories = user_memories.list_active(
            db,
            user_id=first.id,
            query="shared keyword",
        )
        other_memories = user_memories.list_active(
            db,
            user_id=second.id,
            query="shared keyword",
        )

        assert [item["artifact_id"] for item in own_artifacts] == [
            active_artifact["artifact_id"]
        ]
        assert other_artifacts == []
        assert [item["memory_id"] for item in own_memories] == [
            active_memory["memory_id"]
        ]
        assert other_memories == []


def test_snapshot_content_remains_immutable_after_source_message_edit() -> None:
    with _session() as db:
        _user_id, conversation_id, message_id, run_id = _seed_run(db)
        before = context_snapshots.get_by_run(db, run_id=run_id)
        assert before is not None
        assert before["items"][0]["content"] == "Question"

        updated = messages.update(
            db,
            conversation_id=conversation_id,
            message_id=message_id,
            content="Edited question",
        )
        assert updated is not None
        db.commit()

        after = context_snapshots.get_by_run(db, run_id=run_id)
        assert after is not None
        assert after["items"][0]["content"] == "Question"


def test_snapshot_retention_removes_only_records_older_than_cutoff() -> None:
    with _session() as db:
        _user_id, conversation_id, message_id, old_run_id = _seed_run(db)
        old_snapshot = db.query(ContextSnapshot).filter(
            ContextSnapshot.run_id == old_run_id
        ).one()
        old_snapshot.created_at = datetime.now(timezone.utc) - timedelta(days=91)
        recent_bundle = ContextBundle(
            standalone_query="Recent",
            requires_context=False,
            items=[],
            excluded=[],
            budget_tokens=100,
            used_tokens=0,
        )
        context_snapshots.create(
            db,
            run_id="research-recent",
            conversation_id=conversation_id,
            query_message_id=message_id,
            bundle=recent_bundle,
        )
        db.flush()

        removed = context_snapshots.delete_expired(
            db,
            retention_days=90,
            now=datetime.now(timezone.utc),
        )
        db.flush()

        assert removed == 1
        assert context_snapshots.get_by_run(db, run_id=old_run_id) is None
        assert context_snapshots.get_by_run(
            db,
            run_id="research-recent",
        ) is not None


def test_abandon_run_persists_one_non_reusable_system_notice(monkeypatch) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        user = User(username="abandon-run", password_hash="hash")
        db.add(user)
        db.flush()
        created = conversations.create(
            db,
            user_id=user.id,
            title="Abandoned",
            context_mode="auto",
        )
        conversation = conversations.get(
            db,
            user_id=user.id,
            conversation_id=created["conversation_id"],
        )
        assert conversation is not None
        query_message = messages.add(
            db,
            conversation=conversation,
            role="user",
            content="Question",
            run_id="research-abandoned",
        )
        db.commit()
        prepared = PreparedResearchRun(
            request=context_service_module.ResearchRequestContext(
                run_id="research-abandoned",
                conversation_id=created["conversation_id"],
                message_id=query_message["message_id"],
            ),
            bundle=ContextBundle(
                standalone_query="Question",
                requires_context=False,
                items=[],
                excluded=[],
                budget_tokens=100,
                used_tokens=0,
            ),
            user_id=user.id,
            query_message=query_message,
        )

    monkeypatch.setattr(context_service_module, "SessionLocal", factory)
    service = ContextService()

    assert service.abandon_run(
        prepared,
        status="cancelled",
        reason="User cancelled the request.",
    )
    assert service.abandon_run(
        prepared,
        status="cancelled",
        reason="Duplicate cancellation.",
    )

    with factory() as db:
        run_messages = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.run_id == "research-abandoned")
            .order_by(ConversationMessage.sequence.asc())
            .all()
        )
        assert [message.role for message in run_messages] == [
            "user",
            "system_notice",
        ]
        assert run_messages[-1].include_in_context is False
        assert run_messages[-1].metadata_json["run_status"] == "cancelled"
        assert db.query(ResearchArtifact).count() == 0
        assert db.query(UserMemory).count() == 0
