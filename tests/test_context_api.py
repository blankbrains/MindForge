from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from mindforge.api import context_routes, routes
from mindforge.context.models import ContextBundle, ContextCandidate
from mindforge.db import (
    Base,
    ConversationSummary,
    ResearchArtifact,
    User,
    get_default_user_id,
)
from mindforge.repositories import (
    context_items,
    context_snapshots,
    conversations,
    messages,
    research_artifacts,
)
from mindforge.services import context_service


def _api_app() -> FastAPI:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/v1")
    return app


def _session_factory():
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
    return sessionmaker(bind=engine)


def _install_session_factory(monkeypatch, factory) -> None:
    monkeypatch.setattr(context_routes, "SessionLocal", factory)
    monkeypatch.setattr(context_service, "SessionLocal", factory)


@pytest.mark.asyncio
async def test_context_api_conversation_preview_and_memory_crud(monkeypatch) -> None:
    factory = _session_factory()
    _install_session_factory(monkeypatch, factory)

    async with AsyncClient(
        transport=ASGITransport(app=_api_app()),
        base_url="http://test",
    ) as client:
        created_response = await client.post(
            "/api/v1/conversations",
            json={"title": "API context", "context_mode": "manual"},
        )
        assert created_response.status_code == 201
        created = created_response.json()
        conversation_id = created["conversation_id"]

        listed = await client.get("/api/v1/conversations")
        assert listed.status_code == 200
        assert [item["conversation_id"] for item in listed.json()] == [
            conversation_id
        ]

        updated = await client.patch(
            f"/api/v1/conversations/{conversation_id}",
            json={"title": "Updated API context", "version": created["version"]},
        )
        assert updated.status_code == 200
        stale = await client.patch(
            f"/api/v1/conversations/{conversation_id}",
            json={"title": "Stale", "version": created["version"]},
        )
        assert stale.status_code == 409

        with factory() as db:
            user_id = get_default_user_id(db)
            conversation = conversations.get(
                db,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            assert conversation is not None
            message = messages.add(
                db,
                conversation=conversation,
                role="user",
                content="Prior deployment constraint",
                run_id="research-api-prior",
            )
            db.commit()

        preview = await client.post(
            f"/api/v1/conversations/{conversation_id}/context-preview",
            json={
                "task": "new question",
                "context_mode": "manual",
                "selected_context_ids": [f"message:{message['message_id']}"],
            },
        )
        assert preview.status_code == 200
        assert [item["context_id"] for item in preview.json()["items"]] == [
            f"message:{message['message_id']}"
        ]

        memory = await client.post(
            "/api/v1/memories",
            json={
                "category": "preference",
                "content": "Prefer concise reports",
                "confidence": 0.9,
                "user_confirmed": True,
            },
        )
        assert memory.status_code == 201
        memory_id = memory.json()["memory_id"]
        memories_response = await client.get("/api/v1/memories")
        assert memories_response.status_code == 200
        assert [item["memory_id"] for item in memories_response.json()] == [
            memory_id
        ]
        changed_memory = await client.patch(
            f"/api/v1/memories/{memory_id}",
            json={"content": "Prefer structured concise reports"},
        )
        assert changed_memory.status_code == 200
        assert changed_memory.json()["content"] == (
            "Prefer structured concise reports"
        )
        removed_memory = await client.delete(f"/api/v1/memories/{memory_id}")
        assert removed_memory.status_code == 204
        assert (await client.get("/api/v1/memories")).json() == []


@pytest.mark.asyncio
async def test_context_snapshot_endpoint_enforces_conversation_owner(
    monkeypatch,
) -> None:
    factory = _session_factory()
    _install_session_factory(monkeypatch, factory)
    with factory() as db:
        owner = User(username="owner", password_hash="hash")
        other = User(username="other", password_hash="hash")
        db.add_all([owner, other])
        db.flush()
        conversation = conversations.create(
            db,
            user_id=owner.id,
            title="Private",
            context_mode="auto",
        )
        bundle = ContextBundle(
            standalone_query="Private question",
            requires_context=True,
            items=[
                ContextCandidate(
                    source_type="message",
                    source_id="source",
                    title="Private source",
                    content="Private content",
                    token_count=4,
                )
            ],
            excluded=[],
            budget_tokens=100,
            used_tokens=4,
        )
        context_snapshots.create(
            db,
            run_id="research-private",
            conversation_id=conversation["conversation_id"],
            query_message_id=None,
            bundle=bundle,
        )
        db.commit()
        other_id = other.id

    monkeypatch.setattr(context_routes, "get_default_user_id", lambda _db: other_id)
    async with AsyncClient(
        transport=ASGITransport(app=_api_app()),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/api/v1/research-runs/research-private/context"
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_editing_message_invalidates_old_summary_and_artifact(
    monkeypatch,
) -> None:
    factory = _session_factory()
    _install_session_factory(monkeypatch, factory)
    with factory() as db:
        user_id = get_default_user_id(db)
        created = conversations.create(
            db,
            user_id=user_id,
            title="Editable",
            context_mode="auto",
        )
        conversation = conversations.get(
            db,
            user_id=user_id,
            conversation_id=created["conversation_id"],
        )
        assert conversation is not None
        message = messages.add(
            db,
            conversation=conversation,
            role="user",
            content="Original requirement",
            run_id="research-edit",
        )
        summary = context_items.replace_summary(
            db,
            conversation_id=created["conversation_id"],
            from_sequence=1,
            to_sequence=1,
            summary={
                "goal": "Original requirement",
                "constraints": [],
                "decisions": [],
                "open_questions": [],
            },
            source_message_ids=[message["message_id"]],
        )
        artifact = research_artifacts.create(
            db,
            user_id=user_id,
            run_id="research-edit",
            conversation_id=created["conversation_id"],
            artifact_type="decision",
            title="Old decision",
            content="Derived from original requirement",
            source_ids=["source"],
            quality_score=8.0,
            grounding_status="grounded",
            freshness_class="stable",
            expires_at=None,
        )
        context_items.add_lineage(
            db,
            source_type="message",
            source_id=message["message_id"],
            derived_type="artifact",
            derived_id=artifact["artifact_id"],
            relation="extracted_from",
        )
        db.commit()

    async with AsyncClient(
        transport=ASGITransport(app=_api_app()),
        base_url="http://test",
    ) as client:
        response = await client.patch(
            (
                f"/api/v1/conversations/{created['conversation_id']}"
                f"/messages/{message['message_id']}"
            ),
            json={"content": "Changed requirement"},
        )
    assert response.status_code == 200

    with factory() as db:
        summary_row = db.query(ConversationSummary).filter(
            ConversationSummary.summary_id == summary["summary_id"]
        ).one()
        artifact_row = db.query(ResearchArtifact).filter(
            ResearchArtifact.artifact_id == artifact["artifact_id"]
        ).one()
        assert summary_row.status == "invalidated"
        assert artifact_row.enabled is False


def test_complete_context_marks_persistence_failure(monkeypatch) -> None:
    def fail(_self, _prepared, _result):
        raise context_service.ContextServiceError(
            "conversation_deleted_during_run",
            "Conversation was deleted before completion.",
        )

    monkeypatch.setattr(context_service.ContextService, "complete_run", fail)
    result = {
        "success": True,
        "output": "Answer",
        "data": {},
        "metadata": {},
    }

    assert routes._complete_research_context(object(), result) is False
    assert result["metadata"]["context_persistence_status"] == "failed"
    assert "deleted" in result["metadata"]["context_persistence_error"]


@pytest.mark.asyncio
async def test_sse_cancellation_abandons_unfinished_context(monkeypatch) -> None:
    release = asyncio.Event()
    abandoned: list[tuple[str, str]] = []
    prepared = SimpleNamespace(finalized=False)

    async def events(_orch, _task, *, prepared_context=None):
        assert prepared_context is prepared
        yield b"data: first\n\n"
        await release.wait()

    def abandon(_prepared, *, status, reason):
        abandoned.append((status, reason))
        _prepared.finalized = True
        return True

    monkeypatch.setattr(routes, "_stream_response_events", events)
    monkeypatch.setattr(routes, "_abandon_research_context", abandon)
    cancellation = asyncio.Event()
    stream = routes._stream_response(
        None,
        "question",
        request_id="research-cancelled-context",
        cancellation=cancellation,
        prepared_context=prepared,
    )

    assert await anext(stream) == b"data: first\n\n"
    cancellation.set()
    assert [chunk async for chunk in stream] == []

    assert abandoned
    assert abandoned[0][0] == "cancelled"


@pytest.mark.asyncio
async def test_non_stream_exception_abandons_prepared_context(monkeypatch) -> None:
    abandoned: list[tuple[str, str]] = []
    prepared = SimpleNamespace(
        finalized=False,
        bundle=SimpleNamespace(snapshot_id="snapshot"),
    )

    monkeypatch.setattr(
        routes,
        "_prepare_research_context",
        lambda _body, *, run_id: prepared,
    )
    monkeypatch.setattr(routes, "has_llm_credentials", lambda: False)

    async def fail(*_args, **_kwargs):
        raise RuntimeError("execution failed")

    def abandon(_prepared, *, status, reason):
        abandoned.append((status, reason))
        _prepared.finalized = True
        return True

    monkeypatch.setattr(routes, "_execute_query_non_stream", fail)
    monkeypatch.setattr(routes, "_abandon_research_context", abandon)

    with pytest.raises(RuntimeError, match="execution failed"):
        await routes.query(
            routes.QueryRequest(
                request_id="research-non-stream-failure",
                task="question",
                conversation_id="a" * 32,
            )
        )

    assert abandoned == [("failed", "execution failed")]


@pytest.mark.asyncio
async def test_stream_setup_exception_abandons_prepared_context(monkeypatch) -> None:
    abandoned: list[tuple[str, str]] = []
    prepared = SimpleNamespace(
        finalized=False,
        bundle=SimpleNamespace(snapshot_id="snapshot"),
    )

    monkeypatch.setattr(
        routes,
        "_prepare_research_context",
        lambda _body, *, run_id: prepared,
    )

    def fail_credentials():
        raise RuntimeError("credential probe failed")

    def abandon(_prepared, *, status, reason):
        abandoned.append((status, reason))
        _prepared.finalized = True
        return True

    monkeypatch.setattr(routes, "has_llm_credentials", fail_credentials)
    monkeypatch.setattr(routes, "_abandon_research_context", abandon)
    request_id = "research-stream-setup-failure"

    with pytest.raises(RuntimeError, match="credential probe failed"):
        await routes.query(
            routes.QueryRequest(
                request_id=request_id,
                task="question",
                stream=True,
                conversation_id="a" * 32,
            )
        )

    assert abandoned == [("failed", "credential probe failed")]
    assert request_id not in routes._ACTIVE_RESEARCH_CANCELLATIONS
