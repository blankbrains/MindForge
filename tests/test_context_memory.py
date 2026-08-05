from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mindforge.agents.orchestrator import Orchestrator
from mindforge.api import routes
from mindforge.api.schemas import QueryRequest
from mindforge.context.artifacts import extract_artifacts
from mindforge.context.budget import allocate_budget
from mindforge.context.builder import ContextBuilder
from mindforge.context.models import ContextBundle, ContextCandidate, ResearchRequestContext
from mindforge.context.summaries import build_structured_summary
from mindforge.context.resolver import resolve_references


def _message(
    message_id: str,
    content: str,
    *,
    role: str = "user",
    sequence: int = 1,
    pinned: bool = False,
) -> dict:
    return {
        "message_id": message_id,
        "role": role,
        "content": content,
        "sequence": sequence,
        "pinned": pinned,
        "created_at": datetime.now(timezone.utc),
    }


def test_reference_resolver_builds_visible_standalone_follow_up() -> None:
    resolution = resolve_references(
        "第二个方案有什么风险？",
        [
            _message("m1", "比较方案 A 和方案 B"),
            _message(
                "m2",
                "方案 A 成本低；方案 B 隔离性更好。",
                role="assistant",
                sequence=2,
            ),
        ],
    )

    assert resolution.requires_context is True
    assert resolution.referenced_message_ids == ("m1", "m2")
    assert "方案 B 隔离性更好" in resolution.standalone_query
    assert "当前问题：第二个方案有什么风险" in resolution.standalone_query


def test_independent_request_never_carries_history() -> None:
    request = ResearchRequestContext(
        run_id="research-independent",
        conversation_id="c" * 32,
        context_mode="auto",
        independent=True,
    )
    bundle = ContextBuilder(
        recent_messages=[_message("m1", "必须被隔离的旧问题")],
        summary=None,
        artifacts=[],
        memories=[],
    ).build("新问题", request)

    assert bundle.items == []
    assert bundle.used_tokens == 0
    assert bundle.standalone_query == "新问题"


def test_manual_mode_only_uses_explicit_selection() -> None:
    request = ResearchRequestContext(
        run_id="research-manual",
        conversation_id="c" * 32,
        context_mode="manual",
        selected_context_ids=("message:m2",),
    )
    bundle = ContextBuilder(
        recent_messages=[
            _message("m1", "未选择的相关消息", sequence=1),
            _message("m2", "用户明确选择的旧消息", sequence=2),
        ],
        summary=None,
        artifacts=[],
        memories=[],
    ).build("完全不相关的查询", request)

    assert [item.context_id for item in bundle.items] == ["message:m2"]
    assert any(
        selection.candidate.context_id == "message:m1"
        and selection.reason == "not_selected"
        for selection in bundle.excluded
    )


def test_policy_rejects_deleted_equivalent_and_model_only_artifacts() -> None:
    request = ResearchRequestContext(
        run_id="research-policy",
        conversation_id="c" * 32,
        context_mode="auto",
        excluded_context_ids=("message:m1",),
    )
    bundle = ContextBuilder(
        recent_messages=[_message("m1", "Agent 上下文策略")],
        summary=None,
        artifacts=[
            {
                "artifact_id": "a1",
                "title": "无来源旧结论",
                "content": "Agent 上下文策略",
                "grounding_status": "model_only",
                "quality_score": 9,
                "created_at": datetime.now(timezone.utc),
            },
            {
                "artifact_id": "a2",
                "title": "过期结论",
                "content": "Agent 上下文策略",
                "grounding_status": "grounded",
                "expires_at": datetime.now(timezone.utc) - timedelta(seconds=1),
                "created_at": datetime.now(timezone.utc),
            },
        ],
        memories=[],
    ).build("Agent 上下文策略", request)

    reasons = {
        selection.candidate.context_id: selection.reason
        for selection in bundle.excluded
    }
    assert reasons["message:m1"] == "user_excluded"
    assert reasons["artifact:a1"] == "model_only"
    assert reasons["artifact:a2"] == "expired"
    assert bundle.items == []


def test_context_fingerprint_changes_with_selected_content() -> None:
    first = ContextBundle(
        standalone_query="query",
        requires_context=True,
        items=[
            ContextCandidate(
                source_type="message",
                source_id="m1",
                title="message",
                content="version one",
            )
        ],
        excluded=[],
        budget_tokens=100,
        used_tokens=3,
    )
    second = ContextBundle(
        standalone_query="query",
        requires_context=True,
        items=[
            ContextCandidate(
                source_type="message",
                source_id="m1",
                title="message",
                content="version two",
            )
        ],
        excluded=[],
        budget_tokens=100,
        used_tokens=3,
    )

    assert first.fingerprint != second.fingerprint


def test_context_fingerprint_changes_when_rendered_prompt_metadata_changes() -> None:
    first = ContextBundle(
        standalone_query="query",
        requires_context=True,
        items=[
            ContextCandidate(
                source_type="message",
                source_id="m1",
                title="Original title",
                content="same content",
                selection_reason="recent context",
            )
        ],
        excluded=[],
        budget_tokens=100,
        used_tokens=3,
    )
    second = ContextBundle(
        standalone_query="query",
        requires_context=True,
        items=[
            ContextCandidate(
                source_type="message",
                source_id="m1",
                title="Changed title",
                content="same content",
                selection_reason="explicit selection",
            )
        ],
        excluded=[],
        budget_tokens=100,
        used_tokens=3,
    )

    assert first.to_working_chunks() != second.to_working_chunks()
    assert first.fingerprint != second.fingerprint


def test_budget_deduplicates_content_after_snapshot_truncation() -> None:
    candidates = [
        ContextCandidate(
            source_type="message",
            source_id="m1",
            title="first",
            content="same-prefix-AAAA",
            explicitly_selected=True,
        ),
        ContextCandidate(
            source_type="message",
            source_id="m2",
            title="second",
            content="same-prefix-BBBB",
            explicitly_selected=True,
        ),
    ]

    selected, excluded, used = allocate_budget(
        candidates,
        budget_tokens=100,
        chars_per_token=1,
        max_item_chars=12,
    )

    assert [item.source_id for item in selected] == ["m1"]
    assert used == 12
    assert [(item.candidate.source_id, item.reason) for item in excluded] == [
        ("m2", "duplicate")
    ]


def test_cross_conversation_artifact_does_not_receive_same_conversation_bonus() -> None:
    request = ResearchRequestContext(
        run_id="research-cross-conversation",
        conversation_id="current-conversation",
        context_mode="manual",
        selected_context_ids=("artifact:local", "artifact:remote"),
    )
    created_at = datetime.now(timezone.utc)
    bundle = ContextBuilder(
        recent_messages=[],
        summary=None,
        artifacts=[
            {
                "artifact_id": "local",
                "conversation_id": "current-conversation",
                "title": "local",
                "content": "identical content",
                "grounding_status": "grounded",
                "created_at": created_at,
            },
            {
                "artifact_id": "remote",
                "conversation_id": "other-conversation",
                "title": "remote",
                "content": "different content",
                "grounding_status": "grounded",
                "created_at": created_at,
            },
        ],
        memories=[],
    ).build("unrelated", request)

    scores = {item.source_id: item.score for item in bundle.items}
    assert scores["local"] == pytest.approx(scores["remote"] + 0.2)
    remote = next(item for item in bundle.items if item.source_id == "remote")
    assert remote.metadata["same_conversation"] is False


def test_structured_summary_respects_configured_token_limit() -> None:
    summary = build_structured_summary(
        [
            {
                "message_id": f"m{index}",
                "role": "user" if index % 2 == 0 else "assistant",
                "content": ("constraint " if index % 2 == 0 else "# Decision\n")
                + ("x" * 2000),
            }
            for index in range(20)
        ],
        max_tokens=100,
        chars_per_token=4,
    )

    rendered = "\n".join(
        [
            str(summary["goal"]),
            *map(str, summary["constraints"]),
            *map(str, summary["decisions"]),
            *map(str, summary["open_questions"]),
        ]
    )
    assert len(rendered) <= 400


@pytest.mark.asyncio
async def test_orchestrator_prefers_context_bundle_over_legacy_semantic_memory() -> None:
    class LegacyMemory:
        async def recall(self, *_args, **_kwargs):
            raise AssertionError("legacy semantic memory must not be read")

    orchestrator = object.__new__(Orchestrator)
    orchestrator._semantic_memory = LegacyMemory()
    bundle = ContextBundle(
        standalone_query="resolved query",
        requires_context=True,
        items=[
            ContextCandidate(
                source_type="message",
                source_id="m1",
                title="prior answer",
                content="selected visible context",
                score=1.0,
            )
        ],
        excluded=[],
        budget_tokens=100,
        used_tokens=5,
    )

    memory = await orchestrator._create_working_memory(
        "resolved query",
        context_bundle=bundle,
    )

    assert "selected visible context" in memory.get_context_string()
    assert "untrusted_content" not in memory.get_context_string()


@pytest.mark.asyncio
async def test_route_orchestrator_adapter_only_passes_supported_context() -> None:
    captured: dict = {}

    class ContextAware:
        async def run(self, task: str, *, context_bundle=None):
            captured["task"] = task
            captured["bundle"] = context_bundle
            return "ok"

    bundle = SimpleNamespace(fingerprint="fingerprint")
    result = await routes._call_orchestrator_method(
        ContextAware(),
        "run",
        "task",
        request_context=SimpleNamespace(),
        context_bundle=bundle,
    )

    assert result == "ok"
    assert captured == {"task": "task", "bundle": bundle}


def test_query_request_rejects_conflicting_context_controls() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(
            task="test",
            selected_context_ids=["message:m1"],
            excluded_context_ids=["message:m1"],
        )


def test_artifact_extraction_only_keeps_grounded_outputs() -> None:
    result = SimpleNamespace(
        success=True,
        output="# Conclusion\nGrounded report [1].",
        data={
            "sources": [{"url": "https://example.com"}],
            "subtask_outputs": [
                {
                    "task_id": "grounded",
                    "description": "Grounded finding",
                    "success": True,
                    "output": "Evidence-backed result.",
                    "sources": [{"url": "https://example.com"}],
                    "grounding_status": "grounded",
                },
                {
                    "task_id": "model",
                    "description": "Model-only finding",
                    "success": True,
                    "output": "Unsupported result.",
                    "sources": [],
                    "grounding_status": "model_only",
                },
            ],
        },
        metadata={"quality": 8.0, "grounding_status": "grounded"},
    )

    artifacts = extract_artifacts(task="stable question", result=result)

    assert any(item["title"] == "Grounded finding" for item in artifacts)
    assert not any(item["title"] == "Model-only finding" for item in artifacts)
    assert all(item["grounding_status"] != "model_only" for item in artifacts)


def test_artifact_extraction_rejects_missing_grounding_status() -> None:
    result = SimpleNamespace(
        success=True,
        output="# Conclusion\nUnsupported report.",
        data={
            "sources": [{"url": "https://example.com"}],
            "subtask_outputs": [
                {
                    "task_id": "missing-grounding",
                    "success": True,
                    "output": "Unsupported finding.",
                    "sources": [{"url": "https://example.com"}],
                }
            ],
        },
        metadata={},
    )

    assert extract_artifacts(task="question", result=result) == []
