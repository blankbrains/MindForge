from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindforge.agents.base import AgentResult
from mindforge.agents.critic import CriticScore
from mindforge.agents.orchestrator import Orchestrator
from mindforge.agents.planner import ResearchPlan, SubTask
from mindforge.agents.synthesizer import SynthesisStreamEvent
from mindforge.api.schemas import HistorySaveRequest


def _settings(trace_dir: Path, *, remote: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(traces_dir=str(trace_dir)),
        observability=SimpleNamespace(
            enable_tracing=True,
            langfuse_public_key="pk-test" if remote else None,
            langfuse_secret_key="sk-test" if remote else None,
            langfuse_host="https://langfuse.example",
            capture_content=False,
            max_record_chars=20_000,
            max_trace_file_bytes=1024 * 1024,
            trace_retention_days=7,
            trace_list_scan_limit=1000,
            trace_detail_span_limit=2000,
        ),
        agent=SimpleNamespace(
            max_concurrent_research=1,
            max_concurrent_subtasks=2,
            max_concurrent_tool_calls=2,
            queue_timeout=2,
            research_timeout=10,
            subtask_timeout=5,
            max_refine_rounds=1,
            sse_heartbeat_seconds=1,
            stream_chunk_size=128,
        ),
        llm=SimpleNamespace(llm_provider="test"),
    )


def test_tracer_preserves_hierarchy_and_explicit_langfuse_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mindforge.observability import store as store_module
    from mindforge.observability import tracer as tracer_module

    starts: list[dict] = []
    trace_io: list[dict] = []

    class FakeObservation:
        def update(self, **_kwargs):
            return None

    class FakeContext:
        def __enter__(self):
            return FakeObservation()

        def __exit__(self, *_args):
            return False

    class FakeLangfuse:
        def __init__(self, **_kwargs):
            return None

        def start_as_current_observation(self, **kwargs):
            starts.append(kwargs)
            return FakeContext()

        def set_current_trace_io(self, **kwargs):
            trace_io.append(kwargs)

        def get_trace_url(self, *, trace_id: str):
            return f"https://langfuse.example/trace/{trace_id}"

        def flush(self):
            return None

        def shutdown(self):
            return None

    settings = _settings(tmp_path, remote=True)
    monkeypatch.setattr(tracer_module, "get_settings", lambda: settings)
    monkeypatch.setattr(store_module, "get_settings", lambda: settings)
    monkeypatch.setitem(
        sys.modules,
        "langfuse",
        SimpleNamespace(Langfuse=FakeLangfuse),
    )

    tracer = tracer_module.Tracer()
    with tracer.span(
        "orchestrator.research",
        metadata={"transport": "test"},
    ) as root:
        root.input = {"task": "private task"}
        with tracer.span(
            "llm.chat",
            metadata={"model": "test-model"},
        ) as generation:
            generation.output = {
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                }
            }
        root.output = {"success": True, "report_chars": 20}

    assert re.fullmatch(r"[0-9a-f]{32}", root.trace_id)
    assert generation.trace_id == root.trace_id
    assert generation.parent_id == root.span_id
    assert starts[0]["trace_context"] == {"trace_id": root.trace_id}
    assert "trace_context" not in starts[1]
    assert trace_io and trace_io[0]["input"]["redacted"] is True

    repository = store_module.TraceRepository(tmp_path)
    detail = repository.get_trace(root.trace_id)
    assert detail is not None
    assert [item["name"] for item in detail["observations"]] == [
        "orchestrator.research",
        "llm.chat",
    ]
    assert detail["summary"]["span_count"] == 2
    assert detail["summary"]["generation_count"] == 1
    assert detail["summary"]["total_tokens"] == 5
    assert detail["summary"]["cost_usd"] is None
    assert detail["summary"]["cost_status"] == "usage_unavailable"
    assert detail["summary"]["task_preview"] is None
    assert detail["summary"]["remote_url"].endswith(root.trace_id)

    listing = repository.list_traces(limit=20, offset=0)
    assert listing["total"] == 1
    assert listing["traces"][0]["trace_id"] == root.trace_id
    tracer.close()


@pytest.mark.asyncio
async def test_orchestrator_creates_one_top_level_trace_with_agent_children(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mindforge.agents import orchestrator as orchestrator_module
    from mindforge.observability import store as store_module
    from mindforge.observability import tracer as tracer_module

    settings = _settings(tmp_path)
    monkeypatch.setattr(orchestrator_module, "get_settings", lambda: settings)
    monkeypatch.setattr(tracer_module, "get_settings", lambda: settings)
    monkeypatch.setattr(store_module, "get_settings", lambda: settings)

    class PlannerStub:
        async def run(self, task: str) -> ResearchPlan:
            return ResearchPlan(
                plan_id="plan",
                original_task=task,
                subtasks=[
                    SubTask(task_id="one", description="first"),
                    SubTask(task_id="two", description="second"),
                ],
            )

    class ResearcherStub:
        async def run(self, task: str, context: str | None = None) -> AgentResult:
            assert context is None or isinstance(context, str)
            return AgentResult(
                agent_name="researcher",
                success=True,
                output=(task + " finding ") * 100,
            )

    class SynthesizerStub:
        async def synthesize(self, **_kwargs) -> AgentResult:
            return AgentResult(
                agent_name="synthesizer",
                success=True,
                output="Final report.",
            )

    class CriticStub:
        async def evaluate(self, **_kwargs) -> CriticScore:
            return CriticScore(overall=8.5, should_refine=False)

    tracer = tracer_module.Tracer()
    orchestrator = Orchestrator(
        planner=PlannerStub(),
        researcher=ResearcherStub(),
        synthesizer=SynthesizerStub(),
        critic=CriticStub(),
        tracer=tracer,
    )

    result = await orchestrator.run("Trace the research pipeline")

    assert result.success is True
    assert result.trace_id is not None
    assert result.metadata["trace_id"] == result.trace_id
    detail = store_module.TraceRepository(tmp_path).get_trace(result.trace_id)
    assert detail is not None
    names = [item["name"] for item in detail["observations"]]
    assert names.count("orchestrator.research") == 1
    assert names.count("agent.planner") == 1
    assert names.count("agent.researcher") == 2
    assert names.count("agent.synthesizer") == 1
    assert names.count("agent.critic") == 1
    root = next(
        item
        for item in detail["observations"]
        if item["name"] == "orchestrator.research"
    )
    assert all(
        item["parent_id"] == root["span_id"]
        for item in detail["observations"]
        if item["name"].startswith("agent.")
    )


@pytest.mark.asyncio
async def test_streaming_trace_keeps_context_across_streamed_synthesis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mindforge.agents import orchestrator as orchestrator_module
    from mindforge.observability import store as store_module
    from mindforge.observability import tracer as tracer_module

    settings = _settings(tmp_path)
    settings.agent.sse_heartbeat_seconds = 0.001
    monkeypatch.setattr(orchestrator_module, "get_settings", lambda: settings)
    monkeypatch.setattr(tracer_module, "get_settings", lambda: settings)
    monkeypatch.setattr(store_module, "get_settings", lambda: settings)

    class PlannerStub:
        async def run(self, task: str) -> ResearchPlan:
            return ResearchPlan(
                plan_id="stream-plan",
                original_task=task,
                subtasks=[
                    SubTask(task_id="one", description="first"),
                    SubTask(task_id="two", description="second"),
                ],
            )

    class ResearcherStub:
        async def run(self, task: str, context: str | None = None) -> AgentResult:
            del context
            return AgentResult(
                agent_name="researcher",
                success=True,
                output=f"{task} finding",
            )

    class SynthesizerStub:
        async def synthesize_stream(self, **_kwargs):
            await asyncio.sleep(0.005)
            yield SynthesisStreamEvent(type="chunk", content="Final ")
            await asyncio.sleep(0.005)
            result = AgentResult(
                agent_name="synthesizer",
                success=True,
                output="Final report.",
            )
            yield SynthesisStreamEvent(type="chunk", content="report.")
            yield SynthesisStreamEvent(type="done", result=result)

    class CriticStub:
        async def evaluate(self, **_kwargs) -> CriticScore:
            return CriticScore(overall=8.0, should_refine=False)

    tracer = tracer_module.Tracer()
    orchestrator = Orchestrator(
        planner=PlannerStub(),
        researcher=ResearcherStub(),
        synthesizer=SynthesizerStub(),
        critic=CriticStub(),
        tracer=tracer,
    )

    events = [
        event
        async for event in orchestrator.stream_run(
            "Compare Python and Java concurrency models"
        )
    ]

    done = next(event for event in events if event["type"] == "done")
    result = done["result"]
    assert result.success is True
    assert result.trace_id is not None
    assert any(event["type"] == "heartbeat" for event in events)
    assert "".join(
        event["content"]
        for event in events
        if event["type"] == "answer_chunk"
    ) == "Final report."

    detail = store_module.TraceRepository(tmp_path).get_trace(result.trace_id)
    assert detail is not None
    names = [item["name"] for item in detail["observations"]]
    assert names.count("orchestrator.research") == 1
    assert names.count("agent.synthesizer") == 1
    assert detail["summary"]["status"] == "success"


def test_history_trace_id_validation() -> None:
    trace_id = "a" * 32
    request = HistorySaveRequest(task="research", trace_id=trace_id)
    assert request.trace_id == trace_id
    with pytest.raises(ValueError):
        HistorySaveRequest(task="research", trace_id="not-a-trace")


def test_partial_result_marks_top_level_trace_as_degraded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mindforge.observability import store as store_module
    from mindforge.observability import tracer as tracer_module

    settings = _settings(tmp_path)
    monkeypatch.setattr(tracer_module, "get_settings", lambda: settings)
    monkeypatch.setattr(store_module, "get_settings", lambda: settings)

    tracer = tracer_module.Tracer()
    orchestrator = Orchestrator(
        planner=SimpleNamespace(),
        researcher=SimpleNamespace(),
        synthesizer=SimpleNamespace(),
        critic=SimpleNamespace(),
        tracer=tracer,
    )
    with tracer.span(
        "orchestrator.research",
        metadata={"display_name": "partial research"},
    ) as root:
        result = AgentResult(
            agent_name="orchestrator",
            success=True,
            output="Partial report.",
            metadata={
                "outcome": "degraded",
                "failure_reason": "One subtask timed out.",
            },
        )
        orchestrator._finish_root_span(root, result)

    detail = store_module.TraceRepository(tmp_path).get_trace(root.trace_id)
    assert detail is not None
    assert detail["summary"]["status"] == "degraded"


def test_trace_summary_contains_no_secret_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mindforge.observability import tracer as tracer_module

    settings = _settings(tmp_path)
    monkeypatch.setattr(tracer_module, "get_settings", lambda: settings)
    tracer = tracer_module.Tracer()
    with tracer.span("orchestrator.research") as span:
        span.input = {
            "task": "secret task",
            "api_key": "secret-value-do-not-store",
        }
        span.output = {"success": True}

    summary_path = tmp_path / f"trace_{span.trace_id}.summary.json"
    raw = summary_path.read_text(encoding="utf-8")
    assert "secret-value-do-not-store" not in raw
    assert "secret task" not in raw
    summary = json.loads(raw)
    assert summary["input"]["redacted"] is True


def test_trace_retention_zero_keeps_existing_traces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mindforge.observability import tracer as tracer_module

    settings = _settings(tmp_path)
    settings.observability.trace_retention_days = 0
    monkeypatch.setattr(tracer_module, "get_settings", lambda: settings)
    old_trace = tmp_path / "trace_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jsonl"
    old_trace.write_text("{}\n", encoding="utf-8")
    old_time = 1_600_000_000
    old_trace.touch()
    import os

    os.utime(old_trace, (old_time, old_time))

    tracer_module.Tracer()

    assert old_trace.exists()


def test_trace_summary_uses_research_task_as_display_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mindforge.observability import store as store_module
    from mindforge.observability import tracer as tracer_module

    settings = _settings(tmp_path)
    monkeypatch.setattr(tracer_module, "get_settings", lambda: settings)
    monkeypatch.setattr(store_module, "get_settings", lambda: settings)
    tracer = tracer_module.Tracer()
    with tracer.span(
        "orchestrator.research",
        metadata={"display_name": "Python 和 Java 有什么区别"},
    ) as span:
        span.output = {"success": True}

    listing = store_module.TraceRepository(tmp_path).list_traces(
        limit=20,
        offset=0,
    )

    assert listing["traces"][0]["display_name"] == "Python 和 Java 有什么区别"


def test_trace_detail_aggregates_structured_failure_causes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mindforge.observability import store as store_module
    from mindforge.observability import tracer as tracer_module

    settings = _settings(tmp_path)
    monkeypatch.setattr(tracer_module, "get_settings", lambda: settings)
    monkeypatch.setattr(store_module, "get_settings", lambda: settings)
    tracer = tracer_module.Tracer()

    with tracer.span(
        "orchestrator.research",
        metadata={"display_name": "timeout trace"},
    ) as root:
        with tracer.span(
            "agent.researcher",
            metadata={
                "agent": "researcher",
                "subtask_id": "t3",
                "stage": "subtask_execution",
                "status": "error",
                "error_code": "subtask_timeout",
                "error_type": "TimeoutError",
                "timeout_seconds": 60,
            },
        ) as researcher:
            researcher.error = "Subtask 't3' timed out after 60 seconds."
            with tracer.span(
                "llm.chat",
                metadata={
                    "agent": "researcher",
                    "model": "deepseek-v4-flash",
                    "attempt": 1,
                    "stage": "llm_request",
                    "status": "cancelled",
                    "error_code": "llm_request_cancelled",
                    "error_type": "CancelledError",
                },
            ) as generation:
                generation.error = "LLM request was cancelled before completion."
        root.metadata["status"] = "degraded"
        root.error = "One subtask timed out."
        root.output = {"success": True, "outcome": "degraded"}

    detail = store_module.TraceRepository(tmp_path).get_trace(root.trace_id)

    assert detail is not None
    assert detail["summary"]["failure_count"] == 3
    assert "Researcher 子任务 t3 执行超过 60 秒" in detail["summary"][
        "failure_summary"
    ]
    failures = {
        item["error_code"]: item
        for item in detail["failures"]
    }
    assert failures["subtask_timeout"]["stage"] == "subtask_execution"
    assert failures["llm_request_cancelled"]["model"] == "deepseek-v4-flash"
    assert failures["llm_request_cancelled"]["attempt"] == 1


def test_tracer_records_non_empty_timeout_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mindforge.observability import store as store_module
    from mindforge.observability import tracer as tracer_module

    settings = _settings(tmp_path)
    monkeypatch.setattr(tracer_module, "get_settings", lambda: settings)
    monkeypatch.setattr(store_module, "get_settings", lambda: settings)
    tracer = tracer_module.Tracer()

    with pytest.raises(TimeoutError):
        with tracer.span("llm.chat") as generation:
            raise TimeoutError

    detail = store_module.TraceRepository(tmp_path).get_trace(generation.trace_id)

    assert detail is not None
    failure = detail["failures"][0]
    assert failure["error_code"] == "timeout"
    assert failure["error_type"] == "TimeoutError"
    assert failure["message"] == "Operation timed out."


def test_trace_repository_recovers_legacy_subtask_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mindforge.observability import store as store_module

    settings = _settings(tmp_path)
    settings.agent.subtask_timeout = 60
    monkeypatch.setattr(store_module, "get_settings", lambda: settings)
    trace_id = "b" * 32
    root_id = "1" * 16
    researcher_id = "2" * 16
    records = [
        {
            "span_id": root_id,
            "trace_id": trace_id,
            "name": "orchestrator.research",
            "start_time": 1.0,
            "end_time": 91.0,
            "duration_ms": 90_000,
            "parent_id": None,
            "error": (
                "<Token var=<ContextVar name='mindforge_trace_stack'> "
                "was created in a different Context"
            ),
            "metadata": {"status": "error"},
        },
        {
            "span_id": researcher_id,
            "trace_id": trace_id,
            "name": "agent.researcher",
            "start_time": 5.0,
            "end_time": 65.0,
            "duration_ms": 60_001,
            "parent_id": root_id,
            "error": None,
            "metadata": {
                "agent": "researcher",
                "subtask_id": "t3",
            },
        },
        {
            "span_id": "3" * 16,
            "trace_id": trace_id,
            "name": "llm.chat",
            "start_time": 5.0,
            "end_time": 50.0,
            "duration_ms": 45_001,
            "parent_id": researcher_id,
            "error": None,
            "metadata": {
                "agent": "researcher",
                "model": "test-model",
                "attempt": 1,
            },
        },
        {
            "span_id": "4" * 16,
            "trace_id": trace_id,
            "name": "llm.chat",
            "start_time": 51.0,
            "end_time": 54.0,
            "duration_ms": 3_000,
            "parent_id": researcher_id,
            "error": None,
            "output": {"usage": {"total_tokens": 10}},
            "metadata": {
                "agent": "researcher",
                "model": "test-model",
                "attempt": 2,
            },
        },
        {
            "span_id": "5" * 16,
            "trace_id": trace_id,
            "name": "llm.chat",
            "start_time": 57.0,
            "end_time": 65.0,
            "duration_ms": 8_000,
            "parent_id": researcher_id,
            "error": "Operation cancelled before completion.",
            "metadata": {
                "agent": "researcher",
                "model": "test-model",
                "status": "cancelled",
            },
        },
    ]
    trace_path = tmp_path / f"trace_{trace_id}.jsonl"
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    detail = store_module.TraceRepository(tmp_path).get_trace(trace_id)

    assert detail is not None
    assert detail["summary"]["failure_count"] == 4
    assert detail["failures"][1]["error_code"] == "subtask_timeout"
    assert "Researcher 子任务 t3 执行超过 60 秒" in detail["summary"][
        "failure_summary"
    ]
    assert any(
        failure["error_code"] == "llm_request_timeout"
        and failure["attempt"] == 1
        for failure in detail["failures"]
    )
