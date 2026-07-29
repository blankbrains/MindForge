"""Regression tests for confirmed production bugs."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from mindforge.agents.base import AgentResult, BaseAgent
from mindforge.agents.synthesizer import SynthesizerAgent
from mindforge.api import routes, server
from mindforge.api.schemas import (
    LLMProviderUpdate,
    QueryRequest,
    SettingsUpdateRequest,
)
from mindforge.config import LLMConfig
from mindforge.ingestion.chunker import (
    DocumentChunk,
    ElementAwareSplitter,
    TextSplitter,
)
from mindforge.ingestion.embedder import EmbeddingManager
from mindforge.ingestion.parsers import (
    DocumentElement,
    DocumentLimitError,
    DocumentParser,
    DocumentParserCancelledError,
)
from mindforge.ingestion.raptor import RAPTORIndexer
from mindforge.memory.episodic import Episode, EpisodicMemory
from mindforge.retrieval.adaptive import AdaptiveRetriever, QueryMode
from mindforge.retrieval.bm25 import BM25Retriever
from mindforge.retrieval.graphrag import Entity, GraphRAGEngine
from mindforge.retrieval.hybrid import HybridRetriever
from mindforge.services.health import HealthMonitor, HealthSnapshot
from mindforge.services import indexing as indexing_service
from mindforge.services import index_jobs as index_job_service
from mindforge.models.base import ChatResult, LLMConfigurationError, StreamEvent
from mindforge.models.deepseek_adapter import DeepSeekAdapter
from mindforge.models.openai_adapter import OpenAIAdapter
from mindforge.tools.base import ToolResult
from mindforge.tools.citation_verifier import CitationVerifier
from mindforge.tools.rag_tool import RAGTool
from mindforge.tools.web_search import WebSearchTool


def _api_app() -> FastAPI:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/v1")
    return app


@pytest.mark.asyncio
async def test_hybrid_retrieval_starts_independent_paths_in_parallel() -> None:
    direct_embedding_started = asyncio.Event()
    llm_started = asyncio.Event()

    async def embedding(text: str) -> list[float]:
        if text == "parallel query":
            direct_embedding_started.set()
            await asyncio.wait_for(llm_started.wait(), timeout=0.5)
        return [0.1, 0.2]

    async def llm(_: str) -> str:
        await asyncio.wait_for(
            direct_embedding_started.wait(),
            timeout=0.5,
        )
        llm_started.set()
        return "1. expanded query\n2. conceptual query\n3. specific query"

    class VectorStore:
        async def search(self, **_kwargs):
            return [
                (
                    {
                        "chunk_id": "vector-1",
                        "content": "vector result",
                        "source": "test",
                    },
                    0.9,
                )
            ]

    class BM25:
        def search(self, **_kwargs):
            return [
                {
                    "id": "bm25-1",
                    "text": "bm25 result",
                    "source": "test",
                }
            ]

    retriever = HybridRetriever(
        vector_store=VectorStore(),
        bm25_retriever=BM25(),
        embedding_fn=embedding,
        llm_fn=llm,
    )
    results = await asyncio.wait_for(
        retriever.retrieve(
            "parallel query",
            use_hyde=True,
            use_multi_query=True,
        ),
        timeout=1,
    )

    assert {result["id"] for result in results} == {
        "vector-1",
        "bm25-1",
    }


@pytest.mark.asyncio
async def test_indexing_slots_enforce_configured_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        indexing_service,
        "get_settings",
        lambda: SimpleNamespace(
            api=SimpleNamespace(max_concurrent_index_jobs=1)
        ),
    )
    indexing_service.reset_indexing_service()
    active = 0
    max_active = 0

    async def worker() -> None:
        nonlocal active, max_active
        async with indexing_service.index_slot():
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(worker(), worker(), worker())

    assert max_active == 1
    indexing_service.reset_indexing_service()


@pytest.mark.asyncio
async def test_persistent_index_job_records_stage_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source = tmp_path / "data" / "index-jobs" / "job_document.txt"
    source.parent.mkdir(parents=True)
    source.write_text("document", encoding="utf-8")
    job = {
        "job_id": "job",
        "doc_id": None,
        "filename": "document.txt",
        "file_path": str(source),
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "chunk_count": 0,
        "timings": {},
        "error": None,
        "cancel_requested": False,
        "strategy": "auto",
        "use_raptor": False,
        "use_graphrag": False,
    }
    updates: list[dict] = []
    removed: list[object] = []

    def fake_get(_job_id: str) -> dict:
        return dict(job)

    def fake_update(_job_id: str, **values) -> dict:
        updates.append(dict(values))
        job.update(values)
        return dict(job)

    monkeypatch.setattr(index_job_service, "get_index_job", fake_get)
    monkeypatch.setattr(index_job_service, "update_index_job", fake_update)
    monkeypatch.setattr(
        index_job_service,
        "build_index_signature",
        lambda **_kwargs: "signature",
    )

    async def no_reusable_document(**_kwargs):
        return None

    monkeypatch.setattr(
        index_job_service,
        "get_reusable_document",
        no_reusable_document,
    )
    def parse_with_ocr_progress(_self, _path):
        assert _self._progress_callback is not None
        _self._progress_callback("ocr", 1, 2)
        return SimpleNamespace(
            doc_id="doc",
            filename="document.txt",
            content="document",
        )

    monkeypatch.setattr(
        DocumentParser,
        "parse",
        parse_with_ocr_progress,
    )

    async def fake_index(**kwargs):
        await kwargs["progress_callback"](
            "embedding",
            75.0,
            1,
            {"parsing": 0.01, "embedding": 0.02},
        )
        return [SimpleNamespace(chunk_id="chunk")]

    monkeypatch.setattr(routes, "_index_with_lifecycle", fake_index)
    service = index_job_service.IndexJobService()
    monkeypatch.setattr(
        service,
        "_remove_job_file",
        lambda path: removed.append(path),
    )

    await service._run_job("job")

    assert job["status"] == "completed"
    assert job["stage"] == "completed"
    assert job["progress"] == 100.0
    assert job["chunk_count"] == 1
    assert job["timings"]["total"] >= 0
    assert any(update.get("stage") == "embedding" for update in updates)
    assert any(update.get("stage") == "ocr" for update in updates)
    assert removed == [source.resolve()]


@pytest.mark.asyncio
async def test_index_job_reuses_identical_completed_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source = tmp_path / "data" / "index-jobs" / "job_document.txt"
    source.parent.mkdir(parents=True)
    source.write_text("document", encoding="utf-8")
    job = {
        "job_id": "job",
        "doc_id": None,
        "filename": "document.txt",
        "file_path": str(source),
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "chunk_count": 0,
        "timings": {},
        "error": None,
        "cancel_requested": False,
        "strategy": "auto",
        "use_raptor": False,
        "use_graphrag": False,
    }
    removed: list[object] = []

    monkeypatch.setattr(
        index_job_service,
        "get_index_job",
        lambda _job_id: dict(job),
    )

    def fake_update(_job_id: str, **values) -> dict:
        job.update(values)
        return dict(job)

    monkeypatch.setattr(index_job_service, "update_index_job", fake_update)
    monkeypatch.setattr(
        index_job_service,
        "build_index_signature",
        lambda **_kwargs: "signature",
    )

    async def reusable_document(**_kwargs):
        return {
            "doc_id": "stable-doc",
            "status": "indexed",
            "chunk_count": 12,
            "index_signature": "signature",
        }

    monkeypatch.setattr(
        index_job_service,
        "get_reusable_document",
        reusable_document,
    )
    monkeypatch.setattr(
        DocumentParser,
        "parse",
        lambda _self, _path: SimpleNamespace(
            doc_id="stable-doc",
            filename="document.txt",
            content="document",
        ),
    )

    async def unexpected_index(**_kwargs):
        raise AssertionError("duplicate content must not be re-embedded")

    monkeypatch.setattr(routes, "_index_with_lifecycle", unexpected_index)
    service = index_job_service.IndexJobService()
    monkeypatch.setattr(
        service,
        "_remove_job_file",
        lambda path: removed.append(path),
    )

    await service._run_job("job")

    assert job["status"] == "completed"
    assert job["doc_id"] == "stable-doc"
    assert job["chunk_count"] == 12
    assert job["timings"]["total"] >= 0
    assert removed == [source.resolve()]


@pytest.mark.asyncio
async def test_async_upload_returns_accepted_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    now = datetime.now(timezone.utc)
    captured: dict[str, object] = {}

    class FakeService:
        async def create(self, **values):
            captured.update(values)
            return {
                "job_id": values["job_id"],
                "doc_id": None,
                "filename": values["filename"],
                "status": "queued",
                "stage": "queued",
                "progress": 0.0,
                "chunk_count": 0,
                "timings": {},
                "error": None,
                "cancel_requested": False,
                "strategy": values["strategy"],
                "use_raptor": values["use_raptor"],
                "use_graphrag": values["use_graphrag"],
                "created_at": now,
                "updated_at": now,
            }

    monkeypatch.setattr(
        index_job_service,
        "get_index_job_service",
        lambda: FakeService(),
    )
    monkeypatch.setattr(
        routes,
        "resolve_project_path",
        lambda _value: tmp_path,
    )

    async with AsyncClient(
        transport=ASGITransport(app=_api_app()),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/index-jobs",
            files={"file": ("document.txt", b"content", "text/plain")},
        )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert captured["filename"] == "document.txt"
    assert str(captured["file_path"]).startswith(str(tmp_path))


def test_document_id_is_stable_across_upload_temp_names(tmp_path) -> None:
    first = tmp_path / "first-upload.txt"
    second = tmp_path / "different-job-prefix.txt"
    first.write_text("same parsed content", encoding="utf-8")
    second.write_text("same parsed content", encoding="utf-8")

    parser = DocumentParser()
    first_document = parser.parse(first)
    second_document = parser.parse(second)

    assert first_document.doc_id == second_document.doc_id
    assert len(first_document.doc_id) == 24


def test_bm25_replaces_all_chunks_for_existing_document(tmp_path) -> None:
    retriever = BM25Retriever(index_dir=str(tmp_path / "bm25"))
    retriever.build_index(
        [
            {"id": "old-1", "text": "old one", "doc_id": "doc-a"},
            {"id": "old-2", "text": "old two", "doc_id": "doc-a"},
            {"id": "other", "text": "keep", "doc_id": "doc-b"},
        ]
    )

    retriever.replace_document(
        "doc-a",
        [{"id": "new-1", "text": "new", "doc_id": "doc-a"}],
    )

    assert retriever.doc_ids == ["other", "new-1"]
    assert [metadata["doc_id"] for metadata in retriever.metadatas] == [
        "doc-b",
        "doc-a",
    ]
    assert retriever.count_document("doc-a") == 1
    assert retriever.count_document("doc-b") == 1


@pytest.mark.asyncio
async def test_reusable_document_requires_complete_vector_and_bm25_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        indexing_service,
        "get_document",
        lambda _doc_id: {
            "doc_id": "doc",
            "status": "indexed",
            "chunk_count": 2,
            "index_signature": "signature",
        },
    )

    class VectorStore:
        async def count(self, filters=None):
            return 2

    class BM25:
        def count_document(self, _doc_id):
            return 1

    monkeypatch.setattr(
        "mindforge.retrieval.vector_store.get_vector_store",
        lambda: VectorStore(),
    )
    monkeypatch.setattr(
        "mindforge.retrieval.service.get_bm25_retriever",
        lambda: BM25(),
    )

    reusable = await indexing_service.get_reusable_document(
        doc_id="doc",
        index_signature="signature",
    )

    assert reusable is None


@pytest.mark.asyncio
async def test_health_snapshot_does_not_probe_on_each_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = HealthMonitor()
    expected = HealthSnapshot(True, True, True)
    monitor._snapshot = expected

    async def unexpected_refresh():
        raise AssertionError("Cached health reads must not probe dependencies")

    monkeypatch.setattr(monitor, "refresh", unexpected_refresh)

    assert await monitor.get_snapshot() is expected


@pytest.mark.asyncio
async def test_episodic_store_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = EpisodicMemory(redis_client=None)

    def slow_add_episode(**_kwargs) -> None:
        time.sleep(0.08)

    monkeypatch.setattr(memory, "add_episode", slow_add_episode)
    loop_progressed = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.call_later(0.01, loop_progressed.set)

    store_task = asyncio.create_task(
        memory.store("task", {"output": "result"})
    )
    await asyncio.wait_for(loop_progressed.wait(), timeout=0.05)
    await store_task


class _ProbeAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "probe"

    @property
    def system_prompt(self) -> str:
        return "Return a useful answer."


def test_llm_adapters_reject_missing_api_keys_as_configuration_errors() -> None:
    with pytest.raises(LLMConfigurationError):
        DeepSeekAdapter(api_key="")
    with pytest.raises(LLMConfigurationError):
        OpenAIAdapter(api_key="")


@pytest.mark.asyncio
async def test_agent_empty_llm_response_is_not_reported_as_success() -> None:
    class EmptyLLM:
        _model = "empty"

        async def chat(self, *args, **kwargs) -> ChatResult:
            return ChatResult(content="")

    result = await _ProbeAgent(llm=EmptyLLM())._run_tool_loop("test")

    assert result.success is False
    assert result.output == ""


@pytest.mark.asyncio
async def test_synthesizer_empty_llm_response_is_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyLLM:
        _model = "empty"

    async def empty_chat(self, *args, **kwargs) -> ChatResult:
        return ChatResult(content=" \n ")

    monkeypatch.setattr(
        "mindforge.models.base.LLMFactory.create",
        lambda *args, **kwargs: EmptyLLM(),
    )
    monkeypatch.setattr(SynthesizerAgent, "_chat", empty_chat)

    result = await SynthesizerAgent(llm=EmptyLLM()).synthesize(
        task="test",
        subtask_results=[
            {"task_id": "one", "description": "one", "output": "finding"}
        ],
    )

    assert result.success is False


@pytest.mark.asyncio
async def test_synthesizer_stream_awaits_llm_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StreamingLLM:
        _model = "streaming"

        async def chat(self, *args, **kwargs):
            async def events():
                yield StreamEvent(type="chunk", content="part one")
                yield StreamEvent(type="chunk", content=" part two")
                yield StreamEvent(type="done")

            return events()

    llm = StreamingLLM()
    monkeypatch.setattr(
        "mindforge.models.base.LLMFactory.create",
        lambda *args, **kwargs: llm,
    )

    chunks = [
        chunk
        async for chunk in SynthesizerAgent(llm=llm).synthesize_stream(
            task="test",
            subtask_results=[
                {
                    "task_id": "one",
                    "description": "one",
                    "output": "finding",
                }
            ],
        )
    ]

    assert chunks == ["part one", " part two"]


def test_web_search_validates_runtime_arguments() -> None:
    tool = WebSearchTool(tavily_client=SimpleNamespace())

    invalid_calls = [
        {"query": "test", "max_results": 0},
        {"query": "test", "max_results": 21},
        {"query": "test", "max_results": True},
        {"query": "test", "search_depth": "invalid"},
        {"query": "test", "include_answer": "yes"},
        {"query": "test", "include_domains": "example.com"},
        {"query": 123},
    ]

    for kwargs in invalid_calls:
        result = tool.safe_execute(**kwargs)
        assert result.success is False
        assert result.error


def test_web_search_uses_injected_tavily_client_without_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def search(self, **kwargs):
            captured.update(kwargs)
            return {"results": []}

    monkeypatch.setattr("mindforge.tools.web_search.TavilyClient", None)
    result = WebSearchTool(tavily_client=FakeClient()).execute(
        query="mindforge",
        max_results=3,
    )

    assert result.success is True
    assert result.data["backend"] == "tavily"
    assert captured["max_results"] == 3


def test_duckduckgo_parser_handles_redirect_links_and_html_entities() -> None:
    html = """
    <div class="result__body">
      <a class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs">
        MindForge &amp; RAG
      </a>
      <a class="result__snippet">Production &lt;guide&gt;</a>
    </div>
    """

    results = WebSearchTool()._parse_ddg_html(html, max_results=5)

    assert results == [
        {
            "title": "MindForge & RAG",
            "url": "https://example.com/docs",
            "content": "Production <guide>",
        }
    ]


@pytest.mark.asyncio
async def test_graphrag_query_cpu_work_runs_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread = threading.get_ident()
    observed_thread = 0

    def probe_cut(text: str):
        nonlocal observed_thread
        observed_thread = threading.get_ident()
        return [text]

    monkeypatch.setattr("jieba.cut_for_search", probe_cut)
    engine = GraphRAGEngine()
    engine.entities["mindforge"] = Entity(
        id="mindforge",
        name="MindForge",
        description="research assistant",
    )

    await engine.query("MindForge")

    assert observed_thread != event_loop_thread


def test_graphrag_samples_across_entire_document() -> None:
    text = "A" * 4_000 + "MIDDLE" + "B" * 4_000 + "TAIL"

    sampled = GraphRAGEngine._sample_document_text([text], 3_000)

    assert len(sampled) <= 3_000
    assert sampled.startswith("A")
    assert "MIDDLE" in sampled
    assert sampled.endswith("TAIL")


@pytest.mark.asyncio
async def test_graphrag_reuses_unchanged_community_summary() -> None:
    summary_calls = 0

    class FakeLLM:
        async def chat(self, messages, **kwargs):
            nonlocal summary_calls
            del kwargs
            prompt = messages[0].content
            if prompt.startswith("Extract entities"):
                entity_id = "alpha" if "alpha document" in prompt else "beta"
                return SimpleNamespace(
                    content=(
                        '[{"type":"entity","id":"'
                        f'{entity_id}","name":"{entity_id}",'
                        '"entity_type":"concept","description":"stable"}]'
                    )
                )
            summary_calls += 1
            return SimpleNamespace(content="stable summary")

    engine = GraphRAGEngine(llm_fn=FakeLLM())
    engine.min_community_size = 1

    await engine.build_graph(
        [{"doc_id": "doc-alpha", "content": "alpha document"}]
    )
    await engine.build_graph(
        [{"doc_id": "doc-beta", "content": "beta document"}]
    )

    assert summary_calls == 2
    assert len(engine.communities) == 2
    assert all(
        community.summary == "stable summary"
        for community in engine.communities
    )


@pytest.mark.asyncio
async def test_graphrag_query_remains_available_during_build() -> None:
    extraction_started = asyncio.Event()
    release_extraction = asyncio.Event()

    class SlowLLM:
        async def chat(self, messages, **kwargs):
            del kwargs
            prompt = messages[0].content
            if prompt.startswith("Extract entities"):
                extraction_started.set()
                await release_extraction.wait()
                return SimpleNamespace(content="[]")
            return SimpleNamespace(content="")

    engine = GraphRAGEngine(llm_fn=SlowLLM())
    engine.entities["existing"] = Entity(
        id="existing",
        name="Existing",
        description="available during build",
    )
    await engine.query("Existing")

    build_task = asyncio.create_task(
        engine.build_graph(
            [{"doc_id": "new-doc", "content": "new document"}]
        )
    )
    await extraction_started.wait()
    results = await asyncio.wait_for(
        engine.query("Existing"),
        timeout=0.5,
    )
    release_extraction.set()
    await build_task

    assert results
    assert results[0]["entity_id"] == "existing"


@pytest.mark.asyncio
async def test_graph_mode_runs_hybrid_and_graph_retrieval_concurrently() -> None:
    hybrid_started = asyncio.Event()
    graph_started = asyncio.Event()

    class Hybrid:
        async def retrieve(self, **kwargs):
            del kwargs
            hybrid_started.set()
            await graph_started.wait()
            return [{"id": "hybrid", "score": 0.8}]

    class Graph:
        async def query(self, **kwargs):
            del kwargs
            graph_started.set()
            await hybrid_started.wait()
            return [{"id": "graph", "score": 0.9}]

    retriever = AdaptiveRetriever(
        hybrid_retriever=Hybrid(),
        graph_engine=Graph(),
        reranker=None,
    )

    result = await asyncio.wait_for(
        retriever.retrieve("relationships", mode=QueryMode.GRAPH),
        timeout=0.5,
    )

    assert set(result["raw_results"]) == {"hybrid", "graph"}
    assert [item["id"] for item in result["results"]] == [
        "graph",
        "hybrid",
    ]


@pytest.mark.asyncio
async def test_rag_tool_auto_and_graph_modes_reach_adaptive_retriever() -> None:
    observed_modes: list[QueryMode | None] = []

    class Retriever:
        async def retrieve(self, **kwargs):
            observed_modes.append(kwargs["mode"])
            return {"results": []}

    tool = RAGTool(retriever=Retriever())
    await tool.execute_async("automatic query", mode="auto", threshold=0.1)
    await tool.execute_async("graph query", mode="graph", threshold=0.1)

    assert observed_modes == [None, QueryMode.GRAPH]


@pytest.mark.asyncio
async def test_orchestrator_research_queue_is_bounded() -> None:
    orchestrator = object.__new__(routes.Orchestrator)
    orchestrator._research_semaphore = asyncio.Semaphore(1)
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(queue_timeout=0.01)
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_run(task: str) -> AgentResult:
        started.set()
        await release.wait()
        return AgentResult(
            agent_name="orchestrator",
            success=True,
            output=task,
        )

    orchestrator._run_unlimited = slow_run
    first = asyncio.create_task(orchestrator.run("first"))
    await started.wait()
    second = await orchestrator.run("second")
    release.set()
    first_result = await first

    assert first_result.success is True
    assert second.success is False
    assert second.data["error"] == "research_queue_timeout"


@pytest.mark.asyncio
async def test_streaming_orchestrator_emits_planning_and_heartbeat() -> None:
    class SlowPlanner:
        async def run(self, task: str):
            await asyncio.sleep(0.04)
            from mindforge.agents.planner import ResearchPlan, SubTask

            return ResearchPlan(
                plan_id="slow-plan",
                original_task=task,
                subtasks=[SubTask(task_id="one", description=task)],
            )

    class Researcher:
        async def run(self, task: str, *, context=None):
            del context
            return AgentResult(
                agent_name="researcher",
                success=True,
                output=task,
            )

    from mindforge.agents.orchestrator import Orchestrator

    orchestrator = Orchestrator(
        planner=SlowPlanner(),
        researcher=Researcher(),
        synthesizer=SimpleNamespace(),
        critic=SimpleNamespace(),
    )
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(
            research_timeout=1,
            subtask_timeout=1,
            queue_timeout=1,
            sse_heartbeat_seconds=0.01,
            max_refine_rounds=0,
            stream_chunk_size=512,
        ),
        llm=SimpleNamespace(llm_provider="test"),
    )

    events = [
        event async for event in orchestrator.stream_run("heartbeat test")
    ]
    event_types = [event["type"] for event in events]

    assert event_types[0] == "planning"
    assert "heartbeat" in event_types
    assert event_types.index("heartbeat") < event_types.index("plan_ready")


@pytest.mark.asyncio
async def test_missing_llm_credentials_skip_orchestrator_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes, "has_llm_credentials", lambda: False)
    monkeypatch.setattr(
        routes,
        "get_orchestrator",
        lambda: (_ for _ in ()).throw(
            AssertionError("orchestrator must not initialize without a key")
        ),
    )

    async def retrieval_only(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            output="retrieval only",
            data={"quality": 1.0, "sources": []},
        )

    monkeypatch.setattr(RAGTool, "execute_async", retrieval_only)

    response = await routes.query(QueryRequest(task="test task"))

    assert response.report == "retrieval only"
    assert response.iterations == 0


@pytest.mark.asyncio
async def test_failed_agent_result_uses_retrieval_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedOrchestrator:
        async def run(self, task: str) -> AgentResult:
            return AgentResult(
                agent_name="orchestrator",
                success=False,
                output=f"agent failed: {task}",
            )

    monkeypatch.setattr(routes, "get_orchestrator", FailedOrchestrator)

    async def retrieval_fallback(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            output="retrieval fallback",
            data={"quality": 2.5, "sources": []},
        )

    monkeypatch.setattr(
        RAGTool,
        "execute_async",
        retrieval_fallback,
    )
    monkeypatch.setattr(
        RAGTool,
        "safe_execute",
        lambda self, **kwargs: (_ for _ in ()).throw(
            AssertionError("sync fallback must not run in a separate event loop")
        ),
    )

    response = await routes.query(QueryRequest(task="test task"))

    assert response.report == "retrieval fallback"
    assert response.quality_score == 2.5


@pytest.mark.asyncio
async def test_orchestrator_initialization_failure_uses_retrieval_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def failed_orchestrator():
        raise LLMConfigurationError("API key missing")

    monkeypatch.setattr(routes, "get_orchestrator", failed_orchestrator)
    monkeypatch.setattr(routes, "has_llm_credentials", lambda: True)
    caplog.set_level(logging.WARNING, logger=routes.__name__)

    async def retrieval_without_llm(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            output="retrieval without llm",
            data={"quality": 1.5, "sources": []},
        )

    monkeypatch.setattr(
        RAGTool,
        "execute_async",
        retrieval_without_llm,
    )
    monkeypatch.setattr(
        RAGTool,
        "safe_execute",
        lambda self, **kwargs: (_ for _ in ()).throw(
            AssertionError("sync fallback must not run in a separate event loop")
        ),
    )

    response = await routes.query(QueryRequest(task="test task"))

    assert response.report == "retrieval without llm"
    assert response.quality_score == 1.5
    assert any(
        record.levelno == logging.WARNING
        and "using retrieval fallback" in record.getMessage()
        for record in caplog.records
    )
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


@pytest.mark.asyncio
async def test_stream_unsuccessful_agent_result_uses_retrieval_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedOrchestrator:
        async def stream_run(self, task: str):
            yield {
                "type": "done",
                "result": AgentResult(
                    agent_name="orchestrator",
                    success=False,
                    output=f"failed: {task}",
                ),
            }

    class SuccessfulRAG:
        async def execute_async(self, **kwargs) -> ToolResult:
            return ToolResult(
                success=True,
                output="stream retrieval fallback",
                data={"quality": 1.0, "sources": []},
            )

    import mindforge.tools.rag_tool as rag_module

    monkeypatch.setattr(rag_module, "RAGTool", SuccessfulRAG)
    chunks = [
        chunk.decode("utf-8")
        async for chunk in routes._stream_response(
            FailedOrchestrator(),
            "test",
        )
    ]
    events = [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk.startswith("data: {")
    ]

    assert any(
        event.get("type") == "done"
        and event.get("result", {}).get("output")
        == "stream retrieval fallback"
        for event in events
    )
    assert not any(
        event.get("type") == "done"
        and event.get("result", {}).get("success") is False
        for event in events
    )


@pytest.mark.asyncio
async def test_episodic_recall_requires_exact_unexpired_task() -> None:
    memory = EpisodicMemory(redis_ttl=60)
    memory.add_episode(
        task="explain react hooks",
        result="OLD ANSWER",
        sources=[],
    )

    assert await memory.recall("explain react hooks security") is None

    memory._episodes.append(
        Episode(
            task="expired exact task",
            result="EXPIRED",
            sources=[],
            embedding=None,
            timestamp=time.time() - 61,
        )
    )
    assert await memory.recall("expired exact task") is None


def test_citation_verifier_rejects_unsupported_claim() -> None:
    result = CitationVerifier().execute(
        report_text="The moon is made of cheese [1].",
        sources=[
            {
                "index": 1,
                "title": "Python documentation",
                "content": "Python is a programming language.",
            }
        ],
        strict_unused=False,
    )

    assert result.success is False
    assert result.data["validity_score"] == 0.0
    assert result.data["issues"][0]["type"] == "unsupported_claim"


def test_citation_verifier_accepts_lexically_supported_claim() -> None:
    result = CitationVerifier().execute(
        report_text="Python supports asynchronous programming [1].",
        sources=[
            {
                "index": 1,
                "title": "Python asyncio documentation",
                "content": (
                    "The asyncio package provides infrastructure for "
                    "asynchronous programming in Python."
                ),
            }
        ],
        strict_unused=False,
    )

    assert result.success is True
    assert result.data["validity_score"] == 1.0


@pytest.mark.asyncio
async def test_raptor_reuses_precomputed_leaf_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingEmbedder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

    embedder = CountingEmbedder()
    chunks = [
        DocumentChunk(
            chunk_id=f"chunk-{index}",
            doc_id="doc",
            content=f"content {index}",
            embedding=[1.0, float(index)],
        )
        for index in range(6)
    ]
    indexer = RAPTORIndexer(embedder=embedder, llm=None)
    monkeypatch.setattr(indexer, "num_levels", 2)
    monkeypatch.setattr(indexer, "threshold", -1.0)

    nodes = await indexer.build_tree(chunks)

    summary_nodes = [node for node in nodes if node.level > 0]
    assert embedder.calls == [
        [node.content for node in summary_nodes]
    ]
    assert all(node.embedding is not None for node in summary_nodes)


@pytest.mark.asyncio
async def test_raptor_summary_ids_include_source_chunk_identity() -> None:
    class FakeEmbedder:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_single(self, text):
            return [1.0, 0.0]

    async def fixed_summary(prompt: str) -> str:
        return "same summary"

    first_chunks = [
        DocumentChunk(
            chunk_id=f"first-{index}",
            doc_id="first",
            content=f"first content {index}",
        )
        for index in range(4)
    ]
    second_chunks = [
        DocumentChunk(
            chunk_id=f"second-{index}",
            doc_id="second",
            content=f"second content {index}",
        )
        for index in range(4)
    ]
    indexer = RAPTORIndexer(
        embedder=FakeEmbedder(),
        llm=fixed_summary,
    )
    indexer.num_levels = 2

    first_nodes = await indexer.build_tree(first_chunks)
    second_nodes = await indexer.build_tree(second_chunks)
    first_summary_ids = {
        node.node_id for node in first_nodes if node.level > 0
    }
    second_summary_ids = {
        node.node_id for node in second_nodes if node.level > 0
    }

    assert first_summary_ids
    assert first_summary_ids.isdisjoint(second_summary_ids)


@pytest.mark.asyncio
async def test_raptor_does_not_summarize_singleton_clusters() -> None:
    summary_calls = 0

    async def summarize(prompt: str) -> str:
        nonlocal summary_calls
        summary_calls += 1
        return prompt

    chunks = [
        DocumentChunk(
            chunk_id=f"chunk-{index}",
            doc_id="doc",
            content=f"content {index}",
            embedding=[1.0, float(index)],
        )
        for index in range(6)
    ]
    indexer = RAPTORIndexer(llm=summarize)
    indexer.num_levels = 3
    indexer.threshold = 2.0

    nodes = await indexer.build_tree(chunks)

    assert summary_calls == 0
    assert nodes == [
        node for node in nodes if node.level == 0
    ]


@pytest.mark.asyncio
async def test_raptor_checks_node_limit_before_summary_calls() -> None:
    summary_calls = 0

    async def summarize(prompt: str) -> str:
        nonlocal summary_calls
        summary_calls += 1
        return prompt

    chunks = [
        DocumentChunk(
            chunk_id=f"chunk-{index}",
            doc_id="doc",
            content=f"content {index}",
            embedding=[1.0, 0.0],
        )
        for index in range(6)
    ]
    indexer = RAPTORIndexer(llm=summarize)
    indexer.num_levels = 2
    indexer.max_nodes = 6

    with pytest.raises(ValueError, match="node limit"):
        await indexer.build_tree(chunks)

    assert summary_calls == 0


def test_index_points_reject_vector_count_mismatch() -> None:
    chunks = [
        DocumentChunk("one", "doc", "first"),
        DocumentChunk("two", "doc", "second"),
    ]

    with pytest.raises(ValueError, match="vector count"):
        routes._build_chunk_points(
            chunks=chunks,
            vectors=[[1.0, 0.0]],
            doc_id="doc",
            source="source.txt",
            expected_dimension=2,
        )


def test_index_points_preserve_full_content_and_metadata() -> None:
    content = "x" * 2_048
    chunk = DocumentChunk(
        "one",
        "doc",
        content,
        metadata={"chunk_start": 0, "chunk_end": len(content), "team": "rag"},
    )

    points = routes._build_chunk_points(
        chunks=[chunk],
        vectors=[[1.0, 0.0]],
        doc_id="doc",
        source="source.txt",
        expected_dimension=2,
    )

    assert points[0].payload["content"] == content
    assert points[0].payload["metadata"]["team"] == "rag"


def test_document_content_reconstruction_removes_overlap() -> None:
    chunks = [
        {"content": "abcdef", "chunk_start": 0, "chunk_end": 6},
        {"content": "defghi", "chunk_start": 3, "chunk_end": 9},
    ]

    assert routes._reconstruct_document_content(chunks) == "abcdefghi"


def test_fixed_chunks_round_trip_without_losing_whitespace() -> None:
    original = "alpha beta gamma\n\ndelta epsilon"
    chunks = TextSplitter(chunk_size=12, chunk_overlap=4).split(
        "doc",
        original,
    )
    payloads = [
        {
            "content": chunk.content,
            "chunk_index": index,
            **chunk.metadata,
        }
        for index, chunk in enumerate(chunks)
    ]

    assert routes._reconstruct_document_content(payloads) == original


def test_large_pdf_parser_does_not_share_pdf_objects_across_threads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakePage:
        def __init__(self, index: int, owner_thread: int) -> None:
            self.index = index
            self.owner_thread = owner_thread

        def extract_text(self) -> str:
            if threading.get_ident() != self.owner_thread:
                raise RuntimeError("PDF object used from another thread")
            return f"page {self.index}"

    class FakePDF:
        def __init__(self) -> None:
            owner_thread = threading.get_ident()
            self.pages = [
                FakePage(index, owner_thread)
                for index in range(12)
            ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    fake_pdfplumber = SimpleNamespace(open=lambda path: FakePDF())
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)
    pdf_path = tmp_path / "large.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    parser = DocumentParser()
    parser._limits = SimpleNamespace(
        max_pdf_pages=600,
        max_parsed_chars=5_000_000,
        pdf_parallel_page_threshold=10,
        pdf_parse_workers=4,
        pdf_parse_executor="thread",
    )
    content, sections, metadata = parser._parse_pdf(pdf_path)

    assert "page 0" in content
    assert "page 11" in content
    assert len(sections) == 12
    assert metadata["pages"] == 12


def test_pdf_page_limit_reports_detected_and_configured_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakePDF:
        pages = [object()] * 523

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda _path: FakePDF()),
    )
    parser = DocumentParser()
    parser._limits = SimpleNamespace(max_pdf_pages=500)
    pdf_path = tmp_path / "too-many-pages.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    with pytest.raises(DocumentLimitError, match=r"523.*500"):
        parser._parse_pdf(pdf_path)


@pytest.mark.asyncio
async def test_parser_limit_error_is_returned_as_http_413() -> None:
    class Parser:
        def parse(self, _path):
            raise DocumentLimitError(
                "PDF 共 523 页，超过当前上限 500 页。"
            )

    with pytest.raises(HTTPException) as exc_info:
        await routes._parse_document_file(Parser(), "document.pdf")

    assert exc_info.value.status_code == 413
    assert "523" in str(exc_info.value.detail)


def test_history_datetime_is_serialized_as_utc() -> None:
    naive = datetime(2026, 6, 24, 11, 42, 38)
    aware = datetime(
        2026,
        6,
        24,
        11,
        42,
        38,
        tzinfo=timezone.utc,
    )

    assert routes._serialize_datetime_utc(naive).endswith("Z")
    assert routes._serialize_datetime_utc(aware).endswith("Z")


@pytest.mark.asyncio
async def test_mcp_http_endpoint_is_not_exposed() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=_api_app()),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/v1/mcp")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_unknown_api_route_returns_json_404() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=server.app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/v1/not-a-real-endpoint")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_health_schema_has_no_mcp_fields() -> None:
    payload = routes.HealthResponse().model_dump()
    assert not any(key.startswith("mcp_") for key in payload)


def test_runtime_retrieval_limits_are_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    from mindforge.retrieval import service

    captured: dict[str, object] = {}

    class FakeAdaptiveRetriever:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    settings = SimpleNamespace(
        retrieval=SimpleNamespace(
            reranker_model="",
            reranker_model_revision=None,
            reranker_max_candidates=100,
            max_request_top_k=50,
            vector_top_k=24,
            rerank_top_k=7,
        ),
        graphrag=SimpleNamespace(graph_enabled=False),
    )
    monkeypatch.setattr(service, "_retriever", None)
    monkeypatch.setattr(service, "get_settings", lambda: settings)
    monkeypatch.setattr(service, "get_embedder", lambda: SimpleNamespace())
    monkeypatch.setattr(service, "get_vector_store", lambda: SimpleNamespace())
    monkeypatch.setattr(service, "get_bm25_retriever", lambda: SimpleNamespace())
    monkeypatch.setattr(service, "AdaptiveRetriever", FakeAdaptiveRetriever)

    service.get_retriever()

    assert captured["retrieval_top_k"] == 24
    assert captured["rerank_top_k"] == 7


def test_embedding_provider_change_requires_empty_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(embedding_provider="bge")
        ),
    )
    monkeypatch.setattr(
        routes,
        "get_vector_store",
        lambda: SimpleNamespace(get_point_count=lambda: 3),
    )

    with pytest.raises(routes.HTTPException) as exc_info:
        routes._update_settings_locked(
            SettingsUpdateRequest(embedding_provider="openai")
        )

    assert exc_info.value.status_code == 409


def test_settings_response_exposes_unified_provider_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ApiKey:
        is_active = object()

    rows = [
        SimpleNamespace(
            provider="openai_compatible",
            key_encrypted="encrypted-cloud-key",
        )
    ]

    class Query:
        def filter(self, *args):
            del args
            return self

        def all(self):
            return rows

    class Session:
        def query(self, model):
            assert model is ApiKey
            return Query()

        def close(self):
            return None

    fake_db = SimpleNamespace(
        ApiKey=ApiKey,
        SessionLocal=Session,
        decrypt_api_key=lambda value: (
            "cloud-secret-1234"
            if value == "encrypted-cloud-key"
            else ""
        ),
    )
    settings = SimpleNamespace(
        llm=LLMConfig(
            llm_provider="local",
            compatible_base_url="https://cloud.example/v1",
            compatible_model="cloud-model",
            local_base_url="http://host.docker.internal:11434/v1",
            local_model="qwen3",
            local_api_key_required=False,
        ),
        retrieval=SimpleNamespace(vector_top_k=20, rerank_top_k=6),
        agent=SimpleNamespace(
            max_iterations=3,
            max_refine_rounds=1,
            critic_threshold=7.0,
            subtask_timeout=30,
            research_timeout=180,
        ),
    )
    monkeypatch.setitem(sys.modules, "mindforge.db", fake_db)
    monkeypatch.setattr(
        "mindforge.config.get_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        routes,
        "has_llm_credentials",
        lambda provider=None: provider == "local",
    )

    response = routes.get_settings_api()
    providers = {
        item.provider: item for item in response.llm_providers
    }

    assert set(providers) == {
        "openai",
        "deepseek",
        "openai_compatible",
        "local",
    }
    assert response.llm_provider == "local"
    assert response.llm_configured is True
    assert providers["local"].configured is True
    assert providers["local"].api_key_required is False
    assert providers["local"].default_model == "qwen3"
    assert providers["openai_compatible"].api_key == "***1234"


def test_settings_update_persists_multiple_provider_configs_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Column:
        def __eq__(self, other):
            del other
            return True

    class ApiKey:
        provider = Column()

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Query:
        def filter(self, *args):
            del args
            return self

        def first(self):
            return None

    class Session:
        def query(self, model):
            assert model is ApiKey
            return Query()

        def add(self, row):
            del row

        def delete(self, row):
            del row

        def commit(self):
            return None

        def rollback(self):
            return None

        def close(self):
            return None

    fake_db = SimpleNamespace(
        ApiKey=ApiKey,
        SessionLocal=Session,
        encrypt_api_key=lambda value: f"encrypted:{value}",
        get_default_user_id=lambda db: 1,
    )
    captured: dict[str, str] = {}
    environment_keys = {
        "LLM_COMPATIBLE_API_KEY",
        "LLM_COMPATIBLE_BASE_URL",
        "LLM_COMPATIBLE_MODEL",
        "LLM_COMPATIBLE_SUPPORTS_JSON_SCHEMA",
        "LLM_LOCAL_API_KEY",
        "LLM_LOCAL_BASE_URL",
        "LLM_LOCAL_MODEL",
        "LLM_LOCAL_API_KEY_REQUIRED",
        "LLM_LLM_PROVIDER",
    }
    for key in environment_keys:
        monkeypatch.setenv(key, "previous")

    monkeypatch.setitem(sys.modules, "mindforge.db", fake_db)
    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(embedding_provider="bge")
        ),
    )
    monkeypatch.setattr(
        routes,
        "_snapshot_env_file",
        lambda keys: {key: (False, "") for key in keys},
    )
    monkeypatch.setattr(
        routes,
        "_sync_env_file",
        lambda updates: captured.update(updates),
    )
    monkeypatch.setattr(
        "mindforge.config.reload_settings",
        lambda: None,
    )
    monkeypatch.setattr(routes, "reset_runtime_components", lambda: None)

    result = routes._update_settings_locked(
        SettingsUpdateRequest(
            llm_provider="local",
            llm_provider_configs=[
                LLMProviderUpdate(
                    provider="openai_compatible",
                    base_url="https://cloud.example/v1",
                    api_key="cloud-key",
                    default_model="cloud-model",
                    supports_json_schema=True,
                ),
                LLMProviderUpdate(
                    provider="local",
                    base_url="http://host.docker.internal:8001/v1",
                    api_key="",
                    api_key_required=False,
                    default_model="qwen3",
                ),
            ],
        )
    )

    assert result == {"status": "saved"}
    assert captured["LLM_LLM_PROVIDER"] == "local"
    assert captured["LLM_COMPATIBLE_API_KEY"] == "cloud-key"
    assert (
        captured["LLM_COMPATIBLE_BASE_URL"]
        == "https://cloud.example/v1"
    )
    assert captured["LLM_COMPATIBLE_MODEL"] == "cloud-model"
    assert captured["LLM_COMPATIBLE_SUPPORTS_JSON_SCHEMA"] == "true"
    assert captured["LLM_LOCAL_API_KEY"] == ""
    assert (
        captured["LLM_LOCAL_BASE_URL"]
        == "http://host.docker.internal:8001/v1"
    )
    assert captured["LLM_LOCAL_MODEL"] == "qwen3"
    assert captured["LLM_LOCAL_API_KEY_REQUIRED"] == "false"


def test_runtime_component_reset_continues_after_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mindforge.ingestion import embedder
    from mindforge.retrieval import service, vector_store

    calls: list[str] = []

    class BrokenOrchestrator:
        def close(self) -> None:
            calls.append("orchestrator")
            raise RuntimeError("close failed")

    monkeypatch.setattr(routes, "_orchestrator", BrokenOrchestrator())
    monkeypatch.setattr(
        embedder,
        "reset_embedder",
        lambda: calls.append("embedder"),
    )
    monkeypatch.setattr(
        vector_store,
        "reset_vector_store",
        lambda: calls.append("vector_store"),
    )
    monkeypatch.setattr(
        service,
        "reset_retrieval_service",
        lambda: calls.append("retrieval"),
    )

    routes.reset_runtime_components()

    assert routes._orchestrator is None
    assert calls == [
        "orchestrator",
        "embedder",
        "vector_store",
        "retrieval",
    ]


def test_explicit_embedding_provider_never_silently_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(self) -> None:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(EmbeddingManager, "_init_openai", unavailable)

    with pytest.raises(RuntimeError, match="Refusing"):
        EmbeddingManager(provider="openai", dim=2)


def test_bm25_runtime_dependencies_are_available() -> None:
    import bm25s  # noqa: F401
    import jieba  # noqa: F401
    from mindforge.retrieval import bm25

    assert bm25._BM25S_AVAILABLE is True


@pytest.mark.asyncio
async def test_stream_fallback_does_not_report_failed_retrieval_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenOrchestrator:
        async def stream_run(self, task: str):
            if False:
                yield {"task": task}
            raise RuntimeError("agent failed")

    class FailedRAG:
        async def execute_async(self, **kwargs) -> ToolResult:
            return ToolResult(success=False, error="retrieval failed")

        def safe_execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=False, error="retrieval failed")

    import mindforge.tools.rag_tool as rag_module

    monkeypatch.setattr(rag_module, "RAGTool", FailedRAG)
    chunks = [
        chunk.decode("utf-8")
        async for chunk in routes._stream_response(
            BrokenOrchestrator(),
            "test",
        )
    ]
    events = [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk.startswith("data: {")
    ]

    assert any(event.get("type") == "error" for event in events)
    assert not any(
        event.get("type") == "done"
        and event.get("result", {}).get("success") is True
        for event in events
    )


@pytest.mark.asyncio
async def test_sync_retrieval_work_runs_off_the_event_loop() -> None:
    event_loop_thread = threading.get_ident()
    observed_threads: dict[str, int] = {}

    class ProbeBM25:
        def search(self, **kwargs):
            observed_threads["bm25"] = threading.get_ident()
            return []

    class ProbeReranker:
        def rerank(self, **kwargs):
            observed_threads["reranker"] = threading.get_ident()
            return kwargs["candidates"]

    hybrid = HybridRetriever(bm25_retriever=ProbeBM25())
    await hybrid.retrieve("test")

    class FakeHybrid:
        async def retrieve(self, **kwargs):
            return [{"id": "one", "text": "result", "score": 1.0}]

    adaptive = AdaptiveRetriever(
        hybrid_retriever=FakeHybrid(),
        reranker=ProbeReranker(),
    )
    await adaptive.retrieve("test", mode=QueryMode.FACTUAL)

    assert observed_threads["bm25"] != event_loop_thread
    assert observed_threads["reranker"] != event_loop_thread


@pytest.mark.asyncio
async def test_rag_retriever_initialization_runs_off_the_event_loop() -> None:
    event_loop_thread = threading.get_ident()
    observed_thread = 0

    class FakeRetriever:
        async def retrieve(self, **kwargs):
            return {"results": []}

    class ProbeRAGTool(RAGTool):
        def _get_retriever(self):
            nonlocal observed_thread
            observed_thread = threading.get_ident()
            return FakeRetriever()

    result = await ProbeRAGTool().execute_async(
        query="test",
        threshold=0.1,
    )

    assert result.success is True
    assert observed_thread != event_loop_thread


def _parser_limits() -> SimpleNamespace:
    return SimpleNamespace(
        max_pdf_pages=20,
        max_parsed_chars=5_000_000,
        pdf_parallel_page_threshold=10,
        pdf_parse_workers=2,
        pdf_parse_executor="thread",
    )


def _parser_config(**overrides) -> SimpleNamespace:
    values = {
        "mode": "auto",
        "ocr_enabled": True,
        "ocr_language": "ch",
        "ocr_device": "cpu",
        "ocr_model_source": "BOS",
        "ocr_enable_mkldnn": False,
        "ocr_dpi": 200,
        "ocr_min_native_text_chars": 30,
        "ocr_min_printable_ratio": 0.65,
        "ocr_max_pages": 20,
        "layout_enabled": False,
        "table_extraction_enabled": False,
        "table_max_cells": 10_000,
        "image_extraction_enabled": False,
        "image_max_per_page": 20,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_native_pdf_page_does_not_run_ocr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakePage:
        images: list[dict] = []

        @staticmethod
        def extract_text() -> str:
            return "This is a native PDF page with enough text for extraction."

    class FakePDF:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    from mindforge.ingestion import parsers as parser_module

    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda _path: FakePDF()),
    )
    ocr_calls: list[int] = []
    monkeypatch.setattr(
        parser_module._PaddleOCRAdapter,
        "extract",
        lambda _self, _image, *, page: ocr_calls.append(page) or [],
    )
    parser = DocumentParser()
    parser._limits = _parser_limits()
    parser._parser_config = _parser_config()
    pdf_path = tmp_path / "native.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    content, _, metadata, elements, _ = parser._parse_pdf_structured(pdf_path)

    assert "native PDF page" in content
    assert metadata["ocr_pages"] == 0
    assert ocr_calls == []
    assert elements[0].source_method == "native_text"


def test_scanned_pdf_page_falls_back_to_ocr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakePage:
        images: list[dict] = []

        @staticmethod
        def extract_text() -> str:
            return ""

        @staticmethod
        def to_image(*, resolution: int):
            assert resolution == 200
            return SimpleNamespace(original=SimpleNamespace())

    class FakePDF:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    from mindforge.ingestion import parsers as parser_module

    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda _path: FakePDF()),
    )
    monkeypatch.setattr(
        parser_module._PaddleOCRAdapter,
        "extract",
        lambda _self, _image, *, page: [
            DocumentElement(
                kind="text",
                content="OCR extracted content",
                page=page,
                bbox=(1.0, 2.0, 10.0, 12.0),
                confidence=0.98,
                source_method="ocr",
            )
        ],
    )
    parser = DocumentParser()
    parser._limits = _parser_limits()
    parser._parser_config = _parser_config()
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    content, _, metadata, elements, _ = parser._parse_pdf_structured(pdf_path)

    assert content == "OCR extracted content"
    assert metadata["ocr_pages"] == 1
    assert elements[0].page == 1
    assert elements[0].confidence == 0.98
    assert elements[0].source_method == "ocr"


def test_native_pdf_table_becomes_markdown_element(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakeTable:
        bbox = (10.0, 20.0, 120.0, 80.0)

        @staticmethod
        def extract():
            return [["Name", "Score"], ["Ada", "100"]]

    class FakePage:
        images: list[dict] = []

        @staticmethod
        def extract_text() -> str:
            return "Name  Score\nAda  100\nGrace  99"

        @staticmethod
        def find_tables():
            return [FakeTable()]

    class FakePDF:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda _path: FakePDF()),
    )
    parser = DocumentParser()
    parser._limits = _parser_limits()
    parser._parser_config = _parser_config(
        table_extraction_enabled=True,
        ocr_min_native_text_chars=1,
    )
    pdf_path = tmp_path / "table.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    _, _, metadata, elements, _ = parser._parse_pdf_structured(pdf_path)
    table_elements = [element for element in elements if element.kind == "table"]

    assert metadata["table_count"] == 1
    assert table_elements[0].content == (
        "| Name | Score |\n| --- | --- |\n| Ada | 100 |"
    )
    assert table_elements[0].source_method == "native_table"
    assert table_elements[0].metadata["table_html"].startswith("<table>")
    assert table_elements[0].metadata["table_cells"] == [
        {
            "row": 0,
            "column": 0,
            "text": "Name",
            "rowspan": 1,
            "colspan": 1,
            "header": True,
            "is_merged": False,
        },
        {
            "row": 0,
            "column": 1,
            "text": "Score",
            "rowspan": 1,
            "colspan": 1,
            "header": True,
            "is_merged": False,
        },
        {
            "row": 1,
            "column": 0,
            "text": "Ada",
            "rowspan": 1,
            "colspan": 1,
            "header": False,
            "is_merged": False,
        },
        {
            "row": 1,
            "column": 1,
            "text": "100",
            "rowspan": 1,
            "colspan": 1,
            "header": False,
            "is_merged": False,
        },
    ]


def test_element_aware_chunks_preserve_parser_metadata() -> None:
    elements = [
        DocumentElement(
            kind="text",
            content="Native page text",
            page=1,
            source_method="native_text",
            start=0,
            end=16,
        ),
        DocumentElement(
            kind="table",
            content="| Name |\n| --- |\n| Ada |",
            page=2,
            bbox=(1.0, 2.0, 30.0, 40.0),
            confidence=0.91,
            source_method="ocr_table",
            start=18,
            end=45,
        ),
    ]

    chunks = ElementAwareSplitter().split("doc", elements)

    assert len(chunks) == 2
    assert chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["source_method"] == "native_text"
    assert chunks[1].metadata["element_type"] == "table"
    assert chunks[1].metadata["bbox"] == [1.0, 2.0, 30.0, 40.0]
    assert chunks[1].metadata["ocr_confidence"] == 0.91


def test_element_chunks_exclude_large_table_structure_payloads() -> None:
    element = DocumentElement(
        kind="table",
        content="| Name |\n| --- |\n| Ada |",
        page=1,
        source_method="native_table",
        metadata={
            "table_html": "<table><tr><td>Ada</td></tr></table>",
            "table_cells": [{"row": 0, "column": 0, "text": "Ada"}],
            "asset_id": "asset",
        },
    )

    chunk = ElementAwareSplitter().split("doc", [element])[0]

    assert chunk.metadata["asset_id"] == "asset"
    assert "table_html" not in chunk.metadata
    assert "table_cells" not in chunk.metadata


def test_native_layout_orders_detected_columns_before_right_column() -> None:
    from mindforge.ingestion.parsers import (
        _group_native_blocks,
        _group_native_lines,
    )

    words = [
        {"text": "left-one", "x0": 10, "top": 10, "x1": 55, "bottom": 20},
        {"text": "right-one", "x0": 300, "top": 10, "x1": 355, "bottom": 20},
        {"text": "left-two", "x0": 10, "top": 32, "x1": 55, "bottom": 42},
        {"text": "right-two", "x0": 300, "top": 32, "x1": 355, "bottom": 42},
    ]

    blocks = _group_native_blocks(_group_native_lines(words))

    assert [block["content"] for block in blocks] == [
        "left-one\nleft-two",
        "right-one\nright-two",
    ]


def test_parser_cancellation_is_exposed_as_a_typed_error(tmp_path) -> None:
    source = tmp_path / "cancelled.txt"
    source.write_text("content", encoding="utf-8")
    parser = DocumentParser()
    parser.set_cancellation_callback(lambda: True)

    with pytest.raises(DocumentParserCancelledError):
        parser.parse(source)


def test_asset_persistence_records_source_and_table_structure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from mindforge.services import document_assets as asset_service

    source = tmp_path / "source.txt"
    source.write_text("source content", encoding="utf-8")
    parsed = SimpleNamespace(
        doc_id="a" * 24,
        filename="source.txt",
        metadata={},
        elements=[
            DocumentElement(
                kind="table",
                content="| Name |\n| --- |\n| Ada |",
                page=1,
                source_method="native_table",
                metadata={
                    "table_html": "<table><tr><td>Ada</td></tr></table>",
                    "table_cells": [
                        {"row": 0, "column": 0, "text": "Ada"}
                    ],
                    "row_count": 2,
                    "column_count": 1,
                },
            )
        ],
    )
    parser_config = SimpleNamespace(
        asset_persistence_enabled=True,
        source_retention_enabled=True,
        asset_storage_dir="assets",
        asset_dpi=144,
        asset_max_per_document=10,
        asset_max_total_mb=10,
    )
    monkeypatch.setattr(
        asset_service,
        "get_settings",
        lambda: SimpleNamespace(parser=parser_config),
    )
    monkeypatch.setattr(
        asset_service,
        "resolve_project_path",
        lambda _value: tmp_path / "assets",
    )
    captured: list[dict] = []
    monkeypatch.setattr(
        asset_service,
        "replace_document_assets",
        lambda _doc_id, assets: captured.extend(assets) or assets,
    )

    assets = asset_service.persist_document_assets(
        source_path=source,
        parsed=parsed,
    )

    assert len(assets) == 2
    assert (
        tmp_path / "assets" / ("a" * 24) / "source" / "source.txt"
    ).is_file()
    assert parsed.elements[0].metadata["asset_id"]
    table_asset = next(asset for asset in captured if asset["kind"] == "table")
    assert table_asset["metadata"]["cells"][0]["text"] == "Ada"
