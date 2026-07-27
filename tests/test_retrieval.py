"""Test RRF fusion algorithm, adaptive retriever strategy selection, and GraphRAG."""

from __future__ import annotations

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from mindforge.agents.base import AgentResult
from mindforge.agents.critic import CriticScore
from mindforge.agents.orchestrator import Orchestrator
from mindforge.agents.planner import ResearchPlan, SubTask
from mindforge.api import routes
from mindforge.api import server
from mindforge.api.schemas import HistorySaveRequest, SettingsUpdateRequest
from mindforge.ingestion.embedder import EmbeddingManager
from mindforge.mcp.registry import MCPRegistry
from mindforge.retrieval.graphrag import Entity, GraphRAGEngine
from mindforge.retrieval.hybrid import HybridRetriever
from mindforge.retrieval.reranker import CrossEncoderReranker
from mindforge.tools.code_executor import CodeExecutor
from mindforge.tools.rag_tool import RAGTool


def test_settings_api_keys_reject_control_characters():
    with pytest.raises(ValueError, match="control characters"):
        SettingsUpdateRequest(deepseek_api_key="valid-prefix\nINJECTED=value")


def test_env_sync_quotes_values_and_prevents_entry_injection(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(routes, "get_project_root", lambda: tmp_path)

    routes._sync_env_file(
        {"LLM_DEEPSEEK_API_KEY": "key with spaces and # punctuation"}
    )

    from dotenv import dotenv_values

    values = dotenv_values(tmp_path / ".env")
    assert values == {
        "LLM_DEEPSEEK_API_KEY": "key with spaces and # punctuation"
    }


def test_env_sync_does_not_replace_bind_mount_target(tmp_path, monkeypatch):
    from dotenv import main as dotenv_main

    env_path = tmp_path / ".env"
    env_path.write_text("EXISTING='value'\n", encoding="utf-8")
    monkeypatch.setattr(routes, "get_project_root", lambda: tmp_path)

    real_replace = dotenv_main.os.replace

    def reject_target_replace(source, destination):
        if Path(destination) == env_path:
            raise OSError(16, "Device or resource busy")
        return real_replace(source, destination)

    monkeypatch.setattr(dotenv_main.os, "replace", reject_target_replace)

    routes._sync_env_file({"LLM_DEEPSEEK_API_KEY": "secret-value"})

    from dotenv import dotenv_values

    assert dotenv_values(env_path) == {
        "EXISTING": "value",
        "LLM_DEEPSEEK_API_KEY": "secret-value",
    }


def test_history_save_request_enforces_body_bounds():
    with pytest.raises(ValueError):
        HistorySaveRequest(task="")
    with pytest.raises(ValueError):
        HistorySaveRequest(task="ok", model_used="m" * 201)
    with pytest.raises(ValueError, match="token_usage"):
        HistorySaveRequest(
            task="ok",
            token_usage={"payload": "x" * 100_001},
        )


def test_public_service_url_removes_credentials_and_query():
    assert routes._public_service_url(
        "redis://user:password@cache.internal:6379/0?token=secret"
    ) == "redis://cache.internal:6379/0"


def test_critic_score_normalizes_untrusted_model_values():
    score = CriticScore.from_dict(
        {
            "scores": {
                "overall": float("nan"),
                "accuracy": 99,
                "depth": -5,
            },
            "issues": [{"bad": "shape"}, "valid"],
            "suggestions": ["x" * 3000],
        }
    )

    assert score.overall == 0
    assert score.accuracy == 10
    assert score.depth == 0
    assert score.issues == ["valid"]
    assert len(score.suggestions[0]) == 2000


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion) tests
# ---------------------------------------------------------------------------


def rrf_fusion(
    rankings: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Minimal RRF implementation for testing."""
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}

    for rank_list in rankings:
        for rank, item in enumerate(rank_list):
            doc_id = item.get("id", item.get("doc_id", str(id(item))))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            items[doc_id] = item

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)  # type: ignore
    return [items[did] for did in sorted_ids]


class TestRRFFusion:
    """Test Reciprocal Rank Fusion for hybrid search results."""

    def test_basic_fusion(self):
        result_a = [{"id": "doc1"}, {"id": "doc2"}, {"id": "doc3"}]
        result_b = [{"id": "doc2"}, {"id": "doc3"}, {"id": "doc4"}]
        fused = rrf_fusion([result_a, result_b], k=60)
        ids = [r["id"] for r in fused]
        assert ids[0] == "doc2", f"Expected doc2 first, got {ids}"
        assert ids[1] == "doc3", f"Expected doc3 second, got {ids}"

    def test_empty_input(self):
        assert rrf_fusion([[], []]) == []

    def test_single_rank_list(self):
        docs = [{"id": "a"}, {"id": "b"}]
        fused = rrf_fusion([docs])
        assert fused == docs

    def test_k_parameter_effect(self):
        result_a = [{"id": "x"}, {"id": "y"}]
        result_b = [{"id": "y"}, {"id": "z"}]
        fused_small_k = rrf_fusion([result_a, result_b], k=1)
        fused_large_k = rrf_fusion([result_a, result_b], k=100)
        assert len(fused_small_k) == 3
        assert len(fused_large_k) == 3

    def test_score_distribution(self):
        result_a = [{"id": "a"}, {"id": "b"}]
        result_b = [{"id": "b"}, {"id": "c"}]
        fused = rrf_fusion([result_a, result_b], k=60)
        scores = {}
        for rank_list in [result_a, result_b]:
            for rank, item in enumerate(rank_list):
                did = item["id"]
                scores[did] = scores.get(did, 0.0) + 1.0 / (60 + rank + 1)
        assert fused[0]["id"] == "b"
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["c"]


# ---------------------------------------------------------------------------
# Adaptive retriever — QueryMode routing tests
# ---------------------------------------------------------------------------


class TestAdaptiveRetrieverRouting:
    """Test query classification and strategy routing."""

    QUERY_MODES = {
        "factual": ["BM25", "Vector"],
        "conceptual": ["RAPTOR", "Vector"],
        "comparative": ["GraphRAG", "Vector"],
        "procedural": ["BM25", "Vector"],
        "analytical": ["GraphRAG", "RAPTOR"],
        "graph": ["GraphRAG"],
    }

    def classify_query(self, query: str) -> str:
        q = query.lower()
        # Check specific intent patterns first, broad patterns last
        if any(w in q for w in ["concept", "theory", "idea", "概念"]):
            return "conceptual"
        if any(w in q for w in ["compare", "difference", "vs", "versus", "区别", "比较"]):
            return "comparative"
        if any(w in q for w in ["how to", "steps", "process", "how do", "如何"]):
            return "procedural"
        if any(w in q for w in ["analyze", "why", "impact", "原因", "分析"]):
            return "analytical"
        if any(w in q for w in ["relation", "connection", "link", "关系", "联系"]):
            return "graph"
        if any(w in q for w in ["what is", "define", "explain", "什么是"]):
            return "factual"
        return "factual"

    def test_classify_factual(self):
        assert self.classify_query("What is RAG?") == "factual"
        assert self.classify_query("Define self-attention") == "factual"

    def test_classify_conceptual(self):
        assert self.classify_query("Explain the concept of embeddings") == "conceptual"

    def test_classify_comparative(self):
        assert self.classify_query("Compare RAG and fine-tuning") == "comparative"
        assert self.classify_query("Difference between GPT and BERT") == "comparative"

    def test_classify_procedural(self):
        assert self.classify_query("How to implement a vector database") == "procedural"

    def test_classify_analytical(self):
        assert self.classify_query("Analyze the impact of attention mechanisms") == "analytical"

    def test_classify_graph(self):
        assert self.classify_query("Relation between transformers and LSTMs") == "graph"

    def test_routing_strategies(self):
        queries = {
            "what is attention": "factual",
            "compare RAG and GraphRAG": "comparative",
            "how to build an agent": "procedural",
            "analyze transformer performance": "analytical",
            "concept of transfer learning": "conceptual",
            "relation between tokens and embeddings": "graph",
        }
        for query, expected_mode in queries.items():
            mode = self.classify_query(query)
            assert mode == expected_mode, f"{query} → {mode}, expected {expected_mode}"
            strategies = self.QUERY_MODES[mode]
            assert len(strategies) > 0, f"No strategies for {mode}"


# ---------------------------------------------------------------------------
# GraphRAG entity extraction tests
# ---------------------------------------------------------------------------


class TestGraphRAGLogic:
    """Test GraphRAG entity and relationship extraction logic."""

    def test_entity_extraction_pattern(self):
        text = "Apple Inc. was founded by Steve Jobs in Cupertino."
        entities = {
            "Apple Inc.": {"type": "organization"},
            "Steve Jobs": {"type": "person"},
            "Cupertino": {"type": "location"},
        }
        assert text
        assert "Apple Inc." in entities
        assert "Steve Jobs" in entities
        assert entities["Cupertino"]["type"] == "location"

    def test_relationship_extraction(self):
        relations = [("OpenAI", "developed", "GPT-4"), ("Microsoft", "invested", "OpenAI")]
        openai_relations = [r for r in relations if r[0] == "OpenAI" or r[2] == "OpenAI"]
        assert len(openai_relations) == 2
        assert ("Microsoft", "invested", "OpenAI") in relations

    def test_community_detection_basic(self):
        nodes = ["A", "B", "C", "D"]
        communities = {"comm_0": ["A", "B"], "comm_1": ["C", "D"]}
        all_nodes = set()
        for comm_nodes in communities.values():
            all_nodes.update(comm_nodes)
        assert all_nodes == set(nodes)

    def test_summary_generation_format(self):
        community_data = {
            "comm_0": {"nodes": ["Transformer", "Self-Attention"], "relations": [("Transformer", "uses", "Self-Attention")]},
        }
        summary = f"Community comm_0 contains entities: {', '.join(community_data['comm_0']['nodes'])}"
        assert "Transformer" in summary
        assert "Self-Attention" in summary


# ---------------------------------------------------------------------------
# Production pipeline regression tests
# ---------------------------------------------------------------------------


def test_real_rrf_ignores_incompatible_raw_score_scales() -> None:
    rankings = {
        "vector": [
            {
                "id": "shared",
                "text": "shared",
                "score": 0.8,
                "source": "vector",
            },
            {
                "id": "vector-only",
                "text": "vector",
                "score": 0.7,
                "source": "vector",
            },
        ],
        "bm25": [
            {
                "id": "shared",
                "text": "shared",
                "score": 1.0,
                "source": "bm25",
            },
            {
                "id": "bm25-only",
                "text": "bm25",
                "score": 1_000_000.0,
                "source": "bm25",
            },
        ],
    }

    fused = HybridRetriever._rrf_fuse(
        rankings,
        top_k=3,
        vector_weight=0.5,
        bm25_weight=0.5,
    )

    assert fused[0]["id"] == "shared"
    assert fused[0]["score"] <= 1.0
    assert fused[1]["score"] == fused[2]["score"]


@pytest.mark.asyncio
async def test_real_graphrag_accepts_ingestion_document_shape() -> None:
    class FakeLLM:
        async def chat(self, messages, **kwargs):
            del messages, kwargs
            return SimpleNamespace(
                content=(
                    '[{"type":"entity","id":"mindforge",'
                    '"name":"MindForge","entity_type":"project",'
                    '"description":"Research assistant"}]'
                )
            )

    engine = GraphRAGEngine(llm_fn=FakeLLM())
    await engine.build_graph(
        [
            {
                "doc_id": "doc-1",
                "content": "MindForge is a research assistant.",
            }
        ]
    )

    assert "mindforge" in engine.entities


@pytest.mark.asyncio
async def test_graphrag_delete_recomputes_shared_entity_provenance(
    tmp_path,
) -> None:
    responses = iter(
        [
            (
                '[{"type":"entity","id":"shared","name":"Shared",'
                '"entity_type":"concept","description":"poisoned marker"}]'
            ),
            (
                '[{"type":"entity","id":"shared","name":"Shared",'
                '"entity_type":"concept","description":"clean description"}]'
            ),
        ]
    )

    class FakeLLM:
        async def chat(self, messages, **kwargs):
            del kwargs
            prompt = messages[0].content
            if prompt.startswith("Extract entities"):
                return SimpleNamespace(content=next(responses))
            return SimpleNamespace(content="Shared concept")

    engine = GraphRAGEngine(llm_fn=FakeLLM())
    await engine.build_graph([{"doc_id": "doc-a", "content": "first"}])
    await engine.build_graph([{"doc_id": "doc-b", "content": "second"}])

    assert engine.entities["shared"].description == "poisoned marker"

    engine.delete_document("doc-a")

    entity = engine.entities["shared"]
    assert entity.description == "clean description"
    assert entity.metadata["doc_ids"] == ["doc-b"]
    assert "poisoned marker" not in str(await engine.query("poisoned"))

    graph_path = tmp_path / "graph.json"
    engine.save(str(graph_path))
    restored = GraphRAGEngine()
    restored.load(str(graph_path))

    assert restored.entities["shared"].description == "clean description"
    assert "poisoned marker" not in str(await restored.query("poisoned"))


def test_graphrag_legacy_shared_entity_is_removed_conservatively() -> None:
    engine = GraphRAGEngine()
    engine.entities["shared"] = Entity(
        id="shared",
        name="Shared",
        description="unattributed legacy text",
        metadata={"doc_ids": ["doc-a", "doc-b"]},
    )

    engine.delete_document("doc-a")

    assert "shared" not in engine.entities


def test_graphrag_relation_weight_is_finite_and_bounded() -> None:
    assert GraphRAGEngine._coerce_relation_weight("not-a-number") == 1.0
    assert GraphRAGEngine._coerce_relation_weight(float("nan")) == 1.0
    assert GraphRAGEngine._coerce_relation_weight(-5) == 0.0
    assert GraphRAGEngine._coerce_relation_weight(12) == 1.0


@dataclass
class _FakePlanner:
    async def run(self, task: str) -> ResearchPlan:
        return ResearchPlan(
            plan_id="plan",
            original_task=task,
            subtasks=[
                SubTask(task_id="t1", description="first"),
                SubTask(
                    task_id="t2",
                    description="second",
                    dependencies=["t1"],
                ),
            ],
        )


class _ContextResearcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def run(
        self,
        task: str,
        *,
        context: str | None = None,
    ) -> AgentResult:
        self.calls.append((task, context))
        return AgentResult(
            agent_name="researcher",
            output=f"result for {task}",
            cost_usd=0.25,
            token_usage={
                "prompt_tokens": 2,
                "completion_tokens": 3,
            },
        )


class _NoopSynthesizer:
    async def synthesize(self, **kwargs) -> AgentResult:
        del kwargs
        return AgentResult(
            agent_name="synthesizer",
            output="combined",
        )


class _NoopCritic:
    async def evaluate(self, **kwargs):
        del kwargs
        raise AssertionError("simple pipeline should skip critic")


@pytest.mark.asyncio
async def test_orchestrator_passes_dependency_context_and_cost() -> None:
    researcher = _ContextResearcher()
    orchestrator = Orchestrator(
        planner=_FakePlanner(),
        researcher=researcher,
        synthesizer=_NoopSynthesizer(),
        critic=_NoopCritic(),
    )
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(
            research_timeout=30,
            subtask_timeout=5,
            max_refine_rounds=0,
        ),
        llm=SimpleNamespace(llm_provider="test"),
        observability=SimpleNamespace(enable_tracing=False),
    )

    result = await orchestrator.run("task")

    assert researcher.calls[0] == ("first", None)
    assert researcher.calls[1][0] == "second"
    assert "result for first" in (researcher.calls[1][1] or "")
    assert result.cost_usd == pytest.approx(0.5)
    assert "cost_usd" not in result.token_usage


class _FailedResearcher:
    async def run(
        self,
        task: str,
        *,
        context: str | None = None,
    ) -> AgentResult:
        del context
        return AgentResult(
            agent_name="researcher",
            success=False,
            output=f"failed: {task}",
        )


class _UnexpectedSynthesizer:
    async def synthesize(self, **kwargs):
        del kwargs
        raise AssertionError(
            "synthesizer must not run when all subtasks fail"
        )

    async def synthesize_stream(self, **kwargs):
        del kwargs
        raise AssertionError(
            "streaming synthesizer must not run when all subtasks fail"
        )
        yield ""


def _failed_orchestrator() -> Orchestrator:
    orchestrator = Orchestrator(
        planner=_FakePlanner(),
        researcher=_FailedResearcher(),
        synthesizer=_UnexpectedSynthesizer(),
        critic=_NoopCritic(),
    )
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(
            research_timeout=30,
            subtask_timeout=5,
            max_refine_rounds=1,
        ),
        llm=SimpleNamespace(llm_provider="test"),
        observability=SimpleNamespace(enable_tracing=False),
    )
    return orchestrator


@pytest.mark.asyncio
async def test_orchestrator_does_not_report_all_failed_tasks_as_success() -> None:
    result = await _failed_orchestrator().run("task")

    assert result.success is False
    assert "all subtasks failed" in result.output
    assert result.data["subtask_outputs"]
    assert all(
        not item["success"]
        for item in result.data["subtask_outputs"]
    )


@pytest.mark.asyncio
async def test_streaming_orchestrator_emits_failed_done_result() -> None:
    events = [
        event
        async for event in _failed_orchestrator().stream_run("task")
    ]

    done = [event for event in events if event["type"] == "done"]
    assert len(done) == 1
    assert done[0]["result"].success is False
    assert not any(
        event["type"] == "synthesizing"
        for event in events
    )


def test_collect_sources_deduplicates_and_reindexes() -> None:
    outputs = [
        {
            "sources": [
                {
                    "index": 1,
                    "title": "A",
                    "url": "https://a.example",
                },
                {
                    "index": 2,
                    "title": "B",
                    "url": "https://b.example",
                },
            ]
        },
        {
            "sources": [
                {
                    "index": 1,
                    "title": "B duplicate",
                    "url": "https://b.example",
                },
                {
                    "index": 2,
                    "title": "C",
                    "url": "https://c.example",
                },
            ]
        },
    ]

    sources = Orchestrator._collect_sources(outputs)

    assert [source["url"] for source in sources] == [
        "https://a.example",
        "https://b.example",
        "https://c.example",
    ]
    assert [source["index"] for source in sources] == [1, 2, 3]


def test_code_executor_uses_isolated_subprocess() -> None:
    result = CodeExecutor().execute(
        "print(sum(values))\n_return = len(values)",
        vars={"values": [1, 2, 3]},
        timeout=3,
    )

    assert result.success is True
    assert result.output.strip() == "6"
    assert result.data["return_value"] == 3


def test_code_executor_rejects_non_whitelisted_import() -> None:
    result = CodeExecutor().execute(
        "import os\nprint(os.getcwd())"
    )

    assert result.success is False
    assert "Module not allowed" in (result.error or "")


def test_code_executor_blocks_scientific_library_file_access() -> None:
    result = CodeExecutor().execute(
        "import numpy as np\n"
        "print(np.fromfile(target, dtype=np.uint8, count=8))",
        vars={"target": str(Path("pyproject.toml").resolve())},
        timeout=5,
    )

    assert result.success is False
    assert "File API not allowed" in (result.error or "")


def test_code_executor_bounds_child_output_before_parent_capture() -> None:
    executor = CodeExecutor()
    result = executor.execute('print("x" * 200000)', timeout=5)

    assert result.success is True
    assert result.truncated is True
    assert len(result.output) <= executor._settings.sandbox.max_output_length


@pytest.mark.asyncio
async def test_rag_tool_enforces_runtime_top_k_limit() -> None:
    class UnexpectedRetriever:
        async def retrieve(self, **kwargs):
            raise AssertionError(f"retriever must not run: {kwargs}")

    result = await RAGTool(retriever=UnexpectedRetriever()).execute_async(
        query="bounded",
        top_k=10_000,
    )

    assert result.success is False
    assert "top_k" in (result.error or "")


def test_reranker_caps_candidates_before_model_inference() -> None:
    class FakeModel:
        def __init__(self):
            self.batch_size = 0

        def predict(self, pairs):
            self.batch_size = len(pairs)
            return [0.5] * len(pairs)

    model = FakeModel()
    reranker = CrossEncoderReranker("fake", max_candidates=3)
    reranker._model = model
    results = reranker.rerank(
        "query",
        [{"text": str(index)} for index in range(10)],
        top_k=10,
    )

    assert model.batch_size == 3
    assert len(results) == 3


def test_reranker_pins_configured_model_revision(monkeypatch) -> None:
    calls = []

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    reranker = CrossEncoderReranker(
        "example/reranker",
        model_revision="immutable-revision",
    )

    assert isinstance(reranker.model, FakeCrossEncoder)
    assert calls == [
        (
            "example/reranker",
            {"revision": "immutable-revision"},
        )
    ]


def test_embedding_manager_pins_configured_model_revision(
    monkeypatch,
) -> None:
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

        def get_embedding_dimension(self):
            return 1024

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    manager = EmbeddingManager.__new__(EmbeddingManager)
    manager._dim = None
    manager._model_name = None
    manager._provider = "sentence-transformers"
    manager._model = None
    manager._client = None
    manager._native_dim = None

    manager._init_st()

    assert calls[0][1]["revision"]
    assert calls[0][1]["local_files_only"] is True


def test_embedding_dimension_adapter_preserves_geometry() -> None:
    manager = EmbeddingManager.__new__(EmbeddingManager)
    manager._dim = 4

    assert manager._fit_dimension([0.6, 0.8]) == [
        0.6,
        0.8,
        0.0,
        0.0,
    ]


def test_research_plan_rejects_invalid_dependencies() -> None:
    cyclic = ResearchPlan(
        plan_id="cycle",
        original_task="task",
        subtasks=[
            SubTask("t1", "one", dependencies=["t2"]),
            SubTask("t2", "two", dependencies=["t1"]),
        ],
    )
    with pytest.raises(ValueError, match="cyclic"):
        cyclic.validate()

    unknown = ResearchPlan(
        plan_id="unknown",
        original_task="task",
        subtasks=[
            SubTask("t1", "one", dependencies=["missing"])
        ],
    )
    with pytest.raises(ValueError, match="unknown task"):
        unknown.validate()

    oversized = ResearchPlan(
        plan_id="oversized",
        original_task="task",
        subtasks=[
            SubTask(f"t{index}", f"task {index}")
            for index in range(6)
        ],
    )
    with pytest.raises(ValueError, match="allowed range"):
        oversized.validate(max_subtasks=5)


def test_accumulate_usage_accepts_raw_usage_dict() -> None:
    usage: dict[str, int] = {}
    Orchestrator._accumulate_usage(
        usage,
        {"prompt_tokens": 3, "completion_tokens": 5},
    )

    assert usage == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
    }


# ---------------------------------------------------------------------------
# Real FastAPI route tests with isolated dependencies
# ---------------------------------------------------------------------------


def _api_app() -> FastAPI:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/v1")
    return app


async def _api_get(path: str):
    async with AsyncClient(
        transport=ASGITransport(app=_api_app()),
        base_url="http://test",
    ) as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_qdrant_probe_checks_service_without_requiring_collection(
    monkeypatch,
) -> None:
    class ReachableStore:
        async def ping(self) -> None:
            return None

        async def get_stats(self):
            raise AssertionError("Probe must not require an existing collection")

    monkeypatch.setattr(
        routes,
        "get_vector_store",
        lambda: ReachableStore(),
    )

    assert await routes._probe_qdrant_connection() is True


@pytest.mark.asyncio
async def test_health_reports_real_core_status(monkeypatch) -> None:
    async def connected() -> bool:
        return True

    registry = SimpleNamespace(
        servers={"configured": object()},
        is_any_running=True,
    )
    monkeypatch.setattr(
        routes,
        "_probe_qdrant_connection",
        connected,
    )
    monkeypatch.setattr(
        routes,
        "_probe_redis_connection",
        connected,
    )
    monkeypatch.setattr(
        routes,
        "_probe_postgres_connection",
        connected,
    )
    monkeypatch.setattr(
        routes,
        "get_mcp_registry",
        lambda: registry,
    )

    response = await _api_get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["postgres_connected"] is True
    assert payload["mcp_configured"] is True
    assert payload["mcp_tools_available"] is True


@pytest.mark.asyncio
async def test_health_reports_degraded_when_core_dependency_fails(
    monkeypatch,
) -> None:
    async def connected() -> bool:
        return True

    async def disconnected() -> bool:
        return False

    monkeypatch.setattr(
        routes,
        "_probe_qdrant_connection",
        disconnected,
    )
    monkeypatch.setattr(
        routes,
        "_probe_redis_connection",
        connected,
    )
    monkeypatch.setattr(
        routes,
        "_probe_postgres_connection",
        connected,
    )
    monkeypatch.setattr(
        routes,
        "get_mcp_registry",
        lambda: None,
    )

    response = await _api_get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["qdrant_connected"] is False
    assert payload["mcp_configured"] is False


@pytest.mark.asyncio
async def test_readiness_uses_http_status_for_core_health(
    monkeypatch,
) -> None:
    async def healthy():
        return routes.HealthResponse(
            status="ok",
            qdrant_connected=True,
            redis_connected=True,
            postgres_connected=True,
        )

    monkeypatch.setattr(routes, "health", healthy)
    ready = await _api_get("/api/v1/ready")
    assert ready.status_code == 200

    async def degraded():
        return routes.HealthResponse(
            status="degraded",
            qdrant_connected=False,
            redis_connected=True,
            postgres_connected=True,
        )

    monkeypatch.setattr(routes, "health", degraded)
    unavailable = await _api_get("/api/v1/ready")
    assert unavailable.status_code == 503
    assert unavailable.json()["status"] == "degraded"


@pytest.mark.asyncio
async def test_history_pagination_validation_happens_before_database() -> None:
    invalid_page = await _api_get("/api/v1/history?page=0")
    oversized_page = await _api_get(
        "/api/v1/history?page_size=101"
    )

    assert invalid_page.status_code == 422
    assert oversized_page.status_code == 422


@pytest.mark.asyncio
async def test_history_detail_returns_complete_report(
    monkeypatch,
) -> None:
    entry = SimpleNamespace(
        id=7,
        user_id=1,
        task="full report",
        report="x" * 4_000,
        quality_score=8.5,
        model_used="test-model",
        created_at=None,
    )

    class FakeSession:
        class Query:
            def filter(self, *conditions):
                del conditions
                return self

            def first(self):
                return entry

        def query(self, model):
            del model
            return self.Query()

        def close(self):
            return None

    fake_db_module = SimpleNamespace(
        ResearchHistory=SimpleNamespace(id=7, user_id=1),
        SessionLocal=FakeSession,
        get_default_user_id=lambda db: 1,
    )
    monkeypatch.setitem(
        sys.modules,
        "mindforge.db",
        fake_db_module,
    )

    response = await _api_get("/api/v1/history/7")

    assert response.status_code == 200
    assert response.json()["report"] == "x" * 4_000


def test_mcp_registry_loads_inline_json_and_validates_shape() -> None:
    registry = MCPRegistry()
    registry.load_config_json(
        '{"mcpServers":{"demo":{"command":"python",'
        '"args":["-m","demo"],"enabled":true}}}'
    )

    assert list(registry.servers) == ["demo"]
    assert registry.servers["demo"].config.args == ["-m", "demo"]

    with pytest.raises(ValueError, match="invalid JSON"):
        registry.load_config_json("{")


def test_frontend_path_rejects_sibling_prefix_traversal(
    monkeypatch,
    tmp_path,
) -> None:
    frontend = tmp_path / "dist"
    frontend.mkdir()
    monkeypatch.setattr(server, "_FRONTEND_PATH", frontend.resolve())

    assert server._safe_frontend_candidate("assets/app.js") == (
        frontend / "assets" / "app.js"
    ).resolve()
    assert server._safe_frontend_candidate(
        "../dist-private/secret.txt"
    ) is None


@pytest.mark.asyncio
async def test_root_returns_metadata_without_frontend(
    monkeypatch,
    tmp_path,
) -> None:
    missing_frontend = tmp_path / "missing-dist"
    monkeypatch.setattr(
        server,
        "_FRONTEND_DIR",
        str(missing_frontend),
    )

    response = await server.root()
    payload = json.loads(response.body)

    assert payload["name"] == "MindForge"
    assert payload["readiness"] == "/api/v1/ready"
