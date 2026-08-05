"""Test RRF fusion algorithm, adaptive retriever strategy selection, and GraphRAG."""

from __future__ import annotations

import asyncio
import sys
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.responses import Response
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from mindforge.agents.base import AgentResult
from mindforge.agents.critic import CriticAgent, CriticScore
from mindforge.agents.orchestrator import Orchestrator
from mindforge.agents.planner import PlannerAgent, ResearchPlan, SubTask
from mindforge.agents.researcher import ResearcherAgent
from mindforge.api import routes
from mindforge.api import server
from mindforge.api.schemas import (
    HistorySaveRequest,
    LLMProviderUpdate,
    SettingsUpdateRequest,
)
from mindforge.ingestion.embedder import EmbeddingManager
from mindforge.config import get_settings
from mindforge.retrieval.bm25 import BM25Retriever
from mindforge.retrieval.graphrag import Entity, GraphRAGEngine
from mindforge.retrieval.hybrid import HybridRetriever
from mindforge.retrieval.reranker import CrossEncoderReranker
from mindforge.tools.code_executor import CodeExecutor
from mindforge.tools.rag_tool import RAGTool


def test_settings_api_keys_reject_control_characters():
    with pytest.raises(ValueError, match="control characters"):
        SettingsUpdateRequest(deepseek_api_key="valid-prefix\nINJECTED=value")


def test_provider_update_validates_base_url_and_duplicates() -> None:
    update = LLMProviderUpdate(
        provider="local",
        base_url="http://127.0.0.1:8001/v1/",
    )
    assert update.base_url == "http://127.0.0.1:8001/v1"

    with pytest.raises(ValueError, match="credentials"):
        LLMProviderUpdate(
            provider="local",
            base_url="http://user:secret@127.0.0.1:8001/v1",
        )

    with pytest.raises(ValueError, match="only be updated once"):
        SettingsUpdateRequest(
            llm_provider_config=update,
            llm_provider_configs=[update],
        )


@pytest.mark.parametrize(
    ("update", "expected_detail"),
    [
        (
            SettingsUpdateRequest(
                llm_request_timeout=90,
                subtask_timeout=60,
                research_timeout=180,
            ),
            "LLM request timeout",
        ),
        (
            SettingsUpdateRequest(
                queue_timeout=200,
                research_timeout=180,
            ),
            "tool queue timeout",
        ),
        (
            SettingsUpdateRequest(
                native_web_search_timeout_seconds=90,
                subtask_timeout=60,
            ),
            "native web-search timeout",
        ),
        (
            SettingsUpdateRequest(
                llm_request_timeout=10,
                sandbox_timeout=15,
                subtask_timeout=10,
            ),
            "sandbox timeout",
        ),
    ],
)
def test_settings_reject_contradictory_timeout_budgets(
    monkeypatch: pytest.MonkeyPatch,
    update: SettingsUpdateRequest,
    expected_detail: str,
) -> None:
    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: SimpleNamespace(
            agent=SimpleNamespace(
                llm_request_timeout=45,
                subtask_timeout=60,
                research_timeout=180,
                queue_timeout=30,
            ),
            web_search=SimpleNamespace(native_timeout_seconds=5),
            sandbox=SimpleNamespace(sandbox_timeout=5),
        ),
    )

    with pytest.raises(Exception) as exc_info:
        routes._update_settings_locked(update)

    assert getattr(exc_info.value, "status_code", None) == 422
    assert expected_detail in str(getattr(exc_info.value, "detail", ""))


def test_env_sync_quotes_values_and_prevents_entry_injection(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(routes, "get_project_root", lambda: tmp_path)

    routes._sync_env_file({"LLM_DEEPSEEK_API_KEY": "key with spaces and # punctuation"})

    from dotenv import dotenv_values

    values = dotenv_values(tmp_path / ".env")
    assert values == {"LLM_DEEPSEEK_API_KEY": "key with spaces and # punctuation"}


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


def test_database_url_must_be_configured(monkeypatch):
    from mindforge.config import require_environment_variable

    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL is required"):
        require_environment_variable("DATABASE_URL")


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
    with pytest.raises(ValueError, match="at most 200"):
        HistorySaveRequest(
            task="ok",
            sources=[
                {
                    "index": index + 1,
                    "title": "source",
                }
                for index in range(201)
            ],
        )
    request = HistorySaveRequest(
        task="ok",
        sources=[
            {
                "index": 1,
                "title": "unsafe",
                "url": "javascript:alert(1)",
            }
        ],
    )
    assert request.sources[0].url == ""


def test_public_service_url_removes_credentials_and_query():
    assert (
        routes._public_service_url(
            "redis://user:password@cache.internal:6379/0?token=secret"
        )
        == "redis://cache.internal:6379/0"
    )


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
        if any(
            w in q for w in ["compare", "difference", "vs", "versus", "区别", "比较"]
        ):
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
        assert (
            self.classify_query("Analyze the impact of attention mechanisms")
            == "analytical"
        )

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
        relations = [
            ("OpenAI", "developed", "GPT-4"),
            ("Microsoft", "invested", "OpenAI"),
        ]
        openai_relations = [
            r for r in relations if r[0] == "OpenAI" or r[2] == "OpenAI"
        ]
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
            "comm_0": {
                "nodes": ["Transformer", "Self-Attention"],
                "relations": [("Transformer", "uses", "Self-Attention")],
            },
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
    assert fused[0]["semantic_score"] == pytest.approx(0.8)
    assert fused[0]["keyword_score"] == pytest.approx(1.0)
    assert fused[0]["rrf_score"] == fused[0]["score"]


def test_bm25_keyword_fallback_excludes_disabled_documents() -> None:
    retriever = object.__new__(BM25Retriever)
    retriever.documents = [
        "python async disabled",
        "python async enabled",
    ]
    retriever.doc_ids = ["disabled-chunk", "enabled-chunk"]
    retriever.metadatas = [
        {"doc_id": "doc-disabled"},
        {"doc_id": "doc-enabled"},
    ]

    results = retriever._keyword_fallback(
        "python async",
        top_k=1,
        excluded_doc_ids={"doc-disabled"},
    )

    assert [result["id"] for result in results] == ["enabled-chunk"]


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


class _EmptySynthesizer:
    async def synthesize(self, **kwargs) -> AgentResult:
        del kwargs
        return AgentResult(
            agent_name="synthesizer",
            success=False,
            output="",
        )

    async def synthesize_stream(self, **kwargs):
        del kwargs
        if False:
            yield ""


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
            research_mode="fast",
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
    assert (researcher.calls[1][1] or "").count("result for first") == 1
    assert result.cost_usd == pytest.approx(0.5)
    assert "cost_usd" not in result.token_usage


@pytest.mark.asyncio
async def test_orchestrator_adds_semantic_memory_to_working_context() -> None:
    class SemanticMemoryStub:
        async def recall(self, task: str, top_k: int = 5):
            assert task == "task"
            assert top_k == 3
            return [
                SimpleNamespace(
                    fact_id="fact-1",
                    content="Prior verified research.",
                    confidence=0.8,
                    sources=["https://example.com"],
                )
            ]

    orchestrator = Orchestrator(
        planner=_FakePlanner(),
        researcher=_ContextResearcher(),
        synthesizer=_NoopSynthesizer(),
        critic=_NoopCritic(),
        semantic_memory=SemanticMemoryStub(),
    )

    memory = await orchestrator._create_working_memory("task")

    assert "Prior verified research." in memory.get_context_string()


@pytest.mark.asyncio
async def test_orchestrator_rejects_cached_factual_answer_without_citations() -> None:
    class EpisodicMemoryStub:
        async def recall(self, task: str):
            assert task == "什么是 RAG"
            return AgentResult(
                agent_name="orchestrator",
                success=True,
                output="RAG 是一种检索增强生成技术。",
                data={"sources": []},
                metadata={"outcome": "success"},
            ).to_dict()

    orchestrator = object.__new__(Orchestrator)
    orchestrator._episodic_memory = EpisodicMemoryStub()

    result = await orchestrator._recall_cached_result(
        "什么是 RAG",
        start_time=time.perf_counter(),
    )

    assert result is None


def test_success_without_critic_is_marked_not_evaluated() -> None:
    orchestrator = Orchestrator(
        planner=SimpleNamespace(),
        researcher=SimpleNamespace(),
        synthesizer=SimpleNamespace(),
        critic=SimpleNamespace(),
    )
    orchestrator._settings = SimpleNamespace(
        llm=SimpleNamespace(llm_provider="test"),
    )
    plan = ResearchPlan(
        plan_id="simple",
        original_task="什么是异步编程",
        subtasks=[SubTask(task_id="t1", description="什么是异步编程")],
    )

    result = orchestrator._build_success_result(
        output="异步编程允许任务在等待期间让出执行权。",
        plan=plan,
        subtask_outputs=[
            {
                "task_id": "t1",
                "description": "什么是异步编程",
                "success": True,
                "output": "异步编程允许任务在等待期间让出执行权。",
                "sources": [],
            }
        ],
        sources=[],
        final_critic=None,
        refine_count=0,
        total_usage={"total_tokens": 10},
        elapsed_ms=100,
        total_cost_usd=0.001,
        cost_status="estimated",
    )

    assert result.metadata["quality"] is None
    assert result.metadata["quality_status"] == "not_evaluated"


@pytest.mark.asyncio
async def test_critic_failure_is_not_reported_as_a_numeric_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    critic = object.__new__(CriticAgent)
    critic._model_name = "critic-model"
    critic._provider_name = "test"

    async def invalid_json(*_args, **_kwargs):
        return SimpleNamespace(
            content="not-json",
            usage={"total_tokens": 4},
            model="critic-model",
        )

    critic._chat = invalid_json
    monkeypatch.setattr(
        "mindforge.agents.critic.get_settings",
        lambda: SimpleNamespace(
            agent=SimpleNamespace(critic_threshold=7.0),
        ),
    )
    monkeypatch.setattr(
        "mindforge.agents.critic._estimate_cost_details",
        lambda *_args, **_kwargs: SimpleNamespace(
            amount_usd=None,
            status="pricing_unconfigured",
        ),
    )

    score = await critic.evaluate(
        task="研究问题",
        draft="研究报告",
    )

    assert score.evaluation_status == "failed"
    assert score.overall == 0.0
    assert score.should_refine is False


@pytest.mark.asyncio
async def test_critic_retries_one_invalid_structured_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    critic = object.__new__(CriticAgent)
    critic._model_name = "critic-model"
    critic._provider_name = "test"
    calls = 0

    async def repaired_json(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(
                content="",
                usage={"total_tokens": 4},
                model="critic-model",
            )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "scores": {
                        "completeness": 8,
                        "accuracy": 8,
                        "depth": 7,
                        "clarity": 9,
                        "citations": 7,
                        "overall": 7,
                    },
                    "issues": [],
                    "suggestions": [],
                    "should_refine": True,
                },
                ensure_ascii=False,
            ),
            usage={"total_tokens": 6},
            model="critic-model",
        )

    critic._chat = repaired_json
    monkeypatch.setattr(
        "mindforge.agents.critic.get_settings",
        lambda: SimpleNamespace(
            agent=SimpleNamespace(critic_threshold=7.0),
        ),
    )
    monkeypatch.setattr(
        "mindforge.agents.critic._estimate_cost_details",
        lambda *_args, **_kwargs: SimpleNamespace(
            amount_usd=None,
            status="usage_unavailable",
        ),
    )

    score = await critic.evaluate(task="研究问题", draft="研究报告")

    assert calls == 2
    assert score.evaluation_status == "evaluated"
    assert score.overall == 7.0
    assert score.should_refine is True
    assert score.token_usage["total_tokens"] == 10


@pytest.mark.asyncio
async def test_critic_missing_required_scores_is_an_evaluation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    critic = object.__new__(CriticAgent)
    critic._model_name = "critic-model"
    critic._provider_name = "test"

    async def incomplete_score(*_args, **_kwargs):
        return SimpleNamespace(
            content='{"scores": {"overall": 8}}',
            usage={},
            model="critic-model",
        )

    critic._chat = incomplete_score
    monkeypatch.setattr(
        "mindforge.agents.critic.get_settings",
        lambda: SimpleNamespace(
            agent=SimpleNamespace(critic_threshold=7.0),
        ),
    )
    monkeypatch.setattr(
        "mindforge.agents.critic._estimate_cost_details",
        lambda *_args, **_kwargs: SimpleNamespace(
            amount_usd=None,
            status="usage_unavailable",
        ),
    )

    score = await critic.evaluate(task="研究问题", draft="研究报告")

    assert score.evaluation_status == "failed"
    assert score.overall == 0.0


def test_balanced_mode_routes_focused_comparison_directly() -> None:
    assert Orchestrator._is_simple_task("什么是异步编程") is True
    assert Orchestrator._is_simple_task("Python 和 Java 有什么区别") is False
    assert Orchestrator._is_simple_task("对比 RAG 与 GraphRAG 的优缺点") is False
    assert Orchestrator._can_use_direct_plan("Python 和 Java 有什么区别") is True
    assert Orchestrator._can_use_direct_plan("全面对比 Python 和 Java") is False


@pytest.mark.asyncio
async def test_balanced_comparison_uses_one_direct_research_task() -> None:
    calls: list[str] = []

    class RecordingPlanner:
        async def run(self, task: str) -> ResearchPlan:
            calls.append(task)
            return ResearchPlan(
                plan_id="comparison",
                original_task=task,
                subtasks=[
                    SubTask(task_id="python", description="研究 Python"),
                    SubTask(task_id="java", description="研究 Java"),
                    SubTask(
                        task_id="compare",
                        description="综合比较 Python 和 Java",
                        dependencies=["python", "java"],
                    ),
                ],
            )

    orchestrator = object.__new__(Orchestrator)
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(research_mode="balanced")
    )
    orchestrator._planner_injected = False
    orchestrator._planner = RecordingPlanner()

    plan = await orchestrator._create_plan("Python 和 Java 有什么区别")

    assert calls == []
    assert plan.planner_status == "direct"
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].description == "Python 和 Java 有什么区别"
    assert ResearcherAgent.source_requirement(
        plan.subtasks[0].description
    ) == "preferred"


@pytest.mark.asyncio
async def test_planner_replans_under_decomposed_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = object.__new__(PlannerAgent)
    planner._model_name = "planner-model"
    planner._provider_name = "test"
    calls: list[list] = []

    async def planned_response(messages, *_args, **_kwargs):
        calls.append(messages)
        if len(calls) == 1:
            subtasks = [
                {
                    "task_id": "t1",
                    "description": "python和go的协程有什么区别？",
                    "task_type": "research",
                    "dependencies": [],
                    "priority": 1,
                    "subtopics": [],
                }
            ]
        else:
            subtasks = [
                {
                    "task_id": "t1",
                    "description": "研究 Python 协程的机制与运行模型",
                    "task_type": "research",
                    "dependencies": [],
                    "priority": 1,
                    "subtopics": ["asyncio", "事件循环"],
                },
                {
                    "task_id": "t2",
                    "description": "研究 Go goroutine 的机制与运行模型",
                    "task_type": "research",
                    "dependencies": [],
                    "priority": 1,
                    "subtopics": ["goroutine", "GMP 调度"],
                },
                {
                    "task_id": "t3",
                    "description": "综合比较两者的调度、通信和适用场景",
                    "task_type": "analysis",
                    "dependencies": ["t1", "t2"],
                    "priority": 2,
                    "subtopics": ["调度差异", "并发模型", "选型"],
                },
            ]
        return SimpleNamespace(
            content=json.dumps(
                {
                    "reasoning": "根据问题结构规划。",
                    "subtasks": subtasks,
                },
                ensure_ascii=False,
            ),
            usage={},
            model="planner-model",
        )

    planner._chat = planned_response
    monkeypatch.setattr(
        "mindforge.agents.planner.get_settings",
        lambda: SimpleNamespace(agent=SimpleNamespace(max_subtasks=5)),
    )

    plan = await planner.run("python和go的协程有什么区别？")

    assert plan.planner_status == "planned"
    assert len(calls) == 2
    assert len(plan.subtasks) == 3
    assert "Python 协程" in plan.subtasks[0].description
    assert "Go goroutine" in plan.subtasks[1].description
    assert plan.subtasks[2].dependencies == ["t1", "t2"]
    assert "未通过质量校验" in calls[1][-1].content


def test_planner_quality_validation_uses_task_shape() -> None:
    simple_plan = ResearchPlan(
        plan_id="simple",
        original_task="什么是协程",
        subtasks=[
            SubTask(
                task_id="t1",
                description="什么是协程",
                subtopics=["协程定义"],
            )
        ],
    )
    comparison_plan = ResearchPlan(
        plan_id="comparison",
        original_task="Python 和 Go 的协程有什么区别",
        subtasks=[
            SubTask(
                task_id="t1",
                description="研究 Python 协程",
                subtopics=["asyncio"],
            ),
            SubTask(
                task_id="t2",
                description="研究 Go goroutine",
                subtopics=["GMP 调度"],
            ),
            SubTask(
                task_id="t3",
                description="综合比较",
                dependencies=["t1", "t2"],
                subtopics=["机制差异"],
            ),
        ],
    )
    duplicate_plan = ResearchPlan(
        plan_id="duplicate",
        original_task="诊断接口超时的根因并修复",
        subtasks=[
            SubTask(task_id="t1", description="检查接口"),
            SubTask(task_id="t2", description="检查接口"),
        ],
    )

    assert PlannerAgent._quality_errors("什么是协程", simple_plan) == []
    assert (
        PlannerAgent._quality_errors(
            "Python 和 Go 的协程有什么区别",
            comparison_plan,
        )
        == []
    )
    assert any(
        "描述重复" in error
        for error in PlannerAgent._quality_errors(
            "诊断接口超时的根因并修复",
            duplicate_plan,
        )
    )
    assert PlannerAgent._minimum_subtask_count("什么是协程") == 1
    assert (
        PlannerAgent._minimum_subtask_count(
            "分析接口超时的原因、影响和解决方案"
        )
        == 3
    )
    assert (
        PlannerAgent._minimum_subtask_count(
            "为什么接口变慢，如何定位并修复？"
        )
        == 2
    )


@pytest.mark.asyncio
async def test_planner_fallback_exposes_failure_and_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = object.__new__(PlannerAgent)
    planner._model_name = "planner-model"
    planner._provider_name = "test"

    async def invalid_json(*_args, **_kwargs):
        return SimpleNamespace(
            content="{invalid",
            usage={"prompt_tokens": 12, "completion_tokens": 1},
            model="planner-model",
        )

    planner._chat = invalid_json
    monkeypatch.setattr(
        "mindforge.agents.planner.get_settings",
        lambda: SimpleNamespace(agent=SimpleNamespace(max_subtasks=5)),
    )
    monkeypatch.setattr(
        "mindforge.agents.planner._estimate_cost_details",
        lambda *_args, **_kwargs: SimpleNamespace(
            amount_usd=None,
            status="pricing_unconfigured",
        ),
    )

    plan = await planner.run("比较 Python 和 Java")
    restored = ResearchPlan.from_dict(plan.to_dict())

    assert plan.planner_status == "fallback"
    assert plan.planner_error
    assert len(plan.subtasks) == 1
    assert restored.planner_status == "fallback"
    assert restored.planner_error == plan.planner_error


def test_balanced_single_subtask_comparison_still_runs_critic() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(
            research_mode="balanced",
            max_refine_rounds=0,
        )
    )
    plan = ResearchPlan(
        plan_id="comparison",
        original_task="Python 和 Java 有什么区别",
        subtasks=[
            SubTask(
                task_id="t1",
                description="Python 和 Java 有什么区别",
            )
        ],
    )

    assert orchestrator._should_run_critic(
        "Python 和 Java 有什么区别",
        plan,
    ) is True
    assert orchestrator._max_refine_rounds(plan) == 0


def test_balanced_single_subtask_honors_configured_refinement() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(
            research_mode="balanced",
            max_refine_rounds=1,
        )
    )
    plan = ResearchPlan(
        plan_id="focused",
        original_task="什么是异步编程",
        subtasks=[
            SubTask(
                task_id="t1",
                description="什么是异步编程",
            )
        ],
    )

    assert orchestrator._should_run_critic(
        "什么是异步编程",
        plan,
    ) is True
    assert orchestrator._max_refine_rounds(plan) == 1


@pytest.mark.asyncio
async def test_conversational_task_skips_critic_and_refinement_in_both_paths() -> None:
    class Planner:
        async def run(self, task: str) -> ResearchPlan:
            return ResearchPlan(
                plan_id="greeting",
                original_task=task,
                subtasks=[SubTask(task_id="t1", description=task)],
            )

    class Researcher:
        async def run(self, task: str, *, context=None) -> AgentResult:
            del context
            return AgentResult(
                agent_name="researcher",
                output=f"{task}！有什么可以帮你？",
            )

    class UnexpectedCritic:
        async def evaluate(self, **_kwargs) -> CriticScore:
            raise AssertionError("conversational tasks must skip Critic")

    orchestrator = Orchestrator(
        planner=Planner(),
        researcher=Researcher(),
        synthesizer=SimpleNamespace(),
        critic=UnexpectedCritic(),
    )
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(
            research_mode="balanced",
            max_refine_rounds=2,
            research_timeout=10,
            subtask_timeout=5,
            queue_timeout=5,
            sse_heartbeat_seconds=1,
            stream_chunk_size=512,
            research_context_max_chars=12_000,
        ),
        llm=SimpleNamespace(llm_provider="test"),
        observability=SimpleNamespace(enable_tracing=False),
    )

    result = await orchestrator.run("你好")
    events = [event async for event in orchestrator.stream_run("你好")]
    streamed_result = next(
        event["result"] for event in events if event["type"] == "done"
    )

    assert result.success is True
    assert result.metadata["quality_status"] == "not_evaluated"
    assert result.metadata["refine_rounds"] == 0
    assert not any(event["type"] == "critic_feedback" for event in events)
    assert not any(event["type"] == "refining" for event in events)
    assert streamed_result.metadata["quality_status"] == "not_evaluated"
    assert streamed_result.metadata["refine_rounds"] == 0


def test_report_without_sources_drops_unbacked_citation_markers() -> None:
    report = (
        "Python 适合快速迭代 [1]，Java 适合企业集成 [2]。\n\n"
        "[文档](https://example.com) 保持不变。"
    )

    cleaned = Orchestrator._strip_unbacked_citations(report, [])

    assert "[1]" not in cleaned
    assert "[2]" not in cleaned
    assert "[文档](https://example.com)" in cleaned
    assert Orchestrator._strip_unbacked_citations(
        report,
        [{"index": 1, "url": "https://example.com"}],
    ) == report


def test_final_citation_status_reflects_report_verification() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator._settings = SimpleNamespace(
        llm=SimpleNamespace(llm_provider="test")
    )
    plan = ResearchPlan(
        plan_id="mixed-grounding",
        original_task="task",
        subtasks=[
            SubTask(task_id="t1", description="grounded"),
            SubTask(task_id="t2", description="model only"),
        ],
    )

    result = orchestrator._build_success_result(
        output="报告 [1]",
        plan=plan,
        subtask_outputs=[
            {
                "task_id": "t1",
                "success": True,
                "outcome": "success",
                "grounding_status": "grounded",
                "sources": [{"index": 1, "url": "https://example.com"}],
            },
            {
                "task_id": "t2",
                "success": True,
                "outcome": "success",
                "grounding_status": "model_only",
                "sources": [],
            },
        ],
        sources=[{"index": 1, "url": "https://example.com"}],
        final_critic=CriticScore(overall=8.0),
        refine_count=0,
        total_usage={},
        elapsed_ms=1,
        total_cost_usd=None,
        cost_status="usage_unavailable",
        citation_verification={"valid": True, "status": "valid"},
    )

    assert result.metadata["grounding_status"] == "model_only"
    assert result.metadata["citation_status"] == "valid"


def test_balanced_mode_refines_only_actionable_content_deficits() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(
            research_mode="balanced",
            critic_threshold=7.0,
        )
    )

    assert orchestrator._should_refine_report(
        CriticScore(overall=7.0, should_refine=True)
    ) is False
    assert orchestrator._should_refine_report(
        CriticScore(
            completeness=6.0,
            accuracy=8.0,
            depth=7.0,
            clarity=8.0,
            citations=8.0,
            overall=6.9,
            should_refine=False,
        )
    ) is True
    assert orchestrator._should_refine_report(
        CriticScore(
            completeness=8.0,
            accuracy=6.0,
            depth=7.0,
            clarity=8.0,
            citations=4.0,
            overall=6.6,
            should_refine=True,
        )
    ) is False


@pytest.mark.asyncio
async def test_deep_mode_evaluates_once_when_refinement_is_disabled() -> None:
    class Planner:
        async def run(self, task: str) -> ResearchPlan:
            return ResearchPlan(
                plan_id="deep",
                original_task=task,
                subtasks=[SubTask(task_id="t1", description=task)],
            )

    class Researcher:
        async def run(self, task: str, *, context=None) -> AgentResult:
            del context
            return AgentResult(agent_name="researcher", output=f"answer: {task}")

    class Critic:
        calls = 0

        async def evaluate(self, **_kwargs) -> CriticScore:
            self.calls += 1
            return CriticScore(overall=8.0, should_refine=False)

    critic = Critic()
    orchestrator = Orchestrator(
        planner=Planner(),
        researcher=Researcher(),
        synthesizer=SimpleNamespace(),
        critic=critic,
    )
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(
            research_mode="deep",
            max_refine_rounds=0,
            research_timeout=10,
            subtask_timeout=5,
            queue_timeout=5,
            research_context_max_chars=12_000,
        ),
        llm=SimpleNamespace(llm_provider="test"),
        observability=SimpleNamespace(enable_tracing=False),
    )

    result = await orchestrator.run("深入解释异步编程")

    assert critic.calls == 1
    assert result.metadata["quality"] == 8.0
    assert result.metadata["quality_status"] == "evaluated"


@pytest.mark.asyncio
async def test_stream_emits_final_critic_score_after_refinement() -> None:
    class Planner:
        async def run(self, task: str) -> ResearchPlan:
            return ResearchPlan(
                plan_id="deep",
                original_task=task,
                subtasks=[SubTask(task_id="t1", description=task)],
            )

    class Researcher:
        async def run(self, task: str, *, context=None) -> AgentResult:
            del context
            return AgentResult(agent_name="researcher", output=f"draft: {task}")

    class Synthesizer:
        async def synthesize(self, **_kwargs) -> AgentResult:
            return AgentResult(agent_name="synthesizer", output="refined")

    class Critic:
        calls = 0

        async def evaluate(self, **_kwargs) -> CriticScore:
            self.calls += 1
            if self.calls == 1:
                return CriticScore(overall=5.0, should_refine=True)
            return CriticScore(overall=8.5, should_refine=False)

    orchestrator = Orchestrator(
        planner=Planner(),
        researcher=Researcher(),
        synthesizer=Synthesizer(),
        critic=Critic(),
    )
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(
            research_mode="deep",
            max_refine_rounds=1,
            research_timeout=10,
            subtask_timeout=5,
            queue_timeout=5,
            sse_heartbeat_seconds=1,
            stream_chunk_size=512,
            research_context_max_chars=12_000,
        ),
        llm=SimpleNamespace(llm_provider="test"),
        observability=SimpleNamespace(enable_tracing=False),
    )

    events = [event async for event in orchestrator.stream_run("深入研究")]
    feedback = [
        event["score"].overall
        for event in events
        if event["type"] == "critic_feedback"
    ]
    result = next(event["result"] for event in events if event["type"] == "done")

    assert feedback == [5.0, 8.5]
    assert result.output == "refined"
    assert result.metadata["quality"] == 8.5


@pytest.mark.asyncio
async def test_refinement_failure_keeps_the_last_valid_report() -> None:
    class Planner:
        async def run(self, task: str) -> ResearchPlan:
            return ResearchPlan(
                plan_id="refine-failure",
                original_task=task,
                subtasks=[SubTask(task_id="t1", description=task)],
            )

    class Researcher:
        async def run(self, task: str, *, context=None) -> AgentResult:
            del context
            return AgentResult(agent_name="researcher", output=f"draft: {task}")

    class Synthesizer:
        async def synthesize(self, **kwargs) -> AgentResult:
            assert kwargs["max_attempts"] == 1
            raise asyncio.TimeoutError

    class Critic:
        async def evaluate(self, **_kwargs) -> CriticScore:
            return CriticScore(overall=6.5, should_refine=True)

    orchestrator = Orchestrator(
        planner=Planner(),
        researcher=Researcher(),
        synthesizer=Synthesizer(),
        critic=Critic(),
    )
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(
            research_mode="deep",
            max_refine_rounds=1,
            research_timeout=10,
            subtask_timeout=5,
            queue_timeout=5,
            sse_heartbeat_seconds=1,
            stream_chunk_size=512,
            research_context_max_chars=12_000,
        ),
        llm=SimpleNamespace(llm_provider="test"),
        observability=SimpleNamespace(enable_tracing=False),
    )

    events = [event async for event in orchestrator.stream_run("深入研究")]
    result = next(event["result"] for event in events if event["type"] == "done")

    assert not any(event["type"] == "error" for event in events)
    assert result.success is True
    assert result.output == "draft: 深入研究"
    assert result.metadata["outcome"] == "degraded"
    assert result.metadata["quality"] == 6.5
    assert result.metadata["refinement_status"] == "failed"
    assert result.data["refinement_failure"]


@pytest.mark.asyncio
async def test_sync_and_stream_cache_restore_the_same_result() -> None:
    cached = AgentResult(
        agent_name="orchestrator",
        output="Cached report [1].",
        data={
            "sources": [
                {
                    "index": 1,
                    "title": "Cached source",
                    "url": "https://example.com/cached",
                }
            ]
        },
        metadata={"quality": 9.0, "cost": 0.01},
        token_usage={"prompt_tokens": 5},
        cost_usd=0.01,
        cost_status="estimated",
    )

    class MemoryStub:
        async def recall(self, task: str):
            assert task == "cached task"
            return cached.to_dict()

    orchestrator = Orchestrator(
        planner=SimpleNamespace(),
        researcher=SimpleNamespace(),
        synthesizer=SimpleNamespace(),
        critic=SimpleNamespace(),
        episodic_memory=MemoryStub(),
    )
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(
            queue_timeout=5,
            sse_heartbeat_seconds=1,
            research_timeout=30,
        ),
        llm=SimpleNamespace(llm_provider="test"),
    )
    cached.metadata["research_cache_fingerprint"] = (
        orchestrator._research_cache_fingerprint()
    )

    sync_result = await orchestrator.run("cached task")
    events = [event async for event in orchestrator.stream_run("cached task")]
    stream_result = next(event["result"] for event in events if event["type"] == "done")

    assert sync_result.output == stream_result.output
    assert sync_result.data["sources"] == stream_result.data["sources"]
    assert sync_result.metadata == stream_result.metadata
    assert sync_result.token_usage == stream_result.token_usage == {}
    assert sync_result.cost_usd is stream_result.cost_usd is None
    assert sync_result.cost_status == stream_result.cost_status == "not_applicable"
    assert sync_result.metadata["cached_generation_token_usage"] == {
        "prompt_tokens": 5
    }
    assert sync_result.metadata["cached_generation_cost_usd"] == 0.01
    assert sync_result.metadata["cached_generation_cost_status"] == "estimated"
    assert sync_result.data["from_cache"] is True
    assert stream_result.data["from_cache"] is True


@pytest.mark.asyncio
async def test_legacy_cached_zero_without_critic_is_normalized_to_unreviewed() -> None:
    cached = AgentResult(
        agent_name="orchestrator",
        output="Legacy cached report.",
        data={"critic_score": None, "sources": []},
        metadata={"quality": 0.0, "outcome": "success"},
    )

    class MemoryStub:
        async def recall(self, task: str):
            assert task == "你好"
            return cached.to_dict()

    orchestrator = Orchestrator(
        planner=SimpleNamespace(),
        researcher=SimpleNamespace(),
        synthesizer=SimpleNamespace(),
        critic=SimpleNamespace(),
        episodic_memory=MemoryStub(),
    )
    cached.metadata["research_cache_fingerprint"] = (
        orchestrator._research_cache_fingerprint()
    )

    result = await orchestrator._recall_cached_result(
        "你好",
        start_time=time.perf_counter(),
    )

    assert result is not None
    assert result.metadata["quality"] is None
    assert result.metadata["quality_status"] == "not_evaluated"


@pytest.mark.asyncio
async def test_orchestrator_rejects_cache_from_an_old_execution_strategy() -> None:
    cached = AgentResult(
        agent_name="orchestrator",
        output="Old two-researcher comparison [1].",
        data={
            "sources": [
                {
                    "index": 1,
                    "title": "Source",
                    "url": "https://example.com/source",
                }
            ],
            "subtask_outputs": [
                {"task_id": "python", "success": True},
                {"task_id": "java", "success": True},
            ],
        },
        metadata={
            "quality": 8.0,
            "quality_status": "evaluated",
            "outcome": "success",
            "research_cache_fingerprint": "old-routing-strategy",
        },
    )

    class MemoryStub:
        async def recall(self, _task: str):
            return cached.to_dict()

    orchestrator = Orchestrator(
        planner=SimpleNamespace(),
        researcher=SimpleNamespace(),
        synthesizer=SimpleNamespace(),
        critic=SimpleNamespace(),
        episodic_memory=MemoryStub(),
    )

    result = await orchestrator._recall_cached_result(
        "Python and Java agent selection",
        start_time=time.perf_counter(),
    )

    assert result is None


@pytest.mark.asyncio
async def test_orchestrator_rejects_legacy_partial_cache_entries() -> None:
    class MemoryStub:
        async def recall(self, task: str):
            assert task == "partial cache"
            return AgentResult(
                agent_name="orchestrator",
                success=True,
                output="Incomplete cached report.",
                data={
                    "subtask_outputs": [
                        {
                            "task_id": "one",
                            "success": True,
                            "output": "finding",
                        },
                        {
                            "task_id": "two",
                            "success": False,
                            "output": "timed out",
                        },
                    ]
                },
            ).to_dict()

    orchestrator = Orchestrator(
        planner=SimpleNamespace(),
        researcher=SimpleNamespace(),
        synthesizer=SimpleNamespace(),
        critic=SimpleNamespace(),
        episodic_memory=MemoryStub(),
    )

    result = await orchestrator._recall_cached_result(
        "partial cache",
        start_time=time.perf_counter(),
    )

    assert result is None


def _empty_synthesis_orchestrator() -> Orchestrator:
    orchestrator = Orchestrator(
        planner=_FakePlanner(),
        researcher=_ContextResearcher(),
        synthesizer=_EmptySynthesizer(),
        critic=_NoopCritic(),
    )
    orchestrator._settings = SimpleNamespace(
        agent=SimpleNamespace(
            research_timeout=30,
            subtask_timeout=5,
            max_refine_rounds=0,
            stream_chunk_size=512,
        ),
        llm=SimpleNamespace(llm_provider="test"),
        observability=SimpleNamespace(enable_tracing=False),
    )
    return orchestrator


@pytest.mark.asyncio
async def test_partial_subtask_failure_is_degraded_and_not_cached() -> None:
    stored: list[AgentResult] = []

    class MemoryStub:
        async def store(self, task: str, result: AgentResult) -> None:
            del task
            stored.append(result)

    orchestrator = Orchestrator(
        planner=SimpleNamespace(),
        researcher=SimpleNamespace(),
        synthesizer=SimpleNamespace(),
        critic=SimpleNamespace(),
        episodic_memory=MemoryStub(),
    )
    orchestrator._settings = SimpleNamespace(
        llm=SimpleNamespace(llm_provider="test"),
    )
    plan = ResearchPlan(
        plan_id="partial",
        original_task="compare",
        subtasks=[
            SubTask(task_id="one", description="completed"),
            SubTask(task_id="two", description="timed out"),
        ],
    )
    result = orchestrator._build_success_result(
        output="Partial report.",
        plan=plan,
        subtask_outputs=[
            {
                "task_id": "one",
                "description": "completed",
                "success": True,
                "output": "finding",
                "sources": [],
            },
            {
                "task_id": "two",
                "description": "timed out",
                "success": False,
                "output": "Subtask 'two' timed out after 60s.",
                "sources": [],
            },
        ],
        sources=[],
        final_critic=CriticScore(overall=8.0, should_refine=False),
        refine_count=0,
        total_usage={"total_tokens": 10},
        elapsed_ms=1000,
        total_cost_usd=0.01,
        cost_status="estimated",
    )

    await orchestrator._store_memories("compare", result)

    assert result.success is True
    assert result.metadata["outcome"] == "degraded"
    assert result.metadata["completed_subtask_count"] == 1
    assert result.metadata["failed_subtask_count"] == 1
    assert "two" in result.metadata["failure_reason"]
    assert stored == []


@pytest.mark.asyncio
async def test_orchestrator_rejects_empty_synthesis() -> None:
    result = await _empty_synthesis_orchestrator().run("task")

    assert result.success is False
    assert "synthesis" in result.output.lower()


@pytest.mark.asyncio
async def test_streaming_orchestrator_rejects_empty_synthesis() -> None:
    events = [
        event async for event in _empty_synthesis_orchestrator().stream_run("task")
    ]

    done = [event for event in events if event["type"] == "done"]
    assert len(done) == 1
    assert done[0]["result"].success is False


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
        raise AssertionError("synthesizer must not run when all subtasks fail")

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
    assert all(not item["success"] for item in result.data["subtask_outputs"])


@pytest.mark.asyncio
async def test_streaming_orchestrator_emits_failed_done_result() -> None:
    events = [event async for event in _failed_orchestrator().stream_run("task")]

    done = [event for event in events if event["type"] == "done"]
    assert len(done) == 1
    assert done[0]["result"].success is False
    assert not any(event["type"] == "synthesizing" for event in events)


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
    Orchestrator._attach_citation_maps(outputs, sources)

    assert [source["url"] for source in sources] == [
        "https://a.example",
        "https://b.example",
        "https://c.example",
    ]
    assert [source["index"] for source in sources] == [1, 2, 3]
    assert outputs[0]["citation_map"] == {"1": 1, "2": 2}
    assert outputs[1]["citation_map"] == {"1": 2, "2": 3}


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
    result = CodeExecutor().execute("import os\nprint(os.getcwd())")

    assert result.success is False
    assert "Module not allowed" in (result.error or "")


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (
            "import numpy as np\nprint(int(np.array([1, 2, 3]).sum()))",
            "6",
        ),
        (
            "import pandas as pd\nprint(int(pd.Series([1, 2, 3]).sum()))",
            "6",
        ),
    ],
)
def test_code_executor_runs_whitelisted_scientific_modules(
    code: str,
    expected: str,
) -> None:
    result = CodeExecutor().execute(code, timeout=10)

    assert result.success is True, (result.data or {}).get(
        "stderr",
        result.error or "",
    )
    assert result.output.strip() == expected


def test_code_executor_blocks_scientific_library_file_access() -> None:
    result = CodeExecutor().execute(
        "import numpy as np\nprint(np.fromfile(target, dtype=np.uint8, count=8))",
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


@pytest.mark.parametrize(
    "code",
    [
        "import pandas as pd\npd.io.common.os.listdir('.')",
        (
            "import pandas as pd\n"
            "os_alias = pd.io.common.os\n"
            "os_alias.remove('zzzz_no_target_4f6a9c')"
        ),
    ],
)
def test_code_executor_blocks_aliased_filesystem_operations(
    code: str,
) -> None:
    result = CodeExecutor().execute(code, timeout=5)

    assert result.success is False
    assert "File API not allowed" in (result.error or "")


def test_code_executor_handles_non_ascii_child_errors() -> None:
    result = CodeExecutor().execute(
        "raise ValueError('中文错误')",
        timeout=5,
    )

    assert result.success is False
    assert "ValueError" in (result.error or "")


def test_bm25_search_is_consistent_during_concurrent_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mindforge.retrieval import bm25 as bm25_module

    monkeypatch.setattr(bm25_module, "_BM25S_AVAILABLE", True)
    monkeypatch.setattr(
        bm25_module,
        "jieba",
        SimpleNamespace(cut_for_search=lambda query: query.split()),
        raising=False,
    )
    retriever = BM25Retriever()
    retriever.documents = ["old document"]
    retriever.doc_ids = ["old"]
    retriever.metadatas = [{}]
    search_entered = threading.Event()
    release_search = threading.Event()
    build_finished = threading.Event()
    search_result: list[dict[str, Any]] = []

    class BlockingIndex:
        def retrieve(self, query_tokens, k):
            import numpy as np

            search_entered.set()
            assert release_search.wait(timeout=5)
            return np.array([[0]]), np.array([[1.0]])

    retriever.retriever = BlockingIndex()
    monkeypatch.setattr(retriever, "_rebuild_retriever", lambda: None)

    search_thread = threading.Thread(
        target=lambda: search_result.extend(retriever.search("old")),
    )

    def rebuild() -> None:
        retriever.build_index(
            [{"id": "new", "text": "new document"}],
        )
        build_finished.set()

    build_thread = threading.Thread(target=rebuild)

    search_thread.start()
    assert search_entered.wait(timeout=5)
    build_thread.start()
    build_completed_while_searching = build_finished.wait(timeout=0.2)
    release_search.set()
    search_thread.join(timeout=5)
    build_thread.join(timeout=5)

    assert build_completed_while_searching is False
    assert search_result[0]["id"] == "old"
    assert build_finished.is_set()


def test_bm25s_0310_result_order_is_indices_then_scores() -> None:
    retriever = BM25Retriever()
    retriever.build_index(
        [
            {"id": "alpha", "text": "alpha beta"},
            {"id": "gamma", "text": "gamma delta"},
        ]
    )

    results = retriever.search("alpha", top_k=2)

    assert results
    assert results[0]["id"] == "alpha"
    assert isinstance(results[0]["score"], float)


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
            {
                "revision": "immutable-revision",
                "device": "cpu",
                "local_files_only": True,
            },
        )
    ]


def test_reranker_model_load_failure_is_cached(monkeypatch) -> None:
    attempts = 0

    class FailingCrossEncoder:
        def __init__(self, *_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("model unavailable")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FailingCrossEncoder),
    )
    reranker = CrossEncoderReranker("unavailable")

    assert reranker.model is None
    assert reranker.model is None
    assert attempts == 1
    results = reranker.rerank(
        "query",
        [{"id": "first", "text": "first"}],
    )
    assert results == [{"id": "first", "text": "first"}]
    assert "rerank_score" not in results[0]


@pytest.mark.asyncio
async def test_rag_tool_does_not_retrieve_for_greeting() -> None:
    class UnexpectedRetriever:
        async def retrieve(self, **kwargs):
            del kwargs
            raise AssertionError("greetings must not query the knowledge base")

    result = await RAGTool(retriever=UnexpectedRetriever()).execute_async("你好")

    assert result.success is True
    assert result.data["intent"] == "conversation"
    assert result.data["sources"] == []
    assert "不会调用大模型进行闲聊" in result.output


@pytest.mark.asyncio
async def test_rag_tool_rejects_rrf_rank_without_relevance_evidence() -> None:
    class Retriever:
        async def retrieve(self, **kwargs):
            del kwargs
            return {
                "results": [
                    {
                        "id": "noise",
                        "text": "unrelated",
                        "score": 1.0,
                        "rrf_score": 1.0,
                        "semantic_score": 0.55,
                        "keyword_score": 0.0,
                        "retrieval_sources": ["vector"],
                    }
                ]
            }

    result = await RAGTool(retriever=Retriever()).execute_async(
        "今天天气怎么样",
        threshold=0.6,
    )

    assert result.success is True
    assert result.data["total"] == 0
    assert result.data["filtered_out"] == 1
    assert "暂无高度相关的资料" in result.output


@pytest.mark.asyncio
async def test_rag_tool_accepts_positive_keyword_evidence() -> None:
    class Retriever:
        async def retrieve(self, **kwargs):
            del kwargs
            return {
                "results": [
                    {
                        "id": "rag",
                        "text": "RAG 使用外部知识增强生成。",
                        "document_source": "rag.md",
                        "score": 1.0,
                        "rrf_score": 1.0,
                        "semantic_score": 0.4,
                        "keyword_score": 2.0,
                        "retrieval_sources": ["bm25", "vector"],
                    }
                ]
            }

    result = await RAGTool(retriever=Retriever()).execute_async(
        "RAG是什么",
        threshold=0.6,
    )

    assert result.success is True
    assert result.data["total"] == 1
    assert "rag.md" in result.output


@pytest.mark.asyncio
async def test_rag_tool_requires_all_explicit_technical_concepts() -> None:
    class Retriever:
        async def retrieve(self, **kwargs):
            del kwargs
            return {
                "results": [
                    {
                        "id": "python-only",
                        "text": "Python 虚拟环境、解释器和第三方库管理。",
                        "document_source": "python.md",
                        "score": 1.0,
                        "rrf_score": 1.0,
                        "semantic_score": 0.42,
                        "keyword_score": 8.0,
                        "retrieval_sources": ["bm25", "vector"],
                    }
                ]
            }

    result = await RAGTool(retriever=Retriever()).execute_async(
        "Python 和 Java 有什么区别",
        threshold=0.6,
    )

    assert result.success is True
    assert result.data["total"] == 0
    assert result.data["retrieval_quality"] == 0.0
    assert "暂无高度相关的资料" in result.output


@pytest.mark.asyncio
async def test_rag_tool_rejects_high_semantic_score_without_query_evidence() -> None:
    class Retriever:
        async def retrieve(self, **kwargs):
            del kwargs
            return {
                "results": [
                    {
                        "id": "false-positive",
                        "text": "这份材料讨论餐饮门店排班和库存管理。",
                        "document_source": "operations.md",
                        "semantic_score": 0.98,
                        "keyword_score": 0.0,
                        "retrieval_sources": ["vector"],
                    }
                ]
            }

    result = await RAGTool(retriever=Retriever()).execute_async(
        "量子计算在药物研发中的应用",
        threshold=0.6,
    )

    assert result.success is True
    assert result.data["total"] == 0
    assert result.data["retrieval_quality"] == 0.0


@pytest.mark.asyncio
async def test_rag_tool_reports_retrieval_quality_not_answer_quality() -> None:
    class Retriever:
        async def retrieve(self, **kwargs):
            del kwargs
            return {
                "results": [
                    {
                        "id": "comparison",
                        "text": "Python 是动态类型语言，Java 是静态类型语言。",
                        "document_source": "comparison.md",
                        "semantic_score": 0.75,
                        "keyword_score": 3.0,
                        "retrieval_sources": ["bm25", "vector"],
                    }
                ]
            }

    result = await RAGTool(retriever=Retriever()).execute_async(
        "Python 和 Java 有什么区别",
        threshold=0.6,
    )

    assert result.data["total"] == 1
    assert "quality" not in result.data
    assert result.data["retrieval_quality"] == pytest.approx(7.5)


def test_reranker_normalizes_unbounded_logits() -> None:
    class Model:
        @staticmethod
        def predict(pairs):
            assert len(pairs) == 2
            return [-2.0, 3.0]

    reranker = CrossEncoderReranker("test-model")
    reranker._model = Model()

    results = reranker.rerank(
        "query",
        [
            {"id": "low", "text": "low"},
            {"id": "high", "text": "high"},
        ],
        top_k=2,
    )

    assert [item["id"] for item in results] == ["high", "low"]
    assert 0.0 < results[0]["rerank_score"] < 1.0
    assert results[0]["rerank_score_raw"] == 3.0


def test_rag_tool_formats_code_evidence_as_fenced_markdown() -> None:
    output = RAGTool()._format_results(
        [
            {
                "text": ('from llama_cpp import Llama\nMODEL_KWARGS = {"n_ctx": 2048}'),
                "document_source": "example.pdf",
                "metadata": {"page": 12},
            },
            {
                "text": '<label>问题</label>\n<input type="text">',
                "document_source": "web.pdf",
            },
        ],
        "示例",
    )

    assert "```python" in output
    assert "```html" in output
    assert "example.pdf · 第 12 页" in output
    assert "未经总结或改写" in output


@pytest.mark.parametrize(
    ("snippet", "language"),
    [
        (
            "from pathlib import Path\ndef load_file(path):\n    return Path(path).read_text()",
            "python",
        ),
        ('<main>\n  <button type="button">Run</button>\n</main>', "html"),
        ("const greet = (name) => console.log(`Hello ${name}`);", "javascript"),
        ("interface User { id: number }\nconst user: User = { id: 1 };", "typescript"),
        (
            "import java.util.List;\n"
            "public class Main {\n"
            "  public static void main(String[] args) {}\n"
            "}",
            "java",
        ),
        ('using System;\nConsole.WriteLine("hello");', "csharp"),
        ('#include <iostream>\nint main() { std::cout << "hi"; }', "cpp"),
        ('#include <stdio.h>\nint main(void) { printf("hi"); }', "c"),
        ('package main\nfunc main() { fmt.Println("hi") }', "go"),
        ('fn main() {\n    println!("hi");\n}', "rust"),
        ('fun main() {\n    println("hi")\n}', "kotlin"),
        ('import Foundation\nprint("hello")', "swift"),
        ('<?php\n$name = "MindForge";\necho $name;', "php"),
        ('require "json"\ndef load_data\n  puts "ready"\nend', "ruby"),
        ("SELECT id, name\nFROM users\nWHERE active = true;", "sql"),
        ("query Research($id: ID!) {\n  document(id: $id) { title }\n}", "graphql"),
        ('{"provider": "local", "enabled": true}', "json"),
        ("services:\n  api:\n    image: mindforge", "yaml"),
        ('[server]\nhost = "127.0.0.1"\nport = 8000', "toml"),
        (".panel {\n  display: flex;\n}", "css"),
        ("FROM python:3.11-slim\nRUN pip install mindforge", "dockerfile"),
        ("Get-ChildItem | Where-Object { $_.Length -gt 0 }", "powershell"),
        ("#!/usr/bin/env bash\nset -euo pipefail\npython -m mindforge", "bash"),
    ],
)
def test_rag_tool_detects_mainstream_code_languages(
    snippet: str,
    language: str,
) -> None:
    assert RAGTool._detect_code_language(snippet) == language


def test_rag_tool_preserves_explicit_fenced_language() -> None:
    snippet = "```java\npublic class Main {}\n```"

    output = RAGTool()._format_results(
        [{"text": snippet, "document_source": "example.md"}],
        "示例",
    )

    assert output.count("```java") == 1
    assert "````java" not in output


def test_rag_tool_uses_text_fence_for_unknown_code() -> None:
    snippet = "BEGIN_WIDGET {\n  APPLY widget;\n}"

    output = RAGTool()._format_results(
        [{"text": snippet, "document_source": "unknown.txt"}],
        "示例",
    )

    assert "```text" in output


def test_rag_tool_keeps_normal_prose_out_of_code_fences() -> None:
    output = RAGTool()._format_results(
        [
            {
                "text": "这是普通说明文字。\n它不应被识别为任何编程语言。",
                "document_source": "notes.txt",
            }
        ],
        "示例",
    )

    assert "```" not in output
    assert "> 这是普通说明文字。" in output


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
    assert calls[0][1]["device"] == get_settings().llm.sentence_transformers_device


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
        subtasks=[SubTask("t1", "one", dependencies=["missing"])],
    )
    with pytest.raises(ValueError, match="unknown task"):
        unknown.validate()

    oversized = ResearchPlan(
        plan_id="oversized",
        original_task="task",
        subtasks=[SubTask(f"t{index}", f"task {index}") for index in range(6)],
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
    snapshot = SimpleNamespace(
        status="ok",
        qdrant_connected=True,
        redis_connected=True,
        postgres_connected=True,
    )

    class Monitor:
        async def get_snapshot(self):
            return snapshot

    monkeypatch.setattr(routes, "get_health_monitor", Monitor)
    response = await _api_get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["postgres_connected"] is True
    assert not any(key.startswith("mcp_") for key in payload)


@pytest.mark.asyncio
async def test_health_reports_degraded_when_core_dependency_fails(
    monkeypatch,
) -> None:
    snapshot = SimpleNamespace(
        status="degraded",
        qdrant_connected=False,
        redis_connected=True,
        postgres_connected=True,
    )

    class Monitor:
        async def get_snapshot(self):
            return snapshot

    monkeypatch.setattr(routes, "get_health_monitor", Monitor)
    response = await _api_get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["qdrant_connected"] is False
    assert not any(key.startswith("mcp_") for key in payload)


@pytest.mark.asyncio
async def test_readiness_uses_http_status_for_core_health(
    monkeypatch,
) -> None:
    class Monitor:
        snapshot = SimpleNamespace(
            status="ok",
            qdrant_connected=True,
            redis_connected=True,
            postgres_connected=True,
        )

        async def refresh(self):
            return self.snapshot

    monitor = Monitor()
    monkeypatch.setattr(routes, "get_health_monitor", lambda: monitor)
    ready = await _api_get("/api/v1/ready")
    assert ready.status_code == 200

    monitor.snapshot = SimpleNamespace(
        status="degraded",
        qdrant_connected=False,
        redis_connected=True,
        postgres_connected=True,
    )
    unavailable = await _api_get("/api/v1/ready")
    assert unavailable.status_code == 503
    assert unavailable.json()["status"] == "degraded"


@pytest.mark.asyncio
async def test_history_pagination_validation_happens_before_database() -> None:
    invalid_page = await _api_get("/api/v1/history?page=0")
    oversized_page = await _api_get("/api/v1/history?page_size=101")

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
        token_usage=None,
        sources=json.dumps(
            [
                {
                    "index": 1,
                    "title": "Example",
                    "url": "https://example.com/source",
                    "source": "web",
                    "content": "legacy unexpected field",
                }
            ]
        ),
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
    assert response.json()["sources"] == [
        {
            "index": 1,
            "title": "Example",
            "url": "https://example.com/source",
            "source": "web",
            "chunk_id": None,
            "doc_id": None,
        }
    ]


def test_frontend_path_rejects_sibling_prefix_traversal(
    monkeypatch,
    tmp_path,
) -> None:
    frontend = tmp_path / "dist"
    frontend.mkdir()
    monkeypatch.setattr(server, "_FRONTEND_PATH", frontend.resolve())

    assert (
        server._safe_frontend_candidate("assets/app.js")
        == (frontend / "assets" / "app.js").resolve()
    )
    assert server._safe_frontend_candidate("../dist-private/secret.txt") is None


@pytest.mark.asyncio
async def test_frontend_cache_headers_distinguish_html_and_assets() -> None:
    middleware = server.SecurityHeadersMiddleware(app=lambda *_: None)

    async def call_next(_request):
        return Response()

    html_request = Request(
        {"type": "http", "method": "GET", "path": "/settings", "headers": []}
    )
    html_response = await middleware.dispatch(html_request, call_next)
    assert html_response.headers["cache-control"] == (
        "no-store, no-cache, must-revalidate"
    )

    asset_request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/assets/settings-page.js",
            "headers": [],
        }
    )
    asset_response = await middleware.dispatch(asset_request, call_next)
    assert asset_response.headers["cache-control"] == (
        "public, max-age=31536000, immutable"
    )


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
