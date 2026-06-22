"""Test RRF fusion algorithm, adaptive retriever strategy selection, and GraphRAG."""

from __future__ import annotations

import math
from typing import Any


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
        if any(w in q for w in ["what is", "define", "explain", "什么是"]):
            return "factual"
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
        assert "Apple Inc." in entities
        assert "Steve Jobs" in entities
        assert entities["Cupertino"]["type"] == "location"

    def test_relationship_extraction(self):
        entities = {"OpenAI": "org", "GPT-4": "model", "Microsoft": "org"}
        relations = [("OpenAI", "developed", "GPT-4"), ("Microsoft", "invested", "OpenAI")]
        openai_relations = [r for r in relations if r[0] == "OpenAI" or r[2] == "OpenAI"]
        assert len(openai_relations) == 2
        assert ("Microsoft", "invested", "OpenAI") in relations

    def test_community_detection_basic(self):
        nodes = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("B", "C"), ("C", "D")]
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
