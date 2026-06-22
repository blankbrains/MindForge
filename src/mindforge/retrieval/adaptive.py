from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

from mindforge.retrieval.hybrid import HybridRetriever
from mindforge.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class QueryMode(str, Enum):
    """Intent modes for classifying a user query."""

    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    ANALYTICAL = "analytical"
    GRAPH = "graph"


@dataclass
class RetrievalConfig:
    """Configuration settings for a given retrieval strategy."""

    use_hyde: bool = False
    use_multi_query: bool = False
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    raptor_levels: int = 0
    use_graph: bool = False
    reasoning: str = ""


# ------------------------------------------------------------------
# Strategy map -- maps each query mode to a tuned retrieval config.
# ------------------------------------------------------------------

STRATEGY_MAP: Dict[QueryMode, RetrievalConfig] = {
    QueryMode.FACTUAL: RetrievalConfig(
        use_hyde=False,
        use_multi_query=False,
        vector_weight=0.7,
        bm25_weight=0.3,
        reasoning="Factual queries prefer high-precision dense vector search. "
        "BM25 provides complementary keyword coverage.",
    ),
    QueryMode.CONCEPTUAL: RetrievalConfig(
        use_hyde=True,
        use_multi_query=False,
        vector_weight=0.8,
        bm25_weight=0.2,
        reasoning="Conceptual queries benefit from HyDE to bridge the lexical gap "
        "and retrieve conceptually similar passages.",
    ),
    QueryMode.COMPARATIVE: RetrievalConfig(
        use_hyde=True,
        use_multi_query=True,
        vector_weight=0.5,
        bm25_weight=0.5,
        reasoning="Comparative queries need broad coverage from both dense and "
        "sparse retrievers; HyDE and multi-query expansion help capture "
        "all sides of the comparison.",
    ),
    QueryMode.PROCEDURAL: RetrievalConfig(
        use_hyde=False,
        use_multi_query=True,
        vector_weight=0.4,
        bm25_weight=0.6,
        reasoning="Procedural queries often contain distinctive keywords (step, "
        "how-to, etc.) that BM25 excels at; multi-query expansion captures "
        "synonymous phrasings.",
    ),
    QueryMode.ANALYTICAL: RetrievalConfig(
        use_hyde=True,
        use_multi_query=True,
        vector_weight=0.6,
        bm25_weight=0.4,
        raptor_levels=2,
        reasoning="Analytical queries require synthesising evidence; HyDE + "
        "multi-query maximise recall and RAPTOR adds hierarchical context.",
    ),
    QueryMode.GRAPH: RetrievalConfig(
        use_hyde=False,
        use_multi_query=False,
        vector_weight=1.0,
        bm25_weight=0.0,
        use_graph=True,
        reasoning="Graph queries route directly to the GraphRAG engine for "
        "entity-relation-community traversal.",
    ),
}


# ------------------------------------------------------------------
# Adaptive Retriever
# ------------------------------------------------------------------


class AdaptiveRetriever:
    """Adaptive retrieval pipeline that classifies query intent, selects a
    strategy, executes hybrid retrieval (+ GraphRAG when enabled), reranks,
    and returns results with an explanation of the reasoning."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        graph_engine=None,
        llm_fn=None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.graph_engine = graph_engine
        self.llm_fn = llm_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        mode: Optional[QueryMode] = None,
    ) -> Dict[str, Any]:
        """Classify the query, select strategy, retrieve, rerank, and return.

        Args:
            query: The user's search query.
            top_k: Number of final results to return.
            mode: Optional explicit mode override. When ``None``, the mode is
                inferred via LLM classification.

        Returns:
            A dict with keys:
              - ``results``: ranked list of result dicts
              - ``mode``: the ``QueryMode`` used
              - ``reasoning``: explanation of the strategy chosen
              - ``raw_results``: unreranked results per source (for debugging)
        """
        # Step 1: Classify intent
        if mode is not None:
            query_mode = mode
        else:
            query_mode = await self._classify_query(query)

        # Step 2: Select strategy
        config = STRATEGY_MAP.get(query_mode, STRATEGY_MAP[QueryMode.FACTUAL])

        logger.info(
            "Adaptive retrieval — mode=%s, hyde=%s, multi=%s, graph=%s",
            query_mode.value,
            config.use_hyde,
            config.use_multi_query,
            config.use_graph,
        )

        # Step 3: Execute retrieval
        all_results: List[Dict[str, Any]] = []
        raw_results: Dict[str, Any] = {}

        # 3a. Hybrid retrieval
        try:
            hybrid_results = await self.hybrid_retriever.retrieve(
                query=query,
                use_hyde=config.use_hyde,
                use_multi_query=config.use_multi_query,
                vector_weight=config.vector_weight,
                bm25_weight=config.bm25_weight,
                top_k=top_k * 2,
            )
            all_results.extend(hybrid_results)
            raw_results["hybrid"] = hybrid_results
        except Exception:
            logger.exception("Hybrid retrieval failed in adaptive pipeline.")

        # 3b. GraphRAG (if enabled and available)
        if config.use_graph and self.graph_engine is not None:
            try:
                graph_results = await self.graph_engine.query(
                    query=query,
                    top_k_entities=top_k,
                    top_k_communities=min(3, top_k),
                )
                all_results.extend(graph_results)
                raw_results["graph"] = graph_results
            except Exception:
                logger.exception("GraphRAG query failed in adaptive pipeline.")

        # Step 4: Rerank
        if self.reranker is not None and all_results:
            try:
                final_results = self.reranker.rerank(
                    query=query,
                    candidates=all_results,
                    top_k=top_k,
                )
            except Exception:
                logger.exception("Reranking failed; falling back to fused order.")
                all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
                final_results = all_results[:top_k]
        else:
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            final_results = all_results[:top_k]

        # Step 5: Return structured output
        return {
            "results": final_results,
            "mode": query_mode,
            "reasoning": config.reasoning,
            "raw_results": raw_results,
        }

    # ------------------------------------------------------------------
    # Private: query classification
    # ------------------------------------------------------------------

    async def _classify_query(self, query: str) -> QueryMode:
        """Use the LLM to classify the query into one of the ``QueryMode``
        values. Defaults to ``FACTUAL`` on failure."""
        if self.llm_fn is None:
            logger.warning("No LLM function provided; defaulting to FACTUAL mode.")
            return QueryMode.FACTUAL

        modes_str = ", ".join(f"'{m.value}'" for m in QueryMode)
        prompt = (
            "Classify the following user query into exactly one of these "
            f"intent categories: {modes_str}.\n\n"
            "Definitions:\n"
            "- factual: Seeking a specific fact or piece of information.\n"
            "- conceptual: Understanding a concept, idea, or definition.\n"
            "- comparative: Comparing two or more items, approaches, or ideas.\n"
            "- procedural: Learning how to do something (step-by-step).\n"
            "- analytical: Analysing data, trends, or relationships.\n"
            "- graph: Exploring entity relationships or graph structures.\n\n"
            "Reply with ONLY the category keyword, nothing else.\n\n"
            f"Query: {query}"
        )

        try:
            result = await self.llm_fn(prompt)
            result_clean = result.strip().lower()

            for mode in QueryMode:
                if mode.value in result_clean:
                    return mode

            logger.warning(
                "Could not parse LLM classification '%s'; defaulting to FACTUAL.",
                result_clean,
            )
            return QueryMode.FACTUAL
        except Exception:
            logger.exception("Query classification failed; defaulting to FACTUAL.")
            return QueryMode.FACTUAL
