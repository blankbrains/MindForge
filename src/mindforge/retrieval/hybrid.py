from __future__ import annotations
from typing import List, Optional, Dict, Any
import logging

from mindforge.retrieval.vector_store import QdrantStore
from mindforge.retrieval.bm25 import BM25Retriever

logger = logging.getLogger(__name__)

_RRF_K = 60


class HybridRetriever:
    """Hybrid retriever combining dense vector search, HyDE, and multi-query BM25
    with Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: Optional[QdrantStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        embedding_fn=None,
        llm_fn=None,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embedding_fn = embedding_fn
        self.llm_fn = llm_fn

    # ------------------------------------------------------------------
    # Main retrieval entry point
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        use_hyde: bool = False,
        use_multi_query: bool = False,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Execute hybrid retrieval with configurable paths.

        Returns a list of result dicts with keys: ``id``, ``text``, ``score``,
        and ``source`` (one of ``vector``, ``hyde``, ``multi_query``).
        """
        rankings: Dict[str, List[Dict[str, Any]]] = {}

        # --- Path 1: Direct vector search ---
        if self.vector_store is not None and self.embedding_fn is not None:
            try:
                dense_vec = await self.embedding_fn(query)
                vector_hits = await self.vector_store.search(
                    vector=dense_vec, top_k=top_k
                )
                rankings["vector"] = [
                    {
                        **payload,
                        "id": payload.get("chunk_id", ""),
                        "text": payload.get("content", ""),
                        "document_source": payload.get("source", ""),
                        "score": float(score),
                        "source": "vector",
                    }
                    for payload, score in vector_hits
                    if payload.get("chunk_id")
                ]
            except Exception:
                logger.exception("Direct vector search failed.")

        # --- Path 2: Direct BM25 search ---
        if self.bm25_retriever is not None:
            try:
                bm25_hits = self.bm25_retriever.search(query=query, top_k=top_k)
                rankings["bm25"] = [
                    {
                        **hit,
                        "document_source": hit.get("source", ""),
                        "source": "bm25",
                    }
                    for hit in bm25_hits
                    if hit.get("id")
                ]
            except Exception:
                logger.exception("Direct BM25 retrieval failed.")

        # --- Path 3: HyDE (Hypothetical Document Embedding) ---
        if use_hyde and self.llm_fn is not None:
            try:
                hyp_doc = await self._generate_hypothetic(query)
                if hyp_doc and self.vector_store is not None and self.embedding_fn is not None:
                    hyp_vec = await self.embedding_fn(hyp_doc)
                    hyde_hits = await self.vector_store.search(
                        vector=hyp_vec, top_k=top_k
                    )
                    rankings["hyde"] = [
                        {
                            **payload,
                            "id": payload.get("chunk_id", ""),
                            "text": payload.get("content", ""),
                            "document_source": payload.get("source", ""),
                            "score": float(score),
                            "source": "hyde",
                        }
                        for payload, score in hyde_hits
                        if payload.get("chunk_id")
                    ]
            except Exception:
                logger.exception("HyDE retrieval failed.")

        # --- Path 4: Multi-Query BM25 ---
        if use_multi_query and self.llm_fn is not None and self.bm25_retriever is not None:
            try:
                multi_queries = await self._generate_multi_queries(query)
                multi_query_hits: List[Dict[str, Any]] = []
                for mq in multi_queries:
                    bm25_hits = self.bm25_retriever.search(query=mq, top_k=top_k)
                    for hit in bm25_hits:
                        multi_query_hits.append(
                            {
                                **hit,
                                "document_source": hit.get("source", ""),
                                "source": "multi_query",
                            }
                        )
                rankings["multi_query"] = multi_query_hits
            except Exception:
                logger.exception("Multi-query BM25 retrieval failed.")

        # --- Fuse with weighted RRF ---
        fused = self._rrf_fuse(
            rankings, top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
        return fused

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _generate_hypothetic(self, query: str) -> Optional[str]:
        """Generate a hypothetical document that would answer the query.

        The LLM produces a concise passage that contains the information needed
        to answer the question. This passage is then embedded and used for
        vector search (HyDE).
        """
        if self.llm_fn is None:
            return None

        prompt = (
            "You are a knowledgeable assistant. Given the following question, "
            "write a concise hypothetical document passage that would contain "
            "the answer to this question. Write only the passage, no extra text.\n\n"
            f"Question: {query}"
        )
        try:
            result = await self.llm_fn(prompt)
            return result.strip() if result else None
        except Exception:
            logger.exception("Failed to generate hypothetical document.")
            return None

    async def _generate_multi_queries(self, query: str) -> List[str]:
        """Generate multiple query reformulations from three distinct angles.

        Returns up to 3 query strings.
        """
        if self.llm_fn is None:
            return [query]

        prompt = (
            "You are a search query expansion assistant. Given the original "
            "user question, produce exactly three reformulations from the "
            "following angles:\n"
            "1. **Factual**: A precise, keyword-focused rephrasing.\n"
            "2. **Conceptual**: A broader phrasing that captures the underlying "
            "concept.\n"
            "3. **Specific**: A narrow, detail-oriented phrasing.\n\n"
            "Output each on its own line, prefixed with '1.','2.','3.'.\n\n"
            f"Original: {query}"
        )
        try:
            result = await self.llm_fn(prompt)
            lines = [
                line.strip()
                for line in result.strip().split("\n")
                if line.strip()
            ]
            queries = []
            for line in lines:
                # Strip leading numbering like "1. " or "1:"
                clean = line.split(".", 1)[-1].strip() if "." in line[:3] else line
                if clean:
                    queries.append(clean)
            return queries[:3] if queries else [query]
        except Exception:
            logger.exception("Failed to generate multi-queries.")
            return [query]

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(
        rankings: Dict[str, List[Dict[str, Any]]],
        top_k: int,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Fuse independently ranked result lists with weighted pure RRF.

        Raw scores are intentionally ignored because cosine similarity,
        BM25, and graph scores have incompatible scales. Each retrieval path
        contributes only its rank position, which is the invariant RRF is
        designed to combine.
        """
        fused: Dict[str, Dict[str, Any]] = {}
        sources_by_doc: Dict[str, set[str]] = {}

        for ranking_name, results in rankings.items():
            if ranking_name in ("vector", "hyde"):
                weight = vector_weight
            elif ranking_name in ("bm25", "multi_query"):
                weight = bm25_weight
            else:
                weight = 1.0

            seen_in_ranking: set[str] = set()
            for rank, doc in enumerate(results, start=1):
                doc_id = str(doc.get("id", ""))
                if not doc_id or doc_id in seen_in_ranking:
                    continue
                seen_in_ranking.add(doc_id)

                if doc_id not in fused:
                    fused[doc_id] = {
                        **doc,
                        "id": doc_id,
                        "text": doc.get("text", doc.get("content", "")),
                        "score": 0.0,
                    }
                    sources_by_doc[doc_id] = set()

                fused[doc_id]["score"] += weight / (_RRF_K + rank)
                sources_by_doc[doc_id].add(ranking_name)

        # Normalize scores to [0, 1]
        max_score = max((d["score"] for d in fused.values()), default=0.0)
        if max_score > 0:
            for d in fused.values():
                d["score"] = d["score"] / max_score

        sorted_docs = sorted(
            fused.values(), key=lambda x: x["score"], reverse=True
        )
        for doc in sorted_docs:
            retrieval_sources = sorted(sources_by_doc.get(doc["id"], set()))
            doc["retrieval_sources"] = retrieval_sources
            doc["source"] = retrieval_sources[0] if retrieval_sources else "unknown"

        return sorted_docs[:top_k]
