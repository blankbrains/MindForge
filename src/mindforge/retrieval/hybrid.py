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
        all_results: List[Dict[str, Any]] = []

        # --- Path 1: Direct vector search ---
        if self.vector_store is not None and self.embedding_fn is not None:
            try:
                dense_vec = await self.embedding_fn(query)
                vector_hits = await self.vector_store.search(
                    vector=dense_vec, top_k=top_k
                )
                for payload, score in vector_hits:
                    all_results.append(
                        {
                            "id": payload.get("chunk_id", ""),
                            "text": payload.get("content", ""),
                            "score": float(score),
                            "source": "vector",
                        }
                    )
            except Exception:
                logger.exception("Direct vector search failed.")

        # --- Path 2: HyDE (Hypothetical Document Embedding) ---
        if use_hyde and self.llm_fn is not None:
            try:
                hyp_doc = await self._generate_hypothetic(query)
                if hyp_doc and self.vector_store is not None and self.embedding_fn is not None:
                    hyp_vec = await self.embedding_fn(hyp_doc)
                    hyde_hits = await self.vector_store.search(
                        vector=hyp_vec, top_k=top_k
                    )
                    for payload, score in hyde_hits:
                        all_results.append(
                            {
                                "id": payload.get("chunk_id", ""),
                                "text": payload.get("content", ""),
                                "score": float(score),
                                "source": "hyde",
                            }
                        )
            except Exception:
                logger.exception("HyDE retrieval failed.")

        # --- Path 3: Multi-Query BM25 ---
        if use_multi_query and self.llm_fn is not None and self.bm25_retriever is not None:
            try:
                multi_queries = await self._generate_multi_queries(query)
                for mq in multi_queries:
                    bm25_hits = self.bm25_retriever.search(query=mq, top_k=top_k)
                    for hit in bm25_hits:
                        all_results.append(
                            {
                                "id": hit["id"],
                                "text": hit["text"],
                                "score": hit["score"],
                                "source": "multi_query",
                            }
                        )
            except Exception:
                logger.exception("Multi-query BM25 retrieval failed.")

        # --- Fuse with weighted RRF ---
        fused = self._rrf_fuse(
            all_results, top_k,
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
        results: List[Dict[str, Any]],
        top_k: int,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Weighted Reciprocal Rank Fusion with constant k = 60.

        RRF scores are scaled by 60 to produce values in [0, 1] range,
        then multiplied by the raw cosine similarity score to preserve
        semantic information from the embedding model.
        """
        fused: Dict[str, Dict[str, Any]] = {}
        source_map: Dict[str, str] = {}

        for rank, doc in enumerate(results):
            doc_id = doc["id"]
            raw_score = doc.get("score", 0.0)

            if doc_id not in fused:
                fused[doc_id] = {
                    "id": doc_id,
                    "text": doc["text"],
                    "score": 0.0,
                }
                source_map[doc_id] = doc.get("source", "unknown")

            # Determine path weight
            src = doc.get("source", "unknown")
            if src in ("vector", "hyde"):
                w = vector_weight
            elif src == "multi_query":
                w = bm25_weight
            else:
                w = 0.5

            # Weighted RRF contribution, scaled and mixed with raw score
            # RRF: 1/(k+rank) ≈ 0.016 → scale by 60 → ~1.0 for top result
            rrf = w * _RRF_K / (rank + _RRF_K)
            # Blend RRF rank signal with raw semantic score (60:40)
            fused[doc_id]["score"] += 0.6 * rrf + 0.4 * raw_score

        # Normalize scores to [0, 1]
        max_score = max((d["score"] for d in fused.values()), default=0.0)
        if max_score > 0:
            for d in fused.values():
                d["score"] = d["score"] / max_score

        sorted_docs = sorted(
            fused.values(), key=lambda x: x["score"], reverse=True
        )
        for doc in sorted_docs:
            doc["source"] = source_map.get(doc["id"], "unknown")

        return sorted_docs[:top_k]
