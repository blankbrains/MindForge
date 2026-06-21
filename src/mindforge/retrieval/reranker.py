from __future__ import annotations
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers.

    The underlying model is loaded lazily on the first call to
    :meth:`rerank` to avoid unnecessary memory usage at import time.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    # ------------------------------------------------------------------
    # Lazy-loading property
    # ------------------------------------------------------------------

    @property
    def model(self):
        """Lazy-loaded CrossEncoder instance."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
                logger.info(
                    "Loaded CrossEncoder model '%s'.", self.model_name
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install it with: pip install sentence-transformers"
                )
            except Exception:
                logger.exception(
                    "Failed to load CrossEncoder model '%s'.", self.model_name
                )
                raise
        return self._model

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Score query-document pairs with the cross-encoder and re-rank.

        Args:
            query: The original user query.
            candidates: A list of result dicts, each containing at least a
                ``text`` key.
            top_k: Number of top results to return. Defaults to all candidates.

        Returns:
            A new list sorted by cross-encoder score descending.
        """
        if not candidates:
            return []

        top_k = top_k or len(candidates)

        # Prepare query-doc pairs
        texts = [c.get("text", "") for c in candidates]
        pairs = [(query, text) for text in texts]

        try:
            scores = self.model.predict(pairs)
        except Exception:
            logger.exception("Cross-encoder scoring failed; returning original order.")
            scores = [0.0] * len(candidates)

        # Attach scores and re-sort
        reranked = []
        for candidate, score in zip(candidates, scores):
            reranked.append({**candidate, "rerank_score": float(score)})

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
