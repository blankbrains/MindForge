from __future__ import annotations
from typing import List, Optional, Dict, Any
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import bm25s
    import jieba

    _BM25S_AVAILABLE = True
except ImportError:
    _BM25S_AVAILABLE = False
    logger.warning("bm25s or jieba not installed; BM25 will use fallback keyword matching.")


class BM25Retriever:
    """BM25 retriever using bm25s with jieba tokenization, with a fallback
    to simple keyword matching when the optional dependencies are unavailable."""

    def __init__(self, index_dir: str = ".bm25_index"):
        self.index_dir = Path(index_dir)
        self.retriever = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """Build a BM25 index from a list of document dicts.

        Each dict should have at least ``id`` and ``text`` keys.
        """
        self.documents = [d.get("text", "") for d in documents]
        self.doc_ids = [d.get("id", str(i)) for i, d in enumerate(documents)]

        if _BM25S_AVAILABLE:
            try:
                tokenized = self._tokenize(self.documents)
                self.retriever = bm25s.BM25()
                self.retriever.index(tokenized)
                logger.info(
                    "Built BM25 index with %d documents.", len(self.documents)
                )
            except Exception:
                logger.exception("Failed to build BM25 index; falling back.")
                self.retriever = None
        else:
            logger.info(
                "bm25s not available; falling back to simple keyword matching."
            )

    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenize a list of texts using jieba for Chinese support."""
        return [list(jieba.cut(t)) for t in texts]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search the BM25 index and return ranked results.

        Returns a list of dicts with keys: ``id``, ``text``, ``score``.
        """
        if not self.documents:
            logger.warning("BM25 index is empty; returning empty results.")
            return []

        if self.retriever is not None and _BM25S_AVAILABLE:
            try:
                # bm25s retrieve 期望 corpus 为 token 列表的列表
                query_tokens = [list(jieba.cut_for_search(query))]
                scores, indices = self.retriever.retrieve(
                    query_tokens, k=min(top_k, len(self.documents))
                )
                results = []
                # bm25s returns shape (1, k) arrays; -1 表示无命中填充位
                for rank in range(indices.shape[1]):
                    doc_idx = indices[0, rank]
                    if doc_idx < 0 or doc_idx >= len(self.documents):
                        continue
                    score = float(scores[0, rank])
                    results.append(
                        {
                            "id": self.doc_ids[doc_idx],
                            "text": self.documents[doc_idx],
                            "score": score,
                        }
                    )
                return results
            except Exception:
                logger.exception("BM25 search failed; falling back to keyword match.")
                return self._keyword_fallback(query, top_k)
        else:
            return self._keyword_fallback(query, top_k)

    def _keyword_fallback(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple keyword matching fallback when bm25s is unavailable."""
        query_lower = query.lower()
        query_terms = query_lower.split()

        scored: List[Dict[str, Any]] = []
        for i, doc_text in enumerate(self.documents):
            doc_lower = doc_text.lower()
            score = sum(1 for term in query_terms if term in doc_lower)
            if score > 0:
                scored.append(
                    {
                        "id": self.doc_ids[i],
                        "text": doc_text,
                        "score": float(score),
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Save the BM25 index to disk."""
        save_dir = Path(path) if path else self.index_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.retriever is not None and _BM25S_AVAILABLE:
            try:
                self.retriever.save(str(save_dir))
                logger.info("BM25 index saved to '%s'.", save_dir)
            except Exception:
                logger.exception("Failed to save BM25 index.")

        # Always persist document metadata so the index is self-contained.
        meta = {"doc_ids": self.doc_ids, "documents": self.documents}
        with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def load(self, path: Optional[str] = None) -> bool:
        """Load a BM25 index from disk. Returns True on success."""
        load_dir = Path(path) if path else self.index_dir
        meta_path = load_dir / "meta.json"

        if not load_dir.exists() or not meta_path.exists():
            logger.warning("No saved BM25 index found at '%s'.", load_dir)
            return False

        # Load metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.doc_ids = meta.get("doc_ids", [])
        self.documents = meta.get("documents", [])

        # Load BM25 retriever
        if _BM25S_AVAILABLE:
            try:
                self.retriever = bm25s.BM25()
                self.retriever.load(str(load_dir))
                logger.info("BM25 index loaded from '%s' (%d docs).", load_dir, len(self.documents))
                return True
            except Exception:
                logger.exception("Failed to load BM25 index; metadata loaded, but retriever unavailable.")
                self.retriever = None
                return False
        else:
            logger.info("bm25s not available; loaded document metadata only.")
            return True
