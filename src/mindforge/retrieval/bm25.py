from __future__ import annotations
from typing import List, Optional, Dict, Any
import json
import logging
import threading
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
        from mindforge.config import get_settings

        self.index_dir = Path(index_dir)
        self.max_chunks = get_settings().retrieval.bm25_max_chunks
        self.retriever = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """Build a BM25 index from a list of document dicts.

        Each dict should have at least ``id`` and ``text`` keys.
        """
        if len(documents) > self.max_chunks:
            raise ValueError(
                "BM25 corpus exceeds the configured chunk limit of "
                f"{self.max_chunks}."
            )
        new_docs = [d.get("text", "") for d in documents]
        new_ids = [str(d.get("id", i)) for i, d in enumerate(documents)]
        new_metadatas = [
            {
                key: value
                for key, value in d.items()
                if key not in {"id", "text", "score"}
            }
            for d in documents
        ]

        with self._lock:
            self.documents = new_docs
            self.doc_ids = new_ids
            self.metadatas = new_metadatas
            self._rebuild_retriever()

    def upsert_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Insert or replace documents by id and rebuild the sparse index."""
        if not documents:
            return
        with self._lock:
            existing = {
                doc_id: {
                    "id": doc_id,
                    "text": self.documents[index],
                    **(
                        self.metadatas[index]
                        if index < len(self.metadatas)
                        else {}
                    ),
                }
                for index, doc_id in enumerate(self.doc_ids)
            }
            for index, document in enumerate(documents):
                doc_id = str(document.get("id", index))
                existing[doc_id] = {**document, "id": doc_id}
            if len(existing) > self.max_chunks:
                raise ValueError(
                    "BM25 corpus exceeds the configured chunk limit of "
                    f"{self.max_chunks}."
                )
            self.build_index(list(existing.values()))

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document and return the count."""
        with self._lock:
            kept: List[Dict[str, Any]] = []
            removed = 0
            for index, chunk_id in enumerate(self.doc_ids):
                metadata = (
                    self.metadatas[index]
                    if index < len(self.metadatas)
                    else {}
                )
                if metadata.get("doc_id") == doc_id:
                    removed += 1
                    continue
                kept.append(
                    {
                        "id": chunk_id,
                        "text": self.documents[index],
                        **metadata,
                    }
                )
            if removed:
                self.build_index(kept)
            return removed

    def _rebuild_retriever(self) -> None:
        if not self.documents:
            self.retriever = None
            return
        if _BM25S_AVAILABLE:
            try:
                tokenized = self._tokenize(self.documents)
                new_retriever = bm25s.BM25()
                new_retriever.index(tokenized)
                self.retriever = new_retriever
                logger.info(
                    "Built BM25 index with %d documents.",
                    len(self.documents),
                )
                return
            except Exception:
                logger.exception("Failed to build BM25 index; falling back.")
        else:
            logger.info(
                "bm25s not available; falling back to simple keyword matching."
            )
        self.retriever = None

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
                            **(
                                self.metadatas[doc_idx]
                                if doc_idx < len(self.metadatas)
                                else {}
                            ),
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
                        **(
                            self.metadatas[i]
                            if i < len(self.metadatas)
                            else {}
                        ),
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

        # Persist the corpus rather than bm25s internals. Rebuilding on load
        # is deterministic and avoids version-specific serialized formats.
        meta = {
            "doc_ids": self.doc_ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }
        tmp_path = save_dir / ".meta.json.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        tmp_path.replace(save_dir / "meta.json")

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
        self.metadatas = meta.get("metadatas", [{} for _ in self.doc_ids])
        if len(self.metadatas) != len(self.doc_ids):
            self.metadatas = [{} for _ in self.doc_ids]
        self._rebuild_retriever()
        logger.info(
            "BM25 corpus loaded from '%s' (%d docs).",
            load_dir,
            len(self.documents),
        )
        return True
