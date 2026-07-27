"""Application-wide retrieval wiring and persistent auxiliary indexes."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from mindforge.config import (
    get_project_root,
    get_settings,
    resolve_project_path,
)
from mindforge.ingestion.embedder import get_embedder
from mindforge.models.base import ChatMessage, LLMFactory
from mindforge.retrieval.adaptive import AdaptiveRetriever
from mindforge.retrieval.bm25 import BM25Retriever
from mindforge.retrieval.graphrag import GraphRAGEngine
from mindforge.retrieval.hybrid import HybridRetriever
from mindforge.retrieval.reranker import CrossEncoderReranker
from mindforge.retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)

_PROJECT_ROOT = get_project_root()
_retriever: AdaptiveRetriever | None = None
_bm25: BM25Retriever | None = None
_graph: GraphRAGEngine | None = None
_index_lock = asyncio.Lock()


def _resolve_project_path(value: str | None, fallback: Path) -> Path:
    if not value:
        return fallback
    return resolve_project_path(value, root=_PROJECT_ROOT)


def _bm25_path() -> Path:
    configured = get_settings().retrieval.bm25_index_dir
    return _resolve_project_path(
        configured,
        _PROJECT_ROOT / "data" / "bm25",
    )


def _graph_path() -> Path:
    configured = get_settings().graphrag.graph_store_path
    return _resolve_project_path(
        configured,
        _PROJECT_ROOT / "data" / "graphrag.json",
    )


async def _llm_text(prompt: str) -> str:
    settings = get_settings()
    llm = LLMFactory.create(
        settings.llm.llm_provider,
        settings.llm.get_model("researcher"),
    )
    result = await llm.chat(
        [ChatMessage(role="user", content=prompt)],
        temperature=0.2,
    )
    return result.content or ""


def get_bm25_retriever() -> BM25Retriever:
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Retriever(index_dir=str(_bm25_path()))
        _bm25.load()
    return _bm25


def get_graph_engine(*, llm: Any = None) -> GraphRAGEngine:
    global _graph
    if _graph is None:
        _graph = GraphRAGEngine(llm_fn=llm)
        path = _graph_path()
        if path.is_file():
            try:
                _graph.load(str(path))
            except Exception:
                logger.exception("Failed to load persistent GraphRAG index.")
    elif llm is not None:
        _graph.llm_fn = llm
    return _graph


def get_retriever() -> AdaptiveRetriever:
    global _retriever
    if _retriever is None:
        settings = get_settings()
        embedder = get_embedder()

        async def _async_embed(text: str) -> list[float]:
            return await asyncio.to_thread(embedder.embed_single, text)

        reranker_model = settings.retrieval.reranker_model or None
        _retriever = AdaptiveRetriever(
            hybrid_retriever=HybridRetriever(
                vector_store=get_vector_store(),
                bm25_retriever=get_bm25_retriever(),
                embedding_fn=_async_embed,
                llm_fn=_llm_text,
            ),
            reranker=(
                CrossEncoderReranker(
                    reranker_model,
                    model_revision=(
                        settings.retrieval.reranker_model_revision
                    ),
                    max_candidates=(
                        settings.retrieval.reranker_max_candidates
                    ),
                )
                if reranker_model
                else None
            ),
            graph_engine=(
                get_graph_engine()
                if settings.graphrag.graph_enabled
                else None
            ),
            llm_fn=_llm_text,
            max_request_top_k=settings.retrieval.max_request_top_k,
        )
    return _retriever


async def index_auxiliary_documents(
    documents: list[dict[str, Any]],
    *,
    graph_llm: Any = None,
    use_graphrag: bool = False,
) -> None:
    """Persist BM25 and optional GraphRAG indexes for uploaded chunks."""
    if not documents:
        return
    async with _index_lock:
        bm25 = get_bm25_retriever()
        await asyncio.to_thread(bm25.upsert_documents, documents)
        await asyncio.to_thread(bm25.save)

        if use_graphrag:
            graph = get_graph_engine(llm=graph_llm)
            await graph.build_graph(documents)
            await asyncio.to_thread(graph.save, str(_graph_path()))


async def delete_auxiliary_document(doc_id: str) -> None:
    """Delete a document from persistent BM25 and GraphRAG indexes."""
    async with _index_lock:
        bm25 = get_bm25_retriever()
        removed = await asyncio.to_thread(bm25.delete_document, doc_id)
        if removed:
            await asyncio.to_thread(bm25.save)

        graph = get_graph_engine()
        graph.delete_document(doc_id)
        await asyncio.to_thread(graph.save, str(_graph_path()))


def reset_retrieval_service() -> None:
    """Drop cached retrieval components after runtime configuration changes."""
    global _retriever, _bm25, _graph
    _retriever = None
    _bm25 = None
    _graph = None
