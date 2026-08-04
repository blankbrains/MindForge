"""Application-wide retrieval wiring and persistent auxiliary indexes."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
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
_reranker: CrossEncoderReranker | None = None
_retrieval_llm: Any = None
_index_lock = asyncio.Lock()
_service_lock = threading.RLock()


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
    llm = _get_retrieval_llm()
    result = await llm.chat(
        [ChatMessage(role="user", content=prompt)],
        temperature=0.2,
    )
    return result.content or ""


def _get_retrieval_llm() -> Any:
    global _retrieval_llm
    with _service_lock:
        if _retrieval_llm is None:
            settings = get_settings()
            _retrieval_llm = LLMFactory.create(
                settings.llm.llm_provider,
                settings.llm.get_model("researcher"),
            )
        return _retrieval_llm


def get_bm25_retriever() -> BM25Retriever:
    global _bm25
    with _service_lock:
        if _bm25 is None:
            _bm25 = BM25Retriever(index_dir=str(_bm25_path()))
            _bm25.load()
        return _bm25


def get_graph_engine(
    *,
    llm: Any = None,
    entity_llm: Any = None,
    summary_llm: Any = None,
) -> GraphRAGEngine:
    global _graph
    with _service_lock:
        if _graph is None:
            _graph = GraphRAGEngine(
                llm_fn=llm,
                entity_llm=entity_llm,
                summary_llm=summary_llm,
            )
            path = _graph_path()
            if path.is_file():
                try:
                    _graph.load(str(path))
                except Exception:
                    logger.exception("Failed to load persistent GraphRAG index.")
        else:
            if llm is not None:
                _graph.llm_fn = llm
            if entity_llm is not None:
                _graph.entity_llm = entity_llm
            if summary_llm is not None:
                _graph.summary_llm = summary_llm
        return _graph


def get_reranker() -> CrossEncoderReranker | None:
    global _reranker
    settings = get_settings().retrieval
    model_name = settings.reranker_model or None
    if model_name is None:
        return None
    with _service_lock:
        if _reranker is None:
            _reranker = CrossEncoderReranker(
                model_name,
                model_revision=settings.reranker_model_revision,
                max_candidates=settings.reranker_max_candidates,
                device=settings.reranker_device,
                local_files_only=getattr(
                    settings,
                    "reranker_local_files_only",
                    True,
                ),
            )
        return _reranker


async def preload_reranker() -> bool:
    reranker = get_reranker()
    if reranker is None:
        return False
    return await asyncio.to_thread(reranker.preload)


def get_reranker_status() -> dict[str, bool]:
    settings = get_settings().retrieval
    configured = bool(settings.reranker_model)
    with _service_lock:
        reranker = _reranker
        return {
            "configured": configured,
            "available": bool(
                reranker is not None and reranker._model is not None
            ),
            "load_failed": bool(
                reranker is not None and reranker._load_failed
            ),
        }


def get_retriever() -> AdaptiveRetriever:
    global _retriever
    with _service_lock:
        if _retriever is None:
            settings = get_settings()
            embedder = get_embedder()

            async def _async_embed(text: str) -> list[float]:
                return await asyncio.to_thread(embedder.embed_single, text)

            from mindforge.repositories.documents import (
                list_disabled_document_ids,
            )

            _retriever = AdaptiveRetriever(
                hybrid_retriever=HybridRetriever(
                    vector_store=get_vector_store(),
                    bm25_retriever=get_bm25_retriever(),
                    embedding_fn=_async_embed,
                    llm_fn=_llm_text,
                    bm25_top_k=settings.retrieval.bm25_top_k,
                ),
                reranker=get_reranker(),
                graph_engine=(
                    get_graph_engine() if settings.graphrag.graph_enabled else None
                ),
                llm_fn=_llm_text,
                max_request_top_k=settings.retrieval.max_request_top_k,
                retrieval_top_k=settings.retrieval.vector_top_k,
                rerank_top_k=settings.retrieval.rerank_top_k,
                disabled_document_ids_fn=list_disabled_document_ids,
            )
        return _retriever


async def index_auxiliary_documents(
    documents: list[dict[str, Any]],
    *,
    graph_entity_llm: Any = None,
    graph_summary_llm: Any = None,
    use_graphrag: bool = False,
    progress_callback: Any = None,
    timings: dict[str, float] | None = None,
    start_progress: float = 88.0,
    commit_callback: Callable[[bool], Awaitable[None]] | None = None,
) -> bool:
    """Persist BM25 and optional GraphRAG indexes for uploaded chunks."""
    if not documents:
        if commit_callback is not None:
            await commit_callback(False)
        return False
    stage_timings = timings if timings is not None else {}
    async with _index_lock:
        bm25 = get_bm25_retriever()
        doc_id = str(documents[0].get("doc_id") or "")
        bm25_snapshot = await asyncio.to_thread(
            bm25.get_document_chunks,
            doc_id,
        ) if doc_id else []
        graph = get_graph_engine(
            entity_llm=graph_entity_llm,
            summary_llm=graph_summary_llm,
        )
        graph_snapshot = await asyncio.to_thread(graph.snapshot_state)
        try:
            if progress_callback is not None:
                await progress_callback(
                    "bm25",
                    start_progress,
                    len(documents),
                    dict(stage_timings),
                )
            started = asyncio.get_running_loop().time()
            if doc_id:
                await asyncio.to_thread(
                    bm25.replace_document,
                    doc_id,
                    documents,
                )
            else:
                await asyncio.to_thread(bm25.upsert_documents, documents)
            await asyncio.to_thread(bm25.save)
            stage_timings["bm25"] = asyncio.get_running_loop().time() - started
            bm25_end = 97.0 if use_graphrag else 99.0
            if progress_callback is not None:
                await progress_callback(
                    "bm25",
                    bm25_end,
                    len(documents),
                    dict(stage_timings),
                )

            graphrag_applied = False
            if use_graphrag:
                if progress_callback is not None:
                    await progress_callback(
                        "graphrag",
                        97.0,
                        len(documents),
                        dict(stage_timings),
                    )
                started = asyncio.get_running_loop().time()
                await graph.build_graph(documents)
                await asyncio.to_thread(graph.save, str(_graph_path()))
                stage_timings["graphrag"] = (
                    asyncio.get_running_loop().time() - started
                )
                if progress_callback is not None:
                    await progress_callback(
                        "graphrag",
                        99.0,
                        len(documents),
                        dict(stage_timings),
                    )
                graphrag_applied = True
            elif doc_id:
                await graph.delete_document_async(doc_id)
                await asyncio.to_thread(graph.save, str(_graph_path()))

            if commit_callback is not None:
                await commit_callback(graphrag_applied)
            return graphrag_applied
        except BaseException:
            try:
                if doc_id:
                    if bm25_snapshot:
                        await asyncio.to_thread(
                            bm25.replace_document,
                            doc_id,
                            bm25_snapshot,
                        )
                    else:
                        await asyncio.to_thread(bm25.delete_document, doc_id)
                    await asyncio.to_thread(bm25.save)
            except Exception:
                logger.exception(
                    "Failed to restore BM25 after indexing document %s.",
                    doc_id,
                )
            try:
                await asyncio.to_thread(graph.restore_state, graph_snapshot)
                await asyncio.to_thread(graph.save, str(_graph_path()))
            except Exception:
                logger.exception(
                    "Failed to restore GraphRAG after indexing document %s.",
                    doc_id,
                )
            raise


async def delete_auxiliary_document(doc_id: str) -> None:
    """Delete a document from persistent BM25 and GraphRAG indexes."""
    async with _index_lock:
        errors: list[Exception] = []
        bm25 = get_bm25_retriever()
        try:
            removed = await asyncio.to_thread(bm25.delete_document, doc_id)
            if removed:
                await asyncio.to_thread(bm25.save)
        except Exception as exc:
            errors.append(exc)
            logger.exception("Failed to delete BM25 document %s.", doc_id)

        graph = get_graph_engine()
        try:
            await graph.delete_document_async(doc_id)
            await asyncio.to_thread(graph.save, str(_graph_path()))
        except Exception as exc:
            errors.append(exc)
            logger.exception("Failed to delete GraphRAG document %s.", doc_id)
        if errors:
            raise RuntimeError(
                f"Failed to delete {len(errors)} auxiliary index backend(s)."
            ) from errors[0]


def reset_retrieval_service() -> None:
    """Drop cached retrieval components after runtime configuration changes."""
    global _retriever, _bm25, _graph, _reranker, _retrieval_llm
    with _service_lock:
        _retriever = None
        _bm25 = None
        _graph = None
        _reranker = None
        _retrieval_llm = None
