"""Indexing concurrency and document-catalog lifecycle helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mindforge.config import get_settings
from mindforge.repositories.documents import (
    bulk_upsert_indexed,
    catalog_is_empty,
    delete_document,
    get_document,
    upsert_document,
)

logger = logging.getLogger(__name__)

_index_semaphore: asyncio.Semaphore | None = None
_index_limit: int | None = None


class IndexingCancelledError(Exception):
    """Cooperative cancellation requested between indexing stages."""


def build_index_signature(
    *,
    strategy: str,
    use_raptor: bool,
    use_graphrag: bool,
) -> str:
    """Hash every setting that changes persisted index output."""
    settings = get_settings()
    embedding_model = (
        settings.llm.local_embedding_model
        if settings.llm.embedding_provider.lower()
        in {"bge", "sentence-transformers", "local"}
        else settings.llm.embedding_model
    )
    payload = {
        "pipeline_version": settings.parser.pipeline_version,
        "strategy": strategy,
        "chunk_size": settings.chunking.chunk_size,
        "chunk_overlap": settings.chunking.chunk_overlap,
        "parser": settings.parser.model_dump(mode="json"),
        "visual_retrieval": {
            key: value
            for key, value in settings.visual_retrieval.model_dump(
                mode="json"
            ).items()
            if key != "api_key"
        },
        "embedding_provider": settings.llm.embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dimension": settings.vector_store.embedding_dim,
        "use_raptor": use_raptor,
        "raptor_levels": settings.raptor.raptor_levels if use_raptor else None,
        "raptor_threshold": (
            settings.raptor.raptor_threshold if use_raptor else None
        ),
        "use_graphrag": use_graphrag,
        "graph_entity_model": (
            settings.graphrag.entity_extraction_model
            if use_graphrag
            else None
        ),
        "graph_community_model": (
            settings.graphrag.community_summary_model
            if use_graphrag
            else None
        ),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


async def get_reusable_document(
    *,
    doc_id: str,
    index_signature: str,
) -> dict | None:
    document = await asyncio.to_thread(get_document, doc_id)
    if (
        document is None
        or document["status"] != "indexed"
        or int(document["chunk_count"]) < 1
        or document["index_signature"] != index_signature
    ):
        return None

    expected_chunks = int(document["chunk_count"])
    try:
        from mindforge.retrieval.service import get_bm25_retriever
        from mindforge.retrieval.vector_store import get_vector_store

        vector_chunks = await get_vector_store().count(
            filters={"doc_id": doc_id, "is_summary": False},
        )
        bm25_chunks = await asyncio.to_thread(
            get_bm25_retriever().count_document,
            doc_id,
        )
    except Exception:
        logger.warning(
            "Existing index verification failed for document %s; "
            "the document will be rebuilt.",
            doc_id,
            exc_info=True,
        )
        return None
    if vector_chunks != expected_chunks or bm25_chunks != expected_chunks:
        logger.warning(
            "Existing index for document %s is incomplete "
            "(catalog=%d, vector=%d, bm25=%d); rebuilding.",
            doc_id,
            expected_chunks,
            vector_chunks,
            bm25_chunks,
        )
        return None
    return document


def _get_index_semaphore() -> asyncio.Semaphore:
    global _index_semaphore, _index_limit
    configured_limit = get_settings().api.max_concurrent_index_jobs
    if _index_semaphore is None or _index_limit != configured_limit:
        _index_semaphore = asyncio.Semaphore(configured_limit)
        _index_limit = configured_limit
    return _index_semaphore


@asynccontextmanager
async def index_slot() -> AsyncIterator[None]:
    """Bound concurrent heavyweight indexing work across HTTP requests."""
    semaphore = _get_index_semaphore()
    async with semaphore:
        yield


async def set_document_status(
    *,
    doc_id: str,
    filename: str,
    status: str,
    chunk_count: int = 0,
    index_signature: str | None = None,
    index_strategy: str = "auto",
    use_raptor: bool = False,
    use_graphrag: bool = False,
    error: str | None = None,
    parser_metadata: dict | None = None,
) -> None:
    await asyncio.to_thread(
        upsert_document,
        doc_id=doc_id,
        filename=filename,
        chunk_count=chunk_count,
        status=status,
        index_signature=index_signature,
        index_strategy=index_strategy,
        use_raptor=use_raptor,
        use_graphrag=use_graphrag,
        error=error,
        parser_metadata=parser_metadata,
    )


async def remove_document_status(doc_id: str) -> None:
    await asyncio.to_thread(delete_document, doc_id)


async def reconcile_document_catalog() -> int:
    """Backfill the relational catalog once for pre-catalog Qdrant data."""
    if not await asyncio.to_thread(catalog_is_empty):
        return 0

    from mindforge.retrieval.vector_store import get_vector_store

    points = await get_vector_store().scroll_all(
        payload_fields=["doc_id", "source", "is_summary"],
    )
    documents: dict[str, dict[str, object]] = defaultdict(
        lambda: {"doc_id": "", "filename": "", "chunk_count": 0}
    )
    for point in points:
        payload = dict(point.payload or {})
        if payload.get("is_summary", False):
            continue
        doc_id = str(payload.get("doc_id") or "")
        if not doc_id:
            continue
        document = documents[doc_id]
        document["doc_id"] = doc_id
        document["filename"] = str(
            payload.get("source") or document["filename"] or doc_id
        )
        document["chunk_count"] = int(document["chunk_count"]) + 1

    values = list(documents.values())
    await asyncio.to_thread(bulk_upsert_indexed, values)
    if values:
        logger.info(
            "Backfilled %d document catalog records from Qdrant.",
            len(values),
        )
    return len(values)


def reset_indexing_service() -> None:
    global _index_semaphore, _index_limit
    _index_semaphore = None
    _index_limit = None
