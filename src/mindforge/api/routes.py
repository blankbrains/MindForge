"""API route definitions for MindForge — 真实实现"""

from __future__ import annotations
import json
import time
import uuid
import logging
import asyncio
import inspect
import os
import shutil
import tempfile
import threading
from contextlib import aclosing, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable, Literal
from urllib.parse import urlsplit, urlunsplit

import os as _os
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from mindforge.api.schemas import (
    DocumentContentResponse,
    DocumentEnabledUpdate,
    DocumentItem,
    HealthResponse,
    HistoryCitationSource,
    HistoryItem,
    HistorySaveRequest,
    IndexJobResponse,
    IndexRequest,
    IndexResponse,
    QueryCancelRequest,
    QueryCancelResponse,
    QueryRequest,
    QueryResponse,
    LLMDiscoveredModel,
    LLMModelDiscoveryRequest,
    LLMModelDiscoveryResponse,
    LLMProviderConfig,
    LLMProviderName,
    ObservabilityStatusResponse,
    SettingsResponse,
    SettingsUpdateRequest,
    TraceDetailResponse,
    TraceListResponse,
)
from mindforge.agents.base import AgentResult
from mindforge.agents.orchestrator import Orchestrator
from mindforge.ingestion.parsers import DocumentParser, DocumentParserError
from mindforge.ingestion.chunker import (
    DocumentChunk,
    ElementAwareSplitter,
    SemanticChunker,
    TextSplitter,
)
from mindforge.ingestion.raptor import RAPTORIndexer
from mindforge.retrieval.vector_store import get_vector_store
from mindforge.memory.episodic import EpisodicMemory
from mindforge.memory.semantic import SemanticMemory
from mindforge.models.base import (
    LLMConfigurationError,
    has_llm_credentials,
)
from mindforge.services.health import get_health_monitor
from mindforge.services.indexing import (
    IndexingCancelledError,
    document_index_slot,
    remove_document_status,
    set_document_status,
)
from mindforge.services.model_discovery import (
    ModelDiscoveryError,
    discover_models,
)
from mindforge.config import (
    get_project_root,
    get_settings,
    resolve_project_path,
)
from mindforge import __version__

logger = logging.getLogger(__name__)
router = APIRouter()
_orchestrator: Orchestrator | None = None
_ORCHESTRATOR_LOCK = threading.Lock()
_ENV_FILE_LOCK = threading.RLock()
_SETTINGS_UPDATE_LOCK = threading.RLock()
_RESEARCH_CANCELLATION_LOCK = threading.RLock()
_ACTIVE_RESEARCH_CANCELLATIONS: dict[str, asyncio.Event] = {}
IndexProgressCallback = Callable[
    [str, float, int, dict[str, float]],
    Awaitable[None],
]


def _register_research_cancellation(
    request_id: str,
) -> asyncio.Event | None:
    cancellation = asyncio.Event()
    with _RESEARCH_CANCELLATION_LOCK:
        if request_id in _ACTIVE_RESEARCH_CANCELLATIONS:
            return None
        _ACTIVE_RESEARCH_CANCELLATIONS[request_id] = cancellation
    return cancellation


def _unregister_research_cancellation(
    request_id: str,
    cancellation: asyncio.Event,
) -> None:
    with _RESEARCH_CANCELLATION_LOCK:
        if _ACTIVE_RESEARCH_CANCELLATIONS.get(request_id) is cancellation:
            _ACTIVE_RESEARCH_CANCELLATIONS.pop(request_id, None)


async def _report_index_progress(
    callback: IndexProgressCallback | None,
    *,
    stage: str,
    progress: float,
    chunk_count: int,
    timings: dict[str, float],
) -> None:
    if callback is not None:
        await callback(stage, progress, chunk_count, dict(timings))


def _public_service_url(value: str) -> str:
    """Return a connection URL without credentials, query, or fragment."""
    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return ""
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = host
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except (TypeError, ValueError):
        return ""


def _sanitize_upload_filename(filename: str | None) -> str:
    import re

    safe_name = filename or "uploaded_doc"
    safe_name = re.sub(r"[\\/]", "_", safe_name)
    safe_name = re.sub(r"\.\.+", "", safe_name)
    safe_name = safe_name.strip() or "uploaded_doc"
    return safe_name[:512]


async def _persist_upload(
    file: UploadFile,
    *,
    target_dir: Path,
    unique_prefix: str,
) -> Path:
    """Stream one validated upload to persistent application storage."""
    safe_name = _sanitize_upload_filename(file.filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in DocumentParser.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported document format. Supported formats: "
                + ", ".join(sorted(DocumentParser.SUPPORTED_EXTENSIONS))
            ),
        )

    max_upload_bytes = get_settings().api.max_upload_mb * 1024 * 1024
    await asyncio.to_thread(target_dir.mkdir, parents=True, exist_ok=True)
    target = target_dir / f"{unique_prefix}_{safe_name}"
    try:
        with target.open("wb") as output:
            total_bytes = 0
            while True:
                block = await file.read(1024 * 1024)
                if not block:
                    break
                total_bytes += len(block)
                if total_bytes > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "File exceeds the configured maximum of "
                            f"{get_settings().api.max_upload_mb}MB."
                        ),
                    )
                await asyncio.to_thread(output.write, block)
    except Exception:
        await asyncio.to_thread(target.unlink, missing_ok=True)
        raise
    finally:
        await file.close()
    return target


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator
    with _ORCHESTRATOR_LOCK:
        if _orchestrator is not None:
            return _orchestrator
        settings = get_settings()
        redis_client = None
        try:
            import redis

            redis_client = redis.from_url(
                settings.cache.redis_url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            redis_client.ping()
        except Exception:
            logger.info("Redis episodic memory unavailable; using process memory.")
            redis_client = None

        semantic_path = Path(settings.app.semantic_memory_dir).expanduser()
        if not semantic_path.is_absolute():
            semantic_path = get_project_root() / semantic_path

        tracer = None
        if settings.observability.enable_tracing:
            try:
                from mindforge.observability.tracer import get_tracer

                tracer = get_tracer()
            except Exception:
                logger.exception("Tracer initialization failed.")

        _orchestrator = Orchestrator(
            episodic_memory=EpisodicMemory(redis_client=redis_client),
            semantic_memory=SemanticMemory(storage_dir=semantic_path),
            tracer=tracer,
        )
    return _orchestrator


def get_retriever():
    from mindforge.retrieval.service import get_retriever as _get_retriever

    return _get_retriever()


async def _parse_document_file(
    parser: DocumentParser,
    file_path: str,
):
    try:
        return await asyncio.to_thread(parser.parse, file_path)
    except DocumentParserError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc


def _split_document(
    *,
    doc_id: str,
    content: str,
    strategy: str,
    metadata: dict[str, Any] | None = None,
    embedder: Any = None,
    elements: list[Any] | None = None,
) -> list[DocumentChunk]:
    """Split one document according to the requested indexing strategy."""
    if elements:
        chunks = ElementAwareSplitter(
            strategy=strategy,
            embedder=embedder,
        ).split(
            doc_id,
            elements,
            metadata=metadata or {},
        )
    elif strategy == "semantic":
        splitter = SemanticChunker(embedder=embedder)
        chunks = splitter.split(doc_id, content, metadata=metadata or {})
    else:
        splitter = TextSplitter()
        chunks = splitter.split(doc_id, content, metadata=metadata or {})
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="Document contains no indexable text.",
        )
    return chunks


def _build_chunk_points(
    *,
    chunks: list[DocumentChunk],
    vectors: list[list[float]],
    doc_id: str,
    source: str,
    expected_dimension: int,
) -> list[Any]:
    """Validate embeddings and build complete Qdrant point payloads."""
    from qdrant_client.models import PointStruct
    import hashlib as _hashlib

    if len(vectors) != len(chunks):
        raise ValueError("Embedding vector count does not match document chunk count.")

    points: list[PointStruct] = []
    for chunk_index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        if len(vector) != expected_dimension:
            raise ValueError(
                f"Embedding vector dimension {len(vector)} does not match "
                f"configured dimension {expected_dimension}."
            )
        chunk.embedding = list(vector)
        points.append(
            PointStruct(
                id=int(
                    _hashlib.md5(chunk.chunk_id.encode()).hexdigest(),
                    16,
                )
                % (2**63),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": doc_id,
                    "content": chunk.content,
                    "source": source,
                    "chunk_index": chunk_index,
                    "chunk_start": chunk.metadata.get("chunk_start"),
                    "chunk_end": chunk.metadata.get("chunk_end"),
                    "metadata": dict(chunk.metadata),
                    "is_summary": False,
                },
            )
        )
    return points


async def _embed_and_store_chunks(
    *,
    chunks: list[DocumentChunk],
    doc_id: str,
    source: str,
    progress_callback: IndexProgressCallback | None = None,
    timings: dict[str, float] | None = None,
) -> tuple[Any, Any]:
    """Embed chunks off the event loop, validate them, and persist them."""
    from mindforge.ingestion.embedder import get_embedder

    embedder = get_embedder()
    store = get_vector_store()
    stage_timings = timings if timings is not None else {}
    await asyncio.to_thread(store.ensure_collection)
    texts = [chunk.content for chunk in chunks]
    logger.info("Embedding %d document chunks.", len(texts))
    await _report_index_progress(
        progress_callback,
        stage="embedding",
        progress=25.0,
        chunk_count=len(chunks),
        timings=stage_timings,
    )
    started = time.perf_counter()
    vectors: list[list[float]] = []
    batch_size = get_settings().api.index_batch_size
    for index in range(0, len(texts), batch_size):
        batch = texts[index : index + batch_size]
        vectors.extend(await asyncio.to_thread(embedder.embed, batch))
        completed = min(index + len(batch), len(texts))
        await _report_index_progress(
            progress_callback,
            stage="embedding",
            progress=25.0 + (50.0 * completed / len(texts)),
            chunk_count=len(chunks),
            timings=stage_timings,
        )
    stage_timings["embedding"] = time.perf_counter() - started
    await _report_index_progress(
        progress_callback,
        stage="embedding",
        progress=75.0,
        chunk_count=len(chunks),
        timings=stage_timings,
    )
    started = time.perf_counter()
    points = _build_chunk_points(
        chunks=chunks,
        vectors=vectors,
        doc_id=doc_id,
        source=source,
        expected_dimension=store.embedding_dim,
    )
    await _report_index_progress(
        progress_callback,
        stage="vector_store",
        progress=75.0,
        chunk_count=len(chunks),
        timings=stage_timings,
    )
    await store.delete(doc_id)
    for index in range(0, len(points), batch_size):
        await store.upsert(points[index : index + batch_size])
        completed = min(index + batch_size, len(points))
        await _report_index_progress(
            progress_callback,
            stage="vector_store",
            progress=75.0 + (13.0 * completed / len(points)),
            chunk_count=len(chunks),
            timings=stage_timings,
        )
    stage_timings["vector_store"] = time.perf_counter() - started
    await _report_index_progress(
        progress_callback,
        stage="vector_store",
        progress=88.0,
        chunk_count=len(chunks),
        timings=stage_timings,
    )
    return embedder, store


async def _index_parsed_document(
    *,
    parsed: Any,
    source: str,
    strategy: str = "auto",
    metadata: dict[str, Any] | None = None,
    use_raptor: bool = False,
    use_graphrag: bool = False,
    progress_callback: IndexProgressCallback | None = None,
    timings: dict[str, float] | None = None,
    cancelled: Callable[[], bool] | None = None,
    completion_callback: Callable[
        [list[DocumentChunk], bool, bool],
        Awaitable[None],
    ] | None = None,
) -> tuple[list[DocumentChunk], bool, bool]:
    """Run the complete, shared indexing pipeline for one parsed document."""
    stage_timings = timings if timings is not None else {}
    splitter_embedder = None
    if strategy == "semantic":
        from mindforge.ingestion.embedder import get_embedder

        splitter_embedder = get_embedder()
    await _report_index_progress(
        progress_callback,
        stage="chunking",
        progress=20.0,
        chunk_count=0,
        timings=stage_timings,
    )
    chunks: list[DocumentChunk] = []
    started = time.perf_counter()
    if parsed.content.strip():
        chunks = await asyncio.to_thread(
            _split_document,
            doc_id=parsed.doc_id,
            content=parsed.content,
            strategy=strategy,
            metadata=metadata,
            embedder=splitter_embedder,
            elements=getattr(parsed, "elements", None),
        )
    stage_timings["chunking"] = time.perf_counter() - started
    await _report_index_progress(
        progress_callback,
        stage="vision",
        progress=25.0,
        chunk_count=len(chunks),
        timings=stage_timings,
    )
    visual_started = time.perf_counter()
    from mindforge.ingestion.visual import build_visual_chunks

    visual_chunks = await build_visual_chunks(
        parsed.doc_id,
        document_metadata=metadata,
        cancelled=cancelled,
    )
    stage_timings["vision"] = time.perf_counter() - visual_started
    chunks.extend(visual_chunks)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=(
                "Document contains no indexable text or visual captions. "
                "Configure visual retrieval for image-only documents."
            ),
        )
    await _report_index_progress(
        progress_callback,
        stage="chunking",
        progress=25.0,
        chunk_count=len(chunks),
        timings=stage_timings,
    )
    if len(chunks) > get_settings().api.max_chunks_per_document:
        raise HTTPException(
            status_code=413,
            detail=(
                "Document produced too many chunks; configured maximum is "
                f"{get_settings().api.max_chunks_per_document}."
            ),
        )

    embedder, store = await _embed_and_store_chunks(
        chunks=chunks,
        doc_id=parsed.doc_id,
        source=source,
        progress_callback=progress_callback,
        timings=stage_timings,
    )

    enable_raptor = use_raptor
    enable_graphrag = use_graphrag
    if (enable_raptor or enable_graphrag) and len(chunks) <= 5:
        logger.info(
            "Skipping RAPTOR/GraphRAG for '%s' (%d chunks): document is too short.",
            source,
            len(chunks),
        )
        enable_raptor = False
        enable_graphrag = False

    raptor_llm = None
    graph_entity_llm = None
    graph_summary_llm = None
    if enable_raptor:
        try:
            from mindforge.models.base import LLMFactory

            settings = get_settings()
            provider = settings.llm.llm_provider
            model = settings.raptor.summary_model.strip() or settings.llm.get_model(
                "researcher", provider
            )
            raptor_llm = LLMFactory.create(
                provider,
                model,
            )
        except Exception as exc:
            logger.warning("RAPTOR LLM init failed: %s", exc)
    if enable_graphrag:
        try:
            from mindforge.models.base import LLMFactory

            settings = get_settings()
            provider = settings.llm.llm_provider
            entity_model = (
                settings.graphrag.entity_extraction_model.strip()
                or settings.llm.get_model("researcher", provider)
            )
            summary_model = (
                settings.graphrag.community_summary_model.strip() or entity_model
            )
            graph_entity_llm = LLMFactory.create(
                provider,
                entity_model,
            )
            if summary_model == entity_model:
                graph_summary_llm = graph_entity_llm
            else:
                graph_summary_llm = LLMFactory.create(
                    provider,
                    summary_model,
                )
        except Exception as exc:
            logger.warning("GraphRAG LLM init failed: %s", exc)

    raptor_applied = False
    if enable_raptor and raptor_llm:
        await _report_index_progress(
            progress_callback,
            stage="raptor",
            progress=88.0,
            chunk_count=len(chunks),
            timings=stage_timings,
        )
        started = time.perf_counter()
        try:
            from qdrant_client.models import PointStruct
            import hashlib as _hashlib

            raptor = RAPTORIndexer(
                embedder=embedder,
                llm=raptor_llm,
            )
            tree_nodes = await raptor.build_tree(chunks)
            raptor_points: list[PointStruct] = []
            for node in tree_nodes:
                if node.level <= 0:
                    continue
                if node.embedding is None:
                    node.embedding = await asyncio.to_thread(
                        embedder.embed_single,
                        node.content,
                    )
                if len(node.embedding) != store.embedding_dim:
                    raise ValueError(
                        "RAPTOR summary embedding dimension does not match "
                        "the Qdrant collection."
                    )
                raptor_points.append(
                    PointStruct(
                        id=int(
                            _hashlib.md5(node.node_id.encode()).hexdigest(),
                            16,
                        )
                        % (2**63),
                        vector=node.embedding,
                        payload={
                            "chunk_id": node.node_id,
                            "doc_id": parsed.doc_id,
                            "content": node.content,
                            "source": source,
                            "raptor_level": node.level,
                            "is_summary": True,
                        },
                    )
                )
            batch_size = get_settings().api.index_batch_size
            for index in range(0, len(raptor_points), batch_size):
                await store.upsert(raptor_points[index : index + batch_size])
            raptor_applied = bool(raptor_points)
            logger.info(
                "RAPTOR: %d summary nodes indexed.",
                len(raptor_points),
            )
        except Exception:
            logger.exception("RAPTOR indexing failed.")
            raise
        finally:
            stage_timings["raptor"] = time.perf_counter() - started
            await _report_index_progress(
                progress_callback,
                stage="raptor",
                progress=94.0,
                chunk_count=len(chunks),
                timings=stage_timings,
            )

    from mindforge.retrieval.service import index_auxiliary_documents

    auxiliary_docs = [
        {
            "id": chunk.chunk_id,
            "text": chunk.content,
            "doc_id": parsed.doc_id,
            "chunk_id": chunk.chunk_id,
            "source": source,
            "metadata": dict(chunk.metadata),
        }
        for chunk in chunks
    ]
    async def commit_auxiliary_indexes(graphrag_applied: bool) -> None:
        if completion_callback is not None:
            await completion_callback(
                chunks,
                raptor_applied,
                graphrag_applied,
            )

    graphrag_applied = await index_auxiliary_documents(
        auxiliary_docs,
        graph_entity_llm=graph_entity_llm,
        graph_summary_llm=graph_summary_llm,
        use_graphrag=bool(enable_graphrag and graph_entity_llm and graph_summary_llm),
        progress_callback=progress_callback,
        timings=stage_timings,
        start_progress=94.0 if enable_raptor and raptor_llm else 88.0,
        commit_callback=(
            commit_auxiliary_indexes
            if completion_callback is not None
            else None
        ),
    )
    return chunks, raptor_applied, graphrag_applied


async def _rollback_document_index(
    doc_id: str,
    *,
    include_auxiliary: bool = True,
    include_assets: bool = True,
) -> None:
    """Best-effort rollback for a partially indexed document."""
    errors: list[Exception] = []
    try:
        await get_vector_store().delete(doc_id)
    except Exception as exc:
        errors.append(exc)
        logger.exception("Failed to roll back vector index for document %s.", doc_id)

    if include_auxiliary:
        try:
            from mindforge.retrieval.service import delete_auxiliary_document

            await delete_auxiliary_document(doc_id)
        except Exception as exc:
            errors.append(exc)
            logger.exception(
                "Failed to roll back auxiliary indexes for document %s.",
                doc_id,
            )

    if include_assets:
        try:
            from mindforge.services.document_assets import remove_document_assets

            await asyncio.to_thread(remove_document_assets, doc_id)
        except Exception as exc:
            errors.append(exc)
            logger.exception("Failed to roll back assets for document %s.", doc_id)

    if errors:
        logger.error(
            "Document %s rollback completed with %d failed backend(s).",
            doc_id,
            len(errors),
        )


async def _restore_previous_document(
    *,
    document: dict[str, Any],
    vector_snapshot: list[Any],
    asset_snapshot: Any,
) -> None:
    """Restore a previously indexed document after a failed rebuild."""
    errors: list[Exception] = []
    try:
        await get_vector_store().restore_document(
            str(document["doc_id"]),
            vector_snapshot,
            batch_size=get_settings().api.index_batch_size,
        )
    except Exception as exc:
        errors.append(exc)
        logger.exception(
            "Failed to restore vector snapshot for document %s.",
            document["doc_id"],
        )

    if asset_snapshot is not None:
        try:
            from mindforge.services.document_assets import restore_document_assets

            await asyncio.to_thread(restore_document_assets, asset_snapshot)
        except Exception as exc:
            errors.append(exc)
            logger.exception(
                "Failed to restore asset snapshot for document %s.",
                document["doc_id"],
            )

    try:
        await set_document_status(
            doc_id=str(document["doc_id"]),
            filename=str(document["filename"]),
            status=str(document["status"]),
            chunk_count=int(document["chunk_count"]),
            index_signature=document.get("index_signature"),
            index_strategy=str(document.get("index_strategy") or "auto"),
            use_raptor=bool(document.get("use_raptor")),
            use_graphrag=bool(document.get("use_graphrag")),
            error=document.get("error"),
            parser_metadata=dict(document.get("parser_metadata") or {}),
        )
    except Exception as exc:
        errors.append(exc)
        logger.exception(
            "Failed to restore catalog snapshot for document %s.",
            document["doc_id"],
        )

    if errors:
        raise RuntimeError(
            f"Failed to restore {len(errors)} document backend(s)."
        ) from errors[0]


async def _index_with_lifecycle(
    *,
    parsed: Any,
    source: str,
    strategy: str = "auto",
    metadata: dict[str, Any] | None = None,
    use_raptor: bool = False,
    use_graphrag: bool = False,
    progress_callback: IndexProgressCallback | None = None,
    timings: dict[str, float] | None = None,
    source_path: str | Path | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[list[DocumentChunk], bool, bool]:
    """Index one document under a bounded slot and persist its state."""
    from mindforge.services.indexing import build_index_signature
    from mindforge.repositories.documents import get_document

    index_signature = build_index_signature(
        strategy=strategy,
        use_raptor=use_raptor,
        use_graphrag=use_graphrag,
    )
    previous_document: dict[str, Any] | None = None
    vector_snapshot: list[Any] = []
    asset_snapshot: Any = None
    lifecycle_started = False
    try:
        async with document_index_slot(parsed.doc_id):
            candidate = await asyncio.to_thread(get_document, parsed.doc_id)
            if candidate is not None and candidate.get("status") == "indexed":
                previous_document = candidate
                vector_snapshot = await get_vector_store().snapshot_document(
                    parsed.doc_id
                )
                if source_path is not None:
                    from mindforge.services.document_assets import (
                        snapshot_document_assets,
                    )

                    asset_snapshot = await asyncio.to_thread(
                        snapshot_document_assets,
                        parsed.doc_id,
                    )

            await set_document_status(
                doc_id=parsed.doc_id,
                filename=source,
                status="indexing",
                index_strategy=strategy,
                use_raptor=use_raptor,
                use_graphrag=use_graphrag,
                parser_metadata=dict(getattr(parsed, "metadata", {}) or {}),
            )
            lifecycle_started = True
            if source_path is not None:
                from mindforge.services.document_assets import (
                    DocumentAssetCancelledError,
                    persist_document_assets,
                )

                await _report_index_progress(
                    progress_callback,
                    stage="assets",
                    progress=20.0,
                    chunk_count=0,
                    timings=timings or {},
                )
                asset_started = time.perf_counter()
                try:
                    await asyncio.to_thread(
                        persist_document_assets,
                        source_path=source_path,
                        parsed=parsed,
                        cancelled=cancelled,
                    )
                except DocumentAssetCancelledError as exc:
                    raise IndexingCancelledError(str(exc)) from exc
                if timings is not None:
                    timings["asset_persistence"] = time.perf_counter() - asset_started
                await set_document_status(
                    doc_id=parsed.doc_id,
                    filename=source,
                    status="indexing",
                    index_strategy=strategy,
                    use_raptor=use_raptor,
                    use_graphrag=use_graphrag,
                    parser_metadata=dict(getattr(parsed, "metadata", {}) or {}),
                )
                if cancelled is not None and cancelled():
                    raise IndexingCancelledError("Indexing was cancelled.")

            async def complete_index(
                completed_chunks: list[DocumentChunk],
                applied_raptor: bool,
                applied_graphrag: bool,
            ) -> None:
                await set_document_status(
                    doc_id=parsed.doc_id,
                    filename=source,
                    status="indexed",
                    chunk_count=len(completed_chunks),
                    index_signature=index_signature,
                    index_strategy=strategy,
                    use_raptor=applied_raptor,
                    use_graphrag=applied_graphrag,
                    parser_metadata=dict(
                        getattr(parsed, "metadata", {}) or {}
                    ),
                )

            (
                chunks,
                applied_raptor,
                applied_graphrag,
            ) = await _index_parsed_document(
                parsed=parsed,
                source=source,
                strategy=strategy,
                metadata=metadata,
                use_raptor=use_raptor,
                use_graphrag=use_graphrag,
                progress_callback=progress_callback,
                timings=timings,
                cancelled=cancelled,
                completion_callback=complete_index,
            )
    except (asyncio.CancelledError, IndexingCancelledError):
        if lifecycle_started:
            await _rollback_document_index(
                parsed.doc_id,
                include_auxiliary=previous_document is None,
                include_assets=source_path is not None,
            )
            if previous_document is not None:
                await _restore_previous_document(
                    document=previous_document,
                    vector_snapshot=vector_snapshot,
                    asset_snapshot=asset_snapshot,
                )
            else:
                await set_document_status(
                    doc_id=parsed.doc_id,
                    filename=source,
                    status="cancelled",
                    index_strategy=strategy,
                    use_raptor=use_raptor,
                    use_graphrag=use_graphrag,
                )
        raise
    except Exception as exc:
        if lifecycle_started:
            await _rollback_document_index(
                parsed.doc_id,
                include_auxiliary=previous_document is None,
                include_assets=source_path is not None,
            )
            if previous_document is not None:
                await _restore_previous_document(
                    document=previous_document,
                    vector_snapshot=vector_snapshot,
                    asset_snapshot=asset_snapshot,
                )
            else:
                await set_document_status(
                    doc_id=parsed.doc_id,
                    filename=source,
                    status="failed",
                    index_strategy=strategy,
                    use_raptor=use_raptor,
                    use_graphrag=use_graphrag,
                    error=str(exc)[:2000],
                )
        raise
    finally:
        if asset_snapshot is not None:
            try:
                from mindforge.services.document_assets import (
                    discard_document_asset_snapshot,
                )

                await asyncio.to_thread(
                    discard_document_asset_snapshot,
                    asset_snapshot,
                )
            except Exception:
                logger.warning(
                    "Failed to discard asset snapshot for document %s.",
                    parsed.doc_id,
                    exc_info=True,
                )
    return chunks, applied_raptor, applied_graphrag


def _reconstruct_document_content(
    chunks: list[dict[str, Any]],
) -> str:
    """Rebuild original content without duplicating overlapping regions."""
    if not chunks:
        return ""
    ordered = sorted(
        chunks,
        key=lambda chunk: (
            chunk.get("chunk_start") is None,
            chunk.get("chunk_start")
            if chunk.get("chunk_start") is not None
            else chunk.get("chunk_index") or 0,
        ),
    )
    if any(chunk.get("chunk_start") is None for chunk in ordered):
        return "\n\n".join(str(chunk.get("content", "")) for chunk in ordered)

    output = ""
    covered_until = 0
    for chunk in ordered:
        content = str(chunk.get("content", ""))
        start = int(chunk.get("chunk_start") or 0)
        end = chunk.get("chunk_end")
        if end is None:
            end = start + len(content)
        overlap = max(0, covered_until - start)
        if overlap < len(content):
            output += content[overlap:]
        covered_until = max(covered_until, int(end))
    return output


def _serialize_datetime_utc(value: datetime | None) -> str | None:
    """Serialize database timestamps with an explicit UTC designator."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _parse_history_token_usage(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_history_sources(value: str | None) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []

    sources: list[dict[str, Any]] = []
    for item in parsed[:200]:
        try:
            source = HistoryCitationSource.model_validate(item)
        except (TypeError, ValueError):
            continue
        sources.append(source.model_dump(exclude_none=True))
    return sources


def _research_trace_context(task: str, *, transport: str):
    settings = get_settings()
    if not settings.observability.enable_tracing:
        return nullcontext(None)
    try:
        from mindforge.observability.tracer import get_tracer

        tracer = get_tracer()
        if tracer.current_trace_id is not None:
            return nullcontext(None)
        return tracer.span(
            "orchestrator.research",
            metadata={
                "component": "orchestrator",
                "transport": transport,
                "task_chars": len(task),
                "display_name": task,
            },
        )
    except Exception:
        logger.exception("Research trace initialization failed.")
        return nullcontext(None)


def _call_orchestrator_method(
    orchestrator: Any,
    method_name: str,
    task: str,
) -> Any:
    """Call an orchestrator entry point without creating a second root trace."""
    method = getattr(orchestrator, method_name)
    try:
        parameters = inspect.signature(method).parameters.values()
        supports_trace_control = any(
            parameter.name == "create_root_trace"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        supports_trace_control = False
    if supports_trace_control:
        return method(task, create_root_trace=False)
    return method(task)


def _finish_research_trace(
    span: Any,
    *,
    success: bool,
    latency_ms: float,
    cost_usd: float | None,
    cost_status: str,
    total_tokens: int = 0,
    report_chars: int = 0,
    fallback: bool = False,
    outcome: str | None = None,
    failure_reason: str | None = None,
) -> None:
    if span is None:
        return
    span.output = {
        "success": success,
        "latency_ms": round(latency_ms, 3),
        "cost_usd": cost_usd,
        "cost_status": cost_status,
        "total_tokens": total_tokens,
        "report_chars": report_chars,
        "fallback": fallback,
        "outcome": outcome or ("success" if success else "failed"),
        "failure_reason": failure_reason,
    }
    span.metadata["status"] = (
        "degraded"
        if outcome == "degraded"
        else ("success" if success else "error")
    )
    if failure_reason:
        span.error = failure_reason[:1000]
    if not success and not span.error:
        span.error = "Research request failed."


async def _execute_query_non_stream(
    body: QueryRequest,
    *,
    llm_available: bool,
    start: float,
) -> QueryResponse:
    """Execute one non-streaming request inside the caller's trace context."""
    primary_failure: AgentResult | None = None
    primary_failure_reason: str | None = None
    if llm_available:
        try:
            orch = await asyncio.to_thread(get_orchestrator)
            result = await _call_orchestrator_method(
                orch,
                "run",
                body.task,
            )
            if not result.success:
                primary_failure = result
                primary_failure_reason = (
                    result.output
                    or "Agent pipeline returned an unsuccessful result."
                )
            else:
                latency = (time.time() - start) * 1000
                cost_value = result.metadata.get("cost")
                trace_id = result.trace_id
                result_outcome = str(
                    result.metadata.get("outcome") or "success"
                )
                if result_outcome not in {"success", "degraded"}:
                    result_outcome = "success"
                result_failure_reason = (
                    str(result.metadata.get("failure_reason"))
                    if result.metadata.get("failure_reason")
                    else None
                )
                return QueryResponse(
                    task_id=trace_id[:12] if trace_id else uuid.uuid4().hex[:12],
                    trace_id=trace_id,
                    report=result.output,
                    sources=list(result.data.get("sources", [])),
                    quality_score=(
                        float(result.metadata["quality"])
                        if isinstance(
                            result.metadata.get("quality"),
                            (int, float),
                        )
                        else None
                    ),
                    quality_status=str(
                        result.metadata.get(
                            "quality_status",
                            "not_evaluated",
                        )
                    ),
                    latency_ms=round(latency, 2),
                    cost_usd=(
                        round(float(cost_value), 10)
                        if isinstance(cost_value, (int, float))
                        else None
                    ),
                    cost_status=str(
                        result.metadata.get(
                            "cost_status",
                            result.cost_status,
                        )
                    ),
                    iterations=int(result.metadata.get("subtask_count", 0)),
                    outcome=result_outcome,
                    failure_reason=result_failure_reason,
                )
        except LLMConfigurationError as exc:
            primary_failure_reason = str(exc)
            logger.warning(
                "Configured Agent provider became unavailable; "
                "using retrieval fallback: %s",
                exc,
            )
        except Exception as exc:
            primary_failure_reason = str(exc)
            logger.exception("Agent pipeline failed, falling back to retrieval-only.")
    else:
        logger.info("No LLM credentials configured; using retrieval-only.")

    fallback_enabled = get_settings().agent.fallback_enabled
    if llm_available and not fallback_enabled:
        raise HTTPException(
            status_code=502,
            detail=primary_failure_reason or "研究任务执行失败。",
        )

    try:
        from mindforge.tools.rag_tool import RAGTool

        rag = RAGTool()
        result = await rag.execute_async(
            query=body.task,
            mode="hybrid",
        )
        if not result.success:
            raise RuntimeError(result.error or "Retrieval fallback failed.")
        result_data = result.data or {}
        has_relevant_results = (
            bool(result_data.get("total"))
            if "total" in result_data
            else bool(result.output.strip())
        )
        if primary_failure_reason and not has_relevant_results:
            raise HTTPException(
                status_code=502,
                detail=primary_failure_reason,
            )
        latency = (time.time() - start) * 1000
        degraded = primary_failure_reason is not None
        trace_id = primary_failure.trace_id if primary_failure else None
        return QueryResponse(
            task_id=trace_id[:12] if trace_id else uuid.uuid4().hex[:12],
            trace_id=trace_id,
            report=result.output,
            sources=list(result_data.get("sources", [])),
            quality_score=None,
            quality_status="not_evaluated",
            latency_ms=round(latency, 2),
            cost_usd=primary_failure.cost_usd if primary_failure else None,
            cost_status=(
                primary_failure.cost_status
                if primary_failure
                else "not_applicable"
            ),
            iterations=0,
            outcome="degraded" if degraded else "retrieval_only",
            failure_reason=primary_failure_reason,
            retrieval_quality=float(
                result_data.get("retrieval_quality", 0.0)
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Retrieval-only query failed.")
        raise HTTPException(
            status_code=503,
            detail="Knowledge base retrieval is temporarily unavailable.",
        ) from exc


@router.post("/query/cancel", response_model=QueryCancelResponse)
async def cancel_query(body: QueryCancelRequest) -> QueryCancelResponse:
    """Cancel one active streaming research request."""
    with _RESEARCH_CANCELLATION_LOCK:
        cancellation = _ACTIVE_RESEARCH_CANCELLATIONS.get(body.request_id)
    if cancellation is None:
        return QueryCancelResponse(
            request_id=body.request_id,
            cancelled=False,
        )
    cancellation.set()
    return QueryCancelResponse(
        request_id=body.request_id,
        cancelled=True,
    )


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    """Submit a research task. Falls back to retrieval-only if LLM is unavailable."""
    start = time.time()

    if body.stream:
        request_id = body.request_id or uuid.uuid4().hex
        cancellation = _register_research_cancellation(request_id)
        if cancellation is None:
            raise HTTPException(
                status_code=409,
                detail="A research request with this request_id is already active.",
            )
        try:
            llm_available = await asyncio.to_thread(has_llm_credentials)
            orch = None
            if llm_available:
                try:
                    orch = await asyncio.to_thread(get_orchestrator)
                except LLMConfigurationError as exc:
                    logger.warning(
                        "Configured Agent provider could not initialize; "
                        "using retrieval fallback: %s",
                        exc,
                    )
                except Exception:
                    logger.exception(
                        "Agent initialization failed for SSE; "
                        "using retrieval fallback."
                    )
            else:
                logger.info(
                    "No LLM credentials configured; using retrieval-only SSE."
                )
            return StreamingResponse(
                _stream_response(
                    orch,
                    body.task,
                    request_id=request_id,
                    cancellation=cancellation,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-MindForge-Request-ID": request_id,
                },
            )
        except BaseException:
            _unregister_research_cancellation(request_id, cancellation)
            raise

    llm_available = await asyncio.to_thread(has_llm_credentials)

    with _research_trace_context(body.task, transport="rest") as trace_span:
        if trace_span is not None:
            trace_span.input = {"task": body.task}
        response = await _execute_query_non_stream(
            body,
            llm_available=llm_available,
            start=start,
        )
        trace_id = (
            trace_span.trace_id
            if trace_span is not None
            else response.trace_id
        )
        response.trace_id = trace_id
        if trace_id:
            response.task_id = trace_id[:12]
        _finish_research_trace(
            trace_span,
            success=True,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
            cost_status=response.cost_status,
            report_chars=len(response.report or ""),
            fallback=response.iterations == 0,
        )
        return response


@router.post("/index", response_model=IndexResponse)
async def index_document(body: IndexRequest):
    """Ingest a document into the Qdrant knowledge base."""
    if not get_settings().api.allow_local_file_index:
        raise HTTPException(
            status_code=403,
            detail="Local file indexing is disabled by configuration.",
        )

    file_path = body.file_path
    data_root = Path(get_settings().app.data_dir).expanduser()
    if not data_root.is_absolute():
        data_root = get_project_root() / data_root
    resolved_file = Path(file_path).expanduser().resolve()
    try:
        resolved_file.relative_to(data_root.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail="file_path must be inside MINDFORGE_DATA_DIR.",
        ) from exc
    file_path = str(resolved_file)
    parser = DocumentParser()
    doc = await _parse_document_file(parser, file_path)
    chunks, applied_raptor, applied_graphrag = await _index_with_lifecycle(
        parsed=doc,
        source=doc.filename,
        strategy=body.strategy,
        metadata=body.metadata,
        use_raptor=body.use_raptor,
        use_graphrag=body.use_graphrag,
        source_path=file_path,
    )

    logger.info(f"Indexed {doc.filename}: {len(chunks)} chunks")
    return IndexResponse(
        doc_id=doc.doc_id,
        filename=doc.filename,
        chunk_count=len(chunks),
        status="indexed",
        index_strategy=body.strategy,
        use_raptor=applied_raptor,
        use_graphrag=applied_graphrag,
    )


@router.post(
    "/index-jobs",
    response_model=IndexJobResponse,
    status_code=202,
)
async def create_index_job(
    file: UploadFile = File(...),
    strategy: Literal["auto", "fixed", "semantic"] = Form("auto"),
    use_raptor: bool = Form(False),
    use_graphrag: bool = Form(False),
):
    """Persist an upload and return before heavyweight indexing starts."""
    from mindforge.services.index_jobs import get_index_job_service

    job_id = uuid.uuid4().hex
    filename = _sanitize_upload_filename(file.filename)
    job_dir = resolve_project_path(get_settings().app.data_dir) / "index-jobs"
    file_path = await _persist_upload(
        file,
        target_dir=job_dir,
        unique_prefix=job_id,
    )
    try:
        return await get_index_job_service().create(
            job_id=job_id,
            filename=filename,
            file_path=str(file_path),
            strategy=strategy,
            use_raptor=use_raptor,
            use_graphrag=use_graphrag,
        )
    except Exception:
        await asyncio.to_thread(file_path.unlink, missing_ok=True)
        raise


@router.get("/index-jobs", response_model=list[IndexJobResponse])
async def list_index_job_records(
    active: bool = Query(False),
    limit: int = Query(20, ge=1, le=100),
):
    from mindforge.services.index_jobs import get_index_job_service

    return await get_index_job_service().list(
        active_only=active,
        limit=limit,
    )


@router.get(
    "/index-jobs/{job_id}",
    response_model=IndexJobResponse,
)
async def get_index_job_record(job_id: str):
    from mindforge.services.index_jobs import get_index_job_service

    job = await get_index_job_service().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Index job not found.")
    return job


@router.delete(
    "/index-jobs/{job_id}",
    response_model=IndexJobResponse,
    status_code=202,
)
async def cancel_index_job(job_id: str):
    from mindforge.services.index_jobs import get_index_job_service

    job = await get_index_job_service().cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Index job not found.")
    return job


async def _probe_qdrant_connection() -> bool:
    try:
        await get_vector_store().ping()
        return True
    except Exception:
        return False


async def _probe_redis_connection() -> bool:
    return await get_health_monitor()._probe_redis()


async def _probe_postgres_connection() -> bool:
    return await get_health_monitor()._probe_postgres()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Return the latest background dependency-health snapshot."""
    snapshot = await get_health_monitor().get_snapshot()
    return HealthResponse(
        status=snapshot.status,
        version=__version__,
        qdrant_connected=snapshot.qdrant_connected,
        redis_connected=snapshot.redis_connected,
        postgres_connected=snapshot.postgres_connected,
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
)
async def readiness():
    """Strict readiness probe for deployment orchestration."""
    snapshot = await get_health_monitor().refresh()
    result = HealthResponse(
        status=snapshot.status,
        version=__version__,
        qdrant_connected=snapshot.qdrant_connected,
        redis_connected=snapshot.redis_connected,
        postgres_connected=snapshot.postgres_connected,
    )
    if result.status != "ok":
        return JSONResponse(
            status_code=503,
            content=result.model_dump(),
        )
    return result


@router.get("/stats")
async def stats():
    """System statistics from Qdrant — counts unique documents, not chunks."""
    from mindforge.repositories.documents import get_document_stats

    try:
        document_count, chunk_count = await asyncio.to_thread(get_document_stats)
    except Exception:
        logger.exception("Failed to load document statistics.")
        document_count, chunk_count = 0, 0
    snapshot = await get_health_monitor().get_snapshot()
    from mindforge.ingestion.embedder import get_embedder_status

    embedding = get_embedder_status()
    return {
        "documents_indexed": document_count,
        "chunks_indexed": chunk_count,
        "qdrant_connected": snapshot.qdrant_connected,
        "qdrant_url": _public_service_url(get_settings().vector_store.qdrant_url),
        "redis_url": _public_service_url(get_settings().cache.redis_url),
        "max_upload_mb": get_settings().api.max_upload_mb,
        "max_pdf_pages": get_settings().api.max_pdf_pages,
        "embedding_provider": embedding["provider"],
        "embedding_device": embedding["device"],
    }


@router.get("/documents", response_model=list[DocumentItem])
async def list_documents():
    """List all indexed documents with metadata."""
    from mindforge.repositories.documents import (
        list_documents as list_document_records,
    )

    try:
        return await asyncio.to_thread(list_document_records)
    except Exception:
        logger.exception("Failed to list documents.")
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is temporarily unavailable.",
        )


@router.patch(
    "/documents/{doc_id}/enabled",
    response_model=DocumentItem,
)
async def update_document_enabled(
    doc_id: str,
    request: DocumentEnabledUpdate,
):
    """Enable or disable one document for retrieval without reindexing."""
    from mindforge.repositories.documents import set_document_enabled

    try:
        document = await asyncio.to_thread(
            set_document_enabled,
            doc_id,
            enabled=request.enabled,
        )
    except Exception as exc:
        logger.exception(
            "Failed to update retrieval availability for document %s.",
            doc_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Document availability could not be updated.",
        ) from exc
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str):
    """Delete a document from Qdrant."""
    errors: list[Exception] = []
    try:
        await get_vector_store().delete(doc_id)
    except Exception as exc:
        errors.append(exc)
        logger.exception("Vector deletion failed for document %s.", doc_id)
    try:
        from mindforge.retrieval.service import delete_auxiliary_document

        await delete_auxiliary_document(doc_id)
    except Exception as exc:
        errors.append(exc)
        logger.exception("Auxiliary deletion failed for document %s.", doc_id)
    try:
        from mindforge.services.document_assets import remove_document_assets

        await asyncio.to_thread(remove_document_assets, doc_id)
    except Exception as exc:
        errors.append(exc)
        logger.exception("Asset deletion failed for document %s.", doc_id)
    if errors:
        raise HTTPException(
            status_code=503,
            detail=(
                "Document deletion was incomplete; retry to finish cleaning "
                "all storage backends."
            ),
        ) from errors[0]
    try:
        await remove_document_status(doc_id)
    except Exception as exc:
        logger.exception("Catalog deletion failed for document %s.", doc_id)
        raise HTTPException(
            status_code=503,
            detail=(
                "Document data was removed, but the catalog update failed; "
                "retry the deletion."
            ),
        ) from exc
    return None


@router.get("/documents/{doc_id}/assets")
async def list_document_asset_records(doc_id: str):
    """List persisted visual and structural assets without exposing source files."""
    from mindforge.repositories.document_assets import list_document_assets

    assets = await asyncio.to_thread(list_document_assets, doc_id)
    return [
        {
            **asset,
            "url": (
                f"/api/v1/documents/{doc_id}/assets/{asset['asset_id']}"
                if asset.get("relative_path")
                and asset.get("kind") in {"image", "page_preview"}
                else None
            ),
        }
        for asset in assets
    ]


@router.get("/documents/{doc_id}/assets/{asset_id}")
async def get_document_asset_file(doc_id: str, asset_id: str):
    """Serve only rendered visual assets registered for the requested document."""
    from mindforge.repositories.document_assets import get_document_asset
    from mindforge.services.document_assets import (
        DocumentAssetError,
        resolve_asset_path,
    )

    asset = await asyncio.to_thread(get_document_asset, asset_id)
    if asset is None or asset["doc_id"] != doc_id:
        raise HTTPException(status_code=404, detail="Document asset not found.")
    if asset.get("kind") not in {"image", "page_preview"}:
        raise HTTPException(status_code=404, detail="Document asset is not visual.")
    try:
        path = resolve_asset_path(str(asset.get("relative_path") or ""))
    except DocumentAssetError as exc:
        raise HTTPException(
            status_code=404, detail="Document asset is unavailable."
        ) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Document asset is unavailable.")
    return FileResponse(
        path,
        media_type=str(asset.get("content_type") or "image/png"),
        filename=path.name,
    )


# ------------------------------------------------------------------
# Document content
# ------------------------------------------------------------------


@router.get("/documents/{doc_id}/content", response_model=DocumentContentResponse)
async def get_document_content(
    doc_id: str,
    include_chunks: bool = Query(True),
):
    """Return full content of an indexed document (all chunks)."""
    store = get_vector_store()
    points = await store.scroll_all(filters={"doc_id": doc_id})
    chunks = []
    filename = ""
    for p in points:
        pl = p.payload or {}
        if pl.get("doc_id") == doc_id and not pl.get("is_summary", False):
            chunks.append(
                {
                    "chunk_id": pl.get("chunk_id", ""),
                    "content": pl.get("content", ""),
                    "chunk_index": pl.get("chunk_index"),
                    "chunk_start": pl.get("chunk_start"),
                    "chunk_end": pl.get("chunk_end"),
                }
            )
            if not filename:
                filename = pl.get("source", "")
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks.sort(
        key=lambda chunk: (
            chunk["chunk_index"] is None,
            chunk["chunk_index"]
            if chunk["chunk_index"] is not None
            else chunk.get("chunk_start") or 0,
        )
    )
    full_content = _reconstruct_document_content(chunks)
    return DocumentContentResponse(
        doc_id=doc_id,
        filename=filename,
        content=full_content,
        chunk_count=len(chunks),
        chunks=chunks if include_chunks else [],
    )


# ------------------------------------------------------------------
# File upload
# ------------------------------------------------------------------


@router.post("/upload", response_model=IndexResponse)
async def upload_document(
    file: UploadFile = File(...),
    use_raptor: bool = Form(False),
    use_graphrag: bool = Form(False),
):
    """Upload a document file for indexing into the knowledge base."""
    upload_dir = resolve_project_path(get_settings().app.data_dir)
    file_path = await _persist_upload(
        file,
        target_dir=upload_dir,
        unique_prefix=uuid.uuid4().hex[:8],
    )
    parsed = None
    try:
        parser = DocumentParser()
        parsed = await _parse_document_file(parser, str(file_path))
        source = file.filename or parsed.filename
        (
            chunks,
            applied_raptor,
            applied_graphrag,
        ) = await _index_with_lifecycle(
            parsed=parsed,
            source=source,
            use_raptor=use_raptor,
            use_graphrag=use_graphrag,
            source_path=file_path,
        )

        return IndexResponse(
            doc_id=parsed.doc_id,
            filename=source,
            chunk_count=len(chunks),
            status="indexed",
            index_strategy="auto",
            use_raptor=applied_raptor,
            use_graphrag=applied_graphrag,
        )
    finally:
        # Always attempt cleanup of the uploaded temp file after indexing
        try:
            await asyncio.to_thread(file_path.unlink, missing_ok=True)
        except OSError:
            pass


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

_PROVIDER_ENV_PREFIXES: dict[LLMProviderName, str] = {
    "openai": "OPENAI",
    "deepseek": "DEEPSEEK",
    "kimi": "KIMI",
    "glm": "GLM",
    "openai_compatible": "COMPATIBLE",
    "local": "LOCAL",
}
_CONFIGURABLE_PROVIDER_PREFIXES: dict[LLMProviderName, str] = {
    "kimi": "kimi",
    "glm": "glm",
    "openai_compatible": "compatible",
    "local": "local",
}
_PROVIDER_LABELS: dict[LLMProviderName, str] = {
    "openai": "OpenAI",
    "deepseek": "DeepSeek",
    "kimi": "Kimi",
    "glm": "GLM",
    "openai_compatible": "通用接口",
    "local": "本地模型",
}
_PROVIDER_ORDER: tuple[LLMProviderName, ...] = (
    "openai",
    "deepseek",
    "kimi",
    "glm",
    "openai_compatible",
    "local",
)


def _stored_provider_api_key(provider: LLMProviderName) -> str:
    """Resolve a provider key without returning it to the browser."""
    llm_settings = get_settings().llm
    get_api_key = getattr(llm_settings, "get_api_key", None)
    settings_key = (
        get_api_key(provider)
        if callable(get_api_key)
        else getattr(
            llm_settings,
            {
                "openai": "openai_api_key",
                "deepseek": "deepseek_api_key",
                "kimi": "kimi_api_key",
                "glm": "glm_api_key",
                "openai_compatible": "compatible_api_key",
                "local": "local_api_key",
            }[provider],
            "",
        )
    )
    settings_key = str(settings_key or "").strip()
    try:
        from mindforge.db import (
            ApiKey,
            SessionLocal,
            decrypt_api_key,
        )

        db = SessionLocal()
        try:
            row = (
                db.query(ApiKey)
                .filter(
                    ApiKey.provider == provider,
                    ApiKey.is_active,
                )
                .first()
            )
            if row and row.key_encrypted:
                return decrypt_api_key(row.key_encrypted).strip()
        finally:
            db.close()
    except Exception:
        logger.warning(
            "Stored API key lookup failed for model discovery provider %s.",
            provider,
        )
    return settings_key


@router.post(
    "/settings/models",
    response_model=LLMModelDiscoveryResponse,
)
async def discover_provider_models(
    body: LLMModelDiscoveryRequest,
) -> LLMModelDiscoveryResponse:
    """Fetch model IDs using unsaved provider connection values."""
    settings = get_settings()
    if body.use_stored_api_key:
        configured_base_url = (
            settings.llm.get_base_url(body.provider) or ""
        ).rstrip("/")
        if body.base_url.rstrip("/") != configured_base_url:
            raise HTTPException(
                status_code=409,
                detail=(
                    "A stored API key may only be used with the provider's "
                    "saved Base URL. Save the new URL with a reconfirmed key first."
                ),
            )
    api_key = (
        _stored_provider_api_key(body.provider)
        if body.use_stored_api_key
        else body.api_key.strip()
    )
    api_key_required = (
        True
        if body.provider in {"openai", "deepseek", "kimi", "glm"}
        else body.api_key_required
    )
    if api_key_required and not api_key:
        raise HTTPException(
            status_code=400,
            detail="请先填写或保存该 Provider 的 API Key。",
        )

    try:
        models, truncated = await discover_models(
            base_url=body.base_url,
            api_key=api_key,
            allow_private=body.provider == "local",
            timeout_seconds=(settings.api.model_discovery_timeout_seconds),
            max_response_bytes=(settings.api.model_discovery_max_response_bytes),
            max_models=settings.api.model_discovery_max_models,
        )
    except ModelDiscoveryError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc

    return LLMModelDiscoveryResponse(
        models=[
            LLMDiscoveredModel(
                id=model.id,
                owned_by=model.owned_by,
            )
            for model in models
        ],
        count=len(models),
        truncated=truncated,
    )


@router.get("/settings", response_model=SettingsResponse)
def get_settings_api():
    """Return current user settings (API keys masked)."""
    from mindforge.db import (
        ApiKey,
        SessionLocal,
        decrypt_api_key,
    )
    from mindforge.config import get_settings

    db = SessionLocal()
    try:
        keys = {k.provider: k for k in db.query(ApiKey).filter(ApiKey.is_active).all()}
        s = get_settings()

        def _masked(provider: str, db_keys: dict, settings_key: str) -> str:
            if provider in db_keys:
                decrypted = decrypt_api_key(db_keys[provider].key_encrypted)
                return "***" + decrypted[-4:] if decrypted else "***configured"
            if settings_key:
                return "***" + settings_key[-4:]
            return ""

        def _masked_value(value: str | None) -> str:
            normalized = str(value or "").strip()
            return "***" + normalized[-4:] if normalized else ""

        def _provider_config(
            provider: LLMProviderName,
        ) -> LLMProviderConfig:
            if provider == "openai":
                default_model = ""
                planner_model = s.llm.planner_model
                researcher_model = s.llm.researcher_model
                critic_model = s.llm.critic_model
                synthesizer_model = s.llm.synthesizer_model
            elif provider == "deepseek":
                default_model = ""
                planner_model = s.llm.deepseek_planner
                researcher_model = s.llm.deepseek_researcher
                critic_model = s.llm.deepseek_critic
                synthesizer_model = s.llm.deepseek_synthesizer
            else:
                prefix = _CONFIGURABLE_PROVIDER_PREFIXES[provider]
                default_model = getattr(s.llm, f"{prefix}_model")
                planner_model = getattr(
                    s.llm,
                    f"{prefix}_planner_model",
                )
                researcher_model = getattr(
                    s.llm,
                    f"{prefix}_researcher_model",
                )
                critic_model = getattr(
                    s.llm,
                    f"{prefix}_critic_model",
                )
                synthesizer_model = getattr(
                    s.llm,
                    f"{prefix}_synthesizer_model",
                )
            return LLMProviderConfig(
                provider=provider,
                label=_PROVIDER_LABELS[provider],
                base_url=s.llm.get_base_url(provider) or "",
                api_key=_masked(
                    provider,
                    keys,
                    s.llm.get_api_key(provider),
                ),
                api_key_required=s.llm.requires_api_key(provider),
                default_model=default_model,
                planner_model=planner_model,
                researcher_model=researcher_model,
                critic_model=critic_model,
                synthesizer_model=synthesizer_model,
                supports_tools=s.llm.supports_tools(provider),
                supports_json_mode=s.llm.supports_json_mode(provider),
                supports_json_schema=s.llm.supports_json_schema(provider),
                native_web_search_protocol=(
                    s.llm.get_native_web_search_protocol(provider)
                ),
                native_web_search_endpoint=(
                    s.llm.get_native_web_search_endpoint(provider) or ""
                ),
                configured=has_llm_credentials(provider),
            )

        provider_configs = [
            _provider_config(provider) for provider in _PROVIDER_ORDER
        ]
        from mindforge.retrieval.service import get_reranker_status

        reranker_status = get_reranker_status()
        selected_native_protocol = s.llm.get_native_web_search_protocol()
        web_search_config = getattr(s, "web_search", None)
        native_enabled = bool(
            getattr(web_search_config, "native_enabled", True)
        )
        duckduckgo_enabled = bool(
            getattr(web_search_config, "duckduckgo_enabled", False)
        )
        model_only_fallback = bool(
            getattr(web_search_config, "model_only_fallback", True)
        )
        native_web_search_supported = bool(
            native_enabled
            and selected_native_protocol != "none"
            and has_llm_credentials(s.llm.llm_provider)
        )
        tavily_configured = bool(
            os.environ.get("TAVILY_API_KEY", "").strip()
        )
        return SettingsResponse(
            llm_provider=s.llm.llm_provider,
            llm_configured=has_llm_credentials(s.llm.llm_provider),
            llm_providers=provider_configs,
            deepseek_api_key=_masked("deepseek", keys, s.llm.deepseek_api_key),
            openai_api_key=_masked("openai", keys, s.llm.openai_api_key),
            compatible_api_key=_masked(
                "openai_compatible",
                keys,
                s.llm.compatible_api_key,
            ),
            local_api_key=_masked(
                "local",
                keys,
                s.llm.local_api_key,
            ),
            embedding_provider=s.llm.embedding_provider,
            research_mode=getattr(s.agent, "research_mode", "balanced"),
            source_policy=getattr(s.agent, "source_policy", "auto"),
            fallback_enabled=getattr(s.agent, "fallback_enabled", True),
            retrieval_top_k=s.retrieval.vector_top_k,
            rerank_top_k=s.retrieval.rerank_top_k,
            retrieval_min_score=getattr(s.retrieval, "min_score", 0.60),
            keyword_min_coverage=getattr(
                s.retrieval,
                "keyword_min_coverage",
                0.60,
            ),
            max_iterations=s.agent.max_iterations,
            max_refine_rounds=s.agent.max_refine_rounds,
            critic_threshold=s.agent.critic_threshold,
            subtask_timeout=s.agent.subtask_timeout,
            research_timeout=s.agent.research_timeout,
            llm_request_timeout=getattr(s.agent, "llm_request_timeout", 45),
            max_subtasks=getattr(s.agent, "max_subtasks", 5),
            max_tool_calls_total=getattr(
                s.agent,
                "max_tool_calls_total",
                12,
            ),
            max_history_entries=getattr(
                getattr(s, "api", None),
                "max_history_entries",
                0,
            ),
            langfuse_public_key=_masked_value(
                getattr(
                    getattr(s, "observability", None),
                    "langfuse_public_key",
                    "",
                )
            ),
            langfuse_secret_key=_masked_value(
                getattr(
                    getattr(s, "observability", None),
                    "langfuse_secret_key",
                    "",
                )
            ),
            langfuse_host=getattr(
                getattr(s, "observability", None),
                "langfuse_host",
                "https://cloud.langfuse.com",
            ),
            observability_capture_content=getattr(
                getattr(s, "observability", None),
                "capture_content",
                False,
            ),
            trace_retention_days=getattr(
                getattr(s, "observability", None),
                "trace_retention_days",
                0,
            ),
            tavily_configured=tavily_configured,
            native_web_search_enabled=native_enabled,
            native_web_search_protocol=selected_native_protocol,
            native_web_search_supported=native_web_search_supported,
            duckduckgo_enabled=duckduckgo_enabled,
            model_only_fallback_enabled=model_only_fallback,
            web_search_available=bool(
                native_web_search_supported
                or tavily_configured
                or duckduckgo_enabled
            ),
            reranker_configured=reranker_status["configured"],
            reranker_available=reranker_status["available"],
            reranker_load_failed=reranker_status["load_failed"],
        )
    finally:
        db.close()


def _rewrite_env_file(updates: dict[str, str | None]) -> None:
    """Update a possibly bind-mounted .env file without replacing its inode."""
    from dotenv import set_key, unset_key

    env_path = get_project_root() / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)

    with _ENV_FILE_LOCK:
        if not env_path.exists():
            env_path.touch(mode=0o600)

        file_descriptor, stage_name = tempfile.mkstemp(
            prefix=".mindforge-env-",
            suffix=".tmp",
            dir=env_path.parent,
        )
        os.close(file_descriptor)
        stage_path = Path(stage_name)
        try:
            shutil.copyfile(env_path, stage_path)
            for key, value in updates.items():
                if value is None:
                    unset_key(str(stage_path), key)
                else:
                    set_key(
                        str(stage_path),
                        key,
                        value,
                        quote_mode="always",
                    )

            with env_path.open("wb") as env_file:
                env_file.write(stage_path.read_bytes())
                env_file.flush()
                os.fsync(env_file.fileno())
            env_path.chmod(0o600)
        finally:
            stage_path.unlink(missing_ok=True)


def _sync_env_file(updates: dict[str, str]) -> None:
    """Persist settings to .env so they survive server restarts."""
    _rewrite_env_file(updates)


def _snapshot_env_file(keys: set[str]) -> dict[str, tuple[bool, str]]:
    """Capture selected .env values so a failed DB commit can restore them."""
    from dotenv import dotenv_values

    env_path = get_project_root() / ".env"
    with _ENV_FILE_LOCK:
        values = dotenv_values(env_path) if env_path.is_file() else {}
    return {key: (key in values, values.get(key) or "") for key in keys}


def _restore_env_file(
    snapshot: dict[str, tuple[bool, str]],
) -> None:
    _rewrite_env_file(
        {
            key: value if was_present else None
            for key, (was_present, value) in snapshot.items()
        }
    )


@router.put("/settings")
def update_settings_api(body: SettingsUpdateRequest):
    with _SETTINGS_UPDATE_LOCK:
        return _update_settings_locked(body)


def _update_settings_locked(body: SettingsUpdateRequest):
    """Save user settings (API keys encrypted in DB, synced to .env)."""
    current_settings = get_settings()
    current_agent = getattr(current_settings, "agent", None)
    effective_llm_timeout = (
        body.llm_request_timeout
        if body.llm_request_timeout is not None
        else getattr(current_agent, "llm_request_timeout", 45)
    )
    effective_subtask_timeout = (
        body.subtask_timeout
        if body.subtask_timeout is not None
        else getattr(current_agent, "subtask_timeout", 60)
    )
    effective_research_timeout = (
        body.research_timeout
        if body.research_timeout is not None
        else getattr(current_agent, "research_timeout", 180)
    )
    timeout_update_requested = any(
        value is not None
        for value in (
            body.llm_request_timeout,
            body.subtask_timeout,
            body.research_timeout,
        )
    )
    if timeout_update_requested and effective_llm_timeout > effective_subtask_timeout:
        raise HTTPException(
            status_code=422,
            detail=(
                "The LLM request timeout must not exceed the subtask timeout."
            ),
        )
    if timeout_update_requested and effective_subtask_timeout > effective_research_timeout:
        raise HTTPException(
            status_code=422,
            detail=(
                "The subtask timeout must not exceed the research timeout."
            ),
        )
    embedding_provider_changed = (
        body.embedding_provider is not None
        and body.embedding_provider != current_settings.llm.embedding_provider
    )
    if embedding_provider_changed:
        try:
            point_count = get_vector_store().get_point_count()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Embedding provider cannot be changed while the "
                    "knowledge-base state is unavailable."
                ),
            ) from exc
        if point_count > 0:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Embedding provider cannot be changed while indexed "
                    "documents exist. Delete and reindex the knowledge base "
                    "with the new provider."
                ),
            )
        from mindforge.repositories.index_jobs import list_index_jobs

        try:
            active_jobs = list_index_jobs(active_only=True, limit=1)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Embedding provider cannot be changed while the "
                    "index-job state is unavailable."
                ),
            ) from exc
        if active_jobs:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Embedding provider cannot be changed while indexing "
                    "jobs are active."
                ),
            )

    from mindforge.db import (
        ApiKey,
        SessionLocal,
        encrypt_api_key,
        get_default_user_id,
    )

    db = SessionLocal()
    try:
        db.query(ApiKey).first()  # ensure table exists for single-user mode
        user_id = get_default_user_id(db)
        env_updates: dict[str, str] = {}
        _env_key_map = {
            provider: f"LLM_{prefix}_API_KEY"
            for provider, prefix in _PROVIDER_ENV_PREFIXES.items()
        }
        key_updates = {
            "deepseek": body.deepseek_api_key,
            "openai": body.openai_api_key,
            "kimi": None,
            "glm": None,
            "openai_compatible": body.compatible_api_key,
            "local": body.local_api_key,
        }
        provider_updates = list(body.llm_provider_configs or [])
        if body.llm_provider_config is not None:
            provider_updates.append(body.llm_provider_config)
        for provider_update in provider_updates:
            key_updates[provider_update.provider] = provider_update.api_key

        for provider, key_val in key_updates.items():
            existing = db.query(ApiKey).filter(ApiKey.provider == provider).first()
            # 拒绝脱敏值（***开头）被当作真实 key 保存
            if key_val is None:
                continue  # undefined → 不修改
            if key_val and key_val.startswith("***"):
                continue  # 脱敏值 → 不修改
            if key_val:
                if existing:
                    existing.key_encrypted = encrypt_api_key(key_val)
                else:
                    db.add(
                        ApiKey(
                            provider=provider,
                            key_encrypted=encrypt_api_key(key_val),
                            user_id=user_id,
                        )
                    )
                env_updates[_env_key_map[provider]] = key_val
            else:
                if existing:
                    db.delete(existing)
                env_updates[_env_key_map[provider]] = ""

        for provider_update in provider_updates:
            provider = provider_update.provider
            get_base_url = getattr(current_settings.llm, "get_base_url", None)
            current_base_url = (
                (
                    get_base_url(provider)
                    if callable(get_base_url)
                    else getattr(
                        current_settings.llm,
                        {
                            "openai": "openai_base_url",
                            "deepseek": "deepseek_base_url",
                            "kimi": "kimi_base_url",
                            "glm": "glm_base_url",
                            "openai_compatible": "compatible_base_url",
                            "local": "local_base_url",
                        }[provider],
                        "",
                    )
                )
                or ""
            ).rstrip("/")
            requested_base_url = (
                provider_update.base_url.rstrip("/")
                if provider_update.base_url is not None
                else current_base_url
            )
            get_native_search_endpoint = getattr(
                current_settings.llm,
                "get_native_web_search_endpoint",
                None,
            )
            current_native_search_endpoint = (
                (
                    get_native_search_endpoint(provider)
                    if callable(get_native_search_endpoint)
                    else (
                        getattr(
                            current_settings.llm,
                            (
                                f"{_CONFIGURABLE_PROVIDER_PREFIXES[provider]}"
                                "_native_web_search_endpoint"
                            ),
                            "",
                        )
                        if provider in _CONFIGURABLE_PROVIDER_PREFIXES
                        else ""
                    )
                )
                or ""
            ).rstrip("/")
            requested_native_search_endpoint = (
                provider_update.native_web_search_endpoint.rstrip("/")
                if provider_update.native_web_search_endpoint is not None
                else current_native_search_endpoint
            )
            key_reconfirmed = (
                provider_update.api_key is not None
                and not provider_update.api_key.startswith("***")
            )
            get_api_key = getattr(current_settings.llm, "get_api_key", None)
            configured_key = (
                get_api_key(provider)
                if callable(get_api_key)
                else getattr(
                    current_settings.llm,
                    {
                        "openai": "openai_api_key",
                        "deepseek": "deepseek_api_key",
                        "kimi": "kimi_api_key",
                        "glm": "glm_api_key",
                        "openai_compatible": "compatible_api_key",
                        "local": "local_api_key",
                    }[provider],
                    "",
                )
            )
            existing_key = str(configured_key or "").strip()
            if not existing_key:
                existing_key = _stored_provider_api_key(provider)
            if (
                (
                    requested_base_url != current_base_url
                    or requested_native_search_endpoint
                    != current_native_search_endpoint
                )
                and existing_key
                and not key_reconfirmed
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Changing a provider endpoint requires re-entering "
                        "or explicitly clearing its API key."
                    ),
                )
            base_url_keys = {
                name: f"LLM_{prefix}_BASE_URL"
                for name, prefix in _PROVIDER_ENV_PREFIXES.items()
            }
            role_key_maps = {
                "openai": {
                    "planner_model": "LLM_PLANNER_MODEL",
                    "researcher_model": "LLM_RESEARCHER_MODEL",
                    "critic_model": "LLM_CRITIC_MODEL",
                    "synthesizer_model": "LLM_SYNTHESIZER_MODEL",
                },
                "deepseek": {
                    "planner_model": "LLM_DEEPSEEK_PLANNER",
                    "researcher_model": "LLM_DEEPSEEK_RESEARCHER",
                    "critic_model": "LLM_DEEPSEEK_CRITIC",
                    "synthesizer_model": "LLM_DEEPSEEK_SYNTHESIZER",
                },
                "kimi": {
                    "planner_model": "LLM_KIMI_PLANNER_MODEL",
                    "researcher_model": "LLM_KIMI_RESEARCHER_MODEL",
                    "critic_model": "LLM_KIMI_CRITIC_MODEL",
                    "synthesizer_model": "LLM_KIMI_SYNTHESIZER_MODEL",
                },
                "glm": {
                    "planner_model": "LLM_GLM_PLANNER_MODEL",
                    "researcher_model": "LLM_GLM_RESEARCHER_MODEL",
                    "critic_model": "LLM_GLM_CRITIC_MODEL",
                    "synthesizer_model": "LLM_GLM_SYNTHESIZER_MODEL",
                },
                "openai_compatible": {
                    "planner_model": "LLM_COMPATIBLE_PLANNER_MODEL",
                    "researcher_model": ("LLM_COMPATIBLE_RESEARCHER_MODEL"),
                    "critic_model": "LLM_COMPATIBLE_CRITIC_MODEL",
                    "synthesizer_model": ("LLM_COMPATIBLE_SYNTHESIZER_MODEL"),
                },
                "local": {
                    "planner_model": "LLM_LOCAL_PLANNER_MODEL",
                    "researcher_model": "LLM_LOCAL_RESEARCHER_MODEL",
                    "critic_model": "LLM_LOCAL_CRITIC_MODEL",
                    "synthesizer_model": "LLM_LOCAL_SYNTHESIZER_MODEL",
                },
            }
            default_model_keys = {
                "kimi": "LLM_KIMI_MODEL",
                "glm": "LLM_GLM_MODEL",
                "openai_compatible": "LLM_COMPATIBLE_MODEL",
                "local": "LLM_LOCAL_MODEL",
            }
            capability_key_maps = {
                "kimi": {
                    "api_key_required": "LLM_KIMI_API_KEY_REQUIRED",
                    "supports_tools": "LLM_KIMI_SUPPORTS_TOOLS",
                    "supports_json_mode": "LLM_KIMI_SUPPORTS_JSON_MODE",
                    "supports_json_schema": "LLM_KIMI_SUPPORTS_JSON_SCHEMA",
                },
                "glm": {
                    "api_key_required": "LLM_GLM_API_KEY_REQUIRED",
                    "supports_tools": "LLM_GLM_SUPPORTS_TOOLS",
                    "supports_json_mode": "LLM_GLM_SUPPORTS_JSON_MODE",
                    "supports_json_schema": "LLM_GLM_SUPPORTS_JSON_SCHEMA",
                },
                "openai_compatible": {
                    "api_key_required": ("LLM_COMPATIBLE_API_KEY_REQUIRED"),
                    "supports_tools": ("LLM_COMPATIBLE_SUPPORTS_TOOLS"),
                    "supports_json_mode": ("LLM_COMPATIBLE_SUPPORTS_JSON_MODE"),
                    "supports_json_schema": ("LLM_COMPATIBLE_SUPPORTS_JSON_SCHEMA"),
                },
                "local": {
                    "api_key_required": "LLM_LOCAL_API_KEY_REQUIRED",
                    "supports_tools": "LLM_LOCAL_SUPPORTS_TOOLS",
                    "supports_json_mode": ("LLM_LOCAL_SUPPORTS_JSON_MODE"),
                    "supports_json_schema": ("LLM_LOCAL_SUPPORTS_JSON_SCHEMA"),
                },
            }
            native_search_key_maps = {
                "kimi": {
                    "native_web_search_protocol": (
                        "LLM_KIMI_NATIVE_WEB_SEARCH_PROTOCOL"
                    ),
                    "native_web_search_endpoint": (
                        "LLM_KIMI_NATIVE_WEB_SEARCH_ENDPOINT"
                    ),
                },
                "glm": {
                    "native_web_search_protocol": (
                        "LLM_GLM_NATIVE_WEB_SEARCH_PROTOCOL"
                    ),
                    "native_web_search_endpoint": (
                        "LLM_GLM_NATIVE_WEB_SEARCH_ENDPOINT"
                    ),
                },
                "openai_compatible": {
                    "native_web_search_protocol": (
                        "LLM_COMPATIBLE_NATIVE_WEB_SEARCH_PROTOCOL"
                    ),
                    "native_web_search_endpoint": (
                        "LLM_COMPATIBLE_NATIVE_WEB_SEARCH_ENDPOINT"
                    ),
                },
                "local": {
                    "native_web_search_protocol": (
                        "LLM_LOCAL_NATIVE_WEB_SEARCH_PROTOCOL"
                    ),
                    "native_web_search_endpoint": (
                        "LLM_LOCAL_NATIVE_WEB_SEARCH_ENDPOINT"
                    ),
                },
            }
            if provider_update.base_url is not None:
                env_updates[base_url_keys[provider]] = provider_update.base_url
            if (
                provider_update.default_model is not None
                and provider in default_model_keys
            ):
                env_updates[default_model_keys[provider]] = (
                    provider_update.default_model
                )
            for field_name, env_key in role_key_maps[provider].items():
                value = getattr(provider_update, field_name)
                if value is not None:
                    env_updates[env_key] = value
            for field_name, env_key in capability_key_maps.get(
                provider,
                {},
            ).items():
                value = getattr(provider_update, field_name)
                if value is not None:
                    env_updates[env_key] = "true" if value else "false"
            for field_name, env_key in native_search_key_maps.get(
                provider,
                {},
            ).items():
                value = getattr(provider_update, field_name)
                if value is not None:
                    env_updates[env_key] = value

        if body.llm_provider:
            env_updates["LLM_LLM_PROVIDER"] = body.llm_provider
        if body.embedding_provider:
            env_updates["LLM_EMBEDDING_PROVIDER"] = body.embedding_provider
        if body.research_mode is not None:
            env_updates["AGENT_RESEARCH_MODE"] = body.research_mode
        if body.source_policy is not None:
            env_updates["AGENT_SOURCE_POLICY"] = body.source_policy
        if body.fallback_enabled is not None:
            env_updates["AGENT_FALLBACK_ENABLED"] = (
                "true" if body.fallback_enabled else "false"
            )
        if body.retrieval_top_k is not None:
            env_updates["RETRIEVAL_VECTOR_TOP_K"] = str(body.retrieval_top_k)
        if body.rerank_top_k is not None:
            env_updates["RETRIEVAL_RERANK_TOP_K"] = str(body.rerank_top_k)
        if body.retrieval_min_score is not None:
            env_updates["RETRIEVAL_MIN_SCORE"] = str(
                body.retrieval_min_score
            )
        if body.keyword_min_coverage is not None:
            env_updates["RETRIEVAL_KEYWORD_MIN_COVERAGE"] = str(
                body.keyword_min_coverage
            )
        if body.max_iterations is not None:
            env_updates["AGENT_MAX_ITERATIONS"] = str(body.max_iterations)
        if body.max_refine_rounds is not None:
            env_updates["AGENT_MAX_REFINE_ROUNDS"] = str(body.max_refine_rounds)
        if body.critic_threshold is not None:
            env_updates["AGENT_CRITIC_THRESHOLD"] = str(body.critic_threshold)
        if body.subtask_timeout is not None:
            env_updates["AGENT_SUBTASK_TIMEOUT"] = str(body.subtask_timeout)
        if body.research_timeout is not None:
            env_updates["AGENT_RESEARCH_TIMEOUT"] = str(body.research_timeout)
        if body.llm_request_timeout is not None:
            env_updates["AGENT_LLM_REQUEST_TIMEOUT"] = str(
                body.llm_request_timeout
            )
        if body.max_subtasks is not None:
            env_updates["AGENT_MAX_SUBTASKS"] = str(body.max_subtasks)
        if body.max_tool_calls_total is not None:
            env_updates["AGENT_MAX_TOOL_CALLS_TOTAL"] = str(
                body.max_tool_calls_total
            )
        if body.max_history_entries is not None:
            env_updates["API_MAX_HISTORY_ENTRIES"] = str(
                body.max_history_entries
            )
        for field_name, env_key in (
            ("langfuse_public_key", "OBSERVABILITY_LANGFUSE_PUBLIC_KEY"),
            ("langfuse_secret_key", "OBSERVABILITY_LANGFUSE_SECRET_KEY"),
        ):
            value = getattr(body, field_name)
            if value is not None and not value.startswith("***"):
                env_updates[env_key] = value
        if body.langfuse_host is not None:
            env_updates["OBSERVABILITY_LANGFUSE_HOST"] = body.langfuse_host
        if body.observability_capture_content is not None:
            env_updates["OBSERVABILITY_CAPTURE_CONTENT"] = (
                "true" if body.observability_capture_content else "false"
            )
        if body.trace_retention_days is not None:
            env_updates["OBSERVABILITY_TRACE_RETENTION_DAYS"] = str(
                body.trace_retention_days
            )

        env_snapshot: dict[str, tuple[bool, str]] = {}
        if env_updates:
            env_snapshot = _snapshot_env_file(set(env_updates))
            try:
                _sync_env_file(env_updates)
            except Exception as e:
                db.rollback()
                try:
                    _restore_env_file(env_snapshot)
                except Exception:
                    logger.critical(
                        ".env sync and rollback both failed.",
                        exc_info=True,
                    )
                logger.exception(
                    "Settings .env sync failed; database changes rolled back."
                )
                raise HTTPException(
                    status_code=500,
                    detail="Settings could not be persisted.",
                ) from e

        try:
            db.commit()
        except Exception as exc:
            db.rollback()
            if env_snapshot:
                try:
                    _restore_env_file(env_snapshot)
                except Exception:
                    logger.critical(
                        "Database commit and .env rollback both failed.",
                        exc_info=True,
                    )
            raise HTTPException(
                status_code=500,
                detail="Settings could not be persisted.",
            ) from exc

        if env_updates:
            for key, value in env_updates.items():
                if value:
                    _os.environ[key] = value
                else:
                    _os.environ.pop(key, None)

        from mindforge.config import reload_settings

        reload_settings()
        provider_updates_present = bool(
            body.llm_provider_configs or body.llm_provider_config is not None
        )
        direct_key_update = any(
            value is not None
            for value in (
                body.deepseek_api_key,
                body.openai_api_key,
                body.compatible_api_key,
                body.local_api_key,
            )
        )
        llm_changed = (
            provider_updates_present
            or direct_key_update
            or (
                body.llm_provider is not None
                and body.llm_provider
                != getattr(
                    current_settings.llm,
                    "llm_provider",
                    None,
                )
            )
        )
        current_retrieval = getattr(
            current_settings,
            "retrieval",
            None,
        )
        retrieval_changed = any(
            value is not None and value != current
            for value, current in (
                (
                    body.retrieval_top_k,
                    getattr(current_retrieval, "vector_top_k", None),
                ),
                (
                    body.rerank_top_k,
                    getattr(current_retrieval, "rerank_top_k", None),
                ),
                (
                    body.retrieval_min_score,
                    getattr(current_retrieval, "min_score", None),
                ),
                (
                    body.keyword_min_coverage,
                    getattr(current_retrieval, "keyword_min_coverage", None),
                ),
            )
        )
        current_agent = getattr(current_settings, "agent", None)
        agent_changed = any(
            value is not None and value != current
            for value, current in (
                (
                    body.max_iterations,
                    getattr(current_agent, "max_iterations", None),
                ),
                (
                    body.max_refine_rounds,
                    getattr(current_agent, "max_refine_rounds", None),
                ),
                (
                    body.critic_threshold,
                    getattr(current_agent, "critic_threshold", None),
                ),
                (
                    body.subtask_timeout,
                    getattr(current_agent, "subtask_timeout", None),
                ),
                (
                    body.research_timeout,
                    getattr(current_agent, "research_timeout", None),
                ),
                (
                    body.research_mode,
                    getattr(current_agent, "research_mode", None),
                ),
                (
                    body.source_policy,
                    getattr(current_agent, "source_policy", None),
                ),
                (
                    body.fallback_enabled,
                    getattr(current_agent, "fallback_enabled", None),
                ),
                (
                    body.llm_request_timeout,
                    getattr(current_agent, "llm_request_timeout", None),
                ),
                (
                    body.max_subtasks,
                    getattr(current_agent, "max_subtasks", None),
                ),
                (
                    body.max_tool_calls_total,
                    getattr(current_agent, "max_tool_calls_total", None),
                ),
            )
        )
        observability_changed = any(
            value is not None
            for value in (
                body.langfuse_public_key,
                body.langfuse_secret_key,
                body.langfuse_host,
                body.observability_capture_content,
                body.trace_retention_days,
            )
        )
        if observability_changed:
            from mindforge.observability.tracer import close_tracer

            close_tracer()
        reset_runtime_components(
            reset_orchestrator=llm_changed or agent_changed,
            reset_embedder=embedding_provider_changed,
            reset_vector_store=False,
            reset_retrieval=(
                llm_changed or embedding_provider_changed or retrieval_changed
            ),
            reset_indexing=False,
        )
        return {"status": "saved"}
    finally:
        db.close()


def reset_runtime_components(
    *,
    reset_orchestrator: bool = True,
    reset_embedder: bool = True,
    reset_vector_store: bool = True,
    reset_retrieval: bool = True,
    reset_indexing: bool = True,
) -> None:
    """Recreate configuration-bound singletons after settings updates."""
    global _orchestrator
    previous_orchestrator = _orchestrator if reset_orchestrator else None
    if reset_orchestrator:
        _orchestrator = None
    if previous_orchestrator is not None:
        try:
            previous_orchestrator.close()
        except Exception:
            logger.exception("Failed to close the previous orchestrator during reset.")

    from mindforge.ingestion.embedder import (
        reset_embedder as reset_embedder_component,
    )
    from mindforge.retrieval.service import reset_retrieval_service
    from mindforge.retrieval.vector_store import (
        reset_vector_store as reset_vector_store_component,
    )
    from mindforge.services.indexing import reset_indexing_service

    resets = []
    if reset_embedder:
        resets.append(("embedder", reset_embedder_component))
    if reset_vector_store:
        resets.append(("vector store", reset_vector_store_component))
    if reset_retrieval:
        resets.append(("retrieval service", reset_retrieval_service))
    if reset_indexing:
        resets.append(("indexing service", reset_indexing_service))
    for component_name, reset in resets:
        try:
            reset()
        except Exception:
            logger.exception(
                "Failed to reset the %s after settings update.",
                component_name,
            )


# ------------------------------------------------------------------
# Observability
# ------------------------------------------------------------------


@router.get(
    "/observability/status",
    response_model=ObservabilityStatusResponse,
)
def get_observability_status():
    """Return public tracing status without exposing credentials or paths."""
    from mindforge.observability.store import TraceRepository

    return TraceRepository().status()


@router.get(
    "/observability/traces",
    response_model=TraceListResponse,
)
def list_observability_traces(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Literal["success", "degraded", "error", "cancelled"] | None = Query(
        None
    ),
    search: str = Query("", max_length=200),
):
    """Return bounded local trace summaries for the operations UI."""
    from mindforge.observability.store import TraceRepository

    return TraceRepository().list_traces(
        limit=limit,
        offset=offset,
        status=status,
        search=search,
    )


@router.get(
    "/observability/traces/{trace_id}",
    response_model=TraceDetailResponse,
)
def get_observability_trace(trace_id: str):
    """Return one local trace and its bounded observation chain."""
    from mindforge.observability.store import TraceRepository

    detail = TraceRepository().get_trace(trace_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return detail


@router.delete("/observability/traces/{trace_id}", status_code=204)
def delete_observability_trace(trace_id: str):
    """Delete one local trace without affecting research history."""
    from mindforge.observability.store import TraceRepository

    if not TraceRepository().delete_trace(trace_id):
        raise HTTPException(status_code=404, detail="Trace not found.")
    return None


@router.delete("/observability/traces")
def clear_observability_traces():
    """Delete all local traces without affecting research history."""
    from mindforge.observability.store import TraceRepository

    return {"deleted": TraceRepository().clear_traces()}


# ------------------------------------------------------------------
# History
# ------------------------------------------------------------------


@router.get("/history")
def list_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Return paginated research history entries."""
    from mindforge.db import (
        ResearchHistory,
        SessionLocal,
        get_default_user_id,
    )

    db = SessionLocal()
    try:
        user_id = get_default_user_id(db)
        offset = max(0, (page - 1)) * page_size
        base_query = db.query(ResearchHistory).filter(
            ResearchHistory.user_id == user_id
        )
        total = base_query.count()
        entries = (
            base_query.order_by(ResearchHistory.created_at.desc())
            .offset(offset)
            .limit(page_size)
            .all()
        )
        return {
            "entries": [
                HistoryItem(
                    id=e.id,
                    task=e.task,
                    report=e.report[:500] if e.report else None,
                    quality_score=e.quality_score,
                    model_used=e.model_used,
                    token_usage=_parse_history_token_usage(
                        getattr(e, "token_usage", None)
                    ),
                    trace_id=getattr(e, "trace_id", None),
                    created_at=_serialize_datetime_utc(e.created_at),
                ).model_dump()
                for e in entries
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    finally:
        db.close()


@router.get("/history/{history_id}", response_model=HistoryItem)
def get_history_entry(history_id: int):
    """Return one complete research-history record."""
    from mindforge.db import (
        ResearchHistory,
        SessionLocal,
        get_default_user_id,
    )

    db = SessionLocal()
    try:
        entry = (
            db.query(ResearchHistory)
            .filter(
                ResearchHistory.id == history_id,
                ResearchHistory.user_id == get_default_user_id(db),
            )
            .first()
        )
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail="History entry not found.",
            )
        return HistoryItem(
            id=entry.id,
            task=entry.task,
            report=entry.report,
            quality_score=entry.quality_score,
            model_used=entry.model_used,
            token_usage=_parse_history_token_usage(getattr(entry, "token_usage", None)),
            sources=_parse_history_sources(getattr(entry, "sources", None)),
            trace_id=getattr(entry, "trace_id", None),
            created_at=_serialize_datetime_utc(entry.created_at),
        )
    finally:
        db.close()


@router.post("/history")
def save_history(body: HistorySaveRequest):
    """Save a research result to history."""
    from mindforge.db import (
        ResearchHistory,
        SessionLocal,
        get_default_user_id,
    )
    import json as _json

    db = SessionLocal()
    try:
        user_id = get_default_user_id(db)
        entry = ResearchHistory(
            user_id=user_id,
            task=body.task,
            report=body.report,
            quality_score=body.quality_score,
            model_used=body.model_used,
            token_usage=_json.dumps(body.token_usage),
            sources=_json.dumps(
                [source.model_dump(exclude_none=True) for source in body.sources],
                ensure_ascii=False,
            ),
            trace_id=body.trace_id,
        )
        db.add(entry)
        try:
            db.flush()
            max_entries = get_settings().api.max_history_entries
            if max_entries > 0:
                stale_entries = (
                    db.query(ResearchHistory)
                    .filter(ResearchHistory.user_id == user_id)
                    .order_by(
                        ResearchHistory.created_at.desc(),
                        ResearchHistory.id.desc(),
                    )
                    .offset(max_entries)
                    .all()
                )
                for stale in stale_entries:
                    db.delete(stale)
            db.commit()
        except Exception:
            db.rollback()
            raise
        return {"id": entry.id, "status": "saved"}
    finally:
        db.close()


@router.delete("/history/{entry_id}", status_code=204)
def delete_history_entry(entry_id: int):
    """Delete a single research history entry."""
    from mindforge.db import (
        SessionLocal,
        ResearchHistory,
        get_default_user_id,
    )

    db = SessionLocal()
    try:
        entry = (
            db.query(ResearchHistory)
            .filter(
                ResearchHistory.id == entry_id,
                ResearchHistory.user_id == get_default_user_id(db),
            )
            .first()
        )
        if entry is None:
            raise HTTPException(status_code=404, detail="History entry not found")
        db.delete(entry)
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise
        return None
    finally:
        db.close()


@router.delete("/history", status_code=204)
def clear_history():
    """Delete all research history entries."""
    from mindforge.db import (
        SessionLocal,
        ResearchHistory,
        get_default_user_id,
    )

    db = SessionLocal()
    try:
        db.query(ResearchHistory).filter(
            ResearchHistory.user_id == get_default_user_id(db)
        ).delete()
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise
        return None
    finally:
        db.close()


def _serialize_event(event: dict) -> dict:
    """Convert dataclass values in an event dict to plain dicts for JSON serialization."""
    import dataclasses as _dc

    serialized: dict[str, Any] = {}
    for key, val in event.items():
        if _dc.is_dataclass(val) and not isinstance(val, type):
            if hasattr(val, "to_dict"):
                serialized[key] = val.to_dict()
            else:
                serialized[key] = _dc.asdict(val)
        else:
            serialized[key] = val
    return serialized


async def _stream_response(
    orch: Orchestrator | None,
    task: str,
    *,
    request_id: str | None = None,
    cancellation: asyncio.Event | None = None,
) -> AsyncGenerator[bytes, None]:
    """Stream SSE bytes while one producer task owns the trace context."""
    chunk_queue: asyncio.Queue[bytes | object] = asyncio.Queue()
    stream_finished = object()

    async def pump_chunks() -> None:
        async with aclosing(_stream_response_events(orch, task)) as events:
            try:
                async for chunk in events:
                    chunk_queue.put_nowait(chunk)
            finally:
                chunk_queue.put_nowait(stream_finished)

    producer_task: asyncio.Task[None] | None = None
    cancellation_task: asyncio.Task[bool] | None = None
    chunk_task: asyncio.Task[bytes | object] | None = None
    try:
        producer_task = asyncio.create_task(pump_chunks())
        if cancellation is not None:
            cancellation_task = asyncio.create_task(cancellation.wait())
        # A client can disconnect immediately after receiving response headers.
        # Keep startup inside the cleanup boundary so the producer cannot
        # outlive its SSE connection.
        await asyncio.sleep(0)
        while True:
            chunk_task = asyncio.create_task(chunk_queue.get())
            if cancellation_task is not None:
                done, _pending = await asyncio.wait(
                    {chunk_task, cancellation_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancellation_task in done:
                    if not chunk_task.done():
                        chunk_task.cancel()
                        await asyncio.gather(
                            chunk_task,
                            return_exceptions=True,
                        )
                    chunk_task = None
                    return
            chunk = await chunk_task
            chunk_task = None
            if chunk is stream_finished:
                await producer_task
                return
            if not isinstance(chunk, bytes):
                raise RuntimeError("SSE producer returned an invalid chunk.")
            yield chunk
    finally:
        if chunk_task is not None and not chunk_task.done():
            chunk_task.cancel()
            await asyncio.gather(chunk_task, return_exceptions=True)
        if cancellation_task is not None and not cancellation_task.done():
            cancellation_task.cancel()
            await asyncio.gather(cancellation_task, return_exceptions=True)
        if producer_task is not None and not producer_task.done():
            producer_task.cancel()
            await asyncio.gather(producer_task, return_exceptions=True)
        if request_id is not None and cancellation is not None:
            _unregister_research_cancellation(request_id, cancellation)


async def _stream_response_events(
    orch: Orchestrator | None,
    task: str,
) -> AsyncGenerator[bytes, None]:
    """SSE streaming — with automatic fallback to retrieval-only on LLM failure."""
    started = time.perf_counter()
    completed_result: AgentResult | dict[str, Any] | None = None
    primary_failure: AgentResult | None = None
    primary_failure_reason: str | None = None
    with _research_trace_context(task, transport="sse") as trace_span:
        if trace_span is not None:
            trace_span.input = {"task": task}
        trace_id = trace_span.trace_id if trace_span is not None else None
        if trace_id:
            started_event = {
                "type": "trace_started",
                "trace_id": trace_id,
            }
            yield (
                f"data: {json.dumps(started_event, ensure_ascii=False)}\n\n"
            ).encode()

        use_retrieval_fallback = orch is None
        fallback_enabled = get_settings().agent.fallback_enabled
        if orch is not None:
            try:
                research_stream_source = _call_orchestrator_method(
                    orch,
                    "stream_run",
                    task,
                )
                async with aclosing(research_stream_source) as research_stream:
                    async for event in research_stream:
                        if trace_id:
                            event = {**event, "trace_id": trace_id}
                        if event.get("type") == "done":
                            result = event.get("result")
                            success = (
                                result.success
                                if isinstance(result, AgentResult)
                                else (
                                    result.get("success")
                                    if isinstance(result, dict)
                                    else None
                                )
                            )
                            if success is False:
                                primary_failure = (
                                    result
                                    if isinstance(result, AgentResult)
                                    else None
                                )
                                primary_failure_reason = (
                                    result.output
                                    if isinstance(result, AgentResult)
                                    else str(result.get("output", ""))
                                )
                                primary_failure_reason = (
                                    primary_failure_reason
                                    or (
                                        "Agent pipeline returned an "
                                        "unsuccessful result."
                                    )
                                )
                                use_retrieval_fallback = fallback_enabled
                                if not fallback_enabled:
                                    completed_result = result
                                    yield (
                                        "data: "
                                        f"{json.dumps(_serialize_event(event), ensure_ascii=False)}"
                                        "\n\n"
                                    ).encode()
                                break
                            completed_result = result
                        elif event.get("type") == "error":
                            primary_failure_reason = str(
                                event.get("content") or "研究任务执行失败。"
                            )
                            use_retrieval_fallback = fallback_enabled
                            if fallback_enabled:
                                continue
                        try:
                            payload = json.dumps(
                                _serialize_event(event),
                                ensure_ascii=False,
                            )
                        except TypeError:
                            payload = json.dumps(
                                {
                                    "type": "info",
                                    "content": str(event)[:200],
                                    "trace_id": trace_id,
                                },
                                ensure_ascii=False,
                            )
                        yield f"data: {payload}\n\n".encode()
            except Exception as exc:
                primary_failure_reason = str(exc)
                logger.warning(
                    "Agent SSE stream failed: %s; using retrieval fallback.",
                    exc,
                )
                use_retrieval_fallback = fallback_enabled

        if use_retrieval_fallback:
            from mindforge.tools.rag_tool import RAGTool

            try:
                rag = RAGTool()
                result = await rag.execute_async(
                    query=task,
                    mode="hybrid",
                    top_k=5,
                )
                result_data = result.data or {}
                has_relevant_results = (
                    bool(result_data.get("total"))
                    if "total" in result_data
                    else bool(result.output.strip())
                )
                if result.success and (
                    primary_failure_reason is None or has_relevant_results
                ):
                    degraded = primary_failure_reason is not None
                    token_usage = (
                        dict(primary_failure.token_usage)
                        if primary_failure is not None
                        else {}
                    )
                    cost_usd = (
                        primary_failure.cost_usd
                        if primary_failure is not None
                        else None
                    )
                    cost_status = (
                        primary_failure.cost_status
                        if primary_failure is not None
                        else "not_applicable"
                    )
                    completed_result = {
                        "agent_name": "orchestrator",
                        "success": True,
                        "output": result.output,
                        "data": {
                            "plan": None,
                            "subtask_outputs": [],
                            "critic_score": None,
                            "refine_rounds": 0,
                            "fallback": True,
                            "primary_failure": primary_failure_reason,
                            "retrieval_quality": float(
                                result_data.get("retrieval_quality", 0.0)
                            ),
                            "sources": list(result_data.get("sources", [])),
                        },
                        "metadata": {
                            "quality": None,
                            "quality_status": "not_evaluated",
                            "retrieval_quality": float(
                                result_data.get("retrieval_quality", 0.0)
                            ),
                            "outcome": (
                                "degraded" if degraded else "retrieval_only"
                            ),
                            "failure_reason": primary_failure_reason,
                            "cost": cost_usd,
                            "cost_status": cost_status,
                            "subtask_count": 0,
                            "refine_rounds": 0,
                            "model": "fallback-retrieval",
                            "trace_id": trace_id,
                        },
                        "token_usage": token_usage,
                        "cost_usd": cost_usd,
                        "cost_status": cost_status,
                        "trace_id": trace_id,
                    }
                    fallback = {
                        "type": "done",
                        "trace_id": trace_id,
                        "result": completed_result,
                    }
                else:
                    fallback_error = (
                        primary_failure_reason
                        if primary_failure_reason
                        else (
                            "知识库中没有检索到高度相关资料。"
                            if result.success
                            else "知识库检索回退也未成功："
                            f"{result.error or '未知错误'}"
                        )
                    )
                    fallback = {
                        "type": "error",
                        "trace_id": trace_id,
                        "content": fallback_error,
                    }
                yield (
                    f"data: {json.dumps(fallback, ensure_ascii=False)}\n\n"
                ).encode()
            except Exception:
                logger.exception("Retrieval fallback failed during SSE streaming.")
                fallback = {
                    "type": "error",
                    "trace_id": trace_id,
                    "content": "研究服务暂时不可用，请稍后重试。",
                }
                yield (
                    f"data: {json.dumps(fallback, ensure_ascii=False)}\n\n"
                ).encode()

        elapsed_ms = (time.perf_counter() - started) * 1000
        if isinstance(completed_result, AgentResult):
            _finish_research_trace(
                trace_span,
                success=completed_result.success,
                latency_ms=elapsed_ms,
                cost_usd=completed_result.cost_usd,
                cost_status=completed_result.cost_status,
                total_tokens=int(
                    completed_result.token_usage.get("total_tokens", 0)
                ),
                report_chars=len(completed_result.output),
                fallback=bool(completed_result.data.get("fallback")),
                outcome=str(completed_result.metadata.get("outcome") or "success"),
                failure_reason=(
                    str(completed_result.metadata.get("failure_reason"))
                    if completed_result.metadata.get("failure_reason")
                    else None
                ),
            )
        elif isinstance(completed_result, dict):
            _finish_research_trace(
                trace_span,
                success=bool(completed_result.get("success")),
                latency_ms=elapsed_ms,
                cost_usd=(
                    float(completed_result["cost_usd"])
                    if isinstance(completed_result.get("cost_usd"), (int, float))
                    else None
                ),
                cost_status=str(
                    completed_result.get("cost_status") or "usage_unavailable"
                ),
                report_chars=len(str(completed_result.get("output") or "")),
                fallback=bool(
                    dict(completed_result.get("data") or {}).get("fallback")
                ),
                outcome=str(
                    dict(completed_result.get("metadata") or {}).get("outcome")
                    or "success"
                ),
                failure_reason=(
                    str(
                        dict(completed_result.get("metadata") or {}).get(
                            "failure_reason"
                        )
                    )
                    if dict(completed_result.get("metadata") or {}).get(
                        "failure_reason"
                    )
                    else None
                ),
            )
        else:
            _finish_research_trace(
                trace_span,
                success=False,
                latency_ms=elapsed_ms,
                cost_usd=None,
                cost_status="usage_unavailable",
                failure_reason=primary_failure_reason,
            )
        yield b"data: [DONE]\n\n"
