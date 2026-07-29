"""API route definitions for MindForge — 真实实现"""

from __future__ import annotations
import json
import time
import uuid
import logging
import asyncio
import os
import shutil
import tempfile
import threading
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
    DocumentContentResponse, DocumentItem, HealthResponse,
    HistoryItem, HistorySaveRequest, IndexJobResponse, IndexRequest,
    IndexResponse,
    QueryRequest, QueryResponse,
    LLMProviderConfig, LLMProviderName, SettingsResponse,
    SettingsUpdateRequest,
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
    index_slot,
    remove_document_status,
    set_document_status,
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
IndexProgressCallback = Callable[
    [str, float, int, dict[str, float]],
    Awaitable[None],
]


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
        return urlunsplit(
            (parsed.scheme, netloc, parsed.path, "", "")
        )
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
            logger.info(
                "Redis episodic memory unavailable; using process memory."
            )
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
        raise ValueError(
            "Embedding vector count does not match document chunk count."
        )

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
        batch = texts[index:index + batch_size]
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
        await store.upsert(points[index:index + batch_size])
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
) -> list[DocumentChunk]:
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
            "Skipping RAPTOR/GraphRAG for '%s' (%d chunks): "
            "document is too short.",
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
            model = (
                settings.raptor.summary_model.strip()
                or settings.llm.get_model("researcher", provider)
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
                settings.graphrag.community_summary_model.strip()
                or entity_model
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
                await store.upsert(
                    raptor_points[index:index + batch_size]
                )
            logger.info(
                "RAPTOR: %d summary nodes indexed.",
                len(raptor_points),
            )
        except Exception:
            logger.exception("RAPTOR indexing skipped.")
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
    await index_auxiliary_documents(
        auxiliary_docs,
        graph_entity_llm=graph_entity_llm,
        graph_summary_llm=graph_summary_llm,
        use_graphrag=bool(
            enable_graphrag
            and graph_entity_llm
            and graph_summary_llm
        ),
        progress_callback=progress_callback,
        timings=stage_timings,
        start_progress=94.0 if enable_raptor and raptor_llm else 88.0,
    )
    return chunks


async def _rollback_document_index(doc_id: str) -> None:
    """Best-effort rollback for a partially indexed document."""
    try:
        await get_vector_store().delete(doc_id)
        from mindforge.retrieval.service import delete_auxiliary_document
        from mindforge.services.document_assets import remove_document_assets

        await delete_auxiliary_document(doc_id)
        await asyncio.to_thread(remove_document_assets, doc_id)
    except Exception:
        logger.exception(
            "Failed to roll back partial index for document %s.",
            doc_id,
        )


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
) -> list[DocumentChunk]:
    """Index one document under a bounded slot and persist its state."""
    from mindforge.services.indexing import build_index_signature

    index_signature = build_index_signature(
        strategy=strategy,
        use_raptor=use_raptor,
        use_graphrag=use_graphrag,
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
    try:
        async with index_slot():
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
                    timings["asset_persistence"] = (
                        time.perf_counter() - asset_started
                    )
                await set_document_status(
                    doc_id=parsed.doc_id,
                    filename=source,
                    status="indexing",
                    index_strategy=strategy,
                    use_raptor=use_raptor,
                    use_graphrag=use_graphrag,
                    parser_metadata=dict(
                        getattr(parsed, "metadata", {}) or {}
                    ),
                )
                if cancelled is not None and cancelled():
                    raise IndexingCancelledError("Indexing was cancelled.")
            chunks = await _index_parsed_document(
                parsed=parsed,
                source=source,
                strategy=strategy,
                metadata=metadata,
                use_raptor=use_raptor,
                use_graphrag=use_graphrag,
                progress_callback=progress_callback,
                timings=timings,
                cancelled=cancelled,
            )
    except (asyncio.CancelledError, IndexingCancelledError):
        await _rollback_document_index(parsed.doc_id)
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
        await _rollback_document_index(parsed.doc_id)
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

    await set_document_status(
        doc_id=parsed.doc_id,
        filename=source,
        status="indexed",
        chunk_count=len(chunks),
        index_signature=index_signature,
        index_strategy=strategy,
        use_raptor=use_raptor,
        use_graphrag=use_graphrag,
        parser_metadata=dict(getattr(parsed, "metadata", {}) or {}),
    )
    return chunks


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


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    """Submit a research task. Falls back to retrieval-only if LLM is unavailable."""
    start = time.time()
    llm_available = await asyncio.to_thread(has_llm_credentials)

    if body.stream:
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
            _stream_response(orch, body.task),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Try full Agent pipeline first when the configured provider is available.
    if llm_available:
        try:
            orch = await asyncio.to_thread(get_orchestrator)
            result = await orch.run(body.task)
            if not result.success:
                raise RuntimeError(
                    result.output
                    or "Agent pipeline returned an unsuccessful result."
                )
            latency = (time.time() - start) * 1000
            return QueryResponse(
                task_id=uuid.uuid4().hex[:12],
                report=result.output,
                sources=list(result.data.get("sources", [])),
                quality_score=float(result.metadata.get("quality", 0)),
                latency_ms=round(latency, 2),
                cost_usd=round(float(result.metadata.get("cost", 0)), 6),
                iterations=int(result.metadata.get("subtask_count", 0)),
            )
        except LLMConfigurationError as exc:
            logger.warning(
                "Configured Agent provider became unavailable; "
                "using retrieval fallback: %s",
                exc,
            )
        except Exception:
            logger.exception(
                "Agent pipeline failed, falling back to retrieval-only."
            )
    else:
        logger.info("No LLM credentials configured; using retrieval-only.")

    # Fallback: search knowledge base directly (no LLM needed)
    try:
        from mindforge.tools.rag_tool import RAGTool
        rag = RAGTool()
        result = await rag.execute_async(
            query=body.task,
            mode="hybrid",
        )
        if not result.success:
            raise RuntimeError(
                result.error or "Retrieval fallback failed."
            )
        latency = (time.time() - start) * 1000
        return QueryResponse(
            task_id=uuid.uuid4().hex[:12],
            report=result.output,
            sources=list((result.data or {}).get("sources", [])),
            quality_score=float((result.data or {}).get("quality", 0.0)),
            latency_ms=round(latency, 2),
            cost_usd=0.0,
            iterations=0,
        )
    except Exception as exc:
        logger.exception("Retrieval-only query failed.")
        raise HTTPException(
            status_code=503,
            detail="Knowledge base retrieval is temporarily unavailable.",
        ) from exc


@router.post("/index", response_model=IndexResponse)
async def index_document(body: IndexRequest):
    """Ingest a document into the Qdrant knowledge base."""
    if not body.file_url and not body.file_path:
        raise HTTPException(status_code=422, detail="file_url or file_path required")

    if body.file_url:
        raise HTTPException(
            status_code=400,
            detail=(
                "Remote URL indexing is not supported. Upload the document "
                "through /api/v1/upload."
            ),
        )
    if not get_settings().api.allow_local_file_index:
        raise HTTPException(
            status_code=403,
            detail="Local file indexing is disabled by configuration.",
        )

    file_path = body.file_path or ""
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
    chunks = await _index_with_lifecycle(
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
    job_dir = (
        resolve_project_path(get_settings().app.data_dir)
        / "index-jobs"
    )
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
        document_count, chunk_count = await asyncio.to_thread(
            get_document_stats
        )
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
        "qdrant_url": _public_service_url(
            get_settings().vector_store.qdrant_url
        ),
        "redis_url": _public_service_url(
            get_settings().cache.redis_url
        ),
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


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str):
    """Delete a document from Qdrant."""
    store = get_vector_store()
    try:
        await store.delete(doc_id)
        from mindforge.retrieval.service import delete_auxiliary_document
        from mindforge.services.document_assets import remove_document_assets

        await delete_auxiliary_document(doc_id)
        await asyncio.to_thread(remove_document_assets, doc_id)
        await remove_document_status(doc_id)
    except Exception:
        logger.exception("Delete failed for document %s.", doc_id)
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is temporarily unavailable.",
        )
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
        raise HTTPException(status_code=404, detail="Document asset is unavailable.") from exc
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
        if (
            pl.get("doc_id") == doc_id
            and not pl.get("is_summary", False)
        ):
            chunks.append({
                "chunk_id": pl.get("chunk_id", ""),
                "content": pl.get("content", ""),
                "chunk_index": pl.get("chunk_index"),
                "chunk_start": pl.get("chunk_start"),
                "chunk_end": pl.get("chunk_end"),
            })
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
    # Sanitize filename — prevent path traversal
    import re as _re
    safe_name = file.filename or "uploaded_doc"
    safe_name = _re.sub(r'[\\/]', '_', safe_name)  # strip path separators
    safe_name = _re.sub(r'\.\.+', '', safe_name)     # strip double dots

    suffix = Path(safe_name).suffix.lower()
    if suffix not in DocumentParser.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "不支持的文件格式，支持: "
                + ", ".join(sorted(DocumentParser.SUPPORTED_EXTENSIONS))
            ),
        )

    # Size limit: stream to disk so large uploads do not occupy equal RAM.
    MAX_UPLOAD_BYTES = get_settings().api.max_upload_mb * 1024 * 1024
    upload_dir = resolve_project_path(get_settings().app.data_dir)
    await asyncio.to_thread(
        upload_dir.mkdir,
        parents=True,
        exist_ok=True,
    )
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    file_path = str(upload_dir / unique_name)
    parsed = None
    try:
        with open(file_path, "wb") as f:
            total_bytes = 0
            while True:
                block = await file.read(1024 * 1024)
                if not block:
                    break
                total_bytes += len(block)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "文件过大（最大 "
                            f"{get_settings().api.max_upload_mb}MB）"
                        ),
                    )
                await asyncio.to_thread(f.write, block)

        parser = DocumentParser()
        parsed = await _parse_document_file(parser, file_path)
        source = file.filename or parsed.filename
        chunks = await _index_with_lifecycle(
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
        )
    finally:
        # Always attempt cleanup of the uploaded temp file after indexing
        try:
            _os.remove(file_path)
        except OSError:
            pass


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

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
                decrypted = decrypt_api_key(
                    db_keys[provider].key_encrypted
                )
                return (
                    "***" + decrypted[-4:]
                    if decrypted
                    else "***configured"
                )
            if settings_key:
                return "***" + settings_key[-4:]
            return ""

        provider_labels = {
            "openai": "OpenAI",
            "deepseek": "DeepSeek",
            "openai_compatible": "OpenAI 兼容接口",
            "local": "本地模型",
        }

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
                prefix = (
                    "compatible"
                    if provider == "openai_compatible"
                    else "local"
                )
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
                label=provider_labels[provider],
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
                configured=has_llm_credentials(provider),
            )

        provider_configs = [
            _provider_config(provider)
            for provider in (
                "openai",
                "deepseek",
                "openai_compatible",
                "local",
            )
        ]
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
            retrieval_top_k=s.retrieval.vector_top_k,
            rerank_top_k=s.retrieval.rerank_top_k,
            max_iterations=s.agent.max_iterations,
            max_refine_rounds=s.agent.max_refine_rounds,
            critic_threshold=s.agent.critic_threshold,
            subtask_timeout=s.agent.subtask_timeout,
            research_timeout=s.agent.research_timeout,
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
    return {
        key: (key in values, values.get(key) or "")
        for key in keys
    }


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
    if (
        body.embedding_provider is not None
        and body.embedding_provider
        != current_settings.llm.embedding_provider
    ):
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
            "deepseek": "LLM_DEEPSEEK_API_KEY",
            "openai": "LLM_OPENAI_API_KEY",
            "openai_compatible": "LLM_COMPATIBLE_API_KEY",
            "local": "LLM_LOCAL_API_KEY",
        }
        key_updates = {
            "deepseek": body.deepseek_api_key,
            "openai": body.openai_api_key,
            "openai_compatible": body.compatible_api_key,
            "local": body.local_api_key,
        }
        provider_updates = list(body.llm_provider_configs or [])
        if body.llm_provider_config is not None:
            provider_updates.append(body.llm_provider_config)
        for provider_update in provider_updates:
            key_updates[provider_update.provider] = provider_update.api_key

        for provider, key_val in key_updates.items():
            existing = db.query(ApiKey).filter(
                ApiKey.provider == provider
            ).first()
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
            base_url_keys = {
                "openai": "LLM_OPENAI_BASE_URL",
                "deepseek": "LLM_DEEPSEEK_BASE_URL",
                "openai_compatible": "LLM_COMPATIBLE_BASE_URL",
                "local": "LLM_LOCAL_BASE_URL",
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
                "openai_compatible": {
                    "planner_model": "LLM_COMPATIBLE_PLANNER_MODEL",
                    "researcher_model": (
                        "LLM_COMPATIBLE_RESEARCHER_MODEL"
                    ),
                    "critic_model": "LLM_COMPATIBLE_CRITIC_MODEL",
                    "synthesizer_model": (
                        "LLM_COMPATIBLE_SYNTHESIZER_MODEL"
                    ),
                },
                "local": {
                    "planner_model": "LLM_LOCAL_PLANNER_MODEL",
                    "researcher_model": "LLM_LOCAL_RESEARCHER_MODEL",
                    "critic_model": "LLM_LOCAL_CRITIC_MODEL",
                    "synthesizer_model": "LLM_LOCAL_SYNTHESIZER_MODEL",
                },
            }
            default_model_keys = {
                "openai_compatible": "LLM_COMPATIBLE_MODEL",
                "local": "LLM_LOCAL_MODEL",
            }
            capability_key_maps = {
                "openai_compatible": {
                    "api_key_required": (
                        "LLM_COMPATIBLE_API_KEY_REQUIRED"
                    ),
                    "supports_tools": (
                        "LLM_COMPATIBLE_SUPPORTS_TOOLS"
                    ),
                    "supports_json_mode": (
                        "LLM_COMPATIBLE_SUPPORTS_JSON_MODE"
                    ),
                    "supports_json_schema": (
                        "LLM_COMPATIBLE_SUPPORTS_JSON_SCHEMA"
                    ),
                },
                "local": {
                    "api_key_required": "LLM_LOCAL_API_KEY_REQUIRED",
                    "supports_tools": "LLM_LOCAL_SUPPORTS_TOOLS",
                    "supports_json_mode": (
                        "LLM_LOCAL_SUPPORTS_JSON_MODE"
                    ),
                    "supports_json_schema": (
                        "LLM_LOCAL_SUPPORTS_JSON_SCHEMA"
                    ),
                },
            }
            if provider_update.base_url is not None:
                env_updates[base_url_keys[provider]] = (
                    provider_update.base_url
                )
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
                    env_updates[env_key] = (
                        "true" if value else "false"
                    )

        if body.llm_provider:
            env_updates["LLM_LLM_PROVIDER"] = body.llm_provider
        if body.embedding_provider:
            env_updates["LLM_EMBEDDING_PROVIDER"] = body.embedding_provider
        if body.retrieval_top_k is not None:
            env_updates["RETRIEVAL_VECTOR_TOP_K"] = str(body.retrieval_top_k)
        if body.rerank_top_k is not None:
            env_updates["RETRIEVAL_RERANK_TOP_K"] = str(body.rerank_top_k)
        if body.max_iterations is not None:
            env_updates["AGENT_MAX_ITERATIONS"] = str(body.max_iterations)
        if body.max_refine_rounds is not None:
            env_updates["AGENT_MAX_REFINE_ROUNDS"] = str(
                body.max_refine_rounds
            )
        if body.critic_threshold is not None:
            env_updates["AGENT_CRITIC_THRESHOLD"] = str(body.critic_threshold)
        if body.subtask_timeout is not None:
            env_updates["AGENT_SUBTASK_TIMEOUT"] = str(
                body.subtask_timeout
            )
        if body.research_timeout is not None:
            env_updates["AGENT_RESEARCH_TIMEOUT"] = str(
                body.research_timeout
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
        reset_runtime_components()
        return {"status": "saved"}
    finally:
        db.close()


def reset_runtime_components() -> None:
    """Recreate configuration-bound singletons after settings updates."""
    global _orchestrator
    previous_orchestrator = _orchestrator
    _orchestrator = None
    if previous_orchestrator is not None:
        try:
            previous_orchestrator.close()
        except Exception:
            logger.exception(
                "Failed to close the previous orchestrator during reset."
            )

    from mindforge.ingestion.embedder import reset_embedder
    from mindforge.retrieval.service import reset_retrieval_service
    from mindforge.retrieval.vector_store import reset_vector_store
    from mindforge.services.indexing import reset_indexing_service

    for component_name, reset in (
        ("embedder", reset_embedder),
        ("vector store", reset_vector_store),
        ("retrieval service", reset_retrieval_service),
        ("indexing service", reset_indexing_service),
    ):
        try:
            reset()
        except Exception:
            logger.exception(
                "Failed to reset the %s after settings update.",
                component_name,
            )


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
            base_query
            .order_by(ResearchHistory.created_at.desc())
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
        )
        db.add(entry)
        try:
            db.flush()
            stale_entries = (
                db.query(ResearchHistory)
                .filter(ResearchHistory.user_id == user_id)
                .order_by(
                    ResearchHistory.created_at.desc(),
                    ResearchHistory.id.desc(),
                )
                .offset(get_settings().api.max_history_entries)
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
) -> AsyncGenerator[bytes, None]:
    """SSE streaming — with automatic fallback to retrieval-only on LLM failure."""
    use_retrieval_fallback = orch is None
    if orch is not None:
        try:
            async for event in orch.stream_run(task):
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
                        output = (
                            result.output
                            if isinstance(result, AgentResult)
                            else str(result.get("output", ""))
                        )
                        raise RuntimeError(
                            output
                            or "Agent pipeline returned an unsuccessful result."
                        )
                try:
                    payload = json.dumps(
                        _serialize_event(event),
                        ensure_ascii=False,
                    )
                except TypeError:
                    payload = json.dumps(
                        {
                            "event": "info",
                            "content": str(event)[:200],
                        },
                        ensure_ascii=False,
                    )
                yield f"data: {payload}\n\n".encode()
        except Exception as exc:
            logger.warning(
                "Agent SSE stream failed: %s; using retrieval fallback.",
                exc,
            )
            use_retrieval_fallback = True

    if use_retrieval_fallback:
        from mindforge.tools.rag_tool import RAGTool
        try:
            rag = RAGTool()
            result = await rag.execute_async(
                query=task,
                mode="hybrid",
                top_k=5,
            )
            if result.success:
                fallback = {
                    "type": "done",
                    "result": {
                        "agent_name": "orchestrator",
                        "success": True,
                        "output": result.output,
                        "data": {
                            "plan": None,
                            "subtask_outputs": [],
                            "critic_score": None,
                            "refine_rounds": 0,
                            "fallback": True,
                            "sources": list(
                                (result.data or {}).get("sources", [])
                            ),
                        },
                        "metadata": {
                            "quality": (
                                float(result.data.get("quality", 0.0))
                                if result.data
                                else 0.0
                            ),
                            "cost": 0.0,
                            "subtask_count": 0,
                            "refine_rounds": 0,
                            "model": "fallback-retrieval",
                        },
                    },
                }
            else:
                fallback = {
                    "type": "error",
                    "content": (
                        "研究失败，知识库检索回退也未成功："
                        f"{result.error or '未知错误'}"
                    ),
                }
            yield (
                f"data: {json.dumps(fallback, ensure_ascii=False)}\n\n"
            ).encode()
        except Exception:
            logger.exception("Retrieval fallback failed during SSE streaming.")
            fallback = {
                "type": "error",
                "content": "研究服务暂时不可用，请稍后重试。",
            }
            yield (
                f"data: {json.dumps(fallback, ensure_ascii=False)}\n\n"
            ).encode()
    yield b"data: [DONE]\n\n"
