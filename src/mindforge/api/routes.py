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
from pathlib import Path
from typing import Any, AsyncGenerator
from urllib.parse import urlsplit, urlunsplit

import os as _os
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from mindforge.api.schemas import (
    DocumentContentResponse, DocumentItem, HealthResponse,
    HistoryItem, HistorySaveRequest, IndexRequest, IndexResponse,
    QueryRequest, QueryResponse,
    SettingsResponse, SettingsUpdateRequest,
)
from mindforge.agents.orchestrator import Orchestrator
from mindforge.ingestion.parsers import DocumentParser
from mindforge.ingestion.chunker import TextSplitter
from mindforge.ingestion.raptor import RAPTORIndexer
from mindforge.retrieval.vector_store import get_vector_store
from mindforge.memory.episodic import EpisodicMemory
from mindforge.memory.working import WorkingMemory
from mindforge.memory.semantic import SemanticMemory
from mindforge.config import (
    get_project_root,
    get_settings,
    resolve_project_path,
)
from mindforge import __version__

logger = logging.getLogger(__name__)
router = APIRouter()
_orchestrator: Orchestrator | None = None
_ENV_FILE_LOCK = threading.RLock()


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


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
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
            working_memory=WorkingMemory(),
            episodic_memory=EpisodicMemory(redis_client=redis_client),
            semantic_memory=SemanticMemory(storage_dir=semantic_path),
            tracer=tracer,
        )
    return _orchestrator


def get_retriever():
    from mindforge.retrieval.service import get_retriever as _get_retriever

    return _get_retriever()


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    """Submit a research task. Falls back to retrieval-only if LLM is unavailable."""
    start = time.time()
    orch = get_orchestrator()

    if body.stream:
        return StreamingResponse(
            _stream_response(orch, body.task),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Try full Agent pipeline first, fall back to retrieval-only on failure
    try:
        result = await orch.run(body.task)
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
    except Exception:
        logger.exception("Agent pipeline failed, falling back to retrieval-only.")
        # Fallback: search knowledge base directly (no LLM needed)
        try:
            from mindforge.tools.rag_tool import RAGTool
            rag = RAGTool()
            result = rag.safe_execute(query=body.task, mode="hybrid", top_k=5)
            latency = (time.time() - start) * 1000
            fallback_quality = float(result.data.get("quality", 0.0)) if result.data else 0.0
            return QueryResponse(
                task_id=uuid.uuid4().hex[:12],
                report=result.output if result.success else f"检索失败: {result.error}",
                sources=list(
                    result.data.get("sources", [])
                    if result.data
                    else []
                ),
                quality_score=fallback_quality,
                latency_ms=round(latency, 2),
                cost_usd=0.0,
                iterations=0,
            )
        except Exception:
            logger.exception("Fallback retrieval also failed.")
            logger.exception("All research paths failed")
            raise HTTPException(status_code=500, detail="Research service temporarily unavailable")


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
    doc = parser.parse(file_path)

    splitter = TextSplitter()
    chunks = splitter.split(doc.doc_id, doc.content)
    if len(chunks) > get_settings().api.max_chunks_per_document:
        raise HTTPException(
            status_code=413,
            detail=(
                "Document produced too many chunks; configured maximum is "
                f"{get_settings().api.max_chunks_per_document}."
            ),
        )

    # Embed and store in Qdrant
    from mindforge.ingestion.embedder import get_embedder
    from qdrant_client.models import PointStruct
    import hashlib as _hashlib

    embedder = get_embedder()
    store = get_vector_store()
    store.ensure_collection()

    # Batch embed all chunks at once (GPU-friendly)
    texts = [ch.content for ch in chunks]
    logger.info("嵌入 %d 个文本块...", len(texts))
    vectors = embedder.embed(texts)
    logger.info("嵌入完成，写入 Qdrant...")

    points = []
    for chunk_index, (ch, vec) in enumerate(zip(chunks, vectors)):
        points.append(PointStruct(
            id=int(_hashlib.md5(ch.chunk_id.encode()).hexdigest(), 16) % (2**63),
            vector=vec,
            payload={
                "chunk_id": ch.chunk_id,
                "doc_id": doc.doc_id,
                "content": ch.content[:2000],
                "source": doc.filename,
                "chunk_index": chunk_index,
                "chunk_start": ch.metadata.get("chunk_start"),
                "chunk_end": ch.metadata.get("chunk_end"),
                "is_summary": False,
            },
        ))

    index_batch_size = get_settings().api.index_batch_size
    for i in range(0, len(points), index_batch_size):
        batch = points[i:i + index_batch_size]
        await store.upsert(batch)

    # LLM for enrichment — skip for tiny docs
    if (body.use_raptor or body.use_graphrag) and len(chunks) <= 5:
        logger.info("Skipping RAPTOR/GraphRAG — only %d chunks.", len(chunks))
        body.use_raptor = False
        body.use_graphrag = False

    enrichment_llm = None
    if body.use_raptor or body.use_graphrag:
        try:
            from mindforge.models.base import LLMFactory

            settings = get_settings()
            enrichment_llm = LLMFactory.create(
                settings.llm.llm_provider,
                settings.llm.get_model("researcher"),
            )
        except Exception as e:
            logger.warning(f"Enrichment LLM init failed: {e}")

    # RAPTOR indexing (if requested)
    if body.use_raptor and enrichment_llm:
        try:
            raptor = RAPTORIndexer(embedder=embedder, llm=enrichment_llm)
            tree_nodes = await raptor.build_tree(chunks)
            raptor_points = []
            for node in tree_nodes:
                if node.level > 0:
                    # embedding 已在 build_tree 内生成，避免重复计算
                    vec = node.embedding or embedder.embed_single(node.content)
                    raptor_points.append(PointStruct(
                        id=int(_hashlib.md5(node.node_id.encode()).hexdigest(), 16) % (2**63),
                        vector=vec,
                        payload={
                            "chunk_id": node.node_id,
                            "doc_id": doc.doc_id,
                            "content": node.content[:2000],
                            "source": doc.filename,
                            "raptor_level": node.level,
                            "is_summary": True,
                        },
                    ))
            for i in range(0, len(raptor_points), 100):
                batch = raptor_points[i:i+100]
                await store.upsert(batch)
            logger.info(f"RAPTOR: {len(raptor_points)} summary nodes indexed")
        except Exception as e:
            logger.warning(f"RAPTOR indexing skipped: {e}")

    from mindforge.retrieval.service import index_auxiliary_documents

    auxiliary_docs = [
        {
            "id": ch.chunk_id,
            "text": ch.content,
            "doc_id": doc.doc_id,
            "chunk_id": ch.chunk_id,
            "source": doc.filename,
        }
        for ch in chunks
    ]
    await index_auxiliary_documents(
        auxiliary_docs,
        graph_llm=enrichment_llm,
        use_graphrag=bool(body.use_graphrag and enrichment_llm),
    )

    logger.info(f"Indexed {doc.filename}: {len(chunks)} chunks")
    return IndexResponse(
        doc_id=doc.doc_id,
        filename=doc.filename,
        chunk_count=len(chunks),
        status="indexed",
    )


async def _probe_qdrant_connection() -> bool:
    try:
        store = get_vector_store()
        await store.ping()
        return True
    except Exception:
        return False


async def _probe_redis_connection() -> bool:
    try:
        import redis.asyncio as aioredis

        redis_url = get_settings().cache.redis_url
        rc = aioredis.from_url(redis_url)
        try:
            await rc.ping()
            return True
        finally:
            await rc.aclose()
    except Exception:
        return False


async def _probe_postgres_connection() -> bool:
    def _probe() -> bool:
        from sqlalchemy import text
        from mindforge.db import SessionLocal

        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        return True

    try:
        return await asyncio.to_thread(_probe)
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check with real core-service connectivity."""
    qdrant_ok, redis_ok, postgres_ok = await asyncio.gather(
        _probe_qdrant_connection(),
        _probe_redis_connection(),
        _probe_postgres_connection(),
    )

    # Check MCP registry
    mcp_ok = False
    mcp_configured = False
    try:
        _reg = get_mcp_registry()
        mcp_configured = bool(_reg and _reg.servers)
        mcp_ok = _reg is not None and _reg.is_any_running
    except Exception:
        pass

    return HealthResponse(
        status=(
            "ok"
            if qdrant_ok and redis_ok and postgres_ok
            else "degraded"
        ),
        version=__version__,
        qdrant_connected=qdrant_ok,
        redis_connected=redis_ok,
        postgres_connected=postgres_ok,
        mcp_configured=mcp_configured,
        mcp_tools_available=mcp_ok,
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
)
async def readiness():
    """Strict readiness probe for deployment orchestration."""
    result = await health()
    if result.status != "ok":
        return JSONResponse(
            status_code=503,
            content=result.model_dump(),
        )
    return result


@router.get("/stats")
async def stats():
    """System statistics from Qdrant — counts unique documents, not chunks."""
    store = get_vector_store()
    document_count = 0
    chunk_count = 0
    qdrant_connected = False
    try:
        points = await store.scroll_all()
        qdrant_connected = True
        payloads = [point.payload or {} for point in points]
        doc_ids = {
            payload.get("doc_id")
            for payload in payloads
            if payload.get("doc_id")
        }
        document_count = len(doc_ids)
        chunk_count = sum(
            1 for payload in payloads if not payload.get("is_summary", False)
        )
    except Exception:
        logger.exception("Failed to load Qdrant statistics.")
    return {
        "documents_indexed": document_count,
        "chunks_indexed": chunk_count,
        "qdrant_connected": qdrant_connected,
        "qdrant_url": _public_service_url(
            get_settings().vector_store.qdrant_url
        ),
        "redis_url": _public_service_url(
            get_settings().cache.redis_url
        ),
        "max_upload_mb": get_settings().api.max_upload_mb,
    }


@router.get("/documents", response_model=list[DocumentItem])
async def list_documents():
    """List all indexed documents with metadata."""
    from collections import defaultdict
    store = get_vector_store()
    try:
        points = await store.scroll_all()
        docs: dict[str, dict] = defaultdict(
            lambda: {"doc_id": "", "filename": "", "chunk_count": 0, "status": "indexed"}
        )
        for p in points:
            pl = p.payload or {}
            if pl.get("is_summary", False):
                continue
            did = pl.get("doc_id", "unknown")
            docs[did]["doc_id"] = did
            docs[did]["filename"] = pl.get("source", docs[did]["filename"] or "")
            docs[did]["chunk_count"] += 1
        return list(docs.values())
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

        await delete_auxiliary_document(doc_id)
    except Exception:
        logger.exception("Delete failed for document %s.", doc_id)
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is temporarily unavailable.",
        )
    return None


# ------------------------------------------------------------------
# Document content
# ------------------------------------------------------------------

@router.get("/documents/{doc_id}/content", response_model=DocumentContentResponse)
async def get_document_content(doc_id: str):
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
    full_content = "\n\n".join(c["content"] for c in chunks)
    return DocumentContentResponse(
        doc_id=doc_id,
        filename=filename,
        content=full_content,
        chunk_count=len(chunks),
        chunks=chunks,
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
    upload_dir.mkdir(parents=True, exist_ok=True)
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
                f.write(block)

        parser = DocumentParser()
        parsed = parser.parse(file_path)
        splitter = TextSplitter()
        chunks = splitter.split(doc_id=parsed.doc_id, content=parsed.content)
        if len(chunks) > get_settings().api.max_chunks_per_document:
            raise HTTPException(
                status_code=413,
                detail=(
                    "文档分块数量过多（最大 "
                    f"{get_settings().api.max_chunks_per_document}）"
                ),
            )

        # RAPTOR / GraphRAG enrichment — skip for tiny docs (≤5 chunks, no value)
        if (use_raptor or use_graphrag) and len(chunks) <= 5:
            logger.info(
                "Skipping RAPTOR/GraphRAG for '%s' (%d chunks) — document too short.",
                file.filename, len(chunks),
            )
            use_raptor = False
            use_graphrag = False

        enrichment_llm = None
        if use_raptor or use_graphrag:
            try:
                from mindforge.models.base import LLMFactory

                settings = get_settings()
                enrichment_llm = LLMFactory.create(
                    settings.llm.llm_provider,
                    settings.llm.get_model("researcher"),
                )
            except Exception as e:
                logger.warning("Enrichment LLM init failed: %s", e)

        if use_raptor and enrichment_llm:
            try:
                from mindforge.ingestion.raptor import RAPTORIndexer
                from mindforge.ingestion.embedder import get_embedder
                from qdrant_client.models import PointStruct
                import hashlib as _raptor_hashlib
                _raptor_embedder = get_embedder()
                _raptor_store = get_vector_store()
                raptor = RAPTORIndexer(embedder=_raptor_embedder, llm=enrichment_llm)
                tree_nodes = await raptor.build_tree(chunks)
                raptor_points = []
                for node in tree_nodes:
                    if node.level > 0:
                        vec = node.embedding or _raptor_embedder.embed_single(node.content)
                        raptor_points.append(PointStruct(
                            id=int(_raptor_hashlib.md5(node.node_id.encode()).hexdigest(), 16) % (2**63),
                            vector=vec,
                            payload={
                                "chunk_id": node.node_id,
                                "doc_id": parsed.doc_id,
                                "content": node.content[:2000],
                                "source": file.filename or parsed.filename,
                                "raptor_level": node.level,
                                "is_summary": True,
                            },
                        ))
                for i in range(0, len(raptor_points), 100):
                    await _raptor_store.upsert(raptor_points[i:i+100])
                logger.info("RAPTOR: %d summary nodes indexed", len(raptor_points))
            except Exception as e:
                logger.warning("RAPTOR indexing skipped: %s", e)

        from mindforge.ingestion.embedder import get_embedder
        embedder = get_embedder()
        store = get_vector_store()
        store.ensure_collection()

        from qdrant_client.models import PointStruct
        import hashlib as _hl

        # Batch embed all chunks at once (GPU-friendly)
        texts = [ch.content for ch in chunks]
        logger.info("嵌入 %d 个文本块...", len(texts))
        vectors = embedder.embed(texts)
        logger.info("嵌入完成，写入 Qdrant...")

        points = []
        for chunk_index, (ch, vec) in enumerate(zip(chunks, vectors)):
            stable_id = int(_hl.md5(ch.chunk_id.encode()).hexdigest(), 16) % (2**63)
            points.append(PointStruct(
                id=stable_id,
                vector=vec,
                payload={
                    "chunk_id": ch.chunk_id,
                    "doc_id": parsed.doc_id,
                    "content": ch.content[:2000],
                    "source": file.filename or parsed.filename,
                    "chunk_index": chunk_index,
                    "chunk_start": ch.metadata.get("chunk_start"),
                    "chunk_end": ch.metadata.get("chunk_end"),
                    "is_summary": False,
                },
            ))
        index_batch_size = get_settings().api.index_batch_size
        for i in range(0, len(points), index_batch_size):
            await store.upsert(points[i:i + index_batch_size])

        from mindforge.retrieval.service import index_auxiliary_documents

        auxiliary_docs = [
            {
                "id": ch.chunk_id,
                "text": ch.content,
                "doc_id": parsed.doc_id,
                "chunk_id": ch.chunk_id,
                "source": file.filename or parsed.filename,
            }
            for ch in chunks
        ]
        await index_auxiliary_documents(
            auxiliary_docs,
            graph_llm=enrichment_llm,
            use_graphrag=bool(use_graphrag and enrichment_llm),
        )

        return IndexResponse(
            doc_id=parsed.doc_id,
            filename=file.filename or parsed.filename,
            chunk_count=len(chunks),
            status="indexed",
        )
    except Exception:
        if parsed is not None:
            try:
                await get_vector_store().delete(parsed.doc_id)
                from mindforge.retrieval.service import (
                    delete_auxiliary_document,
                )

                await delete_auxiliary_document(parsed.doc_id)
            except Exception:
                logger.exception(
                    "Failed to roll back partial index for document %s.",
                    parsed.doc_id,
                )
        raise
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

        return SettingsResponse(
            llm_provider=s.llm.llm_provider,
            deepseek_api_key=_masked("deepseek", keys, s.llm.deepseek_api_key),
            openai_api_key=_masked("openai", keys, s.llm.openai_api_key),
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
    """Save user settings (API keys encrypted in DB, synced to .env)."""
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
        }
        for provider, key_val in [
            ("deepseek", body.deepseek_api_key),
            ("openai", body.openai_api_key),
        ]:
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
    _orchestrator = None

    from mindforge.ingestion.embedder import reset_embedder
    from mindforge.retrieval.service import reset_retrieval_service
    from mindforge.retrieval.vector_store import reset_vector_store

    reset_embedder()
    reset_vector_store()
    reset_retrieval_service()


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
                    created_at=e.created_at.isoformat() if e.created_at else None,
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
            created_at=(
                entry.created_at.isoformat()
                if entry.created_at
                else None
            ),
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


@router.post("/mcp")
async def mcp_endpoint(request: dict):
    """MCP JSON-RPC endpoint — exposes MindForce tools via MCP over HTTP.

    Accepts standard MCP JSON-RPC messages (initialize, tools/list, tools/call)
    and delegates to MindForgeMCPServer. Enables external MCP clients to
    call Agent capabilities over HTTP.
    """
    try:
        from mindforge.mcp.server import MindForgeMCPServer
        mcp_server = MindForgeMCPServer()
        result = await mcp_server.handle_request(request)
        return result
    except Exception:
        logger.exception("MCP endpoint error")
        raise HTTPException(status_code=500, detail="MCP service error")


@router.get("/mcp")
async def mcp_info():
    """Return MCP endpoint metadata."""
    return {
        "protocol": "Model Context Protocol",
        "version": "2025-03-26",
        "endpoint": "/api/v1/mcp",
        "transport": "HTTP POST (JSON-RPC)",
        "tools": [
            {"name": "search_knowledge_base", "description": "Search the knowledge base"},
            {"name": "run_research_task", "description": "Run a multi-step research task"},
            {"name": "verify_citation", "description": "Verify citation markers"},
            {"name": "system_status", "description": "Get MindForge system status"},
        ],
    }


# ---------------------------------------------------------------------------
# Module-level MCP registry for preloading (set by server startup)
# ---------------------------------------------------------------------------

_mcp_registry: Any = None


def get_mcp_registry() -> Any:
    """Get the preloaded MCP registry singleton."""
    global _mcp_registry
    return _mcp_registry


def set_mcp_registry(registry: Any) -> None:
    """Set the preloaded MCP registry (called at startup)."""
    global _mcp_registry
    _mcp_registry = registry


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


async def _stream_response(orch: Orchestrator, task: str) -> AsyncGenerator[bytes, None]:
    """SSE streaming — with automatic fallback to retrieval-only on LLM failure."""
    try:
        async for event in orch.stream_run(task):
            try:
                payload = json.dumps(_serialize_event(event), ensure_ascii=False)
            except TypeError:
                payload = json.dumps({"event": "info", "content": str(event)[:200]}, ensure_ascii=False)
            yield f"data: {payload}\n\n".encode("utf-8")
    except Exception as exc:
        logger.warning("Agent SSE stream failed: %s — falling back to retrieval-only", exc)
        from mindforge.tools.rag_tool import RAGTool
        try:
            rag = RAGTool()
            result = rag.safe_execute(query=task, mode="hybrid", top_k=5)
            fallback = {
                "type": "done",
                "result": {
                    "agent_name": "orchestrator",
                    "success": True,
                    "output": result.output if result.success else f"检索失败: {result.error}",
                    "data": {
                        "plan": None,
                        "subtask_outputs": [],
                        "critic_score": None,
                        "refine_rounds": 0,
                        "fallback": True,
                    },
                    "metadata": {
                        "quality": float(result.data.get("quality", 0.0)) if result.data else 0.0,
                        "cost": 0.0,
                        "subtask_count": 0,
                        "refine_rounds": 0,
                        "model": "fallback-retrieval",
                    },
                },
            }
            yield f"data: {json.dumps(fallback, ensure_ascii=False)}\n\n".encode("utf-8")
        except Exception:
            yield f"data: {json.dumps({'type': 'error', 'content': f'研究失败: {exc}'}, ensure_ascii=False)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"
