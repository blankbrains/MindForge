"""API route definitions for MindForge — 真实实现"""

from __future__ import annotations
import json
import time
import uuid
import logging
from typing import Any, AsyncGenerator

import os as _os
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from mindforge.api.schemas import (
    DocumentContentResponse, DocumentItem, HealthResponse,
    HistoryItem, IndexRequest, IndexResponse,
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
from mindforge.retrieval.adaptive import AdaptiveRetriever
from mindforge.retrieval.hybrid import HybridRetriever
from mindforge.retrieval.reranker import CrossEncoderReranker
from mindforge.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
_orchestrator: Orchestrator | None = None
_adaptive_retriever: AdaptiveRetriever | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(
            working_memory=WorkingMemory(),
            episodic_memory=EpisodicMemory(),
            semantic_memory=SemanticMemory(),
        )
    return _orchestrator


def get_retriever() -> AdaptiveRetriever:
    global _adaptive_retriever
    if _adaptive_retriever is None:
        from mindforge.retrieval.vector_store import get_vector_store
        from mindforge.ingestion.embedder import get_embedder

        _embed_fn = get_embedder().embed_single
        _store = get_vector_store()

        _adaptive_retriever = AdaptiveRetriever(
            hybrid_retriever=HybridRetriever(
                vector_store=_store,
                embedding_fn=_embed_fn,
            ),
            reranker=CrossEncoderReranker(),
        )
    return _adaptive_retriever


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    """Submit a research task to the Multi-Agent system."""
    start = time.time()
    orch = get_orchestrator()

    if body.stream:
        return StreamingResponse(
            _stream_response(orch, body.task),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await orch.run(body.task)
    latency = (time.time() - start) * 1000

    return QueryResponse(
        task_id=uuid.uuid4().hex[:12],
        report=result.output,
        sources=[],
        quality_score=float(result.metadata.get("quality", 0)),
        latency_ms=round(latency, 2),
        cost_usd=round(float(result.metadata.get("cost", 0)), 6),
        iterations=int(result.metadata.get("subtask_count", 0)),
    )


@router.post("/index", response_model=IndexResponse)
async def index_document(body: IndexRequest):
    """Ingest a document into the Qdrant knowledge base."""
    if not body.file_url and not body.file_path:
        raise HTTPException(status_code=422, detail="file_url or file_path required")

    file_path = body.file_path or body.file_url or ""
    parser = DocumentParser()
    doc = parser.parse(file_path)

    splitter = TextSplitter()
    chunks = splitter.split(doc.doc_id, doc.content)

    # Embed and store in Qdrant
    from mindforge.ingestion.embedder import get_embedder
    from qdrant_client.models import PointStruct

    embedder = get_embedder()
    store = get_vector_store()
    points = []
    for ch in chunks:
        vec = embedder.embed_single(ch.content)
        points.append(PointStruct(
            id=abs(hash(ch.chunk_id)) % (2**63),
            vector=vec,
            payload={
                "chunk_id": ch.chunk_id,
                "doc_id": doc.doc_id,
                "content": ch.content[:2000],
                "source": doc.filename,
            },
        ))

    for i in range(0, len(points), 100):
        batch = points[i:i+100]
        await store.upsert(batch)

    # LLM for enrichment (shared by RAPTOR and GraphRAG)
    enrichment_llm = None
    if body.use_raptor or body.use_graphrag:
        try:
            from mindforge.models.deepseek_adapter import DeepSeekAdapter
            from mindforge.config import get_settings
            settings = get_settings()
            enrichment_llm = DeepSeekAdapter(
                model=settings.llm.get_model("researcher"),
                api_key=settings.llm.deepseek_api_key,
            )
        except Exception as e:
            logger.warning(f"Enrichment LLM init failed: {e}")

    # RAPTOR indexing (if requested)
    if body.use_raptor and enrichment_llm:
        try:
            raptor = RAPTORIndexer(embedder=embedder, llm=enrichment_llm)
            tree_nodes = raptor.build_tree(chunks)
            raptor_points = []
            for node in tree_nodes:
                if node.level > 0:
                    vec = embedder.embed_single(node.content)
                    raptor_points.append(PointStruct(
                        id=abs(hash(node.node_id)) % (2**63),
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

    # GraphRAG indexing (if requested)
    if body.use_graphrag and enrichment_llm:
        try:
            from mindforge.retrieval.graphrag import GraphRAGEngine
            graphrag = GraphRAGEngine(llm_fn=enrichment_llm, embedding_fn=embedder.embed_single)
            graph_docs = [{"doc_id": doc.doc_id, "content": ch.content, "source": doc.filename} for ch in chunks]
            await graphrag.build_graph(graph_docs)
            logger.info(f"GraphRAG: built graph from {len(graph_docs)} chunks")
        except Exception as e:
            logger.warning(f"GraphRAG indexing skipped: {e}")

    logger.info(f"Indexed {doc.filename}: {len(chunks)} chunks")
    return IndexResponse(
        doc_id=doc.doc_id,
        filename=doc.filename,
        chunk_count=len(chunks),
        status="indexed",
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check with real Qdrant/Redis connectivity."""
    qdrant_ok = redis_ok = False
    try:
        store = get_vector_store()
        stats = await store.get_stats()
        qdrant_ok = "points" in stats
    except Exception:
        pass
    try:
        import redis.asyncio as aioredis
        redis_url = get_settings().cache.redis_url or "redis://localhost:6380"
        rc = aioredis.from_url(redis_url)
        await rc.ping()
        redis_ok = True
        await rc.close()
    except Exception:
        pass
    # Check MCP registry
    mcp_ok = False
    try:
        _reg = get_mcp_registry()
        mcp_ok = _reg is not None and _reg.is_any_running
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version="0.1.0",
        qdrant_connected=qdrant_ok,
        redis_connected=redis_ok,
        mcp_tools_available=mcp_ok,
    )


@router.get("/stats")
async def stats():
    """System statistics from Qdrant — counts unique documents, not chunks."""
    store = get_vector_store()
    count = 0
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=get_settings().vector_store.qdrant_url, timeout=5)
        points, _ = client.scroll(
            collection_name=store.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        doc_ids = {p.payload.get("doc_id") for p in points if p.payload}
        count = len(doc_ids)
    except Exception:
        pass
    return {
        "documents_indexed": count,
        "qdrant_url": get_settings().vector_store.qdrant_url,
        "redis_url": get_settings().cache.redis_url,
    }


@router.get("/documents", response_model=list[DocumentItem])
async def list_documents():
    """List all indexed documents with metadata."""
    from qdrant_client import QdrantClient
    from collections import defaultdict
    store = get_vector_store()
    try:
        client = QdrantClient(url=get_settings().vector_store.qdrant_url, timeout=5)
        points, _ = client.scroll(
            collection_name=store.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        docs: dict[str, dict] = defaultdict(
            lambda: {"doc_id": "", "filename": "", "chunk_count": 0, "status": "indexed"}
        )
        for p in points:
            pl = p.payload or {}
            did = pl.get("doc_id", "unknown")
            docs[did]["doc_id"] = did
            docs[did]["filename"] = pl.get("source", docs[did]["filename"] or "")
            docs[did]["chunk_count"] += 1
        return list(docs.values())
    except Exception:
        logger.exception("Failed to list documents.")
        return []


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str):
    """Delete a document from Qdrant."""
    store = get_vector_store()
    try:
        await store.delete(doc_id)
    except Exception as e:
        logger.warning(f"Delete failed for {doc_id}: {e}")
    return None


# ------------------------------------------------------------------
# Document content
# ------------------------------------------------------------------

@router.get("/documents/{doc_id}/content", response_model=DocumentContentResponse)
async def get_document_content(doc_id: str):
    """Return full content of an indexed document (all chunks)."""
    from qdrant_client import QdrantClient
    store = get_vector_store()
    client = QdrantClient(url=get_settings().vector_store.qdrant_url, timeout=5)
    points, _ = client.scroll(
        collection_name=store.collection_name,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    chunks = []
    filename = ""
    for p in points:
        pl = p.payload or {}
        if pl.get("doc_id") == doc_id:
            chunks.append({
                "chunk_id": pl.get("chunk_id", ""),
                "content": pl.get("content", ""),
                "raptor_level": pl.get("raptor_level", 0),
            })
            if not filename:
                filename = pl.get("source", "")
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found")
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
    # Save uploaded file to data/ directory
    upload_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "data")
    _os.makedirs(upload_dir, exist_ok=True)
    file_path = _os.path.join(upload_dir, file.filename or "uploaded_doc")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Parse and index
    parser = DocumentParser()
    parsed = parser.parse(file_path)
    splitter = TextSplitter()
    chunks = splitter.split(doc_id=parsed.doc_id, content=parsed.content)

    from mindforge.ingestion.embedder import get_embedder
    embedder = get_embedder()
    store = get_vector_store()
    store.ensure_collection()

    from qdrant_client.models import PointStruct
    points = []
    for ch in chunks:
        vec = embedder.embed_single(ch.content)
        points.append(PointStruct(
            id=abs(hash(ch.chunk_id)) % (2**63),
            vector=vec,
            payload={
                "chunk_id": ch.chunk_id,
                "doc_id": parsed.doc_id,
                "content": ch.content[:2000],
                "source": file.filename or parsed.filename,
            },
        ))
    for i in range(0, len(points), 100):
        await store.upsert(points[i:i+100])

    return IndexResponse(
        doc_id=parsed.doc_id,
        filename=file.filename or parsed.filename,
        chunk_count=len(chunks),
        status="indexed",
    )


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

@router.get("/settings", response_model=SettingsResponse)
def get_settings_api():
    """Return current user settings (API keys masked)."""
    from mindforge.db import SessionLocal, ApiKey
    db = SessionLocal()
    try:
        keys = {k.provider: k for k in db.query(ApiKey).filter(ApiKey.is_active).all()}
        return SettingsResponse(
            llm_provider=get_settings().llm.llm_provider,
            deepseek_api_key="***" + keys["deepseek"].key_encrypted[-4:] if "deepseek" in keys else "",
            openai_api_key="***" + keys["openai"].key_encrypted[-4:] if "openai" in keys else "",
            embedding_provider=_os.getenv("LLM_EMBEDDING_PROVIDER", "openai"),
        )
    finally:
        db.close()


@router.put("/settings")
def update_settings_api(body: SettingsUpdateRequest):
    """Save user settings (API keys encrypted in DB)."""
    from mindforge.db import SessionLocal, ApiKey, encrypt_api_key
    db = SessionLocal()
    try:
        db.query(ApiKey).first()  # ensure table exists for single-user mode
        for provider, key_val in [
            ("deepseek", body.deepseek_api_key),
            ("openai", body.openai_api_key),
        ]:
            if key_val:
                existing = db.query(ApiKey).filter(
                    ApiKey.provider == provider
                ).first()
                if existing:
                    existing.key_encrypted = encrypt_api_key(key_val)
                else:
                    db.add(ApiKey(provider=provider, key_encrypted=encrypt_api_key(key_val), user_id=1))
        # Update env for current session
        if body.llm_provider:
            _os.environ["LLM_LLM_PROVIDER"] = body.llm_provider
        if body.embedding_provider:
            _os.environ["LLM_EMBEDDING_PROVIDER"] = body.embedding_provider
        db.commit()
        return {"status": "saved"}
    finally:
        db.close()


# ------------------------------------------------------------------
# History
# ------------------------------------------------------------------

@router.get("/history")
def list_history():
    """Return research history entries."""
    from mindforge.db import SessionLocal, ResearchHistory
    db = SessionLocal()
    try:
        entries = (
            db.query(ResearchHistory)
            .order_by(ResearchHistory.created_at.desc())
            .limit(50)
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
            "total": len(entries),
        }
    finally:
        db.close()


@router.post("/history")
def save_history(body: dict):
    """Save a research result to history."""
    from mindforge.db import SessionLocal, ResearchHistory
    import json as _json
    db = SessionLocal()
    try:
        entry = ResearchHistory(
            user_id=1,  # single-user
            task=body.get("task", ""),
            report=body.get("report", ""),
            quality_score=body.get("quality_score"),
            model_used=body.get("model_used"),
            token_usage=_json.dumps(body.get("token_usage", {})),
        )
        db.add(entry)
        db.commit()
        return {"id": entry.id, "status": "saved"}
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


async def _stream_response(orch: Orchestrator, task: str) -> AsyncGenerator[bytes, None]:
    """SSE streaming from real orchestrator."""
    async for event in orch.stream_run(task):
        try:
            payload = json.dumps(event, ensure_ascii=False)
        except TypeError:
            payload = json.dumps({"event": "info", "content": str(event)[:200]}, ensure_ascii=False)
        yield f"data: {payload}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"
