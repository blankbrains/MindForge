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

        embedder = get_embedder()
        async def _async_embed(text: str):
            return embedder.embed_single(text)
        _embed_fn = _async_embed
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
            sources=[],
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
                sources=[],
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

    file_path = body.file_path or body.file_url or ""
    parser = DocumentParser()
    doc = parser.parse(file_path)

    splitter = TextSplitter()
    chunks = splitter.split(doc.doc_id, doc.content)

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
    for ch, vec in zip(chunks, vectors):
        points.append(PointStruct(
            id=int(_hashlib.md5(ch.chunk_id.encode()).hexdigest(), 16) % (2**63),
            vector=vec,
            payload={
                "chunk_id": ch.chunk_id,
                "doc_id": doc.doc_id,
                "content": ch.content[:2000],
                "source": doc.filename,
            },
        ))

    for i in range(0, len(points), 500):
        batch = points[i:i+500]
        await store.upsert(batch)

    # LLM for enrichment — skip for tiny docs
    if (body.use_raptor or body.use_graphrag) and len(chunks) <= 5:
        logger.info("Skipping RAPTOR/GraphRAG — only %d chunks.", len(chunks))
        body.use_raptor = False
        body.use_graphrag = False

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

    # GraphRAG indexing (if requested)
    if body.use_graphrag and enrichment_llm:
        try:
            from mindforge.retrieval.graphrag import GraphRAGEngine
            graphrag = GraphRAGEngine(llm_fn=enrichment_llm)
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
        logger.exception(f"Delete failed for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
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
    # Sanitize filename — prevent path traversal
    import re as _re
    safe_name = file.filename or "uploaded_doc"
    safe_name = _re.sub(r'[\\/]', '_', safe_name)  # strip path separators
    safe_name = _re.sub(r'\.\.+', '', safe_name)     # strip double dots

    # Size limit: 200 MB
    MAX_UPLOAD_BYTES = 200 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件过大（最大 200MB）")

    upload_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "data")
    _os.makedirs(upload_dir, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    file_path = _os.path.join(upload_dir, unique_name)
    try:
        with open(file_path, "wb") as f:
            f.write(content)

        parser = DocumentParser()
        parsed = parser.parse(file_path)
        splitter = TextSplitter()
        chunks = splitter.split(doc_id=parsed.doc_id, content=parsed.content)

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
                from mindforge.models.deepseek_adapter import DeepSeekAdapter
                from mindforge.config import get_settings
                settings = get_settings()
                enrichment_llm = DeepSeekAdapter(
                    model=settings.llm.get_model("researcher"),
                    api_key=settings.llm.deepseek_api_key,
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

        if use_graphrag and enrichment_llm:
            try:
                from mindforge.retrieval.graphrag import GraphRAGEngine
                graphrag = GraphRAGEngine(llm_fn=enrichment_llm)
                graph_docs = [{"doc_id": parsed.doc_id, "content": ch.content, "source": file.filename or parsed.filename} for ch in chunks]
                await graphrag.build_graph(graph_docs)
                logger.info("GraphRAG: built graph from %d chunks", len(graph_docs))
            except Exception as e:
                logger.warning("GraphRAG indexing skipped: %s", e)

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
        for ch, vec in zip(chunks, vectors):
            stable_id = int(_hl.md5(ch.chunk_id.encode()).hexdigest(), 16) % (2**63)
            points.append(PointStruct(
                id=stable_id,
                vector=vec,
                payload={
                    "chunk_id": ch.chunk_id,
                    "doc_id": parsed.doc_id,
                    "content": ch.content[:2000],
                    "source": file.filename or parsed.filename,
                },
            ))
        for i in range(0, len(points), 500):
            await store.upsert(points[i:i+500])

        return IndexResponse(
            doc_id=parsed.doc_id,
            filename=file.filename or parsed.filename,
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
    from mindforge.db import SessionLocal, ApiKey
    from mindforge.config import get_settings
    db = SessionLocal()
    try:
        keys = {k.provider: k for k in db.query(ApiKey).filter(ApiKey.is_active).all()}
        s = get_settings()

        def _masked(provider: str, db_keys: dict, settings_key: str) -> str:
            if provider in db_keys:
                return "***" + db_keys[provider].key_encrypted[-4:]
            if settings_key:
                return "***" + settings_key[-4:]
            return ""

        return SettingsResponse(
            llm_provider=s.llm.llm_provider,
            deepseek_api_key=_masked("deepseek", keys, s.llm.deepseek_api_key),
            openai_api_key=_masked("openai", keys, s.llm.openai_api_key),
            embedding_provider=_os.getenv("LLM_EMBEDDING_PROVIDER", "openai"),
        )
    finally:
        db.close()


def _sync_env_file(updates: dict[str, str]) -> None:
    """同步写入 .env 文件，保证 key 在服务器重启后仍然生效。"""
    _env_path = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "..", "..", ".env")
    )
    if not _os.path.exists(_env_path):
        return
    try:
        with open(_env_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception:
        return
    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        # 跳过注释和空行
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        matched = False
        for key, value in updates.items():
            if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} "):
                new_lines.append(f"{key}={value}\n")
                updated_keys.add(key)
                matched = True
                break
        if not matched:
            new_lines.append(line)
    # 追加未在 .env 中出现的 key
    for key, value in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}\n")
    try:
        with open(_env_path, "w", encoding="utf-8") as fh:
            fh.writelines(new_lines)
    except Exception:
        pass


@router.put("/settings")
def update_settings_api(body: SettingsUpdateRequest):
    """Save user settings (API keys encrypted in DB, synced to .env)."""
    from mindforge.db import SessionLocal, ApiKey, encrypt_api_key
    db = SessionLocal()
    try:
        db.query(ApiKey).first()  # ensure table exists for single-user mode
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
                # 保存新 key → DB + os.environ + .env
                if existing:
                    existing.key_encrypted = encrypt_api_key(key_val)
                else:
                    db.add(ApiKey(provider=provider, key_encrypted=encrypt_api_key(key_val), user_id=1))
                _os.environ[_env_key_map[provider]] = key_val
                env_updates[_env_key_map[provider]] = key_val
            else:
                # key_val 为空 → 删除 key: DB + os.environ + .env
                if existing:
                    db.delete(existing)
                _os.environ.pop(_env_key_map[provider], None)
                env_updates[_env_key_map[provider]] = ""
        # Update env for current session — LLM
        if body.llm_provider:
            _os.environ["LLM_LLM_PROVIDER"] = body.llm_provider
            env_updates["LLM_LLM_PROVIDER"] = body.llm_provider
        if body.embedding_provider:
            _os.environ["LLM_EMBEDDING_PROVIDER"] = body.embedding_provider
            env_updates["LLM_EMBEDDING_PROVIDER"] = body.embedding_provider
        # — Retrieval config
        if body.retrieval_top_k is not None:
            _os.environ["RETRIEVAL_VECTOR_TOP_K"] = str(body.retrieval_top_k)
            env_updates["RETRIEVAL_VECTOR_TOP_K"] = str(body.retrieval_top_k)
        if body.rerank_top_k is not None:
            _os.environ["RETRIEVAL_RERANK_TOP_K"] = str(body.rerank_top_k)
            env_updates["RETRIEVAL_RERANK_TOP_K"] = str(body.rerank_top_k)
        # — Agent config
        if body.max_iterations is not None:
            _os.environ["AGENT_MAX_ITERATIONS"] = str(body.max_iterations)
            env_updates["AGENT_MAX_ITERATIONS"] = str(body.max_iterations)
        if body.critic_threshold is not None:
            _os.environ["AGENT_CRITIC_THRESHOLD"] = str(body.critic_threshold)
            env_updates["AGENT_CRITIC_THRESHOLD"] = str(body.critic_threshold)
        # 刷新缓存的 Settings 实例
        from mindforge.config import reload_settings
        reload_settings()
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise
        # 同步写入 .env 文件（在 DB 提交成功后，.env 失败不影响 DB 数据）
        if env_updates:
            try:
                _sync_env_file(env_updates)
            except Exception as e:
                logger.error("Settings saved to DB but .env sync failed: %s", e)
        return {"status": "saved"}
    finally:
        db.close()


# ------------------------------------------------------------------
# History
# ------------------------------------------------------------------

@router.get("/history")
def list_history(page: int = 1, page_size: int = 20):
    """Return paginated research history entries."""
    from mindforge.db import SessionLocal, ResearchHistory
    db = SessionLocal()
    try:
        offset = max(0, (page - 1)) * page_size
        total = db.query(ResearchHistory).count()
        entries = (
            db.query(ResearchHistory)
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


@router.post("/history")
def save_history(body: HistorySaveRequest):
    """Save a research result to history."""
    from mindforge.db import SessionLocal, ResearchHistory
    import json as _json
    db = SessionLocal()
    try:
        entry = ResearchHistory(
            user_id=1,  # single-user
            task=body.task,
            report=body.report,
            quality_score=body.quality_score,
            model_used=body.model_used,
            token_usage=_json.dumps(body.token_usage),
        )
        db.add(entry)
        try:
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
    from mindforge.db import SessionLocal, ResearchHistory
    db = SessionLocal()
    try:
        entry = db.query(ResearchHistory).filter(ResearchHistory.id == entry_id).first()
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
    from mindforge.db import SessionLocal, ResearchHistory
    db = SessionLocal()
    try:
        db.query(ResearchHistory).delete()
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
