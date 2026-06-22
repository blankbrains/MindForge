"""Qdrant 向量数据库封装 — 兼容 v1.8.x"""
from __future__ import annotations
from typing import List, Optional, Dict
import logging

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter,
    FieldCondition, MatchValue,
)
from mindforge.config import get_settings

logger = logging.getLogger(__name__)


class QdrantStore:
    def __init__(self):
        cfg = get_settings().vector_store
        self._sync_client = QdrantClient(url=cfg.qdrant_url, timeout=30)
        self._async_client = AsyncQdrantClient(url=cfg.qdrant_url, timeout=30)
        self.collection_name = cfg.collection_name
        self.embedding_dim = cfg.embedding_dim

    def ensure_collection(self):
        collections = [c.name for c in self._sync_client.get_collections().collections]
        if self.collection_name not in collections:
            self._sync_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )
            logger.info(f"Collection created: {self.collection_name}")

    async def upsert(self, points: List[PointStruct]):
        result = await self._async_client.upsert(
            collection_name=self.collection_name, points=points, wait=True,
        )
        return result

    async def search(self, vector: List[float], top_k: int = 20) -> List[tuple[Dict, float]]:
        results = await self._async_client.search(
            collection_name=self.collection_name,
            query_vector=vector, limit=top_k, with_payload=True,
        )
        return [(r.payload, r.score) for r in results]

    def _build_filter(self, filters: Optional[Dict]) -> Optional[Filter]:
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    async def delete(self, doc_id: str):
        await self._async_client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        )

    async def get_stats(self) -> Dict:
        info = await self._async_client.get_collection(self.collection_name)
        return {"name": self.collection_name, "points": info.points_count, "status": info.status}


_store: Optional[QdrantStore] = None


def get_vector_store() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore()
    return _store
