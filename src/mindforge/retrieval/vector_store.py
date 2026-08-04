"""Qdrant vector-store adapter for the configured 1.18.x deployment."""

from __future__ import annotations
import asyncio
from typing import List, Optional, Dict
import logging

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    FilterSelector,
)
from mindforge.config import get_settings

logger = logging.getLogger(__name__)


class QdrantStore:
    def __init__(self):
        cfg = get_settings().vector_store
        client_kwargs = {
            "url": cfg.qdrant_url,
            "timeout": 30,
        }
        if cfg.qdrant_api_key:
            client_kwargs["api_key"] = cfg.qdrant_api_key
        self._sync_client = QdrantClient(**client_kwargs)
        self._async_client = AsyncQdrantClient(**client_kwargs)
        self.collection_name = cfg.collection_name
        self.embedding_dim = cfg.embedding_dim
        self.max_scroll_records = cfg.max_scroll_records

    def ensure_collection(self):
        collections = [c.name for c in self._sync_client.get_collections().collections]
        if self.collection_name not in collections:
            self._sync_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim, distance=Distance.COSINE
                ),
            )
            logger.info(f"Collection created: {self.collection_name}")
            return

        info = self._sync_client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        existing_dim = getattr(vectors, "size", None)
        if existing_dim is not None and existing_dim != self.embedding_dim:
            raise ValueError(
                f"Qdrant collection '{self.collection_name}' has vector "
                f"dimension {existing_dim}, but configuration requires "
                f"{self.embedding_dim}. Set VECTOR_EMBEDDING_DIM to the "
                "existing collection dimension or reindex the collection."
            )

    async def upsert(self, points: List[PointStruct]):
        result = await self._async_client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return result

    async def search(
        self,
        vector: List[float],
        top_k: int = 20,
        filters: Optional[Dict] = None,
        excluded_doc_ids: set[str] | None = None,
    ) -> List[tuple[Dict, float]]:
        """Vector search using query_points (qdrant-client >= 1.18)."""
        results = await self._async_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=self._build_filter(
                filters,
                excluded_doc_ids=excluded_doc_ids,
            ),
            limit=top_k,
            with_payload=True,
        )
        return [(r.payload, r.score) for r in results.points]

    def _build_filter(
        self,
        filters: Optional[Dict],
        *,
        excluded_doc_ids: set[str] | None = None,
    ) -> Optional[Filter]:
        if not filters and not excluded_doc_ids:
            return None
        conditions = []
        for key, value in (filters or {}).items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        excluded_conditions = []
        if excluded_doc_ids:
            excluded_conditions.append(
                FieldCondition(
                    key="doc_id",
                    match=MatchAny(any=sorted(excluded_doc_ids)),
                )
            )
        if not conditions and not excluded_conditions:
            return None
        return Filter(
            must=conditions or None,
            must_not=excluded_conditions or None,
        )

    async def delete(self, doc_id: str):
        await self._async_client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def delete_points(self, point_ids: list[int | str]) -> None:
        """Delete explicit point ids without touching other document generations."""
        if not point_ids:
            return
        await self._async_client.delete(
            collection_name=self.collection_name,
            points_selector=point_ids,
            wait=True,
        )

    async def snapshot_document(self, doc_id: str) -> list[PointStruct]:
        """Capture one document's points so a failed rebuild can restore them."""
        records = await self.scroll_all(
            filters={"doc_id": doc_id},
            with_vectors=True,
        )
        return [
            PointStruct(
                id=record.id,
                vector=record.vector,
                payload=dict(record.payload or {}),
            )
            for record in records
        ]

    async def restore_document(
        self,
        doc_id: str,
        points: list[PointStruct],
        *,
        batch_size: int,
    ) -> None:
        """Replace a partial rebuild with a previously captured snapshot."""
        await self.delete(doc_id)
        for index in range(0, len(points), batch_size):
            await self.upsert(points[index:index + batch_size])

    async def count(self, filters: Optional[Dict] = None) -> int:
        result = await self._async_client.count(
            collection_name=self.collection_name,
            count_filter=self._build_filter(filters),
            exact=True,
        )
        return int(result.count)

    async def ping(self) -> None:
        """Verify that the Qdrant service is reachable."""
        await self._async_client.get_collections()

    def get_point_count(self) -> int:
        """Return the current collection size without scanning payloads."""
        collections = {
            collection.name
            for collection in self._sync_client.get_collections().collections
        }
        if self.collection_name not in collections:
            return 0
        info = self._sync_client.get_collection(self.collection_name)
        return int(info.points_count or 0)

    def close(self) -> None:
        """Close both Qdrant client connection pools."""
        self._sync_client.close()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._async_client.close())
        else:
            loop.create_task(self._async_client.close())

    async def scroll_all(
        self,
        filters: Optional[Dict] = None,
        *,
        with_vectors: bool = False,
        payload_fields: list[str] | None = None,
        page_size: int = 256,
        max_records: int | None = None,
    ) -> list:
        """Return all matching records using Qdrant offset pagination."""
        limit_records = (
            self.max_scroll_records
            if max_records is None
            else min(max_records, self.max_scroll_records)
        )
        if limit_records < 1:
            raise ValueError("max_records must be positive.")
        page_size = min(max(1, page_size), 1000, limit_records)
        records: list = []
        offset = None
        while True:
            request_limit = min(page_size, limit_records - len(records))
            page, offset = await self._async_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._build_filter(filters),
                limit=request_limit,
                offset=offset,
                with_payload=(payload_fields if payload_fields is not None else True),
                with_vectors=with_vectors,
            )
            records.extend(page)
            if offset is None:
                return records
            if len(records) >= limit_records:
                raise RuntimeError("Qdrant scan exceeded the configured record limit.")


_store: Optional[QdrantStore] = None


def get_vector_store() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore()
    return _store


def reset_vector_store() -> None:
    """Drop cached Qdrant clients after connection/configuration changes."""
    global _store
    previous = _store
    _store = None
    if previous is not None:
        previous.close()
