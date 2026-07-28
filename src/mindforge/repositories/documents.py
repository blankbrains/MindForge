"""Persistent document catalog operations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func

from mindforge.db import DocumentCatalog, SessionLocal


def upsert_document(
    *,
    doc_id: str,
    filename: str,
    chunk_count: int = 0,
    status: str,
    index_signature: str | None = None,
    error: str | None = None,
    parser_metadata: dict[str, Any] | None = None,
) -> None:
    with SessionLocal() as db:
        record = db.get(DocumentCatalog, doc_id)
        if record is None:
            record = DocumentCatalog(
                doc_id=doc_id,
                filename=filename,
                chunk_count=chunk_count,
                status=status,
                index_signature=index_signature,
                error=error,
                parser_metadata=dict(parser_metadata or {}),
            )
            db.add(record)
        else:
            record.filename = filename
            record.chunk_count = chunk_count
            record.status = status
            record.index_signature = index_signature
            record.error = error
            if parser_metadata is not None:
                record.parser_metadata = dict(parser_metadata)
            record.updated_at = datetime.now(timezone.utc)
        db.commit()


def get_document(doc_id: str) -> dict[str, Any] | None:
    with SessionLocal() as db:
        record = db.get(DocumentCatalog, doc_id)
        if record is None:
            return None
        return {
            "doc_id": record.doc_id,
            "filename": record.filename,
            "chunk_count": record.chunk_count,
            "status": record.status,
            "index_signature": record.index_signature,
            "parser_metadata": dict(record.parser_metadata or {}),
        }


def delete_document(doc_id: str) -> None:
    with SessionLocal() as db:
        record = db.get(DocumentCatalog, doc_id)
        if record is not None:
            db.delete(record)
            db.commit()


def list_documents() -> list[dict[str, Any]]:
    with SessionLocal() as db:
        records = (
            db.query(DocumentCatalog)
            .order_by(
                DocumentCatalog.updated_at.desc(),
                DocumentCatalog.doc_id.asc(),
            )
            .all()
        )
        return [
            {
                "doc_id": record.doc_id,
                "filename": record.filename,
                "chunk_count": record.chunk_count,
                "status": record.status,
            }
            for record in records
        ]


def get_document_stats() -> tuple[int, int]:
    with SessionLocal() as db:
        row = (
            db.query(
                func.count(DocumentCatalog.doc_id),
                func.coalesce(func.sum(DocumentCatalog.chunk_count), 0),
            )
            .filter(DocumentCatalog.status == "indexed")
            .one()
        )
        return int(row[0]), int(row[1])


def catalog_is_empty() -> bool:
    with SessionLocal() as db:
        return db.query(DocumentCatalog.doc_id).first() is None


def bulk_upsert_indexed(documents: list[dict[str, Any]]) -> None:
    if not documents:
        return
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        for document in documents:
            doc_id = str(document["doc_id"])
            record = db.get(DocumentCatalog, doc_id)
            if record is None:
                db.add(
                    DocumentCatalog(
                        doc_id=doc_id,
                        filename=str(document["filename"]),
                        chunk_count=int(document["chunk_count"]),
                        status="indexed",
                    )
                )
            elif record.status != "indexed":
                record.filename = str(document["filename"])
                record.chunk_count = int(document["chunk_count"])
                record.status = "indexed"
                record.error = None
                record.updated_at = now
        db.commit()
