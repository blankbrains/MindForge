"""Database operations for persisted document assets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mindforge.db import DocumentAsset, SessionLocal


def _serialize(record: DocumentAsset) -> dict[str, Any]:
    return {
        "asset_id": record.asset_id,
        "doc_id": record.doc_id,
        "kind": record.kind,
        "page": record.page,
        "element_index": record.element_index,
        "relative_path": record.relative_path,
        "content_type": record.content_type,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "width": record.width,
        "height": record.height,
        "metadata": dict(record.metadata_json or {}),
        "created_at": record.created_at,
    }


def replace_document_assets(
    doc_id: str,
    assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Atomically replace database rows for one document's asset manifest."""
    with SessionLocal() as db:
        db.query(DocumentAsset).filter(DocumentAsset.doc_id == doc_id).delete()
        records = [
            DocumentAsset(
                asset_id=str(asset["asset_id"]),
                doc_id=doc_id,
                kind=str(asset["kind"]),
                page=asset.get("page"),
                element_index=asset.get("element_index"),
                relative_path=asset.get("relative_path"),
                content_type=asset.get("content_type"),
                sha256=asset.get("sha256"),
                size_bytes=int(asset.get("size_bytes") or 0),
                width=asset.get("width"),
                height=asset.get("height"),
                metadata_json=dict(asset.get("metadata") or {}),
            )
            for asset in assets
        ]
        db.add_all(records)
        db.commit()
        return [_serialize(record) for record in records]


def list_document_assets(
    doc_id: str,
    *,
    include_source: bool = False,
) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        query = db.query(DocumentAsset).filter(DocumentAsset.doc_id == doc_id)
        if not include_source:
            query = query.filter(DocumentAsset.kind != "source")
        records = query.order_by(
            DocumentAsset.page.asc().nullsfirst(),
            DocumentAsset.element_index.asc().nullsfirst(),
            DocumentAsset.created_at.asc(),
        ).all()
        return [_serialize(record) for record in records]


def get_document_asset(asset_id: str) -> dict[str, Any] | None:
    with SessionLocal() as db:
        record = db.get(DocumentAsset, asset_id)
        return _serialize(record) if record is not None else None


def update_asset_caption(
    asset_id: str,
    *,
    caption: str,
    model: str,
    prompt_version: str,
) -> None:
    with SessionLocal() as db:
        record = db.get(DocumentAsset, asset_id)
        if record is None:
            return
        metadata = dict(record.metadata_json or {})
        metadata.update(
            {
                "caption": caption,
                "caption_model": model,
                "caption_prompt_version": prompt_version,
                "captioned_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        record.metadata_json = metadata
        db.commit()


def delete_document_asset_records(doc_id: str) -> None:
    with SessionLocal() as db:
        db.query(DocumentAsset).filter(DocumentAsset.doc_id == doc_id).delete()
        db.commit()
