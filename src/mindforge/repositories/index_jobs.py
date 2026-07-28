"""Persistent indexing-job operations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mindforge.db import IndexJob, SessionLocal

ACTIVE_JOB_STATUSES = {"queued", "running"}
TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}
_UPDATABLE_FIELDS = {
    "doc_id",
    "status",
    "stage",
    "progress",
    "chunk_count",
    "timings",
    "metrics",
    "error",
    "cancel_requested",
}


def _serialize(record: IndexJob) -> dict[str, Any]:
    return {
        "job_id": record.job_id,
        "doc_id": record.doc_id,
        "filename": record.filename,
        "file_path": record.file_path,
        "status": record.status,
        "stage": record.stage,
        "progress": float(record.progress),
        "chunk_count": record.chunk_count,
        "timings": dict(record.timings or {}),
        "metrics": dict(record.metrics or {}),
        "error": record.error,
        "cancel_requested": record.cancel_requested,
        "strategy": record.strategy,
        "use_raptor": record.use_raptor,
        "use_graphrag": record.use_graphrag,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def create_index_job(
    *,
    job_id: str,
    filename: str,
    file_path: str,
    strategy: str,
    use_raptor: bool,
    use_graphrag: bool,
) -> dict[str, Any]:
    with SessionLocal() as db:
        record = IndexJob(
            job_id=job_id,
            filename=filename,
            file_path=file_path,
            strategy=strategy,
            use_raptor=use_raptor,
            use_graphrag=use_graphrag,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return _serialize(record)


def get_index_job(job_id: str) -> dict[str, Any] | None:
    with SessionLocal() as db:
        record = db.get(IndexJob, job_id)
        return _serialize(record) if record is not None else None


def list_index_jobs(
    *,
    active_only: bool = False,
    limit: int = 20,
) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        query = db.query(IndexJob)
        if active_only:
            query = query.filter(IndexJob.status.in_(ACTIVE_JOB_STATUSES))
        records = (
            query.order_by(
                IndexJob.created_at.desc(),
                IndexJob.job_id.asc(),
            )
            .limit(limit)
            .all()
        )
        return [_serialize(record) for record in records]


def update_index_job(
    job_id: str,
    **updates: Any,
) -> dict[str, Any] | None:
    unexpected = set(updates) - _UPDATABLE_FIELDS
    if unexpected:
        raise ValueError(
            "Unsupported index-job fields: "
            + ", ".join(sorted(unexpected))
        )
    with SessionLocal() as db:
        record = db.get(IndexJob, job_id)
        if record is None:
            return None
        for key, value in updates.items():
            if key == "progress":
                value = min(100.0, max(0.0, float(value)))
            elif key == "error" and value is not None:
                value = str(value)[:2000]
            elif key == "timings":
                value = {
                    str(name): round(float(seconds), 3)
                    for name, seconds in dict(value).items()
                }
            elif key == "metrics":
                value = dict(value or {})
            setattr(record, key, value)
        record.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(record)
        return _serialize(record)


def request_index_job_cancellation(
    job_id: str,
) -> dict[str, Any] | None:
    with SessionLocal() as db:
        record = db.get(IndexJob, job_id)
        if record is None:
            return None
        if record.status not in TERMINAL_JOB_STATUSES:
            record.cancel_requested = True
            record.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(record)
        return _serialize(record)


def recover_active_index_jobs() -> list[dict[str, Any]]:
    """Return queued work and reset interrupted running jobs to queued."""
    with SessionLocal() as db:
        records = (
            db.query(IndexJob)
            .filter(IndexJob.status.in_(ACTIVE_JOB_STATUSES))
            .order_by(IndexJob.created_at.asc(), IndexJob.job_id.asc())
            .all()
        )
        now = datetime.now(timezone.utc)
        for record in records:
            if record.status == "running":
                record.status = "queued"
                record.stage = "queued"
                record.progress = 0.0
                record.error = None
                record.updated_at = now
        db.commit()
        return [_serialize(record) for record in records]
