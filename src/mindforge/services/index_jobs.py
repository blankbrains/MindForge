"""Application-owned worker queue for persistent indexing jobs."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from mindforge.config import get_settings, resolve_project_path
from mindforge.repositories.index_jobs import (
    create_index_job,
    get_index_job,
    list_index_jobs,
    recover_active_index_jobs,
    request_index_job_cancellation,
    update_index_job,
)
from mindforge.services.indexing import (
    IndexingCancelledError,
    build_index_signature,
    get_reusable_document,
)

logger = logging.getLogger(__name__)


class IndexJobService:
    """Own background workers and recover persistent jobs after restart."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._queued_ids: set[str] = set()
        self._started = False
        self._lifecycle_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                return
            recovered = await asyncio.to_thread(recover_active_index_jobs)
            worker_count = get_settings().api.max_concurrent_index_jobs
            self._workers = [
                asyncio.create_task(
                    self._worker(worker_index),
                    name=f"mindforge-index-worker-{worker_index}",
                )
                for worker_index in range(worker_count)
            ]
            self._started = True
            for job in recovered:
                await self._enqueue_once(str(job["job_id"]))
            if recovered:
                logger.info(
                    "Recovered %d persistent indexing jobs.",
                    len(recovered),
                )

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            if not self._started:
                return
            workers = list(self._workers)
            self._workers.clear()
            self._started = False
            for worker in workers:
                worker.cancel()
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)
            self._queued_ids.clear()
            self._queue = asyncio.Queue()

    async def enqueue(self, job_id: str) -> None:
        if not self._started:
            await self.start()
        await self._enqueue_once(job_id)

    async def _enqueue_once(self, job_id: str) -> None:
        if job_id in self._queued_ids:
            return
        self._queued_ids.add(job_id)
        await self._queue.put(job_id)

    async def create(
        self,
        *,
        job_id: str,
        filename: str,
        file_path: str,
        strategy: str,
        use_raptor: bool,
        use_graphrag: bool,
    ) -> dict:
        job = await asyncio.to_thread(
            create_index_job,
            job_id=job_id,
            filename=filename,
            file_path=file_path,
            strategy=strategy,
            use_raptor=use_raptor,
            use_graphrag=use_graphrag,
        )
        await self.enqueue(job_id)
        return job

    async def get(self, job_id: str) -> dict | None:
        return await asyncio.to_thread(get_index_job, job_id)

    async def list(
        self,
        *,
        active_only: bool = False,
        limit: int = 20,
    ) -> list[dict]:
        return await asyncio.to_thread(
            list_index_jobs,
            active_only=active_only,
            limit=limit,
        )

    async def cancel(self, job_id: str) -> dict | None:
        return await asyncio.to_thread(
            request_index_job_cancellation,
            job_id,
        )

    async def _worker(self, worker_index: int) -> None:
        while True:
            job_id = await self._queue.get()
            self._queued_ids.discard(job_id)
            try:
                await self._run_job(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Index worker %d failed outside job lifecycle.",
                    worker_index,
                )
            finally:
                self._queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        job = await self.get(job_id)
        if job is None:
            return
        if job["cancel_requested"]:
            await self._finish_cancelled(job_id, job["file_path"])
            return

        path = Path(str(job["file_path"])).resolve()
        if not path.is_file():
            await asyncio.to_thread(
                update_index_job,
                job_id,
                status="failed",
                stage="failed",
                error="Uploaded source file is missing.",
            )
            return

        timings = dict(job.get("timings") or {})
        metrics = dict(job.get("metrics") or {})
        total_started = time.perf_counter()
        cleanup_file = False
        try:
            await asyncio.to_thread(
                update_index_job,
                job_id,
                status="running",
                stage="parsing",
                progress=5.0,
                error=None,
            )
            from mindforge.api.routes import _index_with_lifecycle
            from mindforge.ingestion.parsers import (
                DocumentParser,
                DocumentParserCancelledError,
            )

            loop = asyncio.get_running_loop()
            parser_progress = 5.0
            reported_parser_progress: tuple[str, int] | None = None

            def report_parser_progress(
                stage: str,
                completed: int,
                total: int,
            ) -> None:
                nonlocal parser_progress, reported_parser_progress
                stage_ranges = {
                    "detecting": (5.0, 12.0),
                    "ocr": (12.0, 18.0),
                    "layout": (18.0, 19.0),
                    "table": (19.0, 20.0),
                }
                start, end = stage_ranges.get(stage, (5.0, 20.0))
                ratio = min(1.0, max(0.0, completed / max(total, 1)))
                progress = max(parser_progress, start + (end - start) * ratio)
                progress_bucket = int(progress)
                progress_key = (stage, progress_bucket)
                if progress_key == reported_parser_progress and completed < total:
                    return
                parser_progress = progress
                reported_parser_progress = progress_key

                async def persist() -> None:
                    await self._check_cancelled(job_id)
                    elapsed = max(0.0, time.perf_counter() - total_started)
                    eta_seconds = (
                        elapsed * (total - completed) / completed
                        if completed > 0
                        else None
                    )
                    metrics["parser"] = {
                        "stage": stage,
                        "completed_pages": completed,
                        "total_pages": total,
                        "elapsed_seconds": round(elapsed, 3),
                        "eta_seconds": (
                            round(eta_seconds, 3) if eta_seconds is not None else None
                        ),
                    }
                    await asyncio.to_thread(
                        update_index_job,
                        job_id,
                        status="running",
                        stage=stage,
                        progress=progress,
                        timings=timings,
                        metrics=metrics,
                    )

                future = asyncio.run_coroutine_threadsafe(persist(), loop)
                future.result()

            stage_started = time.perf_counter()
            parser = DocumentParser()
            parser.set_progress_callback(report_parser_progress)
            parser.set_cancellation_callback(
                lambda: bool((get_index_job(job_id) or {}).get("cancel_requested"))
            )
            parsed = await asyncio.to_thread(parser.parse, path)
            timings["parsing"] = time.perf_counter() - stage_started
            parse_metadata = dict(getattr(parsed, "metadata", {}) or {})
            metrics["parse"] = {
                "pages": int(parse_metadata.get("pages") or 0),
                "ocr_pages": int(parse_metadata.get("ocr_pages") or 0),
                "table_count": int(parse_metadata.get("table_count") or 0),
                "image_count": int(parse_metadata.get("image_count") or 0),
                "visual_only": bool(parse_metadata.get("visual_only")),
                "page_metrics": list(parse_metadata.get("page_metrics") or []),
            }
            await self._check_cancelled(job_id)
            index_signature = build_index_signature(
                strategy=str(job["strategy"]),
                use_raptor=bool(job["use_raptor"]),
                use_graphrag=bool(job["use_graphrag"]),
            )
            reusable = await get_reusable_document(
                doc_id=parsed.doc_id,
                index_signature=index_signature,
            )
            if reusable is not None:
                timings["deduplication"] = 0.0
                timings["total"] = time.perf_counter() - total_started
                await asyncio.to_thread(
                    update_index_job,
                    job_id,
                    doc_id=parsed.doc_id,
                    status="completed",
                    stage="completed",
                    progress=100.0,
                    chunk_count=int(reusable["chunk_count"]),
                    timings=timings,
                    metrics=metrics,
                    error=None,
                )
                logger.info(
                    "Reused existing index for duplicate document %s.",
                    parsed.doc_id,
                )
                cleanup_file = True
                return
            await asyncio.to_thread(
                update_index_job,
                job_id,
                doc_id=parsed.doc_id,
                stage="chunking",
                progress=20.0,
                timings=timings,
                metrics=metrics,
            )

            async def report(
                stage: str,
                progress: float,
                chunk_count: int,
                reported_timings: dict[str, float],
            ) -> None:
                await self._check_cancelled(job_id)
                await asyncio.to_thread(
                    update_index_job,
                    job_id,
                    status="running",
                    stage=stage,
                    progress=progress,
                    chunk_count=chunk_count,
                    timings=reported_timings,
                    metrics=metrics,
                )

            chunks = await _index_with_lifecycle(
                parsed=parsed,
                source=str(job["filename"]),
                strategy=str(job["strategy"]),
                use_raptor=bool(job["use_raptor"]),
                use_graphrag=bool(job["use_graphrag"]),
                progress_callback=report,
                timings=timings,
                source_path=path,
                cancelled=lambda: bool(
                    (get_index_job(job_id) or {}).get("cancel_requested")
                ),
            )
            timings["total"] = time.perf_counter() - total_started
            await asyncio.to_thread(
                update_index_job,
                job_id,
                status="completed",
                stage="completed",
                progress=100.0,
                chunk_count=len(chunks),
                timings=timings,
                metrics=metrics,
                error=None,
            )
            cleanup_file = True
        except (IndexingCancelledError, DocumentParserCancelledError):
            await self._finish_cancelled(job_id, str(path))
            cleanup_file = False
        except asyncio.CancelledError:
            await asyncio.to_thread(
                update_index_job,
                job_id,
                status="queued",
                stage="queued",
                progress=0.0,
                error=None,
            )
            raise
        except Exception as exc:
            timings["total"] = time.perf_counter() - total_started
            detail = getattr(exc, "detail", None) or str(exc)
            await asyncio.to_thread(
                update_index_job,
                job_id,
                status="failed",
                stage="failed",
                timings=timings,
                error=detail or type(exc).__name__,
            )
            logger.exception("Index job %s failed.", job_id)
            cleanup_file = True
        finally:
            if cleanup_file:
                await asyncio.to_thread(self._remove_job_file, path)

    async def _check_cancelled(self, job_id: str) -> None:
        job = await self.get(job_id)
        if job is not None and job["cancel_requested"]:
            raise IndexingCancelledError(f"Index job {job_id} was cancelled.")

    async def _finish_cancelled(
        self,
        job_id: str,
        file_path: str,
    ) -> None:
        await asyncio.to_thread(
            update_index_job,
            job_id,
            status="cancelled",
            stage="cancelled",
            error=None,
        )
        await asyncio.to_thread(
            self._remove_job_file,
            Path(file_path).resolve(),
        )

    @staticmethod
    def _remove_job_file(path: Path) -> None:
        job_root = (
            resolve_project_path(get_settings().app.data_dir) / "index-jobs"
        ).resolve()
        try:
            path.relative_to(job_root)
        except ValueError:
            logger.error(
                "Refusing to remove index-job file outside %s: %s",
                job_root,
                path,
            )
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Failed to remove completed index-job file %s.",
                path,
                exc_info=True,
            )


_service: IndexJobService | None = None


def get_index_job_service() -> IndexJobService:
    global _service
    if _service is None:
        _service = IndexJobService()
    return _service
