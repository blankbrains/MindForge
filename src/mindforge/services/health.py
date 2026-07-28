"""Cached dependency health monitoring."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthSnapshot:
    qdrant_connected: bool
    redis_connected: bool
    postgres_connected: bool

    @property
    def status(self) -> str:
        return (
            "ok"
            if (
                self.qdrant_connected
                and self.redis_connected
                and self.postgres_connected
            )
            else "degraded"
        )


class HealthMonitor:
    def __init__(self) -> None:
        self._snapshot: HealthSnapshot | None = None
        self._refresh_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._redis = None

    async def start(self) -> HealthSnapshot:
        snapshot = await self.refresh()
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(
                self._run(),
                name="mindforge-health-monitor",
            )
        return snapshot

    async def stop(self) -> None:
        self._stop_event.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def get_snapshot(self) -> HealthSnapshot:
        if self._snapshot is None:
            return await self.refresh()
        return self._snapshot

    async def refresh(self) -> HealthSnapshot:
        async with self._refresh_lock:
            qdrant_ok, redis_ok, postgres_ok = await asyncio.gather(
                self._probe_qdrant(),
                self._probe_redis(),
                self._probe_postgres(),
            )
            self._snapshot = HealthSnapshot(
                qdrant_connected=qdrant_ok,
                redis_connected=redis_ok,
                postgres_connected=postgres_ok,
            )
            return self._snapshot

    async def _run(self) -> None:
        from mindforge.config import get_settings

        interval = get_settings().api.health_refresh_seconds
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=interval,
                )
            except asyncio.TimeoutError:
                try:
                    await self.refresh()
                except Exception:
                    logger.exception("Background health refresh failed.")

    async def _probe_qdrant(self) -> bool:
        try:
            from mindforge.retrieval.vector_store import get_vector_store

            await get_vector_store().ping()
            return True
        except Exception:
            return False

    async def _probe_redis(self) -> bool:
        try:
            if self._redis is None:
                import redis.asyncio as aioredis
                from mindforge.config import get_settings

                self._redis = aioredis.from_url(
                    get_settings().cache.redis_url,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                )
            await self._redis.ping()
            return True
        except Exception:
            if self._redis is not None:
                try:
                    await self._redis.aclose()
                except Exception:
                    pass
                self._redis = None
            return False

    async def _probe_postgres(self) -> bool:
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


_health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    return _health_monitor
