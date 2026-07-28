"""Episodic memory - cross-session task history."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

TASK_CATEGORIES = frozenset({"comparison", "howto", "analysis", "concept"})

# Simple keyword rules for _classify_task
_CLASSIFIER_RULES: dict[str, set[str]] = {
    "comparison": {"compare", "vs", "versus", "difference", "better", "trade-off"},
    "howto": {"how", "how to", "steps", "guide", "tutorial", "implement", "setup"},
    "analysis": {"analyze", "analysis", "deep dive", "evaluate", "review", "investigate"},
    "concept": {"what is", "explain", "define", "overview", "introduction", "concept"},
}


@dataclass
class Episode:
    """A single recorded task episode."""

    task: str
    result: str
    sources: list[str]
    embedding: list[float] | None
    timestamp: float = field(default_factory=time.time)
    task_type: str = "unknown"


class EpisodicMemory:
    """Cross-session episodic memory that stores task history.

    Data is kept in an in-memory list (capped at ``MAX_EPISODES``). If a
    ``redis_client`` is provided at init, episodes are also persisted with a
    30-day TTL for durable storage.
    """

    def __init__(
        self,
        redis_client: Any = None,
        max_episodes: int | None = None,
        redis_ttl: int | None = None,
    ) -> None:
        from mindforge.config import get_settings

        config = get_settings().memory
        self._episodes: list[Episode] = []
        self._redis = redis_client
        self._max_episodes = max_episodes or config.max_episodes
        self._redis_ttl = redis_ttl or config.episodic_ttl_seconds
        self._max_episode_chars = config.max_episode_chars
        self._lock = asyncio.Lock()
        if self._redis is not None:
            self._load_from_redis()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_episode(
        self,
        task: str,
        result: str,
        sources: list[str],
        embedding: list[float] | None = None,
    ) -> None:
        """Record a new episode."""
        task_type = self._classify_task(task)
        episode = Episode(
            task=task[:20_000],
            result=result[: self._max_episode_chars],
            sources=[str(source)[:1000] for source in sources[:100]],
            embedding=embedding,
            task_type=task_type,
        )

        self._episodes.append(episode)

        # In-memory cap
        if len(self._episodes) > self._max_episodes:
            self._episodes.pop(0)

        # Redis persistence (best-effort)
        if self._redis is not None:
            self._persist_to_redis(episode)

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        days: int | None = None,
    ) -> list[Episode]:
        """Search episodes by simple keyword overlap or Redis scan.

        Parameters
        ----------
        query : str
            The search query.
        top_k : int
            Maximum results to return.
        days : int, optional
            If provided, only consider episodes from the last *days* days.

        Returns
        -------
        list[Episode]
            Matching episodes, scored by word-overlap (descending).
        """
        top_k = min(max(1, top_k), 20)
        cutoff: float | None = None
        if days is not None:
            cutoff = time.time() - days * 86400

        candidates = self._episodes
        if cutoff is not None:
            candidates = [e for e in candidates if e.timestamp >= cutoff]

        if not candidates:
            return []

        query_words = set(query.lower().split())

        def _score(ep: Episode) -> float:
            # Only match on task similarity, NOT result content
            # (matching on result causes false hits for any common word)
            task_words = set(ep.task.lower().split())
            overlap = len(query_words & task_words)
            # Bonus for exact task-type match
            ep_type = self._classify_task(query)
            bonus = 1.0 if ep.task_type == ep_type else 0.0
            return overlap + bonus

        scored = [(ep, _score(ep)) for ep in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Filter out zero-score unless there is nothing else
        scored = [ep for ep, s in scored if s > 0.0]
        if not scored:
            return []

        return scored[:top_k]

    # ------------------------------------------------------------------
    # Aliases used by Orchestrator
    # ------------------------------------------------------------------

    async def store(self, task: str, result: dict) -> None:
        """Async alias for ``add_episode`` — used by Orchestrator.

        Accepts a result dict (with an ``output`` key) and delegates to the
        synchronous ``add_episode``.
        """
        async with self._lock:
            output = result.get("output", str(result)) if isinstance(result, dict) else str(result)
            await asyncio.to_thread(
                self.add_episode,
                task=task,
                result=output,
                sources=[],
            )

    async def recall(self, task: str) -> dict | None:
        """Recall a cached result only for an exact, unexpired task.

        Fuzzy keyword matches are useful for analytics and suggestions, but
        they are not safe as a substitute for executing a new research task.
        """
        async with self._lock:
            cutoff = time.time() - self._redis_ttl
            self._episodes = [
                episode
                for episode in self._episodes
                if episode.timestamp >= cutoff
            ]
            task_clean = task.strip().lower()
            for ep in reversed(self._episodes):
                if ep.task.strip().lower() == task_clean:
                    return {"output": ep.result, "episode": ep}
            return None

    def get_user_profile(self) -> dict[str, float]:
        """Return the distribution of task types across stored episodes.

        Returns
        -------
        dict[str, float]
            Mapping of task_type -> fraction (0.0 – 1.0).
        """
        if not self._episodes:
            return {}

        counts: dict[str, int] = {}
        for ep in self._episodes:
            t = ep.task_type or "unknown"
            counts[t] = counts.get(t, 0) + 1

        total = sum(counts.values())
        return {t: c / total for t, c in counts.items()}

    def close(self) -> None:
        """Release the optional Redis connection."""
        close = getattr(self._redis, "close", None)
        if callable(close):
            close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_task(self, task: str) -> str:
        """Simple keyword-based classification.

        Returns one of ``comparison``, ``howto``, ``analysis``, ``concept``,
        or ``unknown``.
        """
        lower = task.lower()
        for category, keywords in _CLASSIFIER_RULES.items():
            for kw in keywords:
                if kw in lower:
                    return category
        return "unknown"

    def _persist_to_redis(self, episode: Episode) -> None:
        """Store an episode in Redis with a 30-day TTL."""
        try:
            task_hash = hashlib.sha256(
                episode.task.encode("utf-8")
            ).hexdigest()[:16]
            key = f"mindforge:episode:{episode.timestamp:.6f}:{task_hash}"
            payload = {
                "task": episode.task,
                "result": episode.result,
                "sources": episode.sources,
                "embedding": episode.embedding,
                "timestamp": episode.timestamp,
                "task_type": episode.task_type,
            }
            self._redis.setex(key, self._redis_ttl, json.dumps(payload))
            index_key = "mindforge:episodes:index"
            self._redis.zadd(index_key, {key: episode.timestamp})
            count = int(self._redis.zcard(index_key))
            overflow = count - self._max_episodes
            if overflow > 0:
                stale = self._redis.zrange(index_key, 0, overflow - 1)
                if stale:
                    self._redis.delete(*stale)
                    self._redis.zrem(index_key, *stale)
            self._redis.expire(index_key, self._redis_ttl)
        except Exception:
            logger.exception("Failed to persist episode to Redis")

    def _load_from_redis(self) -> None:
        """Restore recent episodes from Redis at process startup."""
        try:
            episodes: list[Episode] = []
            index_key = "mindforge:episodes:index"
            keys = self._redis.zrevrange(
                index_key,
                0,
                self._max_episodes - 1,
            )
            if not keys:
                keys = []
                for key in self._redis.scan_iter(
                    match="mindforge:episode:*",
                    count=min(200, self._max_episodes),
                ):
                    keys.append(key)
                    if len(keys) >= self._max_episodes:
                        break
            for key in keys:
                raw = self._redis.get(key)
                if not raw:
                    continue
                data = json.loads(raw)
                episodes.append(
                    Episode(
                        task=str(data.get("task", ""))[:20_000],
                        result=str(data.get("result", ""))[
                            : self._max_episode_chars
                        ],
                        sources=[
                            str(source)[:1000]
                            for source in list(data.get("sources", []))[:100]
                        ],
                        embedding=data.get("embedding"),
                        timestamp=float(data.get("timestamp", 0.0)),
                        task_type=str(
                            data.get("task_type", "unknown")
                        ),
                    )
                )
            episodes.sort(key=lambda episode: episode.timestamp)
            self._episodes = episodes[-self._max_episodes:]
            logger.info(
                "Restored %d episodic memories from Redis.",
                len(self._episodes),
            )
        except Exception:
            logger.exception("Failed to restore episodic memory from Redis")
