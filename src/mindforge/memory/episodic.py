"""Episodic memory - cross-session task history."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Maximum number of episodes kept in-memory (when no Redis is available)
MAX_EPISODES = 200

# Redis TTL for episodes when Redis *is* available (30 days in seconds)
REDIS_TTL = 30 * 24 * 3600

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

    def __init__(self, redis_client: Any = None) -> None:
        from collections import deque
        self._episodes: deque[Episode] = deque(maxlen=MAX_EPISODES)
        self._redis = redis_client

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
            task=task,
            result=result,
            sources=sources,
            embedding=embedding,
            task_type=task_type,
        )

        self._episodes.append(episode)

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
        cutoff: float | None = None
        if days is not None:
            cutoff = time.time() - days * 86400

        candidates = self._episodes
        if cutoff is not None:
            candidates = [e for e in candidates if e.timestamp >= cutoff]

        if not candidates:
            return []

        query_words = set(query.lower().split())
        query_type = self._classify_task(query)  # 预计算，避免每 episode 重复调用

        def _score(ep: Episode) -> float:
            task_words = set(ep.task.lower().split())
            result_words = set(ep.result.lower().split())
            overlap = len(query_words & task_words) + len(query_words & result_words)
            bonus = 1.0 if ep.task_type == query_type else 0.0
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
        output = result.get("output", str(result)) if isinstance(result, dict) else str(result)
        self.add_episode(task=task, result=output, sources=[])

    async def recall(self, task: str) -> dict | None:
        """Async alias for ``search_similar`` — used by Orchestrator.

        Returns the top episode's result dict, or ``None`` if no match.
        """
        matches = self.search_similar(query=task, top_k=1)
        if matches:
            return {"output": matches[0].result, "episode": matches[0]}
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
            import json

            key = f"episode:{episode.timestamp}:{hash(episode.task) % 10**6}"
            payload = {
                "task": episode.task,
                "result": episode.result,
                "sources": episode.sources,
                "embedding": episode.embedding,
                "timestamp": episode.timestamp,
                "task_type": episode.task_type,
            }
            self._redis.setex(key, REDIS_TTL, json.dumps(payload))
        except Exception:
            logger.exception("Failed to persist episode to Redis")
