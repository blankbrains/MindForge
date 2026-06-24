"""Working memory - single task session memory with capacity management."""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Approximate token budget for the working memory context
CAPACITY_TOKENS = 8000
# Rough char-to-token ratio for estimation
CHARS_PER_TOKEN = 4


@dataclass
class MemoryEntry:
    """A single entry in working memory."""

    key: str
    content: str
    entry_type: str  # "context" | "tool_result" | "thought"
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkingMemory:
    """Session-level memory that holds context chunks, tool results, and reasoning thoughts.

    Manages a capacity budget of ~8000 tokens by evicting low-importance / stale entries when
    the budget is exceeded.
    """

    def __init__(self, capacity_tokens: int = CAPACITY_TOKENS) -> None:
        self._capacity_tokens = capacity_tokens
        self._entries: dict[str, MemoryEntry] = {}  # key -> entry (dedup key)
        self._last_cleanup: float = time.time()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_context(self, chunks: list[dict[str, Any]]) -> None:
        """Add document chunks, deduplicating by key.

        Each *chunk* dict should contain at minimum:
            - id / chunk_id  (used as dedup key)
            - content / text / page_content (the textual content)
            - rerank_score (optional, maps to importance)

        The importance is taken from the rerank_score if present, otherwise
        defaults to 0.5.
        """
        for chunk in chunks:
            key = (
                chunk.get("id")
                or chunk.get("chunk_id")
                or str(hash(chunk.get("content", chunk.get("text", ""))))
            )

            content = (
                chunk.get("content")
                or chunk.get("text")
                or chunk.get("page_content", "")
            )

            importance = chunk.get("rerank_score", 0.5)

            entry = MemoryEntry(
                key=key,
                content=content,
                entry_type="context",
                importance=importance,
                metadata=chunk,
            )

            self._entries[key] = entry

        self._manage_capacity()

    def add_tool_result(self, key: str, content: str, importance: float = 0.8) -> None:
        """Add (or update) a tool result entry."""
        entry = MemoryEntry(
            key=key,
            content=content,
            entry_type="tool_result",
            importance=importance,
        )
        self._entries[key] = entry
        self._manage_capacity()

    def add_thought(self, thought: str) -> None:
        """Add a reasoning-step thought."""
        key = f"thought_{int(time.time() * 1000)}_{hash(thought) % 10**6}"
        entry = MemoryEntry(
            key=key,
            content=thought,
            entry_type="thought",
            importance=0.6,  # moderate default; thoughts can be shifted down
        )
        self._entries[key] = entry
        self._manage_capacity()

    def get_context_string(self, max_chars: int | None = None) -> str:
        """Return a flattened string of the working memory contents.

        Ordering priority (within each group, entries are sorted by
        descending importance):
          1. tool_results
          2. context
          3. thoughts

        Parameters
        ----------
        max_chars : int, optional
            Maximum characters to include (defaults to
            ``capacity_tokens * CHARS_PER_TOKEN``).
        """
        if max_chars is None:
            max_chars = self._capacity_tokens * CHARS_PER_TOKEN

        # Group and sort
        tool_results: list[MemoryEntry] = []
        context: list[MemoryEntry] = []
        thoughts: list[MemoryEntry] = []

        for entry in self._entries.values():
            if entry.entry_type == "tool_result":
                tool_results.append(entry)
            elif entry.entry_type == "context":
                context.append(entry)
            else:
                thoughts.append(entry)

        # Within each group: sort by importance descending
        tool_results.sort(key=lambda e: e.importance, reverse=True)
        context.sort(key=lambda e: e.importance, reverse=True)
        thoughts.sort(key=lambda e: e.importance, reverse=True)

        # Concatenate in priority order, respecting max_chars
        sections: list[str] = []
        remaining = max_chars

        for group in (tool_results, context, thoughts):
            for entry in group:
                snippet = f"[{entry.entry_type}] {entry.content}\n"
                if len(snippet) > remaining:
                    snippet = snippet[:remaining]
                if snippet:
                    sections.append(snippet)
                    remaining -= len(snippet)
                if remaining <= 0:
                    break
            if remaining <= 0:
                break

        return "".join(sections)

    async def clear(self) -> None:
        """Reset working memory to empty."""
        async with self._lock:
            self._entries.clear()
            self._last_cleanup = time.time()

    # ------------------------------------------------------------------
    # Capacity management
    # ------------------------------------------------------------------

    def _estimate_tokens(self) -> int:
        """Rough token estimate based on character length."""
        total_chars = sum(len(e.content) for e in self._entries.values())
        return total_chars // CHARS_PER_TOKEN + len(self._entries) * 2  # overhead

    def _manage_capacity(self) -> None:
        """Evict low-value entries when the token budget is exceeded.

        Eviction score formula::

            score = importance - (now - timestamp) / 3600

        Entries with the lowest scores are removed one-by-one until the
        estimated token count falls back under the capacity limit.
        """
        if self._estimate_tokens() <= self._capacity_tokens:
            return

        now = time.time()
        scored: list[tuple[float, str]] = []

        for key, entry in self._entries.items():
            # Age penalty: 1 hour reduces score by 1.0
            score = entry.importance - (now - entry.timestamp) / 3600
            scored.append((score, key))

        # Ascending order — worst first
        scored.sort(key=lambda x: x[0])

        while scored and self._estimate_tokens() > self._capacity_tokens:
            _, key = scored.pop(0)  # remove worst
            removed = self._entries.pop(key, None)
            if removed:
                logger.debug("Evicted working memory entry: %s (score=%.3f)", key, scored[0][0] if scored else 0)

        self._last_cleanup = now
