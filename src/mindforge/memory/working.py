"""Working memory - single task session memory with capacity management."""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Any

from mindforge.context.ranker import lexical_relevance

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single entry in working memory."""

    key: str
    content: str
    entry_type: str  # "context" | "tool_result" | "thought"
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextView:
    """One task-scoped, observable selection from working memory."""

    text: str
    selected_keys: tuple[str, ...]
    excluded_keys: tuple[str, ...]
    used_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_context_ids": list(self.selected_keys),
            "excluded_context_ids": list(self.excluded_keys),
            "selected_count": len(self.selected_keys),
            "excluded_count": len(self.excluded_keys),
            "used_chars": self.used_chars,
        }


class WorkingMemory:
    """Session-level memory that holds context chunks, tool results, and reasoning thoughts.

    Manages a capacity budget of ~8000 tokens by evicting low-importance / stale entries when
    the budget is exceeded.
    """

    def __init__(
        self,
        capacity_tokens: int | None = None,
        chars_per_token: int | None = None,
    ) -> None:
        from mindforge.config import get_settings

        config = get_settings().memory
        self._capacity_tokens = (
            capacity_tokens
            if capacity_tokens is not None
            else config.working_capacity_tokens
        )
        self._chars_per_token = (
            chars_per_token if chars_per_token is not None else config.chars_per_token
        )
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

    def get_context_string(
        self,
        max_chars: int | None = None,
        *,
        include_types: set[str] | None = None,
    ) -> str:
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
            max_chars = self._capacity_tokens * self._chars_per_token

        # Group and sort
        tool_results: list[MemoryEntry] = []
        context: list[MemoryEntry] = []
        thoughts: list[MemoryEntry] = []

        for entry in self._entries.values():
            if include_types is not None and entry.entry_type not in include_types:
                continue
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

    def get_relevant_context(
        self,
        query: str,
        *,
        max_chars: int,
        include_types: set[str] | None = None,
        min_relevance: float = 0.0,
        max_items: int | None = None,
        allowed_producer_ids: set[str] | None = None,
    ) -> ContextView:
        """Select an isolated context view for one concrete subtask."""
        candidates: list[tuple[bool, float, float, MemoryEntry]] = []
        for entry in self._entries.values():
            if include_types is not None and entry.entry_type not in include_types:
                continue
            producer_id = entry.metadata.get("producer_subtask_id")
            if (
                producer_id is not None
                and allowed_producer_ids is not None
                and producer_id not in allowed_producer_ids
            ):
                continue
            mandatory = bool(
                entry.metadata.get("context_pinned")
                or entry.metadata.get("context_explicitly_selected")
                or entry.metadata.get("context_referenced")
                or entry.metadata.get("context_follow_up_turn")
            )
            relevance = lexical_relevance(query, entry.content)
            try:
                importance = float(entry.importance)
            except (TypeError, ValueError):
                importance = 0.5
            if mandatory or relevance >= min_relevance:
                candidates.append((mandatory, relevance, importance, entry))

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3].timestamp,
            ),
            reverse=True,
        )
        if max_items is not None:
            candidates = candidates[: max(1, max_items)]

        selected: list[str] = []
        sections: list[str] = []
        remaining = max(0, max_chars)
        for _mandatory, _relevance, _importance, entry in candidates:
            if remaining <= 0:
                break
            snippet = f"[{entry.entry_type}] {entry.content}\n"
            bounded = snippet[:remaining]
            if not bounded:
                continue
            sections.append(bounded)
            selected.append(entry.key)
            remaining -= len(bounded)

        eligible_keys = {
            entry.key
            for entry in self._entries.values()
            if include_types is None or entry.entry_type in include_types
        }
        selected_set = set(selected)
        text = "".join(sections)
        return ContextView(
            text=text,
            selected_keys=tuple(selected),
            excluded_keys=tuple(sorted(eligible_keys - selected_set)),
            used_chars=len(text),
        )

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
        return total_chars // self._chars_per_token + len(self._entries) * 2

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
            score, key = scored.pop(0)  # remove worst
            removed = self._entries.pop(key, None)
            if removed:
                logger.debug(
                    "Evicted working memory entry: %s (score=%.3f)",
                    key,
                    score,
                )

        self._last_cleanup = now
