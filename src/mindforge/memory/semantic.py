"""Semantic memory - persistent, reusable research facts."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory where semantic memory files are stored within the configured base.
STORAGE_DIR = ".semantic_memory"

# Categories recognised by the semantic memory store
VALID_CATEGORIES = frozenset(
    {"code", "api", "concept", "workflow", "preference", "general"}
)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


@dataclass
class Fact:
    """A verified fact stored in semantic memory."""

    fact_id: str
    content: str
    sources: list[str]
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    category: str = "general"
    task: str = ""
    match_score: float = field(default=0.0, compare=False)


class SemanticMemory:
    """Persistent semantic memory for successful prior research.

    Facts are persisted as JSON under ``.semantic_memory/`` inside the
    directory specified at init (defaults to current working directory).
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        from mindforge.config import get_settings

        config = get_settings().memory
        self._base = Path(storage_dir or Path.cwd())
        self._store_path = self._base / STORAGE_DIR
        self._max_facts = config.max_semantic_facts
        self._max_fact_chars = config.max_semantic_fact_chars
        self._search_chars = config.semantic_search_chars
        self._recall_fact_chars = config.semantic_recall_fact_chars
        self._min_query_coverage = config.semantic_min_query_coverage
        self._min_similarity = config.semantic_min_similarity
        self._retention_seconds = config.semantic_retention_days * 86400
        self._max_file_bytes = config.semantic_max_file_bytes

        self._facts: dict[str, Fact] = {}
        self._lock = asyncio.Lock()

        self._ensure_store()
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        sources: list[str],
        confidence: float = 0.5,
        task: str = "",
    ) -> str:
        """Add a verified fact, deduplicating by content hash.

        Returns the fact_id.
        """
        import hashlib as _hashlib

        content = content[: self._max_fact_chars]
        task = task.strip()[:2000]
        sources = [str(source)[:1000] for source in sources[:100]]
        fact_id = _hashlib.sha256(content.encode()).hexdigest()[:16]

        if fact_id in self._facts:
            existing = self._facts[fact_id]
            # Merge sources and update confidence (take the higher value)
            existing.sources = list(set(existing.sources + sources))
            existing.confidence = max(existing.confidence, confidence)
            existing.timestamp = time.time()
            if task:
                existing.task = task
            self._save()
            return fact_id

        category = self._infer_category(content)
        fact = Fact(
            fact_id=fact_id,
            content=content,
            sources=sources,
            confidence=confidence,
            category=category,
            task=task,
        )
        self._facts[fact_id] = fact
        self._prune()
        self._save()
        return fact_id

    async def store(
        self,
        task: str,
        output: str,
        sources: list[dict] | None = None,
        confidence: float = 0.5,
    ) -> None:
        """Store a successful research result for later task grounding."""
        source_labels = []
        for source in sources or []:
            label = source.get("url") or source.get("title") or source.get("source")
            if label:
                source_labels.append(str(label))
        if not source_labels:
            source_labels = [f"task: {task[:100]}"]
        async with self._lock:
            await asyncio.to_thread(
                self.add_fact,
                content=output,
                sources=source_labels,
                confidence=max(0.0, min(1.0, confidence)),
                task=task,
            )

    async def recall(self, query: str, top_k: int = 5) -> list[Fact]:
        """Return relevant prior research without blocking the event loop."""
        async with self._lock:
            return await asyncio.to_thread(
                self.search_facts,
                query,
                top_k,
            )

    def search_facts(self, query: str, top_k: int = 5) -> list[Fact]:
        """Simple keyword-based fact retrieval.

        Rank by query coverage and cosine-style token overlap. Matching the
        original task is preferred over incidental terms in a long report.
        """
        top_k = min(max(1, top_k), 50)
        query_words = self._tokenize(query)
        if not query_words:
            return []
        scored: list[tuple[Fact, float]] = []

        for fact in self._facts.values():
            content_words = self._tokenize(fact.content[: self._search_chars])
            task_words = self._tokenize(fact.task)
            content_overlap = len(query_words & content_words)
            task_overlap = len(query_words & task_words)
            best_overlap = max(content_overlap, task_overlap)
            if best_overlap <= 0:
                continue

            query_coverage = best_overlap / len(query_words)
            if query_coverage < self._min_query_coverage:
                continue

            similarities = []
            if content_overlap:
                similarities.append(
                    content_overlap
                    / math.sqrt(len(query_words) * max(1, len(content_words)))
                )
            if task_overlap:
                similarities.append(
                    task_overlap
                    / math.sqrt(len(query_words) * max(1, len(task_words)))
                )
            similarity = max(similarities, default=0.0)
            if similarity < self._min_similarity:
                continue

            age_days = max(0.0, (time.time() - fact.timestamp) / 86400)
            recency_boost = 1.0 + max(0.0, 0.1 - age_days / 300.0)
            score = min(1.0, similarity * fact.confidence * recency_boost)
            fact.match_score = score
            scored.append((fact, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [fact for fact, _ in scored[:top_k]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_store(self) -> None:
        """Create the storage directory if it doesn't exist."""
        self._store_path.mkdir(parents=True, exist_ok=True)

    def _save(self) -> None:
        """Atomically write facts to disk.

        Uses unique temp filenames per call to avoid races when two saves
        overlap (e.g. from concurrent ``add_fact`` / ``add_pattern``).
        """
        import uuid as _uuid

        try:
            self._prune()
            facts_data = {
                fid: {
                    "fact_id": f.fact_id,
                    "content": f.content,
                    "sources": f.sources,
                    "confidence": f.confidence,
                    "timestamp": f.timestamp,
                    "category": f.category,
                    "task": f.task,
                }
                for fid, f in self._facts.items()
            }
            tmp_facts = self._store_path / f".facts.json.{_uuid.uuid4().hex[:8]}.tmp"
            tmp_facts.write_text(
                json.dumps(facts_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp_facts, self._store_path / "facts.json")

        except Exception:
            logger.exception("Failed to save semantic memory to disk")

    def _load(self) -> None:
        """Load persisted facts from disk."""
        facts_file = self._store_path / "facts.json"
        if facts_file.exists() and facts_file.stat().st_size <= self._max_file_bytes:
            try:
                data = json.loads(facts_file.read_text(encoding="utf-8"))
                for fid, d in data.items():
                    self._facts[fid] = Fact(
                        fact_id=d["fact_id"],
                        content=str(d["content"])[: self._max_fact_chars],
                        sources=[
                            str(source)[:1000] for source in d.get("sources", [])[:100]
                        ],
                        confidence=d.get("confidence", 0.5),
                        timestamp=d.get("timestamp", 0.0),
                        category=d.get("category", "general"),
                        task=str(d.get("task") or "")[:2000],
                    )
            except Exception:
                logger.exception("Failed to load facts from %s", facts_file)

        self._prune()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        cutoff = time.time() - self._retention_seconds
        self._facts = {
            fact_id: fact
            for fact_id, fact in self._facts.items()
            if fact.timestamp >= cutoff
        }
        if len(self._facts) > self._max_facts:
            newest = sorted(
                self._facts.values(),
                key=lambda fact: fact.timestamp,
                reverse=True,
            )[: self._max_facts]
            self._facts = {fact.fact_id: fact for fact in newest}

    @staticmethod
    def _infer_category(content: str) -> str:
        """Rough category inference based on keywords in the fact content."""
        lower = content.lower()
        if any(
            kw in lower
            for kw in ("function", "class", "import", "def ", "return", "api")
        ):
            return "code"
        if any(
            kw in lower for kw in ("http", "endpoint", "request", "response", "route")
        ):
            return "api"
        if any(kw in lower for kw in ("is a", "refers to", "defined as", "meaning")):
            return "concept"
        if any(
            kw in lower for kw in ("step", "workflow", "pipeline", "process", "first")
        ):
            return "workflow"
        if any(kw in lower for kw in ("prefer", "like", "always", "never", "favorite")):
            return "preference"
        return "general"

    @staticmethod
    def _tokenize(value: str) -> set[str]:
        tokens: set[str] = set()
        for match in _TOKEN_PATTERN.findall(value.lower()):
            if re.fullmatch(r"[\u4e00-\u9fff]+", match):
                if len(match) == 1:
                    tokens.add(match)
                else:
                    tokens.update(
                        match[index : index + 2] for index in range(len(match) - 1)
                    )
            else:
                tokens.add(match)
        return tokens

    def recall_content(self, fact: Fact) -> str:
        """Return a bounded semantic-memory snippet for prompt injection."""
        return fact.content[: self._recall_fact_chars]
