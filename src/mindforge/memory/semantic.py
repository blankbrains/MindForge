"""Semantic memory - persistent facts and query patterns."""

from __future__ import annotations

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory where semantic memory files are stored (relative to user home / project)
STORAGE_DIR = ".semantic_memory"

# Categories recognised by the semantic memory store
VALID_CATEGORIES = frozenset(
    {"code", "api", "concept", "workflow", "preference", "general"}
)

# Maximum age (seconds) before a fact is considered stale for confidence decay
STALE_AFTER = 90 * 86400  # 90 days


@dataclass
class Fact:
    """A verified fact stored in semantic memory."""

    fact_id: str
    content: str
    sources: list[str]
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    category: str = "general"


@dataclass
class QueryPattern:
    """A logged query-processing pattern."""

    query_type: str
    strategy: str
    success: bool
    quality_score: float
    timestamp: float = field(default_factory=time.time)


class SemanticMemory:
    """Persistent semantic memory that stores facts and query-processing patterns.

    Data is persisted as JSON files under ``.semantic_memory/`` inside the
    directory specified at init (defaults to current working directory).
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self._base = Path(storage_dir or Path.cwd())
        self._store_path = self._base / STORAGE_DIR

        self._facts: dict[str, Fact] = {}
        self._patterns: list[QueryPattern] = []
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
    ) -> str:
        """Add a verified fact, deduplicating by content hash.

        Returns the fact_id.
        """
        fact_id = str(hash(content) % 10**9)

        if fact_id in self._facts:
            existing = self._facts[fact_id]
            # Merge sources and update confidence (take the higher value)
            existing.sources = list(set(existing.sources + sources))
            existing.confidence = max(existing.confidence, confidence)
            existing.timestamp = time.time()
            self._save()
            return fact_id

        category = self._infer_category(content)
        fact = Fact(
            fact_id=fact_id,
            content=content,
            sources=sources,
            confidence=confidence,
            category=category,
        )
        self._facts[fact_id] = fact
        self._save()
        return fact_id

    def add_pattern(
        self,
        query_type: str,
        strategy: str,
        success: bool,
        quality_score: float,
    ) -> None:
        """Log a query-processing pattern."""
        pattern = QueryPattern(
            query_type=query_type,
            strategy=strategy,
            success=success,
            quality_score=quality_score,
        )
        self._patterns.append(pattern)
        self._save()

    def get_strategy_stats(self) -> dict[str, dict[str, Any]]:
        """Analyze which strategies work best per query type.

        Returns
        -------
        dict[str, dict[str, Any]]
            Nested mapping::

                {
                    "<query_type>": {
                        "<strategy>": {
                            "count": ...,
                            "success_rate": ...,
                            "avg_quality": ...,
                        },
                        ...
                    },
                    ...
                }
        """
        stats: dict[str, dict[str, dict[str, Any]]] = {}

        for p in self._patterns:
            qt = stats.setdefault(p.query_type, {})
            strat = qt.setdefault(p.strategy, {"count": 0, "success_count": 0, "quality_total": 0.0})

            strat["count"] += 1
            if p.success:
                strat["success_count"] += 1
            strat["quality_total"] += p.quality_score

        # Convert raw counts to rates / averages
        result: dict[str, dict[str, Any]] = {}
        for query_type, strategies in stats.items():
            result[query_type] = {}
            for strategy, data in strategies.items():
                result[query_type][strategy] = {
                    "count": data["count"],
                    "success_rate": data["success_count"] / data["count"],
                    "avg_quality": data["quality_total"] / data["count"],
                }

        return result

    # ------------------------------------------------------------------
    # Alias used by Orchestrator
    # ------------------------------------------------------------------

    async def store(self, task: str, output: str) -> None:
        """Async alias for ``add_fact`` — used by Orchestrator."""
        self.add_fact(content=output, sources=[f"task: {task[:100]}"], confidence=0.8)

    def search_facts(self, query: str, top_k: int = 5) -> list[Fact]:
        """Simple keyword-based fact retrieval.

        Matches against the ``content`` field of stored facts. Results are
        ranked by a combination of keyword overlap and confidence.
        """
        query_words = set(query.lower().split())
        scored: list[tuple[Fact, float]] = []

        for fact in self._facts.values():
            fact_words = set(fact.content.lower().split())
            overlap = len(query_words & fact_words)
            if overlap > 0:
                score = overlap * fact.confidence
                # Small recency bonus
                age_days = (time.time() - fact.timestamp) / 86400
                recency_boost = max(1.0, 2.0 - age_days / 30.0)  # decays over ~30 days
                score *= recency_boost
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
        """Write facts and patterns to disk as JSON."""
        try:
            facts_data = {
                fid: {
                    "fact_id": f.fact_id,
                    "content": f.content,
                    "sources": f.sources,
                    "confidence": f.confidence,
                    "timestamp": f.timestamp,
                    "category": f.category,
                }
                for fid, f in self._facts.items()
            }
            (self._store_path / "facts.json").write_text(
                json.dumps(facts_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            patterns_data = [
                {
                    "query_type": p.query_type,
                    "strategy": p.strategy,
                    "success": p.success,
                    "quality_score": p.quality_score,
                    "timestamp": p.timestamp,
                }
                for p in self._patterns
            ]
            (self._store_path / "patterns.json").write_text(
                json.dumps(patterns_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Failed to save semantic memory to disk")

    def _load(self) -> None:
        """Load facts and patterns from disk JSON files."""
        facts_file = self._store_path / "facts.json"
        if facts_file.exists():
            try:
                data = json.loads(facts_file.read_text(encoding="utf-8"))
                for fid, d in data.items():
                    self._facts[fid] = Fact(
                        fact_id=d["fact_id"],
                        content=d["content"],
                        sources=d.get("sources", []),
                        confidence=d.get("confidence", 0.5),
                        timestamp=d.get("timestamp", 0.0),
                        category=d.get("category", "general"),
                    )
            except Exception:
                logger.exception("Failed to load facts from %s", facts_file)

        patterns_file = self._store_path / "patterns.json"
        if patterns_file.exists():
            try:
                data = json.loads(patterns_file.read_text(encoding="utf-8"))
                for d in data:
                    self._patterns.append(
                        QueryPattern(
                            query_type=d["query_type"],
                            strategy=d["strategy"],
                            success=d["success"],
                            quality_score=d.get("quality_score", 0.0),
                            timestamp=d.get("timestamp", 0.0),
                        )
                    )
            except Exception:
                logger.exception("Failed to load patterns from %s", patterns_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_category(content: str) -> str:
        """Rough category inference based on keywords in the fact content."""
        lower = content.lower()
        if any(kw in lower for kw in ("function", "class", "import", "def ", "return", "api")):
            return "code"
        if any(kw in lower for kw in ("http", "endpoint", "request", "response", "route")):
            return "api"
        if any(kw in lower for kw in ("is a", "refers to", "defined as", "meaning")):
            return "concept"
        if any(kw in lower for kw in ("step", "workflow", "pipeline", "process", "first")):
            return "workflow"
        if any(kw in lower for kw in ("prefer", "like", "always", "never", "favorite")):
            return "preference"
        return "general"
