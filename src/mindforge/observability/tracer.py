"""Tracing infrastructure for MindForge observability.

Provides Span dataclass and Tracer class for hierarchical trace capture,
with dual export to local JSONL files and optional LangFuse backend.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator

from mindforge.config import get_settings


@dataclass
class Span:
    """A single span within a trace recording an operation's lifecycle."""

    span_id: str
    trace_id: str
    name: str
    start_time: float  # Unix timestamp
    end_time: float | None = None
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    parent_id: str | None = None

    @property
    def duration_ms(self) -> float:
        """Return span duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """Hierarchical tracer that writes spans to JSONL files and optionally
    exports them to LangFuse.

    Usage::

        tracer = get_tracer()
        with tracer.span("research_flow", trace_id="abc") as span:
            ...
    """

    def __init__(self) -> None:
        _traces_base_env = os.getenv("MINDFORGE_TRACES_DIR")
        if _traces_base_env:
            _traces_base = Path(_traces_base_env)
        else:
            _traces_base = Path(__file__).resolve().parent.parent.parent.parent  # MindForge root
        self._traces_dir = _traces_base / ".traces"
        self._traces_dir.mkdir(parents=True, exist_ok=True)

        self._active_stack: list[Span] = []
        self._lock = threading.Lock()
        self._langfuse = None

        # Attempt to initialise LangFuse if the package is installed
        # and config is set via OBSERVABILITY_* env vars or .env.
        try:
            obs_cfg = get_settings().observability
            pub = obs_cfg.langfuse_public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
            sec = obs_cfg.langfuse_secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
            host = obs_cfg.langfuse_host or os.environ.get("LANGFUSE_HOST")
            if all((pub, sec, host)):
                import langfuse  # type: ignore[import-untyped]
                self._langfuse = langfuse.Langfuse(
                    public_key=pub,
                    secret_key=sec,
                    host=host,
                )
        except (ImportError, Exception):
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def span(
        self,
        name: str,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Context manager that creates and automatically closes a span.

        When called inside another active span the new span is nested
        (it uses the outer span as its parent).
        """
        span_id = uuid.uuid4().hex[:16]
        trace_id = trace_id or uuid.uuid4().hex[:16]

        with self._lock:
            parent_id = self._active_stack[-1].span_id if self._active_stack else None

        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            name=name,
            start_time=time.time(),
            parent_id=parent_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._active_stack.append(span)
        try:
            yield span
        except Exception as exc:
            span.error = str(exc)
            raise
        finally:
            span.end_time = time.time()
            try:
                self._active_stack.remove(span)
            except ValueError:
                pass  # already removed or stack corrupted
            self._export(span)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _export(self, span: Span) -> None:
        """Dual-write *span* to local JSONL and (if configured) LangFuse."""
        # -- Local JSONL --------------------------------------------------
        trace_file = self._traces_dir / f"trace_{span.trace_id}.jsonl"
        record = {
            "span_id": span.span_id,
            "trace_id": span.trace_id,
            "name": span.name,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": round(span.duration_ms, 3),
            "parent_id": span.parent_id,
            "error": span.error,
        }
        # Only include potentially large payloads when explicitly set.
        if span.input is not None:
            record["input"] = span.input
        if span.output is not None:
            record["output"] = span.output
        if span.metadata:
            record["metadata"] = span.metadata

        with open(trace_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        # -- LangFuse ----------------------------------------------------
        if self._langfuse is not None:
            try:
                self._langfuse.generation(
                    name=span.name,
                    trace_id=span.trace_id,
                    parent_observation_id=span.parent_id,
                    start_time=span.start_time,
                    end_time=span.end_time,
                    input=span.input,
                    output=span.output,
                    metadata=span.metadata,
                    level="ERROR" if span.error else "DEFAULT",
                    status_message=span.error,
                )
            except Exception:
                pass  # LangFuse errors should never break the application


@lru_cache(maxsize=1)
def get_tracer() -> Tracer:
    """Return the application-wide singleton :class:`Tracer`."""
    return Tracer()
