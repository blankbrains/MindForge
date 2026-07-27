"""Tracing infrastructure for MindForge observability.

Provides Span dataclass and Tracer class for hierarchical trace capture,
with dual export to local JSONL files and optional LangFuse backend.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from contextvars import ContextVar
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
        settings = get_settings()
        _traces_base_env = settings.app.traces_dir
        self._config = settings.observability
        from mindforge.config import get_project_root

        project_root = get_project_root()
        if _traces_base_env:
            configured = Path(_traces_base_env).expanduser()
            self._traces_dir = (
                configured
                if configured.is_absolute()
                else project_root / configured
            ).resolve()
        else:
            self._traces_dir = project_root / ".traces"
        self._traces_dir.mkdir(parents=True, exist_ok=True)

        self._active_stack: ContextVar[tuple[Span, ...]] = ContextVar(
            "mindforge_trace_stack",
            default=(),
        )
        self._export_lock = threading.Lock()
        self._langfuse = None
        self._last_cleanup = 0.0
        self._cleanup_old_traces()

        # Attempt to initialise LangFuse if the package is installed
        # and config is set via OBSERVABILITY_* env vars or .env.
        try:
            obs_cfg = self._config
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

        active_stack = self._active_stack.get()
        parent_id = active_stack[-1].span_id if active_stack else None

        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            name=name,
            start_time=time.time(),
            parent_id=parent_id,
            metadata=metadata or {},
        )

        token = self._active_stack.set((*active_stack, span))
        try:
            yield span
        except Exception as exc:
            span.error = self._redact_text(str(exc))
            raise
        finally:
            span.end_time = time.time()
            self._active_stack.reset(token)
            self._export(span)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _SECRET_KEY_PATTERN = re.compile(
        r"(api[_-]?key|authorization|password|passwd|secret|token|cookie)",
        re.IGNORECASE,
    )
    _SECRET_VALUE_PATTERN = re.compile(
        r"(?i)\b(?:sk|pk|ghp|github_pat|Bearer)[-_A-Za-z0-9.]{8,}"
    )

    def _redact_text(self, value: str) -> str:
        redacted = self._SECRET_VALUE_PATTERN.sub("[REDACTED]", value)
        return redacted[: self._config.max_record_chars]

    def _sanitize(self, value: Any, *, content: bool) -> Any:
        if not content:
            if value is None:
                return None
            if isinstance(value, str):
                return {
                    "redacted": True,
                    "type": "string",
                    "chars": len(value),
                }
            if isinstance(value, (bytes, bytearray)):
                return {
                    "redacted": True,
                    "type": "bytes",
                    "bytes": len(value),
                }
            if isinstance(value, dict):
                return {
                    "redacted": True,
                    "type": "object",
                    "keys": [str(key)[:100] for key in list(value)[:50]],
                    "size": len(value),
                }
            if isinstance(value, (list, tuple, set)):
                return {
                    "redacted": True,
                    "type": type(value).__name__,
                    "size": len(value),
                }
            return {
                "redacted": True,
                "type": type(value).__name__,
            }

        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, item in list(value.items())[:200]:
                key_text = str(key)[:200]
                sanitized[key_text] = (
                    "[REDACTED]"
                    if self._SECRET_KEY_PATTERN.search(key_text)
                    else self._sanitize(item, content=True)
                )
            return sanitized
        if isinstance(value, (list, tuple, set)):
            return [
                self._sanitize(item, content=True)
                for item in list(value)[:200]
            ]
        if isinstance(value, str):
            return self._redact_text(value)
        if isinstance(value, (bytes, bytearray)):
            return f"<{type(value).__name__}:{len(value)} bytes>"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return self._redact_text(str(value))

    def _cleanup_old_traces(self) -> None:
        now = time.time()
        if now - self._last_cleanup < 3600:
            return
        self._last_cleanup = now
        cutoff = now - self._config.trace_retention_days * 86400
        try:
            for path in self._traces_dir.glob("trace_*.jsonl"):
                try:
                    if path.is_file() and path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    continue
        except OSError:
            pass

    def _export(self, span: Span) -> None:
        """Dual-write *span* to local JSONL and (if configured) LangFuse."""
        self._cleanup_old_traces()
        capture = self._config.capture_content
        sanitized_input = self._sanitize(span.input, content=capture)
        sanitized_output = self._sanitize(span.output, content=capture)
        sanitized_metadata = self._sanitize(span.metadata, content=True)

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
            "error": self._redact_text(span.error) if span.error else None,
        }
        # Only include potentially large payloads when explicitly set.
        if span.input is not None:
            record["input"] = sanitized_input
        if span.output is not None:
            record["output"] = sanitized_output
        if span.metadata:
            record["metadata"] = sanitized_metadata

        serialized = json.dumps(
            record,
            ensure_ascii=False,
            default=str,
        )
        if len(serialized) > self._config.max_record_chars:
            record.pop("input", None)
            record.pop("output", None)
            record["payloads_omitted"] = "record size limit exceeded"
            serialized = json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            )

        with self._export_lock:
            try:
                current_size = (
                    trace_file.stat().st_size
                    if trace_file.exists()
                    else 0
                )
            except OSError:
                current_size = self._config.max_trace_file_bytes
            encoded = (serialized + "\n").encode("utf-8")
            if (
                current_size + len(encoded)
                > self._config.max_trace_file_bytes
            ):
                return
            with open(trace_file, "a", encoding="utf-8") as fh:
                fh.write(serialized + "\n")

        # -- LangFuse ----------------------------------------------------
        if self._langfuse is not None:
            try:
                self._langfuse.generation(
                    name=span.name,
                    trace_id=span.trace_id,
                    parent_observation_id=span.parent_id,
                    start_time=span.start_time,
                    end_time=span.end_time,
                    input=sanitized_input,
                    output=sanitized_output,
                    metadata=sanitized_metadata,
                    level="ERROR" if span.error else "DEFAULT",
                    status_message=span.error,
                )
            except Exception:
                pass  # LangFuse errors should never break the application


@lru_cache(maxsize=1)
def get_tracer() -> Tracer:
    """Return the application-wide singleton :class:`Tracer`."""
    return Tracer()
