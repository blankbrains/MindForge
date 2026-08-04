"""Tracing infrastructure for MindForge observability.

Provides Span dataclass and Tracer class for hierarchical trace capture,
with dual export to local JSONL files and optional LangFuse backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from mindforge.config import get_settings

logger = logging.getLogger(__name__)


def resolve_traces_dir() -> Path:
    """Resolve the configured local trace directory."""
    settings = get_settings()
    configured_value = settings.app.traces_dir
    from mindforge.config import get_project_root

    project_root = get_project_root()
    if configured_value:
        configured = Path(configured_value).expanduser()
        return (
            configured if configured.is_absolute() else project_root / configured
        ).resolve()
    return (project_root / ".traces").resolve()


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
        self._config = settings.observability
        self._traces_dir = resolve_traces_dir()
        self._traces_dir.mkdir(parents=True, exist_ok=True)

        self._active_stack: ContextVar[tuple[Span, ...]] = ContextVar(
            "mindforge_trace_stack",
            default=(),
        )
        self._export_lock = threading.Lock()
        self._trace_stats: dict[str, dict[str, int]] = {}
        self._langfuse = None
        self._last_cleanup = 0.0
        self._cleanup_old_traces()

        obs_cfg = self._config
        pub = obs_cfg.langfuse_public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        sec = obs_cfg.langfuse_secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        host = obs_cfg.langfuse_host or os.environ.get("LANGFUSE_HOST")
        if pub or sec:
            if not all((pub, sec, host)):
                logger.warning(
                    "Langfuse is only partially configured; remote tracing is disabled."
                )
            else:
                try:
                    from langfuse import Langfuse

                    self._langfuse = Langfuse(
                        public_key=pub,
                        secret_key=sec,
                        base_url=host,
                    )
                except ImportError:
                    logger.warning(
                        "Langfuse credentials are configured, but the SDK is "
                        "not installed. Reinstall the project dependencies."
                    )
                except Exception:
                    logger.exception("Langfuse initialization failed.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def traces_dir(self) -> Path:
        """Return the local trace storage directory."""
        return self._traces_dir

    @property
    def current_trace_id(self) -> str | None:
        """Return the trace ID active in the current async context."""
        active_stack = self._active_stack.get()
        return active_stack[-1].trace_id if active_stack else None

    @property
    def remote_enabled(self) -> bool:
        """Return whether the Langfuse exporter initialized successfully."""
        return self._langfuse is not None

    def get_trace_url(self, trace_id: str) -> str | None:
        """Return the remote Langfuse URL for *trace_id*, when available."""
        if self._langfuse is None:
            return None
        try:
            get_url = getattr(self._langfuse, "get_trace_url", None)
            if callable(get_url):
                value = get_url(trace_id=trace_id)
                return str(value) if value else None
        except Exception:
            logger.debug("Langfuse trace URL generation failed.", exc_info=True)
        return None

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
        active_stack = self._active_stack.get()
        trace_id = trace_id or (
            active_stack[-1].trace_id if active_stack else uuid.uuid4().hex
        )
        parent_id = active_stack[-1].span_id if active_stack else None

        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            name=name,
            start_time=time.time(),
            parent_id=parent_id,
            metadata=metadata or {},
        )

        remote_context = None
        remote_span = None
        if self._langfuse is not None:
            try:
                is_generation = name == "llm.chat"
                remote_kwargs: dict[str, Any] = {
                    "name": name,
                    "as_type": ("generation" if is_generation else "span"),
                    "metadata": self._sanitize(
                        span.metadata,
                        content=True,
                    ),
                }
                if is_generation and span.metadata.get("model"):
                    remote_kwargs["model"] = str(span.metadata["model"])
                if not active_stack:
                    remote_kwargs["trace_context"] = {
                        "trace_id": trace_id,
                    }
                remote_context = self._langfuse.start_as_current_observation(
                    **remote_kwargs,
                )
                remote_span = remote_context.__enter__()
            except Exception:
                logger.warning(
                    "Langfuse span creation failed; local tracing continues.",
                    exc_info=True,
                )
                remote_context = None
                remote_span = None

        token = self._active_stack.set((*active_stack, span))
        try:
            yield span
        except (asyncio.CancelledError, GeneratorExit) as exc:
            span.metadata.setdefault("status", "cancelled")
            span.metadata.setdefault("error_type", type(exc).__name__)
            span.metadata.setdefault("error_code", "operation_cancelled")
            if not span.error:
                span.error = "Operation cancelled before completion."
            raise
        except Exception as exc:
            span.metadata.setdefault("status", "error")
            span.metadata.setdefault("error_type", type(exc).__name__)
            span.metadata.setdefault(
                "error_code",
                "timeout"
                if isinstance(exc, (asyncio.TimeoutError, TimeoutError))
                else "operation_failed",
            )
            if not span.error:
                span.error = self.describe_exception(exc)
            raise
        finally:
            span.end_time = time.time()
            try:
                self._active_stack.reset(token)
            except ValueError as exc:
                # Observability must never break the research response. Async
                # generators may be finalized by a different consumer context
                # after disconnects or cancellation.
                logger.error(
                    "Trace context reset failed for span %s: %s",
                    span.span_id,
                    exc,
                )
                self._active_stack.set(active_stack)
                span.metadata["status"] = "error"
                span.metadata["error_type"] = type(exc).__name__
                span.metadata["error_code"] = "trace_context_mismatch"
                if not span.error:
                    span.error = (
                        "Tracing context ended in a different async context."
                    )
            if remote_span is not None and remote_context is not None:
                try:
                    update: dict[str, Any] = {
                        "input": self._sanitize(
                            span.input,
                            content=self._config.capture_content,
                        ),
                        "output": self._sanitize(
                            span.output,
                            content=self._config.capture_content,
                        ),
                        "metadata": {
                            **self._sanitize(
                                span.metadata,
                                content=True,
                            ),
                            "local_trace_id": span.trace_id,
                            "local_span_id": span.span_id,
                            "local_parent_id": span.parent_id,
                            "duration_ms": round(span.duration_ms, 3),
                        },
                        "level": "ERROR" if span.error else "DEFAULT",
                        "status_message": span.error,
                    }
                    if (
                        name == "llm.chat"
                        and isinstance(span.output, dict)
                        and isinstance(span.output.get("usage"), dict)
                    ):
                        update["usage_details"] = {
                            str(key): int(value)
                            for key, value in span.output["usage"].items()
                            if isinstance(value, (int, float))
                        }
                    remote_span.update(
                        **update,
                    )
                    if span.parent_id is None:
                        set_trace_io = getattr(
                            self._langfuse,
                            "set_current_trace_io",
                            None,
                        )
                        if callable(set_trace_io):
                            set_trace_io(
                                input=update["input"],
                                output=update["output"],
                            )
                except Exception:
                    logger.warning(
                        "Langfuse span update failed.",
                        exc_info=True,
                    )
                finally:
                    try:
                        remote_context.__exit__(None, None, None)
                    except Exception:
                        logger.warning(
                            "Langfuse span finalization failed.",
                            exc_info=True,
                        )
            self._export(span)

    def close(self) -> None:
        """Flush and stop the optional remote tracing client."""
        if self._langfuse is None:
            return
        try:
            flush = getattr(self._langfuse, "flush", None)
            if callable(flush):
                flush()
            shutdown = getattr(self._langfuse, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            logger.warning("Langfuse shutdown failed.", exc_info=True)

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

    def describe_exception(
        self,
        exc: BaseException,
        *,
        fallback: str | None = None,
    ) -> str:
        """Return a bounded, credential-redacted exception description."""
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return fallback or "Operation timed out."
        message = str(exc).strip()
        return self._redact_text(message or fallback or type(exc).__name__)

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
            return [self._sanitize(item, content=True) for item in list(value)[:200]]
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
        retention_days = self._config.trace_retention_days
        if retention_days <= 0:
            return
        cutoff = now - retention_days * 86400
        try:
            for pattern in ("trace_*.jsonl", "trace_*.summary.json"):
                for path in self._traces_dir.glob(pattern):
                    try:
                        if path.is_file() and path.stat().st_mtime < cutoff:
                            path.unlink()
                    except OSError:
                        continue
        except OSError:
            pass

    def _export(self, span: Span) -> None:
        """Write *span* to local JSONL and maintain a bounded trace summary."""
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
                current_size = trace_file.stat().st_size if trace_file.exists() else 0
            except OSError:
                current_size = self._config.max_trace_file_bytes
            encoded = (serialized + "\n").encode("utf-8")
            if current_size + len(encoded) > self._config.max_trace_file_bytes:
                return
            with open(trace_file, "a", encoding="utf-8") as fh:
                fh.write(serialized + "\n")
            stats = self._trace_stats.setdefault(
                span.trace_id,
                {
                    "span_count": 0,
                    "generation_count": 0,
                    "tool_count": 0,
                    "error_count": 0,
                    "total_tokens": 0,
                },
            )
            stats["span_count"] += 1
            if span.name == "llm.chat":
                stats["generation_count"] += 1
                usage = (
                    span.output.get("usage")
                    if isinstance(span.output, dict)
                    else None
                )
                if isinstance(usage, dict):
                    total_tokens = usage.get("total_tokens")
                    if not isinstance(total_tokens, (int, float)):
                        total_tokens = sum(
                            int(usage.get(key) or 0)
                            for key in ("prompt_tokens", "completion_tokens")
                        )
                    stats["total_tokens"] += int(total_tokens or 0)
            if span.name == "tool.execute":
                stats["tool_count"] += 1
            if span.error:
                stats["error_count"] += 1
            if span.parent_id is None:
                self._write_summary(
                    span,
                    sanitized_input=sanitized_input,
                    sanitized_output=sanitized_output,
                    sanitized_metadata=sanitized_metadata,
                    stats=stats,
                )
                self._trace_stats.pop(span.trace_id, None)

    def _write_summary(
        self,
        span: Span,
        *,
        sanitized_input: Any,
        sanitized_output: Any,
        sanitized_metadata: Any,
        stats: dict[str, int],
    ) -> None:
        """Atomically persist the root-level summary used by trace listing."""
        output = span.output if isinstance(span.output, dict) else {}
        cost_value = output.get("cost_usd")
        cost_usd = (
            float(cost_value)
            if isinstance(cost_value, (int, float))
            else None
        )
        success = output.get("success")
        status = str(span.metadata.get("status") or "").strip().lower()
        if status not in {
            "success",
            "warning",
            "degraded",
            "error",
            "cancelled",
        }:
            status = (
                "error"
                if span.error or success is False
                else "success"
            )
        task_preview = None
        if self._config.capture_content and isinstance(span.input, dict):
            task_value = span.input.get("task")
            if isinstance(task_value, str):
                task_preview = self._redact_text(task_value)[:300]
        summary = {
            "trace_id": span.trace_id,
            "name": span.name,
            "display_name": str(
                span.metadata.get("display_name") or ""
            ).strip()[:500]
            or None,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": round(span.duration_ms, 3),
            "status": status,
            "error": self._redact_text(span.error) if span.error else None,
            "task_preview": task_preview,
            "input": sanitized_input,
            "output": sanitized_output,
            "metadata": sanitized_metadata,
            "span_count": stats["span_count"],
            "generation_count": stats["generation_count"],
            "tool_count": stats["tool_count"],
            "error_count": stats["error_count"],
            "total_tokens": stats["total_tokens"],
            "cost_usd": cost_usd,
            "cost_status": str(
                output.get("cost_status") or "usage_unavailable"
            ),
            "remote_url": self.get_trace_url(span.trace_id),
        }
        summary_path = self._traces_dir / f"trace_{span.trace_id}.summary.json"
        temporary_path = self._traces_dir / (
            f".trace_{span.trace_id}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary_path.write_text(
                json.dumps(summary, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            temporary_path.replace(summary_path)
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


_tracer_instance: Tracer | None = None
_tracer_lock = threading.Lock()


def get_tracer() -> Tracer:
    """Return the application-wide singleton :class:`Tracer`."""
    global _tracer_instance
    with _tracer_lock:
        if _tracer_instance is None:
            _tracer_instance = Tracer()
        return _tracer_instance


def close_tracer() -> None:
    """Flush and release the tracer if it was initialized."""
    global _tracer_instance
    with _tracer_lock:
        tracer = _tracer_instance
        _tracer_instance = None
    if tracer is not None:
        tracer.close()
