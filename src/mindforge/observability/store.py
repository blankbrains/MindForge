"""Read-only access to bounded local observability traces."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from mindforge.config import get_settings
from mindforge.observability.tracer import resolve_traces_dir

_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{16}(?:[0-9a-f]{16})?$")
_LEGACY_DUPLICATE_START_TOLERANCE_SECONDS = 0.001
_LEGACY_DUPLICATE_DURATION_TOLERANCE_MS = 5.0


def _public_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return ""
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = host
        if parsed.port is not None:
            netloc = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))
    except (TypeError, ValueError):
        return ""


class TraceRepository:
    """Expose local JSONL traces without coupling callers to Langfuse."""

    def __init__(self, traces_dir: Path | None = None) -> None:
        self._settings = get_settings()
        self._traces_dir = (traces_dir or resolve_traces_dir()).resolve()
        self._scan_limit = self._settings.observability.trace_list_scan_limit
        self._detail_span_limit = (
            self._settings.observability.trace_detail_span_limit
        )

    def status(self) -> dict[str, Any]:
        config = self._settings.observability
        public_host = _public_url(config.langfuse_host or "")
        remote_configured = bool(
            config.langfuse_public_key
            and config.langfuse_secret_key
            and public_host
        )
        return {
            "enabled": config.enable_tracing,
            "local_storage": True,
            "remote_configured": remote_configured,
            "langfuse_host": public_host if remote_configured else None,
            "capture_content": config.capture_content,
            "retention_days": config.trace_retention_days,
        }

    def list_traces(
        self,
        *,
        limit: int,
        offset: int,
        status: str | None = None,
        search: str = "",
    ) -> dict[str, Any]:
        candidates = self._summary_candidates()
        truncated = len(candidates) > self._scan_limit
        candidates = candidates[: self._scan_limit]
        normalized_search = search.strip().casefold()
        summaries: list[dict[str, Any]] = []
        for path in candidates:
            summary = self._read_summary(path)
            if summary is None:
                continue
            summaries.append(summary)
        summaries = self._collapse_legacy_duplicate_roots(summaries)

        filtered_summaries: list[dict[str, Any]] = []
        for summary in summaries:
            if status and summary.get("status") != status:
                continue
            if normalized_search:
                searchable = " ".join(
                    str(summary.get(key) or "")
                    for key in (
                        "trace_id",
                        "name",
                        "display_name",
                        "task_preview",
                    )
                ).casefold()
                if normalized_search not in searchable:
                    continue
            filtered_summaries.append(summary)
        total = len(filtered_summaries)
        return {
            "traces": filtered_summaries[offset : offset + limit],
            "total": total,
            "limit": limit,
            "offset": offset,
            "truncated": truncated,
        }

    @classmethod
    def _collapse_legacy_duplicate_roots(
        cls,
        summaries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Hide duplicate root traces emitted by the legacy SSE path.

        Older deployments opened one root trace in the API layer and another
        in the Orchestrator. Those roots start and finish effectively
        simultaneously and have the same research label. Keep the richer
        trace while preserving distinct repeated research submissions.
        """
        collapsed: list[dict[str, Any]] = []
        candidate_indexes: dict[tuple[str, str], list[int]] = {}

        for summary in summaries:
            key = cls._legacy_duplicate_key(summary)
            if key is None:
                collapsed.append(summary)
                continue

            duplicate_index = next(
                (
                    index
                    for index in candidate_indexes.get(key, [])
                    if cls._is_legacy_duplicate_root(
                        collapsed[index],
                        summary,
                    )
                ),
                None,
            )
            if duplicate_index is None:
                candidate_indexes.setdefault(key, []).append(len(collapsed))
                collapsed.append(summary)
                continue

            if cls._trace_richness(summary) > cls._trace_richness(
                collapsed[duplicate_index]
            ):
                collapsed[duplicate_index] = summary

        return collapsed

    @staticmethod
    def _legacy_duplicate_key(
        summary: dict[str, Any],
    ) -> tuple[str, str] | None:
        name = str(summary.get("name") or "").strip()
        if name != "orchestrator.research":
            return None
        label = str(
            summary.get("display_name")
            or summary.get("task_preview")
            or ""
        ).strip().casefold()
        if not label:
            return None
        return name, label

    @staticmethod
    def _is_legacy_duplicate_root(
        first: dict[str, Any],
        second: dict[str, Any],
    ) -> bool:
        try:
            start_delta = abs(
                float(first.get("start_time") or 0.0)
                - float(second.get("start_time") or 0.0)
            )
            duration_delta = abs(
                float(first.get("duration_ms") or 0.0)
                - float(second.get("duration_ms") or 0.0)
            )
        except (TypeError, ValueError):
            return False
        return (
            start_delta <= _LEGACY_DUPLICATE_START_TOLERANCE_SECONDS
            and duration_delta <= _LEGACY_DUPLICATE_DURATION_TOLERANCE_MS
        )

    @staticmethod
    def _trace_richness(summary: dict[str, Any]) -> tuple[int, ...]:
        def count(key: str) -> int:
            value = summary.get(key)
            return int(value) if isinstance(value, (int, float)) else 0

        return (
            count("span_count"),
            count("generation_count"),
            count("tool_count"),
            count("total_tokens"),
            count("error_count"),
            int(summary.get("cost_usd") is not None),
        )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        if not _TRACE_ID_PATTERN.fullmatch(trace_id):
            return None
        trace_path = self._traces_dir / f"trace_{trace_id}.jsonl"
        if not trace_path.is_file():
            return None
        observations: list[dict[str, Any]] = []
        try:
            with trace_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if len(observations) >= self._detail_span_limit:
                        break
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if (
                        isinstance(record, dict)
                        and record.get("trace_id") == trace_id
                    ):
                        observations.append(record)
        except OSError:
            return None
        observations = self._order_observations(observations)
        summary_path = self._traces_dir / f"trace_{trace_id}.summary.json"
        summary = self._read_summary(summary_path)
        if summary is None:
            summary = self._derive_summary(trace_id, observations)
        failures = self._collect_failures(observations)
        summary = {
            **summary,
            "failure_count": len(failures),
            "failure_summary": self._summarize_failures(failures),
        }
        return {
            "summary": summary,
            "observations": observations,
            "failures": failures,
            "observations_truncated": (
                len(observations) >= self._detail_span_limit
            ),
        }

    def _collect_failures(
        self,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        cancelled_parent_ids = {
            str(observation.get("parent_id"))
            for observation in observations
            if observation.get("parent_id")
            and (
                str(
                    (
                        observation.get("metadata")
                        if isinstance(observation.get("metadata"), dict)
                        else {}
                    ).get("status")
                    or ""
                ).strip().lower()
                == "cancelled"
                or "cancel" in str(observation.get("error") or "").casefold()
            )
        }
        max_llm_attempt_by_parent: dict[str, int] = {}
        for observation in observations:
            if observation.get("name") != "llm.chat":
                continue
            parent_id = str(observation.get("parent_id") or "")
            metadata = (
                observation.get("metadata")
                if isinstance(observation.get("metadata"), dict)
                else {}
            )
            attempt = metadata.get("attempt")
            if parent_id and isinstance(attempt, (int, float)):
                max_llm_attempt_by_parent[parent_id] = max(
                    max_llm_attempt_by_parent.get(parent_id, 0),
                    int(attempt),
                )
        try:
            subtask_timeout = float(
                getattr(
                    getattr(self._settings, "agent", None),
                    "subtask_timeout",
                    60,
                )
            )
        except (TypeError, ValueError):
            subtask_timeout = 60.0
        try:
            llm_request_timeout = float(
                getattr(
                    getattr(self._settings, "agent", None),
                    "llm_request_timeout",
                    45,
                )
            )
        except (TypeError, ValueError):
            llm_request_timeout = 45.0
        for observation in observations:
            metadata = (
                dict(observation.get("metadata"))
                if isinstance(observation.get("metadata"), dict)
                else {}
            )
            raw_error = str(observation.get("error") or "").strip()
            status = str(metadata.get("status") or "").strip().lower()
            name = str(observation.get("name") or "operation")
            span_id = str(observation.get("span_id") or "")
            try:
                duration_ms = float(observation.get("duration_ms") or 0.0)
            except (TypeError, ValueError):
                duration_ms = 0.0
            inferred_subtask_timeout = (
                name == "agent.researcher"
                and span_id in cancelled_parent_ids
                and observation.get("output") is None
                and duration_ms + 250
                >= subtask_timeout * 1000
            )
            attempt_value = metadata.get("attempt")
            attempt_number = (
                int(attempt_value)
                if isinstance(attempt_value, (int, float))
                else None
            )
            inferred_llm_timeout = (
                name == "llm.chat"
                and observation.get("output") is None
                and attempt_number is not None
                and attempt_number
                < max_llm_attempt_by_parent.get(
                    str(observation.get("parent_id") or ""),
                    0,
                )
                and duration_ms + 250 >= llm_request_timeout * 1000
            )
            if inferred_subtask_timeout:
                metadata.update(
                    {
                        "status": "error",
                        "stage": "subtask_execution",
                        "error_code": "subtask_timeout",
                        "error_type": "TimeoutError",
                        "timeout_seconds": subtask_timeout,
                    }
                )
                raw_error = (
                    "Subtask exceeded the configured execution timeout."
                )
                status = "error"
            elif inferred_llm_timeout:
                metadata.update(
                    {
                        "status": "error",
                        "stage": "llm_request",
                        "error_code": "llm_request_timeout",
                        "error_type": "TimeoutError",
                        "timeout_seconds": llm_request_timeout,
                    }
                )
                raw_error = "LLM request exceeded the configured timeout."
                status = "error"
            elif not raw_error and status not in {"error", "cancelled"}:
                continue

            error_code = str(metadata.get("error_code") or "").strip()
            error_type = str(metadata.get("error_type") or "").strip()
            stage = str(metadata.get("stage") or "").strip()

            if "created in a different Context" in raw_error:
                error_code = "trace_context_mismatch"
                error_type = error_type or "ValueError"
                stage = "observability"
            elif not error_code and name == "llm.chat" and status == "cancelled":
                error_code = "llm_request_cancelled"
            elif not error_code and "timed out" in raw_error.casefold():
                error_code = "timeout"
            elif not error_code and status == "cancelled":
                error_code = "operation_cancelled"
            elif not error_code:
                error_code = "operation_failed"

            if not error_type:
                error_type = (
                    "CancelledError"
                    if status == "cancelled"
                    else "RuntimeError"
                )
            if not stage:
                stage = {
                    "orchestrator.research": "research",
                    "agent.planner": "planning",
                    "agent.researcher": "subtask_execution",
                    "agent.synthesizer": "synthesis",
                    "agent.critic": "quality_review",
                    "llm.chat": "llm_request",
                    "tool.execute": "tool_execution",
                }.get(name, "operation")

            attempt_value = metadata.get("attempt")
            attempt = (
                int(attempt_value)
                if isinstance(attempt_value, (int, float))
                else None
            )
            failure = {
                "span_id": str(observation.get("span_id") or ""),
                "parent_id": observation.get("parent_id"),
                "observation_name": name,
                "stage": stage,
                "error_code": error_code,
                "error_type": error_type,
                "message": self._failure_message(
                    error_code=error_code,
                    raw_error=raw_error,
                    metadata=metadata,
                    attempt=attempt,
                ),
                "status": status or "error",
                "agent": (
                    str(metadata["agent"])
                    if metadata.get("agent") is not None
                    else None
                ),
                "model": (
                    str(metadata["model"])
                    if metadata.get("model") is not None
                    else None
                ),
                "attempt": attempt,
            }
            failures.append(failure)
        return failures

    @staticmethod
    def _failure_message(
        *,
        error_code: str,
        raw_error: str,
        metadata: dict[str, Any],
        attempt: int | None,
    ) -> str:
        timeout = metadata.get("timeout_seconds")
        timeout_text = (
            f"{float(timeout):g} 秒"
            if isinstance(timeout, (int, float))
            else "配置的时限"
        )
        agent = str(metadata.get("agent") or "").strip()
        model = str(metadata.get("model") or "").strip()
        if error_code == "trace_context_mismatch":
            return (
                "可观测追踪上下文跨异步任务结束，这是系统内部错误，"
                "可能覆盖原本的业务失败原因。"
            )
        if error_code == "subtask_timeout":
            subtask_id = str(metadata.get("subtask_id") or "").strip()
            label = f" {subtask_id} " if subtask_id else " "
            return f"Researcher 子任务{label}执行超过 {timeout_text}。"
        if error_code == "llm_request_timeout":
            details = [
                value
                for value in (
                    f"Agent: {agent}" if agent else "",
                    f"模型: {model}" if model else "",
                    f"第 {attempt} 次尝试" if attempt else "",
                )
                if value
            ]
            suffix = f"（{'，'.join(details)}）" if details else ""
            return f"LLM 请求在 {timeout_text} 后超时{suffix}。"
        if error_code == "llm_request_cancelled":
            return (
                "LLM 请求在完成前被取消，通常由上层子任务超时、"
                "研究取消或客户端断开触发。"
            )
        if error_code in {"subtask_cancelled", "operation_cancelled"}:
            return "该执行节点在完成前被取消。"
        return raw_error or str(metadata.get("error_message") or error_code)

    @staticmethod
    def _summarize_failures(
        failures: list[dict[str, Any]],
    ) -> str | None:
        if not failures:
            return None
        consequential_codes = {
            "llm_request_cancelled",
            "subtask_cancelled",
            "operation_cancelled",
        }
        primary = [
            failure
            for failure in failures
            if failure["observation_name"] != "orchestrator.research"
            and failure["error_code"] not in consequential_codes
        ]
        if not primary:
            primary = [
                failure
                for failure in failures
                if failure["error_code"] not in consequential_codes
            ]
        if not primary:
            primary = failures

        messages: list[str] = []
        for failure in primary:
            message = str(failure.get("message") or "").strip()
            if message and message not in messages:
                messages.append(message)
            if len(messages) >= 3:
                break
        return (
            f"检测到 {len(failures)} 个异常节点。主要原因："
            + "；".join(messages)
        )

    def delete_trace(self, trace_id: str) -> bool:
        if not _TRACE_ID_PATTERN.fullmatch(trace_id):
            return False
        deleted = False
        for suffix in (".jsonl", ".summary.json"):
            path = self._traces_dir / f"trace_{trace_id}{suffix}"
            try:
                if path.is_file():
                    path.unlink()
                    deleted = True
            except OSError:
                continue
        return deleted

    def clear_traces(self) -> int:
        deleted_ids: set[str] = set()
        try:
            candidates = [
                *self._traces_dir.glob("trace_*.jsonl"),
                *self._traces_dir.glob("trace_*.summary.json"),
            ]
        except OSError:
            return 0
        for path in candidates:
            match = re.fullmatch(
                r"trace_([0-9a-f]{16}(?:[0-9a-f]{16})?)"
                r"(?:\.jsonl|\.summary\.json)",
                path.name,
            )
            if match is None:
                continue
            try:
                if path.is_file():
                    path.unlink()
                    deleted_ids.add(match.group(1))
            except OSError:
                continue
        return len(deleted_ids)

    @staticmethod
    def _order_observations(
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return a stable parent-before-child observation order."""
        def sort_key(item: dict[str, Any]) -> tuple[float, str]:
            return (
                float(item.get("start_time") or 0.0),
                str(item.get("span_id") or ""),
            )
        by_id = {
            str(item.get("span_id")): item
            for item in observations
            if item.get("span_id")
        }
        children: dict[str, list[dict[str, Any]]] = {}
        roots: list[dict[str, Any]] = []
        for item in observations:
            parent_id = item.get("parent_id")
            if isinstance(parent_id, str) and parent_id in by_id:
                children.setdefault(parent_id, []).append(item)
            else:
                roots.append(item)
        for values in children.values():
            values.sort(key=sort_key)
        roots.sort(key=sort_key)

        ordered: list[dict[str, Any]] = []
        visited: set[str] = set()

        def visit(item: dict[str, Any]) -> None:
            span_id = str(item.get("span_id") or "")
            if not span_id or span_id in visited:
                return
            visited.add(span_id)
            ordered.append(item)
            for child in children.get(span_id, []):
                visit(child)

        for root in roots:
            visit(root)
        for item in sorted(observations, key=sort_key):
            visit(item)
        return ordered

    def _summary_candidates(self) -> list[Path]:
        try:
            summaries = list(self._traces_dir.glob("trace_*.summary.json"))
            summaries.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            if summaries:
                return summaries
            traces = list(self._traces_dir.glob("trace_*.jsonl"))
            traces.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return traces
        except OSError:
            return []

    def _read_summary(self, path: Path) -> dict[str, Any] | None:
        if path.suffix == ".jsonl":
            trace_id = path.name.removeprefix("trace_").removesuffix(".jsonl")
            detail = self.get_trace(trace_id)
            return detail["summary"] if detail else None
        try:
            if path.stat().st_size > 256 * 1024:
                return None
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return None
        if not isinstance(value, dict):
            return None
        trace_id = value.get("trace_id")
        if not isinstance(trace_id, str) or not _TRACE_ID_PATTERN.fullmatch(trace_id):
            return None
        return value

    @staticmethod
    def _derive_summary(
        trace_id: str,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        root = next(
            (
                item
                for item in reversed(observations)
                if item.get("parent_id") is None
            ),
            observations[-1] if observations else {},
        )
        output = root.get("output")
        success = output.get("success") if isinstance(output, dict) else None
        metadata = root.get("metadata") or {}
        configured_status = str(metadata.get("status") or "").strip().lower()
        if configured_status not in {
            "success",
            "degraded",
            "error",
            "cancelled",
        }:
            configured_status = (
                "error"
                if root.get("error") or success is False
                else "success"
            )
        return {
            "trace_id": trace_id,
            "name": str(root.get("name") or "trace"),
            "display_name": str(metadata.get("display_name") or "").strip()
            or None,
            "start_time": float(root.get("start_time") or 0.0),
            "end_time": root.get("end_time"),
            "duration_ms": float(root.get("duration_ms") or 0.0),
            "status": configured_status,
            "error": root.get("error"),
            "failure_summary": None,
            "failure_count": 0,
            "task_preview": None,
            "input": root.get("input"),
            "output": output,
            "metadata": metadata,
            "span_count": len(observations),
            "generation_count": sum(
                item.get("name") == "llm.chat" for item in observations
            ),
            "tool_count": sum(
                item.get("name") == "tool.execute" for item in observations
            ),
            "error_count": sum(bool(item.get("error")) for item in observations),
            "total_tokens": 0,
            "cost_usd": (
                float(output["cost_usd"])
                if isinstance(output, dict)
                and isinstance(output.get("cost_usd"), (int, float))
                else None
            ),
            "cost_status": (
                str(output.get("cost_status") or "usage_unavailable")
                if isinstance(output, dict)
                else "usage_unavailable"
            ),
            "remote_url": None,
        }
