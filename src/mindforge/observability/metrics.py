"""Metrics collection for MindForge observability.

Tracks token usage, tool calls, and agent latency throughout a session
and provides aggregated summaries suitable for cost analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Known per-model pricing in USD per 1M input / 1M output tokens.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}


@dataclass
class MetricPoint:
    """A single data point collected by the metrics system."""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Aggregator for session-level metrics.

    Usage::

        mc = MetricsCollector()
        mc.record_token_usage(100, 50, "gpt-4o")
        mc.record_tool_call("web_search", True, 320)
        summary = mc.get_session_summary()
    """

    def __init__(self) -> None:
        self._points: list[MetricPoint] = []
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_token_usage(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        """Record a token-usage event and its estimated cost."""
        cost = self._estimate_cost(prompt_tokens, completion_tokens, model)
        self._points.extend(
            [
                MetricPoint(
                    name="token_usage.prompt",
                    value=prompt_tokens,
                    labels={"model": model},
                ),
                MetricPoint(
                    name="token_usage.completion",
                    value=completion_tokens,
                    labels={"model": model},
                ),
                MetricPoint(
                    name="token_usage.total",
                    value=prompt_tokens + completion_tokens,
                    labels={"model": model},
                ),
                MetricPoint(
                    name="cost.usd",
                    value=cost,
                    labels={"model": model},
                ),
            ]
        )

    def record_tool_call(self, tool_name: str, success: bool, latency_ms: float) -> None:
        """Record a tool invocation outcome."""
        self._points.append(
            MetricPoint(
                name="tool_call",
                value=1.0,
                labels={"tool": tool_name, "success": str(success).lower()},
            )
        )
        self._points.append(
            MetricPoint(
                name="tool_latency_ms",
                value=latency_ms,
                labels={"tool": tool_name},
            )
        )

    def record_agent_latency(self, agent_name: str, latency_ms: float) -> None:
        """Record the execution latency of an agent step."""
        self._points.append(
            MetricPoint(
                name="agent_latency_ms",
                value=latency_ms,
                labels={"agent": agent_name},
            )
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get_session_summary(self) -> dict[str, Any]:
        """Return aggregated statistics for the current session."""
        total_tokens_prompt = sum(
            p.value for p in self._points if p.name == "token_usage.prompt"
        )
        total_tokens_completion = sum(
            p.value for p in self._points if p.name == "token_usage.completion"
        )
        total_cost = sum(p.value for p in self._points if p.name == "cost.usd")
        total_tool_calls = sum(p.value for p in self._points if p.name == "tool_call")
        tool_latencies = [
            p.value for p in self._points if p.name == "tool_latency_ms"
        ]
        agent_latencies = [
            p.value for p in self._points if p.name == "agent_latency_ms"
        ]

        return {
            "session_duration_s": round(time.time() - self._start_time, 3),
            "total_tokens_prompt": int(total_tokens_prompt),
            "total_tokens_completion": int(total_tokens_completion),
            "total_tokens": int(total_tokens_prompt + total_tokens_completion),
            "total_cost_usd": round(total_cost, 6),
            "total_tool_calls": int(total_tool_calls),
            "avg_tool_latency_ms": round(
                sum(tool_latencies) / len(tool_latencies), 2
            ) if tool_latencies else 0.0,
            "avg_agent_latency_ms": round(
                sum(agent_latencies) / len(agent_latencies), 2
            ) if agent_latencies else 0.0,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, path: str | Path) -> None:
        """Persist all collected metric points as JSON."""
        data = {
            "start_time": self._start_time,
            "summary": self.get_session_summary(),
            "points": [
                {
                    "name": p.name,
                    "value": p.value,
                    "timestamp": p.timestamp,
                    "labels": p.labels,
                }
                for p in self._points
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Estimate cost in USD for the given token counts and model.

        Falls back to the average of known models when *model* is not
        recognised.
        """
        pricing = _MODEL_PRICING.get(model)
        if pricing is None:
            # Average of all known rates as a safe fallback.
            all_rates = list(_MODEL_PRICING.values())
            avg_input = sum(p[0] for p in all_rates) / len(all_rates)
            avg_output = sum(p[1] for p in all_rates) / len(all_rates)
            pricing = (avg_input, avg_output)

        input_cost = (prompt_tokens / 1_000_000) * pricing[0]
        output_cost = (completion_tokens / 1_000_000) * pricing[1]
        return round(input_cost + output_cost, 8)
