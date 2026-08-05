"""Safe extraction of reusable research artifacts."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

_VOLATILE_MARKERS = (
    "今天",
    "当前",
    "最新",
    "实时",
    "价格",
    "天气",
    "比分",
    "股价",
)
_TIME_SENSITIVE_MARKERS = (
    "版本",
    "政策",
    "法规",
    "标准",
    "市场",
    "趋势",
    "排名",
)


def classify_freshness(text: str) -> tuple[str, datetime | None]:
    now = datetime.now(timezone.utc)
    lowered = text.lower()
    if any(marker in lowered for marker in _VOLATILE_MARKERS):
        return "volatile", now + timedelta(days=1)
    if any(marker in lowered for marker in _TIME_SENSITIVE_MARKERS):
        return "time_sensitive", now + timedelta(days=30)
    return "stable", None


def extract_artifacts(
    *,
    task: str,
    result: Any,
) -> list[dict[str, Any]]:
    """Extract grounded subtask findings and bounded report sections."""
    success = bool(
        result.success if hasattr(result, "success") else result.get("success")
    )
    if not success:
        return []
    data = result.data if hasattr(result, "data") else dict(result.get("data") or {})
    metadata = (
        result.metadata
        if hasattr(result, "metadata")
        else dict(result.get("metadata") or {})
    )
    quality = metadata.get("quality")
    quality_score = (
        float(quality) if isinstance(quality, (int, float)) else None
    )
    artifacts: list[dict[str, Any]] = []
    for item in data.get("subtask_outputs") or []:
        if not isinstance(item, dict) or not item.get("success"):
            continue
        grounding = str(item.get("grounding_status") or "")
        sources = item.get("sources") or []
        if grounding not in {"grounded", "not_required"} or not sources:
            continue
        content = str(item.get("output") or "").strip()
        if not content:
            continue
        title = str(item.get("description") or item.get("task_id") or "研究发现")
        freshness, expires_at = classify_freshness(f"{task}\n{title}\n{content}")
        artifacts.append(
            {
                "artifact_type": "subtask_finding",
                "title": title,
                "content": content[:50_000],
                "source_ids": _source_ids(sources),
                "quality_score": quality_score,
                "grounding_status": grounding,
                "freshness_class": freshness,
                "expires_at": expires_at,
                "metadata": {"task_id": item.get("task_id")},
            }
        )

    report = str(
        result.output if hasattr(result, "output") else result.get("output") or ""
    )
    sources = data.get("sources") or []
    grounding = str(metadata.get("grounding_status") or "")
    if report.strip() and sources and grounding in {"grounded", "not_required"}:
        sections = _report_sections(report)
        for title, content in sections[:12]:
            freshness, expires_at = classify_freshness(
                f"{task}\n{title}\n{content}"
            )
            artifacts.append(
                {
                    "artifact_type": "report_section",
                    "title": title,
                    "content": content[:30_000],
                    "source_ids": _source_ids(sources),
                    "quality_score": quality_score,
                    "grounding_status": grounding,
                    "freshness_class": freshness,
                    "expires_at": expires_at,
                    "metadata": {},
                }
            )
    return artifacts[:20]


def _report_sections(report: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^(#{1,3})\s+(.{2,200})\s*$", report))
    if not matches:
        return [("完整研究结论", report[:30_000])]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
        content = report[start:end].strip()
        if content:
            sections.append((match.group(2).strip(), content))
    return sections


def _source_ids(sources: list[Any]) -> list[str]:
    values: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        identity = (
            source.get("chunk_id")
            or source.get("doc_id")
            or source.get("url")
            or source.get("id")
        )
        if identity:
            values.append(str(identity)[:500])
    return list(dict.fromkeys(values))[:100]
