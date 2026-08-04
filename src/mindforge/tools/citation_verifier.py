"""Citation verification tool — validates [N] markers against a source list."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from mindforge.tools.base import BaseTool, ToolResult


@dataclass
class CitationMarker:
    """A single [N] citation reference found in text."""

    index: int
    raw: str
    position: int  # character offset in the original text
    text_surrounding: str = ""


@dataclass
class VerificationIssue:
    """An issue found during citation verification."""

    marker: CitationMarker
    issue_type: str
    detail: str = ""


@dataclass
class VerificationSummary:
    """Full result of a citation verification pass."""

    total_markers: int = 0
    valid_markers: int = 0
    issues: list[VerificationIssue] = field(default_factory=list)
    unused_sources: list[int] = field(default_factory=list)
    sources_used: set[int] = field(default_factory=set)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0 or len(self.unused_sources) > 0

    @property
    def validity_score(self) -> float:
        """Fraction (0-1) of markers that are valid."""
        if self.total_markers == 0:
            return 1.0
        return self.valid_markers / self.total_markers

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_markers": self.total_markers,
            "valid_markers": self.valid_markers,
            "validity_score": self.validity_score,
            "has_issues": self.has_issues,
            "issues": [
                {
                    "marker": str(i.marker.raw),
                    "index": i.marker.index,
                    "type": i.issue_type,
                    "detail": i.detail,
                }
                for i in self.issues
            ],
            "unused_sources": self.unused_sources,
            "sources_used": sorted(self.sources_used),
        }


class CitationVerifier(BaseTool):
    """Verifies citation markers [N] in a report against the provided source list.

    Detects:
    - Markers pointing to non-existent source indices.
    - Markers pointing to empty or invalid sources.
    - Sources that are defined but never cited.
    """

    name = "verify_citation"
    description = (
        "Verify citation markers [N] in a report or text against a list of "
        "sources. Detects missing, out-of-range, or unused citations."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "report_text": {
                "type": "string",
                "description": "The report or text containing [N] citation markers.",
            },
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "Source index (1-based as in [N]).",
                        },
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["index"],
                },
                "description": "List of source definitions to validate against.",
            },
            "strict_unused": {
                "type": "boolean",
                "description": "If True, warn about sources that are defined but not cited.",
                "default": True,
            },
        },
        "required": ["report_text", "sources"],
    }

    MARKER_PATTERN = re.compile(r"\[(\d+)\]")
    WORD_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
    STOP_WORDS = frozenset(
        {
            "about",
            "also",
            "and",
            "are",
            "been",
            "being",
            "for",
            "from",
            "has",
            "have",
            "into",
            "its",
            "made",
            "that",
            "the",
            "their",
            "this",
            "was",
            "were",
            "with",
        }
    )

    def execute(
        self,
        report_text: str,
        sources: list[dict[str, Any]],
        strict_unused: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        start = time.perf_counter()

        if not report_text or not report_text.strip():
            return ToolResult(
                success=False,
                error="report_text must be a non-empty string.",
            )
        if not sources:
            return ToolResult(
                success=False,
                error="sources list must be non-empty.",
            )

        summary = self._verify(report_text, sources, strict_unused)

        elapsed = (time.perf_counter() - start) * 1000

        formatted = self._format_summary(summary, sources)
        return ToolResult(
            success=not summary.has_issues,
            output=formatted,
            data=summary.to_dict(),
            execution_time_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Verification logic
    # ------------------------------------------------------------------

    def _verify(
        self,
        report_text: str,
        sources: list[dict[str, Any]],
        strict_unused: bool,
    ) -> VerificationSummary:
        """Core verification pass."""
        # Build a lookup of source index -> source
        source_map: dict[int, dict[str, Any]] = {}
        for s in sources:
            idx = s.get("index")
            if isinstance(idx, int):
                source_map[idx] = s

        max_index = max(source_map.keys()) if source_map else 0
        summary = VerificationSummary()

        # 1. Find all [N] markers
        markers: list[CitationMarker] = []
        for match in self.MARKER_PATTERN.finditer(report_text):
            idx = int(match.group(1))
            start_pos = match.start()
            ctx_start = max(0, start_pos - 40)
            ctx_end = min(len(report_text), match.end() + 40)
            surrounding = report_text[ctx_start:ctx_end].replace("\n", " ")

            markers.append(
                CitationMarker(
                    index=idx,
                    raw=match.group(0),
                    position=start_pos,
                    text_surrounding=surrounding,
                )
            )

        summary.total_markers = len(markers)

        # 2. Validate each marker
        for marker in markers:
            if marker.index < 1:
                summary.issues.append(
                    VerificationIssue(
                        marker=marker,
                        issue_type="index_out_of_range",
                        detail=f"Citation index {marker.index} is less than 1.",
                    )
                )
            elif marker.index > max_index:
                summary.issues.append(
                    VerificationIssue(
                        marker=marker,
                        issue_type="index_out_of_range",
                        detail=(
                            f"Citation [{marker.index}] exceeds max source "
                            f"index ({max_index})."
                        ),
                    )
                )
            elif marker.index not in source_map:
                summary.issues.append(
                    VerificationIssue(
                        marker=marker,
                        issue_type="missing_source",
                        detail=f"No source defined for index {marker.index}.",
                    )
                )
            else:
                src = source_map[marker.index]
                title = src.get("title", "")
                url = src.get("url", "")
                content = src.get("content", "")
                if not title and not url and not content:
                    summary.issues.append(
                        VerificationIssue(
                            marker=marker,
                            issue_type="empty_source",
                            detail=(
                                f"Source [{marker.index}] exists but has no "
                                f"title, URL, or content."
                            ),
                        )
                    )
                elif not title and not content:
                    summary.issues.append(
                        VerificationIssue(
                            marker=marker,
                            issue_type="insufficient_source_content",
                            detail=(
                                f"Source [{marker.index}] has only a URL, so "
                                "the cited claim cannot be checked."
                            ),
                        )
                    )
                elif (
                    src.get("verification_mode") == "provider_native"
                    and (title or url)
                ):
                    summary.valid_markers += 1
                    summary.sources_used.add(marker.index)
                elif not self._source_supports_marker(
                    report_text,
                    marker,
                    src,
                ):
                    summary.issues.append(
                        VerificationIssue(
                            marker=marker,
                            issue_type="unsupported_claim",
                            detail=(
                                f"Source [{marker.index}] does not contain "
                                "enough lexical evidence for the cited claim."
                            ),
                        )
                    )
                else:
                    summary.valid_markers += 1
                    summary.sources_used.add(marker.index)

        # 3. Check for unused sources
        if strict_unused:
            defined_indices = set(source_map.keys())
            unused = sorted(defined_indices - summary.sources_used)
            summary.unused_sources = unused

        return summary

    def _source_supports_marker(
        self,
        report_text: str,
        marker: CitationMarker,
        source: dict[str, Any],
    ) -> bool:
        claim = self._claim_for_marker(report_text, marker)
        claim_terms = self._terms(claim)
        source_terms = self._terms(
            f"{source.get('title', '')} {source.get('content', '')}"
        )
        if not claim_terms or not source_terms:
            return False
        overlap = claim_terms & source_terms
        if len(claim_terms) <= 2:
            return claim_terms <= source_terms
        return len(overlap) >= 2 and (
            len(overlap) / len(claim_terms)
        ) >= 0.2

    def _claim_for_marker(
        self,
        report_text: str,
        marker: CitationMarker,
    ) -> str:
        sentence_start = max(
            report_text.rfind(delimiter, 0, marker.position)
            for delimiter in (".", "!", "?", "。", "！", "？", "\n")
        )
        following = [
            position
            for delimiter in (".", "!", "?", "。", "！", "？", "\n")
            if (position := report_text.find(
                delimiter,
                marker.position + len(marker.raw),
            ))
            != -1
        ]
        sentence_end = min(following) if following else len(report_text)
        claim = report_text[sentence_start + 1:sentence_end]
        return self.MARKER_PATTERN.sub("", claim)

    def _terms(self, text: str) -> set[str]:
        terms: set[str] = set()
        for token in self.WORD_PATTERN.findall(text.lower()):
            if "\u4e00" <= token[0] <= "\u9fff":
                if len(token) == 1:
                    terms.add(token)
                else:
                    terms.update(
                        token[index:index + 2]
                        for index in range(len(token) - 1)
                    )
            elif len(token) >= 3 and token not in self.STOP_WORDS:
                terms.add(token)
        return terms

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_summary(
        self,
        summary: VerificationSummary,
        sources: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = [
            "Citation Verification Report",
            "=" * 72,
            f"Total markers found:    {summary.total_markers}",
            f"Valid markers:          {summary.valid_markers}",
            f"Validity score:         {summary.validity_score:.1%}",
            f"Issues detected:        {len(summary.issues)}",
            f"Unused sources:         {len(summary.unused_sources)}",
        ]

        if summary.issues:
            lines.append("")
            lines.append("Issues:")
            lines.append("-" * 72)
            for i, issue in enumerate(summary.issues, 1):
                lines.append(
                    f"  {i}. [{issue.issue_type}] {issue.marker.raw} "
                    f"at position {issue.marker.position}"
                )
                lines.append(f"     Context: ...{issue.marker.text_surrounding}...")
                lines.append(f"     Detail:  {issue.detail}")
                lines.append("")

        if summary.unused_sources:
            lines.append("Unused Sources:")
            lines.append("-" * 72)
            for idx in summary.unused_sources:
                src = next(
                    (s for s in sources if s.get("index") == idx), {}
                )
                title = src.get("title", f"Source [{idx}]")
                lines.append(f"  [{idx}] {title}")

        return "\n".join(lines)
