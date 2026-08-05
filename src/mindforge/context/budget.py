"""Context token estimation and deterministic budget allocation."""

from __future__ import annotations

from mindforge.context.models import (
    ContextCandidate,
    ContextSelection,
)


def estimate_tokens(content: str, chars_per_token: int = 4) -> int:
    return max(1, (len(content) + chars_per_token - 1) // chars_per_token)


def allocate_budget(
    candidates: list[ContextCandidate],
    *,
    budget_tokens: int,
    chars_per_token: int,
    max_item_chars: int,
) -> tuple[list[ContextCandidate], list[ContextSelection], int]:
    selected: list[ContextCandidate] = []
    excluded: list[ContextSelection] = []
    used = 0
    seen_hashes: set[int] = set()

    for candidate in candidates:
        bounded = candidate.content.strip()[:max_item_chars]
        if not bounded:
            excluded.append(ContextSelection(candidate, "empty"))
            continue
        content_hash = hash(bounded)
        if content_hash in seen_hashes:
            excluded.append(ContextSelection(candidate, "duplicate"))
            continue
        candidate.content = bounded
        candidate.token_count = estimate_tokens(bounded, chars_per_token)
        if candidate.token_count > budget_tokens:
            max_chars = budget_tokens * chars_per_token
            if not candidate.explicitly_selected and not candidate.pinned:
                excluded.append(ContextSelection(candidate, "budget_exceeded"))
                continue
            candidate.content = bounded[:max_chars]
            candidate.token_count = estimate_tokens(
                candidate.content,
                chars_per_token,
            )
        if used + candidate.token_count > budget_tokens:
            excluded.append(ContextSelection(candidate, "budget_exceeded"))
            continue
        selected.append(candidate)
        seen_hashes.add(content_hash)
        used += candidate.token_count

    return selected, excluded, used
