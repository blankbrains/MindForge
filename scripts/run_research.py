#!/usr/bin/env python3
"""Run one MindForge research task from the command line."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a research task through the MindForge pipeline.",
    )
    parser.add_argument(
        "task",
        nargs="+",
        help="Research question or task.",
    )
    return parser.parse_args()


async def run(task: str) -> int:
    from mindforge.api.routes import get_orchestrator
    from mindforge.models.base import has_llm_credentials

    if not has_llm_credentials():
        print(
            "No usable LLM provider is configured. Update .env or the "
            "application settings first.",
            file=sys.stderr,
        )
        return 2

    orchestrator = await asyncio.to_thread(get_orchestrator)
    result = await orchestrator.run(task)
    if result.output:
        print(result.output)
    if not result.success:
        return 1

    sources = result.data.get("sources", [])
    print(
        "\n"
        f"Sources: {len(sources) if isinstance(sources, list) else 0} | "
        f"Cost status: {result.cost_status} | "
        f"Cost: {result.cost_usd if result.cost_usd is not None else 'N/A'}",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run(" ".join(args.task).strip()))


if __name__ == "__main__":
    raise SystemExit(main())
