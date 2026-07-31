"""Optional vision captioning for persisted document images."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mindforge.config import get_settings
from mindforge.ingestion.chunker import DocumentChunk
from mindforge.repositories.document_assets import (
    list_document_assets,
    update_asset_caption,
)
from mindforge.services.document_assets import resolve_asset_path

logger = logging.getLogger(__name__)
CancellationCallback = Callable[[], bool]


class VisualRetrievalConfigurationError(RuntimeError):
    """Raised when visual retrieval was enabled without usable credentials."""


def is_visual_retrieval_enabled() -> bool:
    config = get_settings().visual_retrieval
    return bool(config.enabled and config.model and config.api_key)


async def _caption_image(
    *,
    image_path: Path,
    content_type: str,
    client: Any,
) -> str:
    config = get_settings().visual_retrieval
    if not config.enabled:
        return ""
    if not config.model or not config.api_key:
        raise VisualRetrievalConfigurationError(
            "Visual retrieval is enabled but VISUAL_MODEL or VISUAL_API_KEY is missing."
        )
    if image_path.stat().st_size > config.max_asset_bytes:
        raise ValueError(
            f"Visual asset exceeds VISUAL_MAX_ASSET_BYTES: {image_path.name}"
        )

    encoded = await asyncio.to_thread(
        lambda: base64.b64encode(image_path.read_bytes()).decode("ascii")
    )
    completion = await client.chat.completions.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Describe this document image for retrieval. State visible "
                    "objects, chart/table type, labels, values, formulas, and "
                    "relationships when legible. Do not invent unreadable text."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Create a concise factual retrieval caption.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{content_type or 'image/png'};base64,"
                                f"{encoded}"
                            ),
                            "detail": config.detail,
                        },
                    },
                ],
            },
        ],
    )
    content = completion.choices[0].message.content
    return content.strip() if content else ""


async def build_visual_chunks(
    doc_id: str,
    *,
    document_metadata: dict[str, Any] | None = None,
    cancelled: CancellationCallback | None = None,
) -> list[DocumentChunk]:
    """Caption stored image assets and expose captions to existing text retrieval."""
    if not is_visual_retrieval_enabled():
        return []

    config = get_settings().visual_retrieval
    assets = await asyncio.to_thread(list_document_assets, doc_id)
    visual_assets = [
        asset
        for asset in assets
        if asset.get("kind") in {"image", "page_preview"}
        and asset.get("relative_path")
        and str(asset.get("content_type") or "").startswith("image/")
    ][: config.max_assets_per_document]

    if not visual_assets:
        return []

    uncached_assets = [
        asset
        for asset in visual_assets
        if not str((asset.get("metadata") or {}).get("caption") or "").strip()
    ]
    client = None
    if uncached_assets:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url or None,
            timeout=config.request_timeout_seconds,
        )
    semaphore = asyncio.Semaphore(config.caption_concurrency)

    async def build_chunk(asset: dict[str, Any]) -> DocumentChunk | None:
        if cancelled is not None and cancelled():
            raise asyncio.CancelledError("Visual captioning was cancelled.")
        metadata = dict(asset.get("metadata") or {})
        caption = str(metadata.get("caption") or "").strip()
        if not caption:
            try:
                if client is None:
                    raise VisualRetrievalConfigurationError(
                        "Visual caption client was not initialized."
                    )
                async with semaphore:
                    if cancelled is not None and cancelled():
                        raise asyncio.CancelledError(
                            "Visual captioning was cancelled."
                        )
                    caption = await _caption_image(
                        image_path=resolve_asset_path(
                            str(asset["relative_path"])
                        ),
                        content_type=str(
                            asset.get("content_type") or "image/png"
                        ),
                        client=client,
                    )
            except Exception:
                logger.warning(
                    "Visual captioning failed for asset %s.",
                    asset["asset_id"],
                    exc_info=True,
                )
                return None
            if not caption:
                return None
            await asyncio.to_thread(
                update_asset_caption,
                str(asset["asset_id"]),
                caption=caption,
                model=config.model,
                prompt_version=config.prompt_version,
            )
        if cancelled is not None and cancelled():
            raise asyncio.CancelledError("Visual captioning was cancelled.")

        page = asset.get("page")
        prefix = f"Visual asset on page {page}: " if page else "Visual asset: "
        content = prefix + caption
        asset_id = str(asset["asset_id"])
        return DocumentChunk(
            chunk_id=hashlib.md5(
                (
                    f"{doc_id}:visual:{asset_id}:"
                    f"{config.model}:{config.prompt_version}"
                ).encode()
            ).hexdigest()[:12],
            doc_id=doc_id,
            content=content,
            metadata={
                **(document_metadata or {}),
                "element_type": "image",
                "source_method": "vision_caption",
                "asset_id": asset_id,
                "asset_page": page,
                "asset_content_type": asset.get("content_type"),
                "visual_model": config.model,
                "visual_prompt_version": config.prompt_version,
                "chunk_start": None,
                "chunk_end": None,
            },
        )

    tasks = [
        asyncio.create_task(build_chunk(asset))
        for asset in visual_assets
    ]
    try:
        chunks = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        if client is not None:
            await client.close()
    return [chunk for chunk in chunks if chunk is not None]
