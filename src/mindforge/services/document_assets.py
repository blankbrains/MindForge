"""Persistent visual and structural document-asset lifecycle."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mindforge.config import get_settings, resolve_project_path
from mindforge.repositories.document_assets import (
    delete_document_asset_records,
    replace_document_assets,
)

logger = logging.getLogger(__name__)


class DocumentAssetError(RuntimeError):
    """Raised when document assets cannot be persisted safely."""


class DocumentAssetCancelledError(DocumentAssetError):
    """Raised when a document asset task is cooperatively cancelled."""


CancellationCallback = Callable[[], bool]


def get_asset_root() -> Path:
    """Return the configured data-volume root for all document assets."""
    root = resolve_project_path(get_settings().parser.asset_storage_dir)
    return root.resolve()


def _safe_doc_directory(doc_id: str) -> Path:
    if not doc_id or any(char not in "0123456789abcdef" for char in doc_id):
        raise DocumentAssetError("Document id cannot be used as an asset path.")
    return get_asset_root() / doc_id


def _ensure_within_root(path: Path) -> Path:
    root = get_asset_root().resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DocumentAssetError(
            f"Asset operation escaped configured root: {resolved}"
        ) from exc
    return resolved


def resolve_asset_path(relative_path: str) -> Path:
    """Resolve a database path only when it remains under the asset root."""
    if not relative_path:
        raise DocumentAssetError("Asset has no persisted file.")
    return _ensure_within_root(get_asset_root() / relative_path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(get_asset_root().resolve())).replace(
        os.sep,
        "/",
    )


def _safe_filename(name: str) -> str:
    result = "".join(
        char if char.isalnum() or char in {".", "-", "_"} else "_"
        for char in name
    ).strip("._")
    return result[:160] or "source"


def _append_file_asset(
    manifest: list[dict[str, Any]],
    *,
    doc_id: str,
    kind: str,
    path: Path,
    page: int | None,
    element_index: int | None,
    content_type: str,
    width: int | None = None,
    height: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    asset = {
        "asset_id": uuid.uuid4().hex,
        "doc_id": doc_id,
        "kind": kind,
        "page": page,
        "element_index": element_index,
        "relative_path": _relative_path(path),
        "content_type": content_type,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "metadata": dict(metadata or {}),
    }
    manifest.append(asset)
    return asset


def _save_crop(
    page: Any,
    bbox: tuple[float, float, float, float] | None,
    target: Path,
    *,
    resolution: int,
) -> tuple[int, int] | None:
    image_page = page
    if bbox is not None:
        x0, top, x1, bottom = bbox
        if x1 <= x0 or bottom <= top:
            return None
        image_page = page.crop((x0, top, x1, bottom), strict=False)
    image = image_page.to_image(resolution=resolution).original
    width, height = image.size
    if width < 8 or height < 8:
        return None
    image.save(target, format="PNG", optimize=True)
    return width, height


def _remove_directory(path: Path) -> None:
    target = _ensure_within_root(path)
    if target.exists():
        shutil.rmtree(target)


def _raise_if_cancelled(cancelled: CancellationCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise DocumentAssetCancelledError(
            "Document asset persistence was cancelled."
        )


def _copy_with_cancel(
    source: Path,
    target: Path,
    *,
    cancelled: CancellationCallback | None,
) -> None:
    with source.open("rb") as input_file, target.open("wb") as output_file:
        while True:
            _raise_if_cancelled(cancelled)
            block = input_file.read(1024 * 1024)
            if not block:
                break
            output_file.write(block)
    shutil.copystat(source, target)


def persist_document_assets(
    *,
    source_path: str | Path,
    parsed: Any,
    cancelled: CancellationCallback | None = None,
) -> list[dict[str, Any]]:
    """Persist the original file, visual crops, and structured table manifest."""
    config = get_settings().parser
    if not config.asset_persistence_enabled:
        return []

    source = Path(source_path).resolve()
    if not source.is_file():
        raise DocumentAssetError("Source file is unavailable for asset persistence.")

    doc_id = str(parsed.doc_id)
    final_directory = _safe_doc_directory(doc_id)
    root = get_asset_root()
    root.mkdir(parents=True, exist_ok=True)
    staging = root / ".staging" / f"{doc_id}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    manifest: list[dict[str, Any]] = []

    try:
        _raise_if_cancelled(cancelled)
        if config.source_retention_enabled:
            source_target = staging / "source" / _safe_filename(parsed.filename)
            source_target.parent.mkdir(parents=True, exist_ok=True)
            _copy_with_cancel(
                source,
                source_target,
                cancelled=cancelled,
            )
            _append_file_asset(
                manifest,
                doc_id=doc_id,
                kind="source",
                path=source_target,
                page=None,
                element_index=None,
                content_type=(
                    "application/pdf"
                    if source.suffix.lower() == ".pdf"
                    else "application/octet-stream"
                ),
                metadata={"filename": parsed.filename},
            )

        table_count = 0
        for element_index, element in enumerate(parsed.elements):
            _raise_if_cancelled(cancelled)
            if element.kind != "table":
                continue
            asset_id = uuid.uuid4().hex
            manifest.append(
                {
                    "asset_id": asset_id,
                    "doc_id": doc_id,
                    "kind": "table",
                    "page": element.page,
                    "element_index": element_index,
                    "relative_path": None,
                    "content_type": "application/json",
                    "sha256": hashlib.sha256(
                        element.content.encode("utf-8")
                    ).hexdigest(),
                    "size_bytes": len(element.content.encode("utf-8")),
                    "width": None,
                    "height": None,
                    "metadata": {
                        "html": element.metadata.get("table_html", ""),
                        "cells": element.metadata.get("table_cells", []),
                        "row_count": element.metadata.get("row_count"),
                        "column_count": element.metadata.get("column_count"),
                        "source_method": element.source_method,
                    },
                }
            )
            element.metadata["asset_id"] = asset_id
            table_count += 1

        visual_candidates = [
            (element_index, element)
            for element_index, element in enumerate(parsed.elements)
            if element.kind == "image" and element.page is not None
        ]
        if source.suffix.lower() == ".pdf" and visual_candidates:
            import pdfplumber

            image_directory = staging / "images"
            image_directory.mkdir(parents=True, exist_ok=True)
            byte_budget = config.asset_max_total_mb * 1024 * 1024
            visual_limit = config.asset_max_per_document
            with pdfplumber.open(str(source)) as pdf:
                for image_number, (element_index, element) in enumerate(
                    visual_candidates[:visual_limit],
                    start=1,
                ):
                    _raise_if_cancelled(cancelled)
                    target = image_directory / (
                        f"page-{element.page:04d}-image-{image_number:03d}.png"
                    )
                    try:
                        dimensions = _save_crop(
                            pdf.pages[element.page - 1],
                            element.bbox,
                            target,
                            resolution=config.asset_dpi,
                        )
                    except Exception:
                        logger.warning(
                            "Skipping image crop on page %s of %s.",
                            element.page,
                            parsed.filename,
                            exc_info=True,
                        )
                        continue
                    if dimensions is None:
                        target.unlink(missing_ok=True)
                        continue
                    if sum(
                        int(asset["size_bytes"])
                        for asset in manifest
                        if asset.get("kind") == "image"
                    ) + target.stat().st_size > byte_budget:
                        target.unlink(missing_ok=True)
                        logger.warning(
                            "Asset byte budget reached for document %s.", doc_id
                        )
                        break
                    width, height = dimensions
                    asset = _append_file_asset(
                        manifest,
                        doc_id=doc_id,
                        kind=(
                            "page_preview"
                            if element.source_method == "rendered_ocr_page"
                            else "image"
                        ),
                        path=target,
                        page=element.page,
                        element_index=element_index,
                        content_type="image/png",
                        width=width,
                        height=height,
                        metadata={
                            **element.metadata,
                            "source_method": element.source_method,
                            "bbox": list(element.bbox) if element.bbox else None,
                        },
                    )
                    element.metadata["asset_id"] = asset["asset_id"]

        _ensure_within_root(staging)
        for asset in manifest:
            relative_path = asset.get("relative_path")
            if relative_path:
                staging_prefix = staging.relative_to(root)
                final_relative = Path(relative_path).relative_to(staging_prefix)
                asset["relative_path"] = str(
                    Path(doc_id) / final_relative
                ).replace(os.sep, "/")

        final_directory.parent.mkdir(parents=True, exist_ok=True)
        backup = root / ".staging" / f"{doc_id}-previous-{uuid.uuid4().hex}"
        published = False
        try:
            if final_directory.exists():
                os.replace(final_directory, backup)
            os.replace(staging, final_directory)
            published = True
            persisted = replace_document_assets(doc_id, manifest)
        except Exception:
            if published and final_directory.exists():
                _remove_directory(final_directory)
            if backup.exists():
                os.replace(backup, final_directory)
            raise
        finally:
            if backup.exists():
                _remove_directory(backup)
        parsed.metadata["asset_count"] = len(persisted)
        parsed.metadata["table_asset_count"] = table_count
        return persisted
    except Exception:
        if staging.exists():
            _remove_directory(staging)
        raise


def remove_document_assets(doc_id: str) -> None:
    """Delete asset records and files owned by one document."""
    directory = _safe_doc_directory(doc_id)
    delete_document_asset_records(doc_id)
    _remove_directory(directory)
