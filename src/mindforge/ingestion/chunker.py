"""文本分块策略 — 递归字符分割"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from dataclasses import dataclass, field
import hashlib
import logging
from mindforge.config import get_settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mindforge.ingestion.parsers import DocumentElement


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class TextSplitter:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        cfg = get_settings().chunking
        self.chunk_size = chunk_size or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            logger.warning(
                "chunk_overlap (%d) >= chunk_size (%d) — clamping overlap to size//4.",
                self.chunk_overlap, self.chunk_size,
            )
            self.chunk_overlap = max(self.chunk_size // 4, 0)

    def split(self, doc_id: str, content: str, metadata: dict = None) -> List[DocumentChunk]:
        separators = ["\n\n", "\n", "。", ".", "，", ",", " "]
        chunks = []
        start = 0
        content_len = len(content)
        if content_len == 0:
            return chunks
        while start < content_len:
            end = min(start + self.chunk_size, content_len)
            if end < content_len:
                for sep in separators:
                    pos = content.rfind(sep, start, end)
                    if pos > start:
                        end = pos + len(sep)
                        break
            chunk_text = content[start:end]
            if chunk_text.strip():
                chunk_id = hashlib.md5(f"{doc_id}:{start}:{end}".encode()).hexdigest()[:12]
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id, doc_id=doc_id, content=chunk_text,
                    metadata={**metadata, "chunk_start": start, "chunk_end": end} if metadata else {"chunk_start": start, "chunk_end": end},
                ))
            if end == content_len:
                break
            new_start = end - self.chunk_overlap
            # 确保 start 单调递增，防止死循环
            if new_start <= start:
                new_start = end
            start = new_start
        return chunks


class SemanticChunker:
    def __init__(self, embedder=None, threshold: float = 0.7):
        self.embedder = embedder
        self.threshold = threshold

    def split(self, doc_id: str, content: str, metadata: dict = None) -> List[DocumentChunk]:
        if not self.embedder:
            return TextSplitter().split(doc_id, content, metadata)
        import re
        sentences: list[tuple[int, int, str]] = []
        cursor = 0
        for match in re.finditer(r'(?<=[。！？\.!?])\s*', content):
            end = match.end()
            sentence = content[cursor:end]
            if sentence.strip():
                sentences.append((cursor, end, sentence))
            cursor = end
        if cursor < len(content):
            sentence = content[cursor:]
            if sentence.strip():
                sentences.append((cursor, len(content), sentence))
        if len(sentences) <= 1:
            return TextSplitter().split(doc_id, content, metadata)
        try:
            embeddings = self.embedder.embed(
                [sentence.strip() for _, _, sentence in sentences]
            )
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = [cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0] for i in range(len(embeddings)-1)]
        except Exception:
            return TextSplitter().split(doc_id, content, metadata)
        chunks: list[DocumentChunk] = []
        chunk_start = sentences[0][0]
        for i, (_, sentence_end, _) in enumerate(sentences):
            if i < len(similarities) and similarities[i] < self.threshold:
                chunk_text = content[chunk_start:sentence_end]
                chunk_id = hashlib.md5(
                    f"{doc_id}:{chunk_start}:{sentence_end}".encode()
                ).hexdigest()[:12]
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        content=chunk_text,
                        metadata={
                            **(metadata or {}),
                            "chunk_start": chunk_start,
                            "chunk_end": sentence_end,
                        },
                    )
                )
                chunk_start = sentences[i + 1][0]
        final_end = sentences[-1][1]
        if chunk_start < final_end:
            chunk_text = content[chunk_start:final_end]
            chunk_id = hashlib.md5(
                f"{doc_id}:{chunk_start}:{final_end}".encode()
            ).hexdigest()[:12]
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=chunk_text,
                    metadata={
                        **(metadata or {}),
                        "chunk_start": chunk_start,
                        "chunk_end": final_end,
                    },
                )
            )
        return chunks


class ElementAwareSplitter:
    """Split parsed elements while preserving page and extraction provenance."""

    def __init__(
        self,
        *,
        strategy: str = "auto",
        embedder=None,
    ) -> None:
        self._text_splitter = TextSplitter()
        self._semantic_splitter = (
            SemanticChunker(embedder=embedder)
            if strategy == "semantic"
            else None
        )

    def split(
        self,
        doc_id: str,
        elements: list["DocumentElement"],
        metadata: dict | None = None,
    ) -> List[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        chunk_size = self._text_splitter.chunk_size
        for element_index, element in enumerate(elements):
            if not element.content.strip():
                continue
            persisted_metadata = {
                key: value
                for key, value in element.metadata.items()
                if key not in {"table_html", "table_cells"}
            }
            element_metadata = {
                **(metadata or {}),
                **persisted_metadata,
                "element_index": element_index,
                "element_type": element.kind,
                "page": element.page,
                "bbox": list(element.bbox) if element.bbox else None,
                "ocr_confidence": element.confidence,
                "source_method": element.source_method,
            }
            if (
                element.kind == "table"
                and len(element.content) <= chunk_size
            ):
                chunks.append(
                    DocumentChunk(
                        chunk_id=hashlib.md5(
                            (
                                f"{doc_id}:element:{element_index}:"
                                f"{element.start}:{element.end}"
                            ).encode()
                        ).hexdigest()[:12],
                        doc_id=doc_id,
                        content=element.content,
                        metadata={
                            **element_metadata,
                            "chunk_start": element.start,
                            "chunk_end": element.end,
                        },
                    )
                )
                continue

            splitter = (
                self._semantic_splitter
                if self._semantic_splitter is not None
                and element.kind == "text"
                else self._text_splitter
            )
            local_chunks = splitter.split(
                doc_id,
                element.content,
                metadata=element_metadata,
            )
            for local_index, chunk in enumerate(local_chunks):
                local_start = int(chunk.metadata.get("chunk_start", 0))
                local_end = int(
                    chunk.metadata.get(
                        "chunk_end",
                        local_start + len(chunk.content),
                    )
                )
                start = (
                    element.start + local_start
                    if element.start is not None
                    else None
                )
                end = (
                    element.start + local_end
                    if element.start is not None
                    else None
                )
                chunk.chunk_id = hashlib.md5(
                    (
                        f"{doc_id}:element:{element_index}:{local_index}:"
                        f"{start}:{end}"
                    ).encode()
                ).hexdigest()[:12]
                chunk.metadata.update(
                    {
                        "chunk_start": start,
                        "chunk_end": end,
                    }
                )
                chunks.append(chunk)
        return chunks
