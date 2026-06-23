"""文本分块策略 — 递归字符分割"""
from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass, field
import hashlib
import logging
from mindforge.config import get_settings

logger = logging.getLogger(__name__)


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
            chunk_text = content[start:end].strip()
            if chunk_text:
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
        sentences = re.split(r'(?<=[。！？\.!?])\s*', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 1:
            return TextSplitter().split(doc_id, content, metadata)
        try:
            embeddings = self.embedder.embed(sentences)
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = [cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0] for i in range(len(embeddings)-1)]
        except Exception:
            return TextSplitter().split(doc_id, content, metadata)
        chunks, current_chunk = [], []
        for i, sent in enumerate(sentences):
            current_chunk.append(sent)
            if i < len(similarities) and similarities[i] < self.threshold:
                chunk_text = "".join(current_chunk)
                chunk_id = hashlib.md5(f"{doc_id}:{i}:{len(chunk_text)}".encode()).hexdigest()[:12]
                chunks.append(DocumentChunk(chunk_id=chunk_id, doc_id=doc_id, content=chunk_text, metadata=metadata or {}))
                current_chunk = []
        if current_chunk:
            chunk_text = "".join(current_chunk)
            chunk_id = hashlib.md5(f"{doc_id}:end:{len(chunk_text)}".encode()).hexdigest()[:12]
            chunks.append(DocumentChunk(chunk_id=chunk_id, doc_id=doc_id, content=chunk_text, metadata=metadata or {}))
        return chunks
