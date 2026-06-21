"""Embedding 生成引擎 — sentence-transformers 语义向量"""
from __future__ import annotations
from typing import List, Optional
import logging
import os

logger = logging.getLogger(__name__)

# 使用国内镜像（服务器无法直连 huggingface.co）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class EmbeddingManager:
    def __init__(self):
        self._model = None
        self._dim = 384

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
        self._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self._dim = self._model.get_embedding_dimension()
        logger.info(f"Model loaded. Dimension: {self._dim}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]


_embedder: Optional[EmbeddingManager] = None


def get_embedder() -> EmbeddingManager:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingManager()
    return _embedder
