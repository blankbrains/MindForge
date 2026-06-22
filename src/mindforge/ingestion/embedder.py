"""Embedding 生成引擎 — 轻量哈希 embedding（零模型加载，适合快速部署后替换）"""

from __future__ import annotations
import hashlib
import math
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Zero-dependency embedding via deterministic hash projection.

    Uses a fixed random seed to project token hashes into a 384-dim space.
    Intended as a bootstrap / demo replacement — swap with a real model
    (sentence-transformers, fastembed, API) for production.
    """

    def __init__(self, dim: Optional[int] = None):
        if dim is None:
            from mindforge.config import get_settings
            dim = get_settings().vector_store.embedding_dim
        self._dim = dim
        self._rng = None

    def _get_proj(self) -> List[float]:
        """Deterministic pseudo-random projection vector (cached)."""
        if self._rng is None:
            import random
            rng = random.Random(42)
            self._rng = [rng.gauss(0, 1) for _ in range(self._dim)]
        return self._rng

    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            words = text.lower().split()
            vec = [0.0] * self._dim
            for word in words:
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                idx = h % self._dim
                vec[idx] += 1.0
            # Normalise
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            results.append(vec)
        return results

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]


_embedder: Optional[EmbeddingManager] = None


def get_embedder() -> EmbeddingManager:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingManager()
    return _embedder
