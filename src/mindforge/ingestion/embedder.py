"""Embedding generation engine with real semantic models.

Supports sentence-transformers (local), OpenAI (API), and a lightweight
hash-based fallback for development only.
"""

from __future__ import annotations
import hashlib
import math
import os
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FALLBACK_DIM = 1024  # must match LLMConfig.embedding_dim (BGE-M3)


class EmbeddingManager:
    """Semantic embedding via sentence-transformers (preferred) or OpenAI API.

    Attempts to load a local sentence-transformers model first. If unavailable,
    falls back to OpenAI embeddings via the configured API key. A lightweight
    hash-based fallback is available for development but logs a prominent warning.

    Parameters
    ----------
    dim : int, optional
        Embedding vector dimension. Inferred from the loaded model by default.
    model_name : str, optional
        Override the model to load. Defaults to ``"BAAI/bge-m3"`` for
        sentence-transformers, or ``"text-embedding-3-small"`` for OpenAI.
    provider : str, optional
        Force a specific provider: ``"sentence-transformers"``, ``"openai"``,
        or ``"fallback"``. Auto-detected when omitted.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self._dim = dim
        self._model_name = model_name
        self._provider = provider
        self._model = None
        self._client = None  # OpenAI client
        self._native_dim: Optional[int] = None
        self._init_backend()

    # ------------------------------------------------------------------
    # Backend initialization
    # ------------------------------------------------------------------

    def _init_backend(self) -> None:
        """Try backends in order: sentence-transformers → OpenAI → fallback."""
        # Normalize provider aliases
        _provider_map = {
            "bge": "sentence-transformers",
            "st": "sentence-transformers",
        }
        resolved = _provider_map.get(self._provider or "", self._provider)
        explicit = bool(resolved)
        if explicit:
            backends = [resolved]
        else:
            backends = ["sentence-transformers", "openai", "fallback"]

        for backend in backends:
            try:
                if backend == "sentence-transformers":
                    self._init_st()
                elif backend == "openai":
                    self._init_openai()
                elif backend == "fallback":
                    self._init_fallback()
                if self._model is not None or self._client is not None or self._provider == "fallback":
                    return
            except Exception as exc:
                logger.warning("Embedding backend %s unavailable: %s", backend, exc)

        if explicit:
            raise RuntimeError(
                f"Configured embedding provider '{resolved}' is unavailable. "
                "Refusing to create incompatible fallback vectors."
            )

        # Automatic mode keeps the development-only fallback.
        self._init_fallback()

    def _init_st(self) -> None:
        """Initialize sentence-transformers backend.

        Uses HuggingFace mirror (hf-mirror.com) for China access.
        Cached models load instantly with local_files_only.
        """
        from mindforge.config import get_settings

        settings = get_settings().llm
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(
            settings.hf_hub_download_timeout
        )

        from sentence_transformers import SentenceTransformer

        model_name = self._model_name or settings.local_embedding_model
        revision = settings.local_embedding_revision
        self._device = settings.sentence_transformers_device
        self._batch_size = settings.embedding_batch_size
        if settings.torch_num_threads > 0:
            import torch

            torch.set_num_threads(settings.torch_num_threads)

        # Try local cache first (instant), then download from HF mirror
        try:
            self._model = SentenceTransformer(
                model_name,
                revision=revision,
                local_files_only=True,
                device=self._device,
            )
        except Exception:
            logger.info("Model '%s' not cached — downloading from mirror...", model_name)
            self._model = SentenceTransformer(
                model_name,
                revision=revision,
                local_files_only=False,
                device=self._device,
            )
        try:
            self._native_dim = self._model.get_embedding_dimension()
        except AttributeError:
            self._native_dim = self._model.get_sentence_embedding_dimension()
        if self._dim is None:
            self._dim = self._native_dim
        self._provider = "sentence-transformers"
        logger.info(
            "Embedding: sentence-transformers/%s "
            "(dim=%d, device=%s, batch=%d)",
            model_name,
            self._dim,
            self._device,
            self._batch_size,
        )

    def _init_openai(self) -> None:
        """Initialize OpenAI embeddings backend."""
        from openai import OpenAI
        from mindforge.config import get_settings

        settings = get_settings().llm
        api_key = settings.openai_api_key
        base_url = settings.openai_base_url
        if not api_key:
            raise RuntimeError("OpenAI API key not configured")

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model_name = self._model_name or settings.embedding_model
        if self._dim is None:
            # Known dimensions for common models
            _OPENAI_DIMS = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            self._dim = _OPENAI_DIMS.get(self._model_name, 1536)
        self._provider = "openai"
        self._device = "remote"
        self._batch_size = 64
        logger.info("Embedding: openai/%s (dim=%d)", self._model_name, self._dim)

    def _init_fallback(self) -> None:
        """Hash-based fallback — for development / bootstrap only."""
        if self._dim is None:
            self._dim = _FALLBACK_DIM
        self._provider = "fallback"
        self._device = "cpu"
        self._batch_size = 1
        logger.warning(
            "Embedding: HASH-BASED FALLBACK (dim=%d) — NO semantic similarity. "
            "Install sentence-transformers or configure an OpenAI API key for production.",
            self._dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def dim(self) -> int:
        return self._dim or _FALLBACK_DIM

    @property
    def device(self) -> str:
        return getattr(self, "_device", "unknown")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts into dense vectors."""
        if not texts:
            return []

        if self._provider == "sentence-transformers" and self._model is not None:
            return self._embed_st(texts)

        if self._provider == "openai" and self._client is not None:
            return self._embed_openai(texts)

        return self._embed_fallback(texts)

    def embed_single(self, text: str) -> List[float]:
        """Embed a single text and return its vector."""
        return self.embed([text])[0]

    async def embed_async(self, texts: List[str]) -> List[List[float]]:
        """Async-friendly alias (delegates to sync embed)."""
        import asyncio
        return await asyncio.to_thread(self.embed, texts)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _embed_st(self, texts: List[str]) -> List[List[float]]:
        result = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self._batch_size,
        )
        return [self._fit_dimension(vector) for vector in result.tolist()]

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        # OpenAI embedding 单次请求有 token / 批量上限，按 64 条分片
        max_batch = 64
        if len(texts) <= max_batch:
            kwargs = {"model": self._model_name, "input": texts}
            if self._model_name.startswith("text-embedding-3"):
                kwargs["dimensions"] = self.dim
            resp = self._client.embeddings.create(**kwargs)
            return [d.embedding for d in resp.data]

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), max_batch):
            batch = texts[i:i + max_batch]
            kwargs = {"model": self._model_name, "input": batch}
            if self._model_name.startswith("text-embedding-3"):
                kwargs["dimensions"] = self.dim
            resp = self._client.embeddings.create(**kwargs)
            all_embeddings.extend(d.embedding for d in resp.data)
        return all_embeddings

    def _fit_dimension(self, vector: List[float]) -> List[float]:
        """Fit native embeddings to the configured Qdrant dimension.

        Zero-padding a normalized vector preserves cosine similarity. This
        allows BGE-M3 and OpenAI v3 embeddings to share a fixed collection
        dimension across runtime provider switches.
        """
        target = self.dim
        if len(vector) == target:
            return vector
        if len(vector) > target:
            resized = vector[:target]
            norm = math.sqrt(sum(value * value for value in resized))
            return (
                [value / norm for value in resized]
                if norm > 0
                else resized
            )
        return vector + [0.0] * (target - len(vector))

    def _embed_fallback(self, texts: List[str]) -> List[List[float]]:
        """Deterministic hash projection (zero model, zero semantic).

        尝试用 jieba 对中文文本分词以改善降级质量；
        若 jieba 不可用则回退到空白分词。
        """
        results = []
        for text in texts:
            lower = text.lower()
            # 中文文本优先用 jieba 分词
            has_cjk = any('一' <= c <= '鿿' for c in text)
            if has_cjk:
                try:
                    import jieba
                    words = list(jieba.cut(lower))
                except Exception:
                    words = lower.split()
            else:
                words = lower.split()
            vec = [0.0] * self.dim
            for word in words:
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                idx = h % self.dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            results.append(vec)
        return results

    def close(self) -> None:
        """Release provider-owned network resources."""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_embedder: Optional[EmbeddingManager] = None


def get_embedder() -> EmbeddingManager:
    global _embedder
    if _embedder is None:
        from mindforge.config import get_settings
        settings = get_settings()
        _embedder = EmbeddingManager(
            model_name=settings.llm.local_embedding_model or "BAAI/bge-m3",
            provider=settings.llm.embedding_provider or None,
            dim=settings.vector_store.embedding_dim,
        )
    return _embedder


def get_embedder_status() -> dict[str, str]:
    """Return configured or loaded backend identity without loading a model."""
    if _embedder is not None:
        return {
            "provider": _embedder.provider,
            "device": _embedder.device,
        }
    from mindforge.config import get_settings

    settings = get_settings().llm
    provider = settings.embedding_provider
    device = (
        settings.sentence_transformers_device
        if provider in {"bge", "st", "sentence-transformers"}
        else "remote"
        if provider == "openai"
        else "cpu"
    )
    return {"provider": provider, "device": device}


def reset_embedder() -> None:
    """Drop the cached embedding backend after configuration changes."""
    global _embedder
    previous = _embedder
    _embedder = None
    if previous is not None:
        previous.close()
