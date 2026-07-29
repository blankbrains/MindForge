"""RAPTOR 层次化索引 — 自底向上构建摘要树"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np

from mindforge.config import get_settings
from mindforge.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RAPTORNode:
    node_id: str
    content: str
    summary: str = ""
    level: int = 0
    children: list[RAPTORNode] = field(default_factory=list)
    embedding: list[float] | None = None


class RAPTORIndexer:
    def __init__(self, embedder=None, llm=None):
        cfg = get_settings().raptor
        self.num_levels = cfg.raptor_levels
        self.threshold = cfg.raptor_threshold
        self.max_nodes = cfg.max_nodes
        self.summary_concurrency = cfg.summary_concurrency
        self.summary_model = cfg.summary_model
        self.embedder = embedder
        self.llm = llm  # 应为 BaseLLM 实例或兼容的 async callable

    async def build_tree(
        self,
        chunks: list[DocumentChunk],
    ) -> list[RAPTORNode]:
        if not chunks:
            return []
        if len(chunks) > self.max_nodes:
            raise ValueError(
                f"RAPTOR input exceeds the configured {self.max_nodes}-node "
                "limit."
            )
        leaves = [RAPTORNode(node_id=ch.chunk_id, content=ch.content, level=0, embedding=ch.embedding) for ch in chunks]
        missing_leaves = [
            node for node in leaves if node.embedding is None
        ]
        if missing_leaves and self.embedder:
            vectors = await asyncio.to_thread(
                self.embedder.embed,
                [node.content for node in missing_leaves],
            )
            if len(vectors) != len(missing_leaves):
                raise ValueError(
                    "RAPTOR leaf embedding count does not match input count."
                )
            for node, vector in zip(missing_leaves, vectors):
                node.embedding = vector
        all_nodes = [leaves]
        total_nodes = len(leaves)
        current_level = leaves
        for level in range(1, self.num_levels):
            if len(current_level) <= 3:
                break
            clusters = await asyncio.to_thread(
                self._cluster_nodes,
                current_level,
            )
            summary_clusters = [
                cluster for cluster in clusters if len(cluster) > 1
            ]
            carried_nodes = [
                cluster[0] for cluster in clusters if len(cluster) == 1
            ]
            prospective_level_size = (
                len(carried_nodes) + len(summary_clusters)
            )
            if (
                not summary_clusters
                or prospective_level_size >= len(current_level)
            ):
                logger.info(
                    "RAPTOR stopped at level %d because clustering did not "
                    "reduce %d nodes.",
                    level,
                    len(current_level),
                )
                break
            if total_nodes + len(summary_clusters) > self.max_nodes:
                raise ValueError(
                    "RAPTOR tree exceeds the configured node limit."
                )

            semaphore = asyncio.Semaphore(self.summary_concurrency)

            async def _summarize_one(
                i: int,
                cluster: list[RAPTORNode],
                *,
                current_level_number: int = level,
                summary_semaphore: asyncio.Semaphore = semaphore,
            ) -> RAPTORNode:
                async with summary_semaphore:
                    summary = await self._summarize_cluster(
                        cluster,
                        current_level_number,
                    )
                child_fingerprint = hashlib.md5(
                    "|".join(
                        child.node_id for child in cluster
                    ).encode()
                ).hexdigest()[:8]
                node = RAPTORNode(
                    node_id=(
                        f"raptor_l{current_level_number}_c{i}_"
                        f"{child_fingerprint}_"
                        f"{hashlib.md5(summary.encode()).hexdigest()[:8]}"
                    ),
                    content=summary,
                    summary=summary,
                    level=current_level_number,
                    children=cluster,
                )
                return node

            summary_nodes = list(
                await asyncio.gather(
                    *[
                        _summarize_one(i, cluster)
                        for i, cluster in enumerate(summary_clusters)
                    ]
                )
            )
            if self.embedder and summary_nodes:
                vectors = await asyncio.to_thread(
                    self.embedder.embed,
                    [node.content for node in summary_nodes],
                )
                if len(vectors) != len(summary_nodes):
                    raise ValueError(
                        "RAPTOR summary embedding count does not match "
                        "summary count."
                    )
                for node, vector in zip(summary_nodes, vectors):
                    node.embedding = vector

            all_nodes.append(summary_nodes)
            total_nodes += len(summary_nodes)
            current_level = carried_nodes + summary_nodes
        nodes = [n for level in all_nodes for n in level]
        logger.info(f"RAPTOR 树: {len(nodes)} 节点, {len(all_nodes)} 层")
        return nodes

    def _cluster_nodes(
        self,
        nodes: list[RAPTORNode],
    ) -> list[list[RAPTORNode]]:
        if len(nodes) <= 3:
            return [nodes]
        embeddings = []
        embedding_indices = []  # maps embedding_idx -> node_idx
        for i, node in enumerate(nodes):
            if node.embedding is not None:
                embeddings.append(node.embedding)
                embedding_indices.append(i)
        if not embeddings:
            cs = max(3, len(nodes) // 3)
            return [nodes[i:i+cs] for i in range(0, len(nodes), cs)]
        embeddings = np.array(embeddings)
        # Build reverse map: node_idx -> embedding_idx
        node_to_emb = {node_idx: emb_idx for emb_idx, node_idx in enumerate(embedding_indices)}
        clusters, used = [], set()
        for i in range(len(nodes)):
            if i in used:
                continue
            if i not in node_to_emb:
                clusters.append([nodes[i]])
                used.add(i)
                continue
            cluster = [nodes[i]]
            used.add(i)
            ei = node_to_emb[i]
            for j in range(i+1, len(nodes)):
                if j in used:
                    continue
                if j not in node_to_emb:
                    continue
                ej = node_to_emb[j]
                sim = np.dot(embeddings[ei], embeddings[ej]) / (np.linalg.norm(embeddings[ei]) * np.linalg.norm(embeddings[ej]) + 1e-8)
                if sim > self.threshold:
                    cluster.append(nodes[j])
                    used.add(j)
            clusters.append(cluster)
        return clusters

    async def _summarize_cluster(
        self,
        cluster: list[RAPTORNode],
        level: int,
    ) -> str:
        if self.llm is None:
            return "\n".join(n.content[:200] for n in cluster[:5])
        texts = "\n\n".join(f"[{i+1}] {n.content[:500]}" for i, n in enumerate(cluster[:10]))
        prompt = f"请为以下 {len(cluster)} 个相关文本片段生成摘要（RAPTOR 第{level}层）：\n{texts}\n摘要："
        try:
            # 兼容 BaseLLM.chat() 接口
            if hasattr(self.llm, "chat"):
                from mindforge.models.base import ChatMessage
                result = await self.llm.chat(
                    [ChatMessage(role="user", content=prompt)],
                    temperature=0.3,
                )
                content = result.content or ""
                if not content.strip():
                    content = "\n".join(n.content[:200] for n in cluster[:3])
                return content.strip()[:1000]
            else:
                # callable fallback: async llm_fn
                result = await self.llm(prompt)
                return str(result).strip()[:1000]
        except Exception:
            logger.exception("RAPTOR summarization failed for level %d", level)
            return "\n".join(n.content[:200] for n in cluster[:3])
