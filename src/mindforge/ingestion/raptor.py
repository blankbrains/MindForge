"""RAPTOR 层次化索引 — 自底向上构建摘要树"""
from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass, field
import hashlib
import logging
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
    children: List["RAPTORNode"] = field(default_factory=list)
    embedding: Optional[List[float]] = None


class RAPTORIndexer:
    def __init__(self, embedder=None, llm=None):
        cfg = get_settings().raptor
        self.num_levels = cfg.raptor_levels
        self.threshold = cfg.raptor_threshold
        self.embedder = embedder
        self.llm = llm

    def build_tree(self, chunks: List[DocumentChunk]) -> List[RAPTORNode]:
        if not chunks:
            return []
        leaves = [RAPTORNode(node_id=ch.chunk_id, content=ch.content, level=0, embedding=ch.embedding) for ch in chunks]
        all_nodes = [leaves]
        current_level = leaves
        for level in range(1, self.num_levels):
            if len(current_level) <= 3:
                break
            clusters = self._cluster_nodes(current_level)
            next_level = []
            for i, cluster in enumerate(clusters):
                summary = self._summarize_cluster(cluster, level)
                node = RAPTORNode(
                    node_id=f"raptor_l{level}_c{i}_{hashlib.md5(summary.encode()).hexdigest()[:8]}",
                    content=summary, summary=summary, level=level, children=cluster,
                )
                next_level.append(node)
            if not next_level:
                break
            all_nodes.append(next_level)
            current_level = next_level
        logger.info(f"RAPTOR 树: {sum(len(n) for n in all_nodes)} 节点, {len(all_nodes)} 层")
        return [n for level in all_nodes for n in level]

    def _cluster_nodes(self, nodes: List[RAPTORNode]) -> List[List[RAPTORNode]]:
        if len(nodes) <= 3:
            return [nodes]
        embeddings = []
        for node in nodes:
            if node.embedding is None and self.embedder:
                node.embedding = self.embedder.embed_single(node.content[:512])
            if node.embedding is not None:
                embeddings.append(node.embedding)
        if not embeddings:
            cs = max(3, len(nodes) // 3)
            return [nodes[i:i+cs] for i in range(0, len(nodes), cs)]
        embeddings = np.array(embeddings)
        clusters, used = [], set()
        for i in range(len(nodes)):
            if i in used:
                continue
            cluster = [nodes[i]]
            used.add(i)
            for j in range(i+1, len(nodes)):
                if j in used:
                    continue
                sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8)
                if sim > self.threshold:
                    cluster.append(nodes[j])
                    used.add(j)
            clusters.append(cluster)
        return clusters

    def _summarize_cluster(self, cluster: List[RAPTORNode], level: int) -> str:
        if self.llm is None:
            return "\n".join(n.content[:200] for n in cluster[:5])
        texts = "\n\n".join(f"[{i+1}] {n.content[:500]}" for i, n in enumerate(cluster[:10]))
        prompt = f"请为以下 {len(cluster)} 个相关文本片段生成摘要（RAPTOR 第{level}层）：\n{texts}\n摘要："
        try:
            response = self.llm.invoke(prompt)
            return response.content[:1000]
        except Exception:
            return "\n".join(n.content[:200] for n in cluster[:3])
