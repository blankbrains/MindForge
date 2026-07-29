from __future__ import annotations
import asyncio
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Tuple
import hashlib
import logging
import math
import threading
from collections import deque

logger = logging.getLogger(__name__)

_ENTITY_CONTRIBUTIONS_KEY = "document_contributions"
_LEGACY_ENTITY_FIELDS_KEY = "legacy_fields"
_LEGACY_DOC_IDS_KEY = "legacy_doc_ids"
_RELATION_WEIGHTS_KEY = "document_weights"
_LEGACY_RELATION_WEIGHT_KEY = "legacy_weight"


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


@dataclass
class Entity:
    """A knowledge-graph entity extracted from documents."""

    id: str
    name: str
    type: str = ""  # e.g. person, organisation, concept
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id


@dataclass
class Relation:
    """A directed relation between two entities."""

    source: str  # entity id
    target: str  # entity id
    relation_type: str = ""
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Community:
    """A community of entities discovered via BFS."""

    id: str
    entities: List[Entity] = field(default_factory=list)
    summary: str = ""


# ------------------------------------------------------------------
# GraphRAG Engine
# ------------------------------------------------------------------


class GraphRAGEngine:
    """Graph-based RAG engine that builds a knowledge graph from documents,
    discovers communities, and retrieves structured results for a query."""

    def __init__(
        self,
        llm_fn=None,
        *,
        entity_llm=None,
        summary_llm=None,
    ):
        from mindforge.config import get_settings

        config = get_settings().graphrag
        self.llm_fn = llm_fn
        self.entity_llm = entity_llm or llm_fn
        self.summary_llm = summary_llm or llm_fn
        self.max_entities_per_doc = config.max_entities_per_doc
        self.max_total_entities = config.max_total_entities
        self.max_communities = config.max_communities
        self.min_community_size = config.min_community_size
        self.extraction_char_budget = config.extraction_char_budget
        self.summary_concurrency = config.summary_concurrency

        # Graph state
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.communities: List[Community] = []

        # Adjacency list for BFS traversal
        self._adjacency: Dict[str, Set[str]] = {}
        self._operation_lock = asyncio.Lock()
        self._state_lock = threading.RLock()

    @staticmethod
    def _normalise_entity_provenance(entity: Entity) -> None:
        """Upgrade legacy metadata without inventing source ownership."""
        metadata = entity.metadata
        contributions = metadata.get(_ENTITY_CONTRIBUTIONS_KEY)
        if not isinstance(contributions, dict):
            contributions = {}
            metadata[_ENTITY_CONTRIBUTIONS_KEY] = contributions

        legacy_doc_ids = metadata.get(_LEGACY_DOC_IDS_KEY)
        if not isinstance(legacy_doc_ids, list):
            legacy_doc_ids = [
                str(value)
                for value in metadata.get("doc_ids", [])
                if str(value) and str(value) not in contributions
            ]
            metadata[_LEGACY_DOC_IDS_KEY] = legacy_doc_ids
            if legacy_doc_ids:
                metadata[_LEGACY_ENTITY_FIELDS_KEY] = {
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                }

        metadata["doc_ids"] = sorted(
            {str(doc_id) for doc_id in contributions if str(doc_id)}
            | {str(doc_id) for doc_id in legacy_doc_ids}
        )

    @staticmethod
    def _refresh_entity_fields(entity: Entity) -> bool:
        """Recompute canonical fields from remaining source contributions."""
        GraphRAGEngine._normalise_entity_provenance(entity)
        metadata = entity.metadata
        candidates: list[dict[str, str]] = []

        legacy_doc_ids = metadata.get(_LEGACY_DOC_IDS_KEY, [])
        legacy_fields = metadata.get(_LEGACY_ENTITY_FIELDS_KEY)
        if legacy_doc_ids and isinstance(legacy_fields, dict):
            candidates.append(legacy_fields)

        contributions = metadata.get(_ENTITY_CONTRIBUTIONS_KEY, {})
        for doc_id in sorted(contributions):
            contribution = contributions[doc_id]
            if isinstance(contribution, dict):
                candidates.append(contribution)

        if not candidates:
            return False

        for field_name in ("name", "type", "description"):
            setattr(
                entity,
                field_name,
                next(
                    (
                        str(candidate.get(field_name, ""))
                        for candidate in candidates
                        if candidate.get(field_name)
                    ),
                    "",
                ),
            )
        return True

    @staticmethod
    def _normalise_relation_provenance(relation: Relation) -> None:
        """Upgrade legacy relation metadata conservatively."""
        metadata = relation.metadata
        weights = metadata.get(_RELATION_WEIGHTS_KEY)
        if not isinstance(weights, dict):
            weights = {}
            metadata[_RELATION_WEIGHTS_KEY] = weights

        legacy_doc_ids = metadata.get(_LEGACY_DOC_IDS_KEY)
        if not isinstance(legacy_doc_ids, list):
            legacy_doc_ids = [
                str(value)
                for value in metadata.get("doc_ids", [])
                if str(value) and str(value) not in weights
            ]
            metadata[_LEGACY_DOC_IDS_KEY] = legacy_doc_ids
            if legacy_doc_ids:
                metadata[_LEGACY_RELATION_WEIGHT_KEY] = relation.weight

        metadata["doc_ids"] = sorted(
            {str(doc_id) for doc_id in weights if str(doc_id)}
            | {str(doc_id) for doc_id in legacy_doc_ids}
        )

    @staticmethod
    def _refresh_relation_weight(relation: Relation) -> bool:
        """Recompute relation weight from remaining source contributions."""
        GraphRAGEngine._normalise_relation_provenance(relation)
        metadata = relation.metadata
        legacy_doc_ids = metadata.get(_LEGACY_DOC_IDS_KEY, [])
        if legacy_doc_ids and _LEGACY_RELATION_WEIGHT_KEY in metadata:
            relation.weight = float(metadata[_LEGACY_RELATION_WEIGHT_KEY])
            return True

        weights = metadata.get(_RELATION_WEIGHTS_KEY, {})
        if not weights:
            return False
        first_doc_id = sorted(weights)[0]
        relation.weight = float(weights[first_doc_id])
        return True

    @staticmethod
    def _coerce_relation_weight(value: Any) -> float:
        """Return a finite relation confidence in the documented score range."""
        try:
            weight = float(value)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(weight):
            return 1.0
        return min(max(weight, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Build graph from documents
    # ------------------------------------------------------------------

    async def build_graph(self, documents: List[Dict[str, Any]]) -> None:
        """Extract entities and relations from documents via LLM, build the
        graph, discover communities, and generate community summaries."""
        async with self._operation_lock:
            staging = self._clone_for_build()
            summary_cache = staging._community_summary_cache()
            replaced_doc_ids = {
                str(document.get("doc_id") or document.get("id", "unknown"))
                for document in documents
            }
            for doc_id in replaced_doc_ids:
                staging.delete_document(doc_id)
            await staging._build_graph(
                documents,
                summary_cache=summary_cache,
            )
            with self._state_lock:
                self.entities = staging.entities
                self.relations = staging.relations
                self.communities = staging.communities
                self._adjacency = staging._adjacency

    def _clone_for_build(self) -> GraphRAGEngine:
        """Create an isolated graph copy for long-running LLM enrichment."""
        staging = GraphRAGEngine(
            llm_fn=self.llm_fn,
            entity_llm=self.entity_llm,
            summary_llm=self.summary_llm,
        )
        staging.max_entities_per_doc = self.max_entities_per_doc
        staging.max_total_entities = self.max_total_entities
        staging.max_communities = self.max_communities
        staging.min_community_size = self.min_community_size
        staging.extraction_char_budget = self.extraction_char_budget
        staging.summary_concurrency = self.summary_concurrency
        with self._state_lock:
            (
                staging.entities,
                staging.relations,
                staging.communities,
                staging._adjacency,
            ) = copy.deepcopy(
                (
                    self.entities,
                    self.relations,
                    self.communities,
                    self._adjacency,
                )
            )
        return staging

    async def _build_graph(
        self,
        documents: List[Dict[str, Any]],
        *,
        summary_cache: dict[str, str] | None = None,
    ) -> None:
        if not documents:
            logger.warning("No documents provided; graph will be empty.")
            return

        # Group chunks by source document so graph nodes can be removed when
        # the corresponding document is deleted.
        grouped: Dict[str, list[str]] = {}
        for doc in documents:
            text = doc.get("text") or doc.get("content", "")
            doc_id = str(doc.get("doc_id") or doc.get("id", "unknown"))
            if not text:
                continue
            grouped.setdefault(doc_id, []).append(str(text))
        if not grouped:
            return
        cached_summaries = (
            summary_cache
            if summary_cache is not None
            else self._community_summary_cache()
        )
        for doc_id, texts in grouped.items():
            combined = self._sample_document_text(
                texts,
                self.extraction_char_budget,
            )
            await self._extract_entities_and_relations(combined, doc_id)

        await asyncio.to_thread(self._rebuild_graph_structure)
        self._restore_community_summaries(cached_summaries)
        await self._summarize_communities()

        logger.info(
            "Graph built: %d entities, %d relations, %d communities.",
            len(self.entities),
            len(self.relations),
            len(self.communities),
        )

    @staticmethod
    def _sample_document_text(
        texts: list[str],
        char_budget: int,
    ) -> str:
        """Sample evenly across a document while respecting a prompt budget."""
        combined = "\n\n---\n\n".join(texts)
        if len(combined) <= char_budget:
            return combined
        segment_count = min(8, max(3, char_budget // 1_000))
        separator = "\n\n[...]\n\n"
        usable_budget = char_budget - len(separator) * (segment_count - 1)
        segment_size = max(1, usable_budget // segment_count)
        max_start = max(0, len(combined) - segment_size)
        starts = [
            round(index * max_start / (segment_count - 1))
            for index in range(segment_count)
        ]
        return separator.join(
            combined[start:start + segment_size]
            for start in starts
        )[:char_budget]

    @staticmethod
    def _community_fingerprint(community: Community) -> str:
        payload = "\n".join(
            "\t".join(
                (entity.id, entity.name, entity.type, entity.description)
            )
            for entity in sorted(
                community.entities,
                key=lambda item: item.id,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _community_summary_cache(self) -> dict[str, str]:
        return {
            self._community_fingerprint(community): community.summary
            for community in self.communities
            if community.summary
        }

    def _restore_community_summaries(
        self,
        summary_cache: dict[str, str],
    ) -> None:
        for community in self.communities:
            community.summary = summary_cache.get(
                self._community_fingerprint(community),
                "",
            )

    def save(self, path: str) -> None:
        """Persist the graph to a JSON file."""
        import json
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._state_lock:
            payload = {
                "entities": {eid: {"id": e.id, "name": e.name, "type": e.type,
                                   "description": e.description, "metadata": e.metadata}
                             for eid, e in self.entities.items()},
                "relations": [{"source": r.source, "target": r.target,
                               "relation_type": r.relation_type, "weight": r.weight,
                               "metadata": r.metadata}
                              for r in self.relations],
                # Community 无 label / entity_ids 字段，只序列化已有字段
                "communities": [{"id": c.id,
                                 "entity_ids": [e.id for e in c.entities],
                                 "summary": c.summary}
                                for c in self.communities],
            }
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("GraphRAG state saved to %s", target)

    def load(self, path: str) -> None:
        """Restore the graph from a JSON file."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.entities = {
            eid: Entity(id=e["id"], name=e["name"], type=e.get("type", ""),
                       description=e.get("description", ""),
                       metadata=e.get("metadata", {}))
            for eid, e in payload.get("entities", {}).items()
        }
        for entity in self.entities.values():
            self._normalise_entity_provenance(entity)
        self.relations = [
            Relation(source=r["source"], target=r["target"],
                     relation_type=r.get("relation_type", ""),
                     weight=r.get("weight", 0.5),
                     metadata=r.get("metadata", {}))
            for r in payload.get("relations", [])
        ]
        for relation in self.relations:
            self._normalise_relation_provenance(relation)
        self.communities = []
        for c in payload.get("communities", []):
            entity_ids = set(c.get("entity_ids", []))
            com_entities = [self.entities[eid] for eid in entity_ids
                            if eid in self.entities]
            self.communities.append(
                Community(id=c["id"], entities=com_entities,
                          summary=c.get("summary", ""))
            )
        self._build_adjacency()
        logger.info("GraphRAG state loaded from %s", path)

    async def _extract_entities_and_relations(
        self, text: str, doc_id: str
    ) -> None:
        """Use the LLM to extract entities and relations from a document."""
        if self.entity_llm is None:
            logger.warning("No LLM function; skipping entity extraction.")
            return

        prompt = (
            "Extract entities and their relationships from the following text. "
            "Return the result as a JSON-like list where each item is either:\n"
            '  {"type": "entity", "id": "<unique_id>", "name": "<name>", '
            '"entity_type": "<type>", "description": "<desc>"}\n'
            '  {"type": "relation", "source": "<entity_id>", '
            '"target": "<entity_id>", "relation_type": "<type>", '
            '"weight": <float>}\n\n'
            "Only output the JSON array, no extra text.\n\n"
            f"Text: {text}"
        )

        try:
            result = await self._call_llm(
                prompt,
                llm=self.entity_llm,
            )
            extracted = self._parse_extraction(result)
        except Exception:
            logger.exception("Entity extraction failed for document '%s'.", doc_id)
            return

        entity_items = 0
        relation_items = 0
        existing_keys = {
            (relation.source, relation.target, relation.relation_type)
            for relation in self.relations
        }
        for item in extracted:
            if item.get("type") == "entity":
                entity_items += 1
                if entity_items > self.max_entities_per_doc:
                    continue
                entity = Entity(
                    id=str(item.get("id", ""))[:200],
                    name=str(item.get("name", ""))[:500],
                    type=str(item.get("entity_type", ""))[:200],
                    description=str(item.get("description", ""))[:2000],
                    metadata={
                        "doc_ids": [doc_id],
                        _ENTITY_CONTRIBUTIONS_KEY: {
                            doc_id: {
                                "name": str(item.get("name", ""))[:500],
                                "type": str(
                                    item.get("entity_type", "")
                                )[:200],
                                "description": str(
                                    item.get("description", "")
                                )[:2000],
                            }
                        },
                        _LEGACY_DOC_IDS_KEY: [],
                    },
                )
                if entity.id:
                    existing = self.entities.get(entity.id)
                    if existing is None:
                        if len(self.entities) >= self.max_total_entities:
                            continue
                        self.entities[entity.id] = entity
                    else:
                        self._normalise_entity_provenance(existing)
                        contributions = existing.metadata[
                            _ENTITY_CONTRIBUTIONS_KEY
                        ]
                        contributions[doc_id] = entity.metadata[
                            _ENTITY_CONTRIBUTIONS_KEY
                        ][doc_id]
                        existing.metadata["doc_ids"] = sorted(
                            set(existing.metadata.get("doc_ids", []))
                            | {doc_id}
                        )
                        self._refresh_entity_fields(existing)

            elif item.get("type") == "relation":
                relation_items += 1
                if relation_items > self.max_entities_per_doc * 4:
                    continue
                relation_weight = self._coerce_relation_weight(
                    item.get("weight", 1.0)
                )
                relation = Relation(
                    source=str(item.get("source", ""))[:200],
                    target=str(item.get("target", ""))[:200],
                    relation_type=str(
                        item.get("relation_type", "")
                    )[:200],
                    weight=relation_weight,
                    metadata={
                        "doc_ids": [doc_id],
                        _RELATION_WEIGHTS_KEY: {
                            doc_id: relation_weight
                        },
                        _LEGACY_DOC_IDS_KEY: [],
                    },
                )
                # 暂存本地列表，由 build_graph 在 gather 后统一合并
                # 避免并发写 self.relations 的竞态
                relation_key = (
                    relation.source,
                    relation.target,
                    relation.relation_type,
                )
                if (
                    relation.source
                    and relation.target
                    and relation_key not in existing_keys
                ):
                    self.relations.append(relation)
                    existing_keys.add(relation_key)
                elif relation_key in existing_keys:
                    for existing in self.relations:
                        if (
                            existing.source,
                            existing.target,
                            existing.relation_type,
                        ) == relation_key:
                            self._normalise_relation_provenance(existing)
                            doc_ids = set(
                                existing.metadata.get("doc_ids", [])
                            )
                            doc_ids.add(doc_id)
                            existing.metadata["doc_ids"] = sorted(doc_ids)
                            existing.metadata[_RELATION_WEIGHTS_KEY][
                                doc_id
                            ] = relation.weight
                            self._refresh_relation_weight(existing)
                            break

    async def _call_llm(self, prompt: str, *, llm=None) -> str:
        """Call either a BaseLLM-style adapter or an async callable."""
        selected_llm = llm or self.llm_fn
        if selected_llm is None:
            return ""
        if hasattr(selected_llm, "chat"):
            from mindforge.models.base import ChatMessage

            result = await selected_llm.chat(
                [ChatMessage(role="user", content=prompt)],
                temperature=0.2,
            )
            return str(getattr(result, "content", result))
        result = await selected_llm(prompt)
        return str(getattr(result, "content", result))

    def delete_document(self, doc_id: str) -> None:
        """Remove graph contributions belonging to one source document."""
        with self._state_lock:
            self._delete_document_locked(doc_id)

    async def delete_document_async(self, doc_id: str) -> None:
        """Serialize deletion with graph builds and queries."""
        async with self._operation_lock:
            await asyncio.to_thread(self.delete_document, doc_id)

    def _delete_document_locked(self, doc_id: str) -> None:
        for entity_id, entity in list(self.entities.items()):
            self._normalise_entity_provenance(entity)
            metadata = entity.metadata
            contributions = metadata[_ENTITY_CONTRIBUTIONS_KEY]
            contributions.pop(doc_id, None)

            legacy_doc_ids = set(metadata.get(_LEGACY_DOC_IDS_KEY, []))
            if doc_id in legacy_doc_ids:
                # Legacy files only stored one merged field set for all
                # sources. Drop that unattributed set to prevent deleted
                # content from being assigned to a remaining document.
                legacy_doc_ids.clear()
                metadata.pop(_LEGACY_ENTITY_FIELDS_KEY, None)
            metadata[_LEGACY_DOC_IDS_KEY] = sorted(legacy_doc_ids)
            metadata["doc_ids"] = sorted(
                set(contributions) | legacy_doc_ids
            )

            if not self._refresh_entity_fields(entity):
                self.entities.pop(entity_id, None)

        retained_relations: list[Relation] = []
        for relation in self.relations:
            self._normalise_relation_provenance(relation)
            metadata = relation.metadata
            weights = metadata[_RELATION_WEIGHTS_KEY]
            weights.pop(doc_id, None)

            legacy_doc_ids = set(metadata.get(_LEGACY_DOC_IDS_KEY, []))
            if doc_id in legacy_doc_ids:
                legacy_doc_ids.clear()
                metadata.pop(_LEGACY_RELATION_WEIGHT_KEY, None)
            metadata[_LEGACY_DOC_IDS_KEY] = sorted(legacy_doc_ids)
            metadata["doc_ids"] = sorted(set(weights) | legacy_doc_ids)

            if (
                relation.source in self.entities
                and relation.target in self.entities
                and self._refresh_relation_weight(relation)
            ):
                retained_relations.append(relation)
        self.relations = retained_relations
        self._build_adjacency()
        self._discover_communities()

    def _rebuild_graph_structure(self) -> None:
        with self._state_lock:
            self._build_adjacency()
            self._discover_communities()

    def _parse_extraction(self, raw: str) -> List[Dict[str, Any]]:
        """Parse LLM JSON array output, handling nested JSON with bracket counting."""
        import json

        raw = raw.strip()
        # Try full parse first
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: bracket-balanced extraction
        start = raw.find("[")
        if start == -1:
            return []
        depth = 0
        end = -1
        for i in range(start, len(raw)):
            if raw[i] == "[":
                depth += 1
            elif raw[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            return []
        try:
            parsed = json.loads(raw[start:end+1])
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    # ------------------------------------------------------------------
    # Adjacency & community discovery (BFS)
    # ------------------------------------------------------------------

    def _build_adjacency(self) -> None:
        """Build an undirected adjacency list from the extracted relations."""
        self._adjacency = {eid: set() for eid in self.entities}
        for rel in self.relations:
            self._adjacency.setdefault(rel.source, set()).add(rel.target)
            self._adjacency.setdefault(rel.target, set()).add(rel.source)

    def _discover_communities(self) -> None:
        """Discover communities using a simplified BFS-based approach.

        Each connected component (or sub-component at increasing depth) is
        treated as a community.
        """
        self.communities = []
        visited: Set[str] = set()
        community_id = 0

        for entity_id in self.entities:
            if entity_id in visited:
                continue

            # BFS to find the connected component
            component: List[str] = []
            queue = deque([entity_id])
            visited.add(entity_id)

            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in self._adjacency.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # Create a community from this component
            community_entities = [
                self.entities[eid] for eid in component if eid in self.entities
            ]
            if len(community_entities) < self.min_community_size:
                continue
            community = Community(
                id=f"community_{community_id}",
                entities=community_entities,
            )
            self.communities.append(community)
            community_id += 1
            if community_id >= self.max_communities:
                break

    async def _summarize_communities(self) -> None:
        """Generate a summary for each discovered community via the LLM."""
        pending = [
            community
            for community in self.communities
            if not community.summary
        ]
        if not pending:
            return
        if self.summary_llm is None:
            for comm in pending:
                names = [e.name for e in comm.entities if e.name]
                comm.summary = "Entities: " + ", ".join(names[:10])
            return

        semaphore = asyncio.Semaphore(self.summary_concurrency)

        async def _summarize_one(comm):
            entity_lines = "\n".join(
                f"- {e.name} ({e.type}): {e.description[:100]}"
                for e in comm.entities[:20]
            )
            prompt = (
                "Summarise the following group of related entities in one or "
                "two sentences. Focus on their collective theme or topic.\n\n"
                f"Entities:\n{entity_lines}"
            )
            try:
                async with semaphore:
                    summary = await self._call_llm(
                        prompt,
                        llm=self.summary_llm,
                    )
                comm.summary = summary.strip()
            except Exception:
                comm.summary = "Summary unavailable."

        await asyncio.gather(*[_summarize_one(c) for c in pending])

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(
        self,
        query: str,
        top_k_entities: int = 5,
        top_k_communities: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find relevant entities and communities for the query.

        Uses keyword overlap for entity relevance, then locates the
        communities those entities belong to.

        Returns a list of result dicts with keys: ``id``, ``text``, ``score``,
        ``source`` (always ``"graph"``).
        """
        if not isinstance(query, str) or not query.strip():
            return []
        if (
            isinstance(top_k_entities, bool)
            or not isinstance(top_k_entities, int)
            or not 1 <= top_k_entities <= self.max_total_entities
        ):
            raise ValueError("top_k_entities is outside the configured range.")
        if (
            isinstance(top_k_communities, bool)
            or not isinstance(top_k_communities, int)
            or not 1 <= top_k_communities <= self.max_communities
        ):
            raise ValueError(
                "top_k_communities is outside the configured range."
            )
        return await asyncio.to_thread(
            self._query_sync,
            query.strip(),
            top_k_entities,
            top_k_communities,
        )

    def _query_sync(
        self,
        query: str,
        top_k_entities: int,
        top_k_communities: int,
    ) -> List[Dict[str, Any]]:
        with self._state_lock:
            return self._query_locked(
                query,
                top_k_entities,
                top_k_communities,
            )

    def _query_locked(
        self,
        query: str,
        top_k_entities: int,
        top_k_communities: int,
    ) -> List[Dict[str, Any]]:
        if not self.entities:
            logger.warning("Graph is empty; returning no results.")
            return []

        # 中文查询使用 jieba 分词，避免整段中文当作单个 term
        try:
            import jieba
            query_terms = {
                term.lower()
                for term in jieba.cut_for_search(query)
                if term.strip()
            }
        except Exception:
            query_terms = set(query.lower().split())

        # Score entities by term overlap
        entity_scores: List[Tuple[str, float]] = []
        for eid, entity in self.entities.items():
            name_lower = entity.name.lower()
            desc_lower = entity.description.lower()
            score = 0.0
            for term in query_terms:
                if term in name_lower:
                    score += 2.0
                if term in desc_lower:
                    score += 1.0
            if score > 0:
                entity_scores.append((eid, score))

        entity_scores.sort(key=lambda x: x[1], reverse=True)
        top_entities = entity_scores[:top_k_entities]

        # 归一化实体分到 [0, 1] 区间，与 hybrid/RRF 分数可比
        max_es = max((s for _, s in entity_scores), default=1.0)
        entity_norm = {eid: s / max_es for eid, s in entity_scores} if max_es > 0 else {}

        # Find communities that contain these entities
        relevant_community_ids: Set[str] = set()
        for eid, _ in top_entities:
            for comm in self.communities:
                if eid in {e.id for e in comm.entities}:
                    relevant_community_ids.add(comm.id)
                    if len(relevant_community_ids) >= top_k_communities:
                        break
            if len(relevant_community_ids) >= top_k_communities:
                break

        # Build results
        results: List[Dict[str, Any]] = []

        # Rank 1: matched entities (使用归一化分数)
        for eid, _ in top_entities:
            entity = self.entities[eid]
            results.append(
                {
                    "id": f"entity_{eid}",
                    "text": f"{entity.name} ({entity.type}): {entity.description}",
                    "score": entity_norm.get(eid, 0.5),
                    "source": "graph",
                    "entity_id": eid,
                }
            )

        # Rank 2: matched communities（仅含≥2实体的社区，且分数基于所含实体平均归一化分）
        for comm in self.communities:
            if comm.id not in relevant_community_ids:
                continue
            comm_norm = sum(
                entity_norm.get(e.id, 0) for e in comm.entities
            ) / max(len(comm.entities), 1)
            results.append(
                {
                    "id": comm.id,
                    "text": comm.summary,
                    "score": round(comm_norm, 4),
                    "source": "graph",
                    "community_id": comm.id,
                }
            )

        # Rank 3: relations from matched entities（也归一化到实体分区间）
        top_eid_set = {eid for eid, _ in top_entities}
        rel_max = max((r.weight for r in self.relations
                       if r.source in top_eid_set or r.target in top_eid_set),
                      default=1.0)
        for rel in self.relations:
            if rel.source in top_eid_set or rel.target in top_eid_set:
                src_name = self.entities.get(rel.source, Entity("", "")).name
                tgt_name = self.entities.get(rel.target, Entity("", "")).name
                rel_norm = (rel.weight / rel_max) * 0.5 if rel_max > 0 else 0.15
                results.append(
                    {
                        "id": f"rel_{rel.source}_{rel.target}",
                        "text": f"{src_name} --[{rel.relation_type}]--> {tgt_name}",
                        "score": round(rel_norm, 4),
                        "source": "graph",
                    }
                )

        # Sort descending by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
