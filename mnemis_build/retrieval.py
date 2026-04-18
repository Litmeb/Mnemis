from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from global_selection.prompts import NODE_SELECTION_PROMPT_TEMPLATE

from .config import BuildConfig
from .llm import OpenAILLMClient
from .models import NodeSelectionList
from .neo4j_store import Neo4jGraphStore
from .prompts import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_user_prompt,
)
from .reranker import RerankCandidate, RerankBackendStatus, build_reranker
from .logging_utils import get_logger


class MnemisRetriever:
    def __init__(self, store: Neo4jGraphStore, llm: OpenAILLMClient, config: BuildConfig):
        self.store = store
        self.llm = llm
        self.config = config
        self.reranker = build_reranker(llm, config)
        self.logger = get_logger("retrieval")

    def _annotate_route(
        self,
        items: list[dict[str, Any]],
        *,
        branch: str,
        score_lookup: dict[str, float] | None = None,
        fallback_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        annotated: list[dict[str, Any]] = []
        for item in items:
            current = dict(item)
            current["system1_branch"] = branch
            current["rrf_score"] = score_lookup.get(current["uuid"], fallback_score) if score_lookup else current.get(
                "rrf_score", fallback_score
            )
            annotated.append(current)
        return annotated

    def _merge_items_by_uuid(
        self,
        *item_groups: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for group in item_groups:
            for item in group:
                current = merged.setdefault(item["uuid"], {})
                for key, value in item.items():
                    if value is not None or key not in current:
                        current[key] = value
                current["rrf_score"] = max(current.get("rrf_score", 0.0), item.get("rrf_score", 0.0))
        return list(merged.values())

    def _normalize_timestamp(self, value: Any) -> str | Any:
        if isinstance(value, datetime):
            return value.strftime("%Y/%m/%d (%a) %H:%M")
        if hasattr(value, "to_native"):
            try:
                native_value = value.to_native()
                if isinstance(native_value, datetime):
                    return native_value.strftime("%Y/%m/%d (%a) %H:%M")
                return str(native_value)
            except Exception:
                return str(value)
        if hasattr(value, "iso_format"):
            try:
                return value.iso_format()
            except Exception:
                return str(value)
        return value

    def _format_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for item in items:
            current = dict(item)
            if "valid_at" in current:
                current["valid_at"] = self._normalize_timestamp(current["valid_at"])
            if "invalid_at" in current:
                current["invalid_at"] = self._normalize_timestamp(current["invalid_at"])
            formatted.append(current)
        return formatted

    def _node_prompt_payload(self, nodes: list[dict[str, Any]]) -> str:
        payload = []
        for node in nodes:
            payload.append(
                {
                    "uuid": node["uuid"],
                    "name": node["name"],
                    "tag": node.get("tag", []),
                }
            )
        return "\n".join(json.dumps(item, ensure_ascii=False) for item in payload)

    def _is_entity_node(self, node: dict[str, Any]) -> bool:
        return node.get("layer") in (None, 0)

    def _split_entity_and_category_nodes(
        self,
        nodes: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        entities: list[dict[str, Any]] = []
        categories: list[dict[str, Any]] = []
        for node in nodes:
            if self._is_entity_node(node):
                entities.append(node)
            else:
                categories.append(node)
        return entities, categories

    async def _resolve_category_entities(
        self,
        group_id: str,
        categories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not categories:
            return []
        return await self.store.fetch_descendant_entities(
            group_id,
            [node["uuid"] for node in categories],
        )

    async def _layer_selection(
        self,
        query: str,
        current_layer_nodes: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        prompt = NODE_SELECTION_PROMPT_TEMPLATE.format(
            query=query,
            nodes_info=self._node_prompt_payload(current_layer_nodes),
        )
        response = await self.llm.complete_json(
            NodeSelectionList,
            [{"role": "user", "content": prompt}],
            stage="retrieval",
            operation="layer_selection",
            use_small_model=True,
        )
        nodes_by_uuid = {node["uuid"]: node for node in current_layer_nodes}
        nodes_by_name = {node["name"]: node for node in current_layer_nodes}
        selected: list[dict[str, Any]] = []
        shortcuts: list[dict[str, Any]] = []
        unmatched: list[dict[str, str]] = []
        for choice in response.selections:
            node = nodes_by_uuid.get(choice.uuid) or nodes_by_name.get(choice.name)
            if not node:
                unmatched.append({"name": choice.name, "uuid": choice.uuid})
                continue
            if choice.get_all_children:
                shortcuts.append(node)
            else:
                selected.append(node)
        if unmatched:
            self.logger.warning(
                "layer selection returned unmatched nodes | query=%r, unmatched_count=%s, unmatched=%s",
                query,
                len(unmatched),
                unmatched[:5],
            )
        if response.selections and not selected and not shortcuts:
            raise ValueError(
                "Layer selection returned only unmatched nodes; aborting System-2 retrieval instead of continuing empty."
            )
        self.logger.info(
            "layer selection parsed | query=%r, candidate_count=%s, selected=%s, shortcuts=%s",
            query,
            len(current_layer_nodes),
            len(selected),
            len(shortcuts),
        )
        return selected, shortcuts

    async def _system2_retrieve(self, query: str, group_id: str) -> dict[str, list[dict[str, Any]]]:
        max_layer = await self.store.fetch_max_layer(group_id)
        previous_layer_nodes: list[dict[str, Any]] = []
        selected_entities: dict[str, dict[str, Any]] = {}
        selected_categories: dict[str, dict[str, Any]] = {}

        for layer in range(max_layer, 0, -1):
            if layer == max_layer:
                current_layer_nodes = await self.store.fetch_nodes_by_layer(group_id, layer)
            elif previous_layer_nodes:
                current_layer_nodes = await self.store.fetch_child_nodes(
                    group_id,
                    [node["uuid"] for node in previous_layer_nodes],
                )
            else:
                break
            selected, shortcuts = await self._layer_selection(query, current_layer_nodes)
            selected_entity_nodes, selected_category_nodes = self._split_entity_and_category_nodes(selected)
            for node in selected_entity_nodes:
                selected_entities[node["uuid"]] = node
            for node in selected_category_nodes:
                selected_categories[node["uuid"]] = node

            shortcut_entity_nodes, shortcut_category_nodes = self._split_entity_and_category_nodes(shortcuts)
            for node in shortcut_entity_nodes:
                selected_entities[node["uuid"]] = node
            descendants = await self.store.fetch_all_descendants(
                group_id,
                [node["uuid"] for node in shortcut_category_nodes],
            )
            descendant_entities, _ = self._split_entity_and_category_nodes(descendants)
            for node in descendant_entities:
                selected_entities[node["uuid"]] = node

            if layer == 1 and selected_category_nodes:
                resolved_entities = await self._resolve_category_entities(group_id, selected_category_nodes)
                for node in resolved_entities:
                    selected_entities[node["uuid"]] = node
                for category in selected_category_nodes:
                    selected_categories.pop(category["uuid"], None)

            previous_layer_nodes = selected_category_nodes

        if selected_categories:
            resolved_entities = await self._resolve_category_entities(group_id, list(selected_categories.values()))
            for node in resolved_entities:
                selected_entities[node["uuid"]] = node

        neighbors = await self.store.fetch_one_hop_neighbors(group_id, list(selected_entities))
        selected_entities.update({node["uuid"]: node for node in neighbors["nodes"]})
        return {
            "episodes": self._format_items(neighbors["episodes"]),
            "edges": self._format_items(neighbors["edges"]),
            "nodes": self._format_items(list(selected_entities.values())),
        }

    async def _system1_retrieve(self, query: str, group_id: str) -> dict[str, list[dict[str, Any]]]:
        query_embedding = (await self.llm.embed([query]))[0]
        candidate_limit = self.config.retrieval_candidate_limit
        entity_candidates = await self.store.search_entities(group_id, query, query_embedding, limit=candidate_limit)
        entity_candidates = entity_candidates[: self.config.entity_top_k]
        candidate_scores = {
            item["uuid"]: item.get("rrf_score", 0.0)
            for item in entity_candidates
        }
        expanded = await self.store.expand_entities_for_retrieval(
            group_id,
            [item["uuid"] for item in entity_candidates],
            limit=candidate_limit,
        )
        expanded_episode_scores = {
            item["uuid"]: max(
                [candidate_scores.get(uuid, 0.0) for uuid in item.get("matched_entity_uuids", [])],
                default=0.0,
            )
            for item in expanded["episodes"]
        }
        expanded_edge_scores = {
            item["uuid"]: max(
                [candidate_scores.get(uuid, 0.0) for uuid in item.get("matched_entity_uuids", [])],
                default=0.0,
            )
            for item in expanded["edges"]
        }
        expanded_node_scores = {
            item["uuid"]: max(
                [candidate_scores.get(uuid, 0.0) for uuid in item.get("matched_entity_uuids", [])],
                default=0.0,
            )
            for item in expanded["nodes"]
        }
        direct_episode_hits = await self.store.search_episodes(group_id, query, query_embedding, limit=candidate_limit)
        nodes = self._annotate_route(entity_candidates, branch="entity_candidates")
        episodes = self._merge_items_by_uuid(
            self._annotate_route(expanded["episodes"], branch="entity_expansion", score_lookup=expanded_episode_scores),
            self._annotate_route(direct_episode_hits, branch="direct_episode", fallback_score=0.0),
        )
        edges = self._annotate_route(expanded["edges"], branch="entity_expansion", score_lookup=expanded_edge_scores)
        expanded_nodes = self._annotate_route(expanded["nodes"], branch="entity_expansion", score_lookup=expanded_node_scores)
        nodes = self._merge_items_by_uuid(nodes, expanded_nodes)
        self.logger.info(
            "system1 entity-first | query=%r, entity_candidates=%s, entity_preview=%s, expanded_episodes=%s, expanded_episode_sources=%s, expanded_edges=%s, expanded_neighbor_nodes=%s, direct_episode_hits=%s",
            query,
            len(entity_candidates),
            [
                {
                    "uuid": item["uuid"],
                    "name": item.get("name"),
                    "rrf_score": round(item.get("rrf_score", 0.0), 4),
                }
                for item in entity_candidates[:5]
            ],
            len(expanded["episodes"]),
            [item.get("source_id") for item in expanded["episodes"][:10]],
            len(expanded["edges"]),
            len(expanded["nodes"]),
            len(direct_episode_hits),
        )
        return {
            "episodes": self._format_items(sorted(
                episodes,
                key=lambda item: (
                    item.get("rrf_score", 0.0),
                    item.get("matched_entity_count", 0),
                    item.get("fulltext_score", 0.0),
                    item.get("similarity_score", 0.0),
                ),
                reverse=True,
            )[:candidate_limit]),
            "nodes": self._format_items(sorted(
                nodes,
                key=lambda item: (
                    item.get("rrf_score", 0.0),
                    item.get("matched_entity_count", 0),
                    item.get("fulltext_score", 0.0),
                    item.get("similarity_score", 0.0),
                ),
                reverse=True,
            )[:candidate_limit]),
            "edges": self._format_items(sorted(
                edges,
                key=lambda item: (
                    item.get("rrf_score", 0.0),
                    item.get("matched_entity_count", 0),
                    item.get("fulltext_score", 0.0),
                    item.get("similarity_score", 0.0),
                ),
                reverse=True,
            )[:candidate_limit]),
        }

    def _merge_route_items(
        self,
        system1: dict[str, list[dict[str, Any]]],
        system2: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        merged: dict[str, list[dict[str, Any]]] = {}
        for key in ("episodes", "nodes", "edges"):
            current: dict[str, dict[str, Any]] = {}
            for item in system1.get(key, []):
                current[item["uuid"]] = {**item, "route_score": item.get("rrf_score", 0.0)}
            for item in system2.get(key, []):
                existing = current.get(item["uuid"], {})
                current[item["uuid"]] = {
                    **existing,
                    **item,
                    "route_score": max(existing.get("route_score", 0.0), 1.0),
                }
            merged[key] = list(current.values())
        return merged

    def _candidate_text(self, item_type: str, item: dict[str, Any]) -> str:
        if item_type == "episodes":
            return f"[{item.get('valid_at', 'unknown')}] {item.get('content', '')}"
        if item_type == "nodes":
            return json.dumps(
                {
                    "name": item.get("name"),
                    "tag": item.get("tag", []),
                    "summary": item.get("summary", ""),
                    "layer": item.get("layer"),
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "fact": item.get("fact"),
                "valid_at": self._normalize_timestamp(item.get("valid_at")),
                "invalid_at": self._normalize_timestamp(item.get("invalid_at")),
                "source_name": item.get("source_name"),
                "target_name": item.get("target_name"),
            },
            ensure_ascii=False,
        )

    async def _rerank_items(
        self,
        query: str,
        item_type: str,
        items: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], RerankBackendStatus]:
        if not items:
            empty_status = self.reranker.last_status or RerankBackendStatus(
                mode="not_used",
                backend="none",
                model=None,
                requested_mode=self.config.rerank_mode,
            )
            return [], empty_status
        candidates = [
            RerankCandidate(
                uuid=item["uuid"],
                text=self._candidate_text(item_type, item),
            )
            for item in items
        ]
        response = await self.reranker.rerank(query=query, item_type=item_type, candidates=candidates)
        ranked = sorted(
            items,
            key=lambda item: (
                response.scores.get(item["uuid"], -1.0),
                item.get("route_score", 0.0),
                item.get("rrf_score", 0.0),
            ),
            reverse=True,
        )
        enriched: list[dict[str, Any]] = []
        for item in ranked[:top_k]:
            enriched.append(
                {
                    **item,
                    "rerank_score": response.scores.get(item["uuid"]),
                    "rerank_mode": response.status.mode,
                }
            )
        return enriched, response.status

    def _format_context(
        self,
        episodes: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> str:
        sections: list[str] = []
        if nodes:
            sections.append("<NODES>")
            for item in nodes:
                sections.append(json.dumps(item, ensure_ascii=False))
            sections.append("</NODES>")
        if edges:
            sections.append("<EDGES>")
            for item in edges:
                sections.append(json.dumps(item, ensure_ascii=False))
            sections.append("</EDGES>")
        if episodes:
            sections.append("<EPISODES>")
            for item in episodes:
                sections.append(json.dumps(item, ensure_ascii=False))
            sections.append("</EPISODES>")
        return "\n".join(sections)

    async def retrieve(self, query: str, group_id: str) -> dict[str, Any]:
        await self.store.ensure_indexes()
        system1 = await self._system1_retrieve(query, group_id)
        system2 = await self._system2_retrieve(query, group_id)
        merged = self._merge_route_items(system1, system2)
        episodes, episode_rerank = await self._rerank_items(query, "episodes", merged["episodes"], self.config.episode_top_k)
        nodes, node_rerank = await self._rerank_items(query, "nodes", merged["nodes"], self.config.entity_top_k)
        edges, edge_rerank = await self._rerank_items(query, "edges", merged["edges"], self.config.edge_top_k)
        active_status = edge_rerank or node_rerank or episode_rerank
        return {
            "system1": system1,
            "system2": system2,
            "rerank": {
                "configured_mode": self.config.rerank_mode,
                "active_mode": active_status.mode,
                "backend": active_status.backend,
                "model": active_status.model,
                "fallback_reason": active_status.fallback_reason,
                "per_type": {
                    "episodes": episode_rerank.to_dict(),
                    "nodes": node_rerank.to_dict(),
                    "edges": edge_rerank.to_dict(),
                },
            },
            "final": {
                "episodes": episodes,
                "nodes": nodes,
                "edges": edges,
            },
            "context": self._format_context(episodes, nodes, edges),
        }

    async def answer(self, query: str, group_id: str) -> dict[str, Any]:
        retrieval = await self.retrieve(query, group_id)
        answer = await self.llm.complete_text(
            [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": build_answer_user_prompt(query, retrieval["context"])},
            ]
        )
        return {
            "query": query,
            "group_id": group_id,
            "answer": answer.strip(),
            "retrieval": retrieval,
        }
