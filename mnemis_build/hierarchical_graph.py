from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from .config import BuildConfig
from .llm import OpenAILLMClient
from .logging_utils import get_logger
from .models import CategoryAssignmentPayload, CategoryDetailsPayload, CategoryRecord, IndexedNode, make_uuid
from .neo4j_store import Neo4jGraphStore
from .prompts import (
    CATEGORY_DETAILS_PROMPT,
    HIERARCHICAL_SYSTEM_PROMPT,
    build_category_details_user_prompt,
    build_hierarchy_user_prompt,
)


PREV_EXAMPLE = """Layer 1:
- Microsoft Research Asia -> Microsoft Research Labs
- Microsoft Research Asia -> NLP-focused Labs
- Microsoft Research Shanghai -> Microsoft Research Labs

Layer 2:
- Microsoft Research Labs -> Tech Company Labs
- Microsoft Research Labs -> AI Research Organizations
"""


@dataclass(slots=True)
class MaterializedLayer:
    categories: list[CategoryRecord]
    natural_child_uuids: set[str]
    promoted_child_uuids: set[str]


class HierarchicalGraphBuilder:
    def __init__(self, store: Neo4jGraphStore, llm: OpenAILLMClient, config: BuildConfig):
        self.store = store
        self.llm = llm
        self.config = config
        self.logger = get_logger("hierarchy")

    async def rebuild(self, group_id: str) -> list[CategoryRecord]:
        self.logger.info("hierarchy rebuild start | group_id=%s", group_id)
        await self.store.clear_hierarchy(group_id)
        current_nodes = [
            IndexedNode(index=i, layer=0, **node)
            for i, node in enumerate(await self.store.fetch_layer_zero_nodes(group_id))
        ]
        created: list[CategoryRecord] = []
        existing_categories: dict[str, str] = {}

        for layer in range(1, self.config.max_hierarchy_layers + 1):
            if not current_nodes:
                break
            self.logger.info("hierarchy layer start | group_id=%s, layer=%s, input_nodes=%s", group_id, layer, len(current_nodes))
            assignable_nodes, speaker_nodes = self._partition_nodes_for_assignment(layer, current_nodes)
            assignments = await self._extract_categories(
                layer,
                assignable_nodes,
                existing_categories,
                self._build_speaker_policy_note(layer, speaker_nodes),
            )
            materialized = await self._materialize_categories(
                group_id,
                layer,
                assignable_nodes,
                speaker_nodes,
                assignments,
            )
            categories = materialized.categories
            if not categories:
                self.logger.info("hierarchy layer empty | group_id=%s, layer=%s", group_id, layer)
                break
            if not self._passes_compression(layer, current_nodes, materialized):
                self.logger.info("hierarchy compression stop | group_id=%s, layer=%s, category_count=%s", group_id, layer, len(categories))
                break

            summary_embeddings = await self.llm.embed([category.summary for category in categories])
            for category, embedding in zip(categories, summary_embeddings):
                await self.store.upsert_category(category, embedding)
                for child_uuid in category.child_uuids:
                    await self.store.connect_category(category.uuid, child_uuid, group_id)
                existing_categories[category.name] = category.summary
            created.extend(categories)
            current_nodes = [
                IndexedNode(
                    index=i,
                    uuid=category.uuid,
                    name=category.name,
                    summary=category.summary,
                    tag=category.tag,
                    layer=layer,
                )
                for i, category in enumerate(categories)
            ]
            self.logger.info("hierarchy layer done | group_id=%s, layer=%s, category_count=%s", group_id, layer, len(categories))
        self.logger.info("hierarchy rebuild done | group_id=%s, created_categories=%s", group_id, len(created))
        return created

    async def _extract_categories(
        self,
        layer: int,
        nodes: list[IndexedNode],
        existing_categories: dict[str, str],
        speaker_policy_note: str | None = None,
    ) -> CategoryAssignmentPayload:
        if not nodes:
            return CategoryAssignmentPayload(assignments=[])
        self.logger.info("category assignment start | layer=%s, node_count=%s", layer, len(nodes))
        content = "\n".join(
            f"{node.index}. {node.name}: [{node.summary}] tags={node.tag}"
            for node in nodes[: self.config.max_categories_per_call] if self.config.max_categories_per_call > 0
        ) or "\n".join(
            f"{node.index}. {node.name}: [{node.summary}] tags={node.tag}"
            for node in nodes
        )
        existing = "\n".join(
            f"- {name}: {summary}"
            for name, summary in sorted(existing_categories.items())
        ) or "None"
        payload = await self.llm.complete_json(
            CategoryAssignmentPayload,
            [
                {"role": "system", "content": HIERARCHICAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_hierarchy_user_prompt(
                        layer,
                        content,
                        existing,
                        PREV_EXAMPLE,
                        speaker_policy_note=speaker_policy_note,
                    ),
                },
            ],
            stage="hierarchical_graph_ingestion",
            operation="category_assignment",
            require_json_object=False,
        )
        self.logger.info("category assignment done | layer=%s, assignment_count=%s", layer, len(payload.assignments))
        return payload

    def _is_appendix_prompt_speaker_node(self, node: IndexedNode) -> bool:
        return node.name.strip().lower() in {"user", "i", "me"}

    def _is_reserved_speaker_node(self, layer: int, node: IndexedNode) -> bool:
        if layer != 1:
            return False
        mode = self.config.speaker_hierarchy_mode
        if mode == "disabled":
            return False
        if mode == "paper_v2":
            return node.is_speaker
        return self._is_appendix_prompt_speaker_node(node)

    def _partition_nodes_for_assignment(
        self,
        layer: int,
        nodes: list[IndexedNode],
    ) -> tuple[list[IndexedNode], list[IndexedNode]]:
        # Speaker handling is resolved here so the LLM only clusters the remaining nodes.
        speaker_nodes: list[IndexedNode] = []
        assignable_nodes: list[IndexedNode] = []
        for node in nodes:
            if self._is_reserved_speaker_node(layer, node):
                speaker_nodes.append(node)
            else:
                assignable_nodes.append(node)
        return assignable_nodes, speaker_nodes

    def _build_speaker_policy_note(self, layer: int, speaker_nodes: list[IndexedNode]) -> str | None:
        if not speaker_nodes:
            return None
        if self.config.speaker_hierarchy_mode == "paper_v2":
            return (
                'Speaker handling follows paper v2 main-text semantics for Layer 1: speaker entities are reserved '
                'outside this categorization step and will be grouped into one dedicated category named "Speaker".'
            )
        if self.config.speaker_hierarchy_mode == "appendix_prompt":
            return (
                'Speaker handling follows the appendix prompt semantics for Layer 1: only nodes literally named '
                '"user", "I", or "me" are reserved outside this categorization step and will be grouped into one '
                'dedicated category named "Speaker".'
            )
        return None

    async def _materialize_categories(
        self,
        group_id: str,
        layer: int,
        assignable_nodes: list[IndexedNode],
        speaker_nodes: list[IndexedNode],
        assignments: CategoryAssignmentPayload,
    ) -> MaterializedLayer:
        nodes_by_index = {node.index: node for node in assignable_nodes}
        grouped: dict[str, list[IndexedNode]] = defaultdict(list)
        natural_child_uuids: set[str] = set()
        promoted_child_uuids: set[str] = set()
        min_children = self.config.min_children_per_category
        for assignment in assignments.assignments:
            name = assignment.category.strip()
            if not name or " and " in f" {name.lower()} ":
                continue
            unique_indexes: list[int] = []
            for idx in assignment.indexes:
                if idx in nodes_by_index and idx not in unique_indexes:
                    unique_indexes.append(idx)
            if not unique_indexes:
                continue
            if len(unique_indexes) < min_children:
                continue
            for idx in unique_indexes:
                node = nodes_by_index[idx]
                natural_child_uuids.add(node.uuid)
                grouped[name].append(node)
        for node in assignable_nodes:
            if node.uuid not in natural_child_uuids:
                grouped[self._build_promoted_category_name(node, grouped)].append(node)
                promoted_child_uuids.add(node.uuid)
        if speaker_nodes:
            grouped["Speaker"].extend(speaker_nodes)

        details = await self._generate_category_details(layer, grouped)
        categories: list[CategoryRecord] = []
        for name, members in grouped.items():
            deduped_members = list({member.uuid: member for member in members}.values())
            detail = details.get(name, {})
            categories.append(
                CategoryRecord(
                    uuid=make_uuid("category"),
                    group_id=group_id,
                    name=name,
                    summary=detail.get("summary") or self._summarize_category(name, deduped_members),
                    tag=detail.get("tag") or self._merge_tags(deduped_members),
                    layer=layer,
                    child_uuids=[member.uuid for member in deduped_members],
                )
            )
        return MaterializedLayer(
            categories=categories,
            natural_child_uuids=natural_child_uuids,
            promoted_child_uuids=promoted_child_uuids,
        )

    async def _generate_category_details(
        self,
        layer: int,
        grouped: dict[str, list[IndexedNode]],
    ) -> dict[str, dict[str, object]]:
        if not grouped:
            return {}
        self.logger.info("category detail generation start | layer=%s, category_count=%s", layer, len(grouped))
        category_blocks: list[str] = []
        for name, members in grouped.items():
            lines = [f"Category: {name}", "Members:"]
            for member in members:
                lines.append(
                    f'- {member.name}: summary="{member.summary}" tags={json.dumps(member.tag, ensure_ascii=False)}'
                )
            category_blocks.append("\n".join(lines))
        payload = await self.llm.complete_json(
            CategoryDetailsPayload,
            [
                {"role": "system", "content": CATEGORY_DETAILS_PROMPT},
                {
                    "role": "user",
                    "content": build_category_details_user_prompt(layer, "\n\n".join(category_blocks)),
                },
            ],
            stage="hierarchical_graph_ingestion",
            operation="category_detail_generation",
        )
        result = {
            category.name: {
                "summary": category.summary.strip(),
                "tag": [tag.strip() for tag in category.tag if tag.strip()][:5],
            }
            for category in payload.categories
            if category.name.strip()
        }
        self.logger.info("category detail generation done | layer=%s, detailed_category_count=%s", layer, len(result))
        return result

    def _passes_compression(
        self,
        layer: int,
        current_nodes: list[IndexedNode],
        materialized: MaterializedLayer,
    ) -> bool:
        min_children = self.config.min_children_per_category
        categories = materialized.categories
        promoted_child_uuids = materialized.promoted_child_uuids

        for category in categories:
            if category.name == "Speaker":
                continue
            child_count = len(category.child_uuids)
            if child_count >= min_children:
                continue
            if child_count != 1:
                return False
            promoted_uuid = category.child_uuids[0]
            if promoted_uuid not in promoted_child_uuids:
                return False
        if layer >= 2 and len(categories) > len(current_nodes):
            return False
        return True

    def _build_promoted_category_name(
        self,
        node: IndexedNode,
        grouped: dict[str, list[IndexedNode]],
    ) -> str:
        base_name = node.name.strip() or node.uuid
        if base_name not in grouped:
            return base_name
        candidate = f"{base_name} (Promoted)"
        suffix = 2
        while candidate in grouped:
            candidate = f"{base_name} (Promoted {suffix})"
            suffix += 1
        return candidate

    def _summarize_category(self, name: str, members: list[IndexedNode]) -> str:
        member_names = ", ".join(member.name for member in members[:6])
        return f"{name} groups semantically related nodes such as {member_names}."

    def _merge_tags(self, members: list[IndexedNode]) -> list[str]:
        merged: list[str] = []
        for member in members:
            for tag in member.tag:
                if tag not in merged:
                    merged.append(tag)
                if len(merged) >= 5:
                    return merged
        return merged
