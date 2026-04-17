from __future__ import annotations

import json
from collections import defaultdict

from .config import BuildConfig
from .llm import OpenAILLMClient
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


class HierarchicalGraphBuilder:
    def __init__(self, store: Neo4jGraphStore, llm: OpenAILLMClient, config: BuildConfig):
        self.store = store
        self.llm = llm
        self.config = config

    async def rebuild(self, group_id: str) -> list[CategoryRecord]:
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
            assignments = await self._extract_categories(layer, current_nodes, existing_categories)
            categories = await self._materialize_categories(group_id, layer, current_nodes, assignments)
            if not categories:
                break
            if not self._passes_compression(layer, current_nodes, categories):
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
        return created

    async def _extract_categories(
        self,
        layer: int,
        nodes: list[IndexedNode],
        existing_categories: dict[str, str],
    ) -> CategoryAssignmentPayload:
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
        return await self.llm.complete_json(
            CategoryAssignmentPayload,
            [
                {"role": "system", "content": HIERARCHICAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_hierarchy_user_prompt(layer, content, existing, PREV_EXAMPLE),
                },
            ],
            require_json_object=False,
        )

    def _is_speaker_node(self, node: IndexedNode) -> bool:
        return node.name.strip().lower() in {"user", "i", "me"}

    async def _materialize_categories(
        self,
        group_id: str,
        layer: int,
        nodes: list[IndexedNode],
        assignments: CategoryAssignmentPayload,
    ) -> list[CategoryRecord]:
        nodes_by_index = {node.index: node for node in nodes}
        grouped: dict[str, list[IndexedNode]] = defaultdict(list)
        covered: set[int] = set()
        speaker_indexes = {node.index for node in nodes if self._is_speaker_node(node)}
        for assignment in assignments.assignments:
            name = assignment.category.strip()
            if not name or " and " in f" {name.lower()} ":
                continue
            unique_indexes: list[int] = []
            for idx in assignment.indexes:
                if idx in nodes_by_index and idx not in unique_indexes:
                    unique_indexes.append(idx)
            if name.lower() != "speaker":
                unique_indexes = [idx for idx in unique_indexes if idx not in speaker_indexes]
            if not unique_indexes:
                continue
            for idx in unique_indexes:
                covered.add(idx)
                grouped[name].append(nodes_by_index[idx])
        for idx in speaker_indexes:
            covered.add(idx)
            grouped["Speaker"].append(nodes_by_index[idx])
        for node in nodes:
            if node.index not in covered:
                if self._is_speaker_node(node):
                    grouped["Speaker"].append(node)
                else:
                    grouped[node.name].append(node)

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
        return categories

    async def _generate_category_details(
        self,
        layer: int,
        grouped: dict[str, list[IndexedNode]],
    ) -> dict[str, dict[str, object]]:
        if not grouped:
            return {}
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
        )
        return {
            category.name: {
                "summary": category.summary.strip(),
                "tag": [tag.strip() for tag in category.tag if tag.strip()][:5],
            }
            for category in payload.categories
            if category.name.strip()
        }

    def _passes_compression(
        self,
        layer: int,
        current_nodes: list[IndexedNode],
        categories: list[CategoryRecord],
    ) -> bool:
        min_children = self.config.min_children_per_category
        if any(len(category.child_uuids) < min_children and category.name != "Speaker" for category in categories):
            oversized_next_layer = len(categories) > len(current_nodes) if layer >= 2 else False
            if oversized_next_layer:
                return False
        if layer >= 2 and len(categories) > len(current_nodes):
            return False
        return True

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
