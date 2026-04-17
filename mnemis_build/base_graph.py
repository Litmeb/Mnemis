from __future__ import annotations

import json
from collections import OrderedDict

from .config import BuildConfig
from .llm import OpenAILLMClient
from .models import (
    EdgeExtractionPayload,
    EntityExtractionPayload,
    EntityNameExtraction,
    EntityRecord,
    EpisodeInput,
    make_uuid,
)
from .neo4j_store import Neo4jGraphStore
from .prompts import (
    EDGE_EXTRACTION_PROMPT,
    EDGE_REFLECTION_PROMPT,
    ENTITY_DETAILS_PROMPT,
    ENTITY_NAME_EXTRACTION_PROMPT,
    ENTITY_REFLECTION_PROMPT,
)


class BaseGraphBuilder:
    def __init__(self, store: Neo4jGraphStore, llm: OpenAILLMClient, config: BuildConfig):
        self.store = store
        self.llm = llm
        self.config = config

    async def build(self, group_id: str, episodes: list[EpisodeInput]) -> list[str]:
        await self.store.ensure_indexes()
        created_episode_ids: list[str] = []
        for episode in episodes:
            episode_uuid = make_uuid("episode")
            created_episode_ids.append(episode_uuid)
            recent = await self.store.fetch_recent_episodes(group_id, self.config.recent_episode_window)
            context = self._format_context(episode, recent)
            episode_embedding = (await self.llm.embed([episode.content]))[0]
            await self.store.upsert_episode(group_id, episode_uuid, episode, episode_embedding)
            deduped_entities = await self._extract_entities(group_id, episode_uuid, episode, context)
            await self._extract_edges(group_id, context, deduped_entities)
        return created_episode_ids

    def _format_context(self, episode: EpisodeInput, recent: list[dict]) -> str:
        lines = [
            f"Current Episode ({episode.source_id})",
            f"Speaker: {episode.speaker}",
            episode.content,
            "",
            "Recent Episodes:",
        ]
        for item in recent:
            lines.append(f"- [{item.get('source_id', item['uuid'])}] {item['content']}")
        return "\n".join(lines)

    def _normalize_text(self, value: str) -> str:
        return " ".join(value.lower().split())

    def _choose_entity_match(self, name: str, candidates: list[dict]) -> dict | None:
        normalized = self._normalize_text(name)
        for candidate in candidates:
            if self._normalize_text(candidate["name"]) == normalized:
                return candidate
        if not candidates:
            return None
        best = candidates[0]
        similarity = float(best.get("similarity_score") or 0.0)
        fulltext = float(best.get("fulltext_score") or 0.0)
        if similarity >= 0.92:
            return best
        if similarity >= 0.88 and fulltext > 0:
            return best
        return None

    def _choose_edge_match(self, fact: str, candidates: list[dict]) -> dict | None:
        normalized = self._normalize_text(fact)
        for candidate in candidates:
            if self._normalize_text(candidate["fact"]) == normalized:
                return candidate
        if not candidates:
            return None
        best = candidates[0]
        similarity = float(best.get("similarity_score") or 0.0)
        fulltext = float(best.get("fulltext_score") or 0.0)
        if similarity >= 0.94:
            return best
        if similarity >= 0.88 and fulltext > 0:
            return best
        return None

    async def _extract_entities(
        self,
        group_id: str,
        episode_uuid: str,
        episode: EpisodeInput,
        context: str,
    ) -> list[EntityRecord]:
        extraction = await self.llm.complete_json(
            EntityNameExtraction,
            [
                {"role": "system", "content": ENTITY_NAME_EXTRACTION_PROMPT},
                {"role": "user", "content": context},
            ],
            use_small_model=True,
        )
        names = OrderedDict((name.strip(), None) for name in extraction.names if name.strip())
        names.setdefault(episode.speaker.strip() or "user", None)
        for _ in range(self.config.max_reflection_rounds):
            reflection = await self.llm.complete_json(
                EntityNameExtraction,
                [
                    {"role": "system", "content": ENTITY_REFLECTION_PROMPT},
                    {"role": "user", "content": f"{context}\n\nAlready extracted: {json.dumps(list(names.keys()), ensure_ascii=False)}"},
                ],
                use_small_model=True,
            )
            for name in reflection.names:
                clean = name.strip()
                if clean:
                    names.setdefault(clean, None)

        embeddings = await self.llm.embed(list(names.keys()))
        deduped: list[EntityRecord] = []
        existing_by_name: dict[str, EntityRecord] = {}
        for name, embedding in zip(names.keys(), embeddings):
            candidates = await self.store.search_entity_dedup_candidates(group_id, name, embedding)
            candidate = self._choose_entity_match(name, candidates)
            if candidate is not None:
                existing = EntityRecord(
                    uuid=candidate["uuid"],
                    group_id=group_id,
                    name=candidate["name"],
                    summary=candidate.get("summary") or "",
                    tag=candidate.get("tag") or [],
                    episode_idx=candidate.get("episode_idx") or [],
                    source_ids=candidate.get("source_ids") or [],
                )
                if episode_uuid not in existing.episode_idx:
                    existing.episode_idx.append(episode_uuid)
                if episode.source_id not in existing.source_ids:
                    existing.source_ids.append(episode.source_id)
                existing_by_name[name] = existing
            else:
                existing_by_name[name] = EntityRecord(
                    group_id=group_id,
                    name=name,
                    summary="",
                    tag=[],
                    episode_idx=[episode_uuid],
                    source_ids=[episode.source_id],
                )

        detail_payload = {
            "group_id": group_id,
            "entities": [
                {
                    "uuid": entity.uuid,
                    "name": entity.name,
                    "episode_idx": entity.episode_idx,
                    "source_ids": entity.source_ids,
                }
                for entity in existing_by_name.values()
            ],
            "context": context,
        }
        details = await self.llm.complete_json(
            EntityExtractionPayload,
            [
                {"role": "system", "content": ENTITY_DETAILS_PROMPT},
                {"role": "user", "content": json.dumps(detail_payload, ensure_ascii=False)},
            ],
        )

        name_embeddings = await self.llm.embed([entity.name for entity in details.entities])
        summary_embeddings = await self.llm.embed([entity.summary for entity in details.entities])
        for entity, name_embedding, summary_embedding in zip(details.entities, name_embeddings, summary_embeddings):
            if not entity.uuid:
                entity.uuid = make_uuid("entity")
            entity.group_id = group_id
            if episode_uuid not in entity.episode_idx:
                entity.episode_idx.append(episode_uuid)
            if episode.source_id not in entity.source_ids:
                entity.source_ids.append(episode.source_id)
            await self.store.upsert_entity(entity, name_embedding, summary_embedding)
            await self.store.connect_entity_to_episode(entity.uuid, episode_uuid, group_id)
            deduped.append(entity)
        return deduped

    async def _extract_edges(self, group_id: str, context: str, entities: list[EntityRecord]) -> None:
        if len(entities) < 2:
            return
        entity_payload = [
            {"name": entity.name, "summary": entity.summary, "tag": entity.tag}
            for entity in entities
        ]
        edge_payload = {
            "group_id": group_id,
            "entities": entity_payload,
            "context": context,
        }
        extraction = await self.llm.complete_json(
            EdgeExtractionPayload,
            [
                {"role": "system", "content": EDGE_EXTRACTION_PROMPT},
                {"role": "user", "content": json.dumps(edge_payload, ensure_ascii=False)},
            ],
        )
        edges = extraction.edges
        for _ in range(self.config.max_reflection_rounds):
            reflection = await self.llm.complete_json(
                EdgeExtractionPayload,
                [
                    {"role": "system", "content": EDGE_REFLECTION_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "group_id": group_id,
                                "entities": entity_payload,
                                "context": context,
                                "existing_edges": [edge.model_dump(mode="json") for edge in edges],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
            seen = {(edge.source_entity_name, edge.target_entity_name, edge.fact) for edge in edges}
            for edge in reflection.edges:
                key = (edge.source_entity_name, edge.target_entity_name, edge.fact)
                if key not in seen:
                    seen.add(key)
                    edges.append(edge)

        entities_by_name = await self.store.fetch_entities_by_name(group_id, [entity.name for entity in entities])
        fact_embeddings = await self.llm.embed([edge.fact for edge in edges])
        for edge, fact_embedding in zip(edges, fact_embeddings):
            if not edge.uuid:
                edge.uuid = make_uuid("fact")
            edge.group_id = group_id
            source = entities_by_name.get(edge.source_entity_name)
            target = entities_by_name.get(edge.target_entity_name)
            if not source or not target or source["uuid"] == target["uuid"]:
                continue
            existing_candidates = await self.store.search_edge_dedup_candidates(
                group_id,
                edge.fact,
                fact_embedding,
                source["uuid"],
                target["uuid"],
            )
            existing = self._choose_edge_match(edge.fact, existing_candidates)
            if existing is not None:
                edge.uuid = existing["uuid"]
            await self.store.upsert_edge(edge, fact_embedding, source["uuid"], target["uuid"])
