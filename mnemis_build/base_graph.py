from __future__ import annotations

import asyncio
import json
from collections import OrderedDict

from .config import BuildConfig
from .llm import OpenAILLMClient
from .logging_utils import get_logger
from .models import (
    EdgeRecord,
    EntityNameExtraction,
    EntityRecord,
    EpisodeInput,
    MinimalEdgeExtractionPayload,
    MinimalEntityExtractionPayload,
    MinimalEntityRecord,
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
        self.logger = get_logger("base_graph")

    async def _gather_limited(self, coroutines: list, *, label: str) -> list:
        if not coroutines:
            return []
        limit = max(1, self.config.max_coroutines)
        self.logger.info("parallel section start | label=%s, task_count=%s, limit=%s", label, len(coroutines), limit)
        semaphore = asyncio.Semaphore(limit)

        async def run_one(coro):
            async with semaphore:
                return await coro

        results = await asyncio.gather(*(run_one(coro) for coro in coroutines))
        self.logger.info("parallel section done | label=%s, task_count=%s, limit=%s", label, len(coroutines), limit)
        return list(results)

    async def build(
        self,
        group_id: str,
        episodes: list[EpisodeInput],
        progress_callback=None,
        start_index: int = 0,
    ) -> list[str]:
        self.logger.info(
            "base graph build start | group_id=%s, total_episodes=%s, start_index=%s",
            group_id,
            len(episodes),
            start_index,
        )
        await self.store.ensure_indexes()
        created_episode_ids: list[str] = []
        total_episodes = len(episodes)
        if start_index < 0:
            start_index = 0
        if start_index >= total_episodes:
            self.logger.info(
                "base graph build skipped | group_id=%s, start_index=%s, total_episodes=%s",
                group_id,
                start_index,
                total_episodes,
            )
            return created_episode_ids
        for completed_count, episode in enumerate(episodes[start_index:], start=start_index + 1):
            self.logger.info(
                "turn start | group_id=%s, turn=%s/%s, source_id=%s, speaker=%s",
                group_id,
                completed_count,
                total_episodes,
                episode.source_id,
                episode.speaker,
            )
            episode_uuid = make_uuid("episode")
            recent = await self.store.fetch_recent_episodes(
                group_id,
                self.config.recent_episode_window,
                exclude_source_id=episode.source_id,
            )
            context = self._format_context(episode, recent)
            episode_embedding = (await self.llm.embed([episode.content]))[0]
            episode_uuid = await self.store.upsert_episode(group_id, episode_uuid, episode, episode_embedding)
            created_episode_ids.append(episode_uuid)
            deduped_entities = await self._extract_entities(group_id, episode_uuid, episode, context)
            await self._extract_edges(group_id, context, deduped_entities)
            mark_episode_ingested = getattr(self.store, "mark_episode_ingested", None)
            if callable(mark_episode_ingested):
                await mark_episode_ingested(group_id, episode.source_id)
            self.logger.info(
                "turn done | group_id=%s, turn=%s/%s, source_id=%s, entity_count=%s",
                group_id,
                completed_count,
                total_episodes,
                episode.source_id,
                len(deduped_entities),
            )
            if progress_callback is not None:
                await progress_callback(completed_count, total_episodes, episode)
        self.logger.info("base graph build done | group_id=%s, created_episodes=%s", group_id, len(created_episode_ids))
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

    def _normalize_generated_uuid(self, value: str | None, prefix: str) -> str:
        raw = (value or "").strip()
        if raw.startswith(f"{prefix}_"):
            return raw
        return make_uuid(prefix)

    def _forced_speaker_name(self, episode: EpisodeInput) -> str | None:
        if not self.config.force_base_speaker_entity:
            return None
        return episode.speaker.strip() or "user"

    def _estimate_entity_detail_batch_size(self) -> int:
        tokens = max(1, self.config.entity_detail_max_completion_tokens)
        # Keep each response comfortably under the configured token budget.
        return max(1, min(8, tokens // 128))

    def _estimate_edge_batch_size(self) -> int:
        tokens = max(1, self.config.edge_extraction_max_completion_tokens)
        # Edge prompts expand quickly because they include entity summaries and context.
        return max(2, min(6, tokens // 128))

    def _build_fallback_entity_details(
        self,
        entities: list[EntityRecord],
        *,
        forced_speaker_names: set[str],
    ) -> MinimalEntityExtractionPayload:
        fallback_entities: list[MinimalEntityRecord] = []
        for entity in entities:
            normalized_name = self._normalize_text(entity.name)
            summary = entity.summary.strip() or f"Entity mentioned in the conversation context: {entity.name}."
            tags = list(entity.tag[:3])
            if normalized_name in forced_speaker_names and "speaker" not in tags:
                tags = ["speaker", *tags][:3]
            fallback_entities.append(
                MinimalEntityRecord(
                    name=entity.name,
                    summary=summary,
                    tag=tags,
                )
            )
        return MinimalEntityExtractionPayload(entities=fallback_entities)

    async def _generate_entity_details(
        self,
        *,
        group_id: str,
        context: str,
        entities: list[EntityRecord],
        forced_speaker_names: set[str],
    ) -> MinimalEntityExtractionPayload:
        batch_size = self._estimate_entity_detail_batch_size()
        detailed_entities: list[MinimalEntityRecord] = []

        for start in range(0, len(entities), batch_size):
            batch = entities[start : start + batch_size]
            detail_payload = {
                "group_id": group_id,
                "entities": [{"name": entity.name} for entity in batch],
                "context": context,
            }
            try:
                details = await self.llm.complete_json(
                    MinimalEntityExtractionPayload,
                    [
                        {"role": "system", "content": ENTITY_DETAILS_PROMPT},
                        {"role": "user", "content": json.dumps(detail_payload, ensure_ascii=False)},
                    ],
                    stage="base_graph_ingestion",
                    operation="entity_detail_generation",
                    max_completion_tokens=self.config.entity_detail_max_completion_tokens,
                )
            except ValueError as exc:
                self.logger.warning(
                    "entity detail generation fallback | group_id=%s, entity_count=%s, batch_start=%s, batch_size=%s, error=%s",
                    group_id,
                    len(entities),
                    start,
                    len(batch),
                    exc,
                )
                details = self._build_fallback_entity_details(
                    batch,
                    forced_speaker_names=forced_speaker_names,
                )

            by_name = {self._normalize_text(item.name): item for item in details.entities}
            for entity in batch:
                normalized_name = self._normalize_text(entity.name)
                detail = by_name.get(normalized_name)
                if detail is None:
                    detail = self._build_fallback_entity_details(
                        [entity],
                        forced_speaker_names=forced_speaker_names,
                    ).entities[0]
                detailed_entities.append(detail)

        return MinimalEntityExtractionPayload(entities=detailed_entities)

    async def _generate_edges(
        self,
        *,
        group_id: str,
        context: str,
        entities: list[EntityRecord],
    ) -> list:
        batch_size = self._estimate_edge_batch_size()
        generated_edges = []
        seen: set[tuple[str, str, str]] = set()

        for start in range(0, len(entities), batch_size):
            batch = entities[start : start + batch_size]
            edge_payload = {
                "group_id": group_id,
                "entities": [
                    {"name": entity.name, "summary": entity.summary, "tag": entity.tag}
                    for entity in batch
                ],
                "context": context,
            }
            try:
                extraction = await self.llm.complete_json(
                    MinimalEdgeExtractionPayload,
                    [
                        {"role": "system", "content": EDGE_EXTRACTION_PROMPT},
                        {"role": "user", "content": json.dumps(edge_payload, ensure_ascii=False)},
                    ],
                    stage="base_graph_ingestion",
                    operation="edge_extraction",
                    max_completion_tokens=self.config.edge_extraction_max_completion_tokens,
                )
                batch_edges = extraction.edges
            except ValueError as exc:
                self.logger.warning(
                    "edge extraction fallback | group_id=%s, entity_count=%s, batch_start=%s, batch_size=%s, error=%s",
                    group_id,
                    len(entities),
                    start,
                    len(batch),
                    exc,
                )
                batch_edges = []

            for edge in batch_edges:
                key = (
                    self._normalize_text(edge.source_entity_name),
                    self._normalize_text(edge.target_entity_name),
                    self._normalize_text(edge.fact),
                )
                if key not in seen:
                    seen.add(key)
                    generated_edges.append(edge)

        return generated_edges

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
        self.logger.info("entity extraction start | group_id=%s, episode_uuid=%s, source_id=%s", group_id, episode_uuid, episode.source_id)
        extraction = await self.llm.complete_json(
            EntityNameExtraction,
            [
                {"role": "system", "content": ENTITY_NAME_EXTRACTION_PROMPT},
                {"role": "user", "content": context},
            ],
            stage="base_graph_ingestion",
            operation="entity_name_extraction",
            use_small_model=True,
            max_completion_tokens=self.config.entity_name_max_completion_tokens,
        )
        names = OrderedDict((name.strip(), None) for name in extraction.names if name.strip())
        self.logger.info("entity names extracted | group_id=%s, episode_uuid=%s, names=%s", group_id, episode_uuid, list(names.keys()))
        forced_speaker_name = self._forced_speaker_name(episode)
        if forced_speaker_name:
            # Paper v2 states that base-graph ingestion forcibly extracts the speaker as an entity.
            names.setdefault(forced_speaker_name, None)
        for _ in range(self.config.max_reflection_rounds):
            reflection = await self.llm.complete_json(
                EntityNameExtraction,
                [
                    {"role": "system", "content": ENTITY_REFLECTION_PROMPT},
                    {"role": "user", "content": f"{context}\n\nAlready extracted: {json.dumps(list(names.keys()), ensure_ascii=False)}"},
                ],
                stage="base_graph_ingestion",
                operation="entity_name_reflection",
                use_small_model=True,
                max_completion_tokens=self.config.entity_reflection_max_completion_tokens,
            )
            for name in reflection.names:
                clean = name.strip()
                if clean:
                    names.setdefault(clean, None)

        embeddings = await self.llm.embed(list(names.keys()))
        deduped: list[EntityRecord] = []
        existing_by_name: dict[str, EntityRecord] = {}
        names_list = list(names.keys())
        candidate_lists = await self._gather_limited(
            [
                self.store.search_entity_dedup_candidates(group_id, name, embedding)
                for name, embedding in zip(names_list, embeddings)
            ],
            label="search_entity_dedup_candidates",
        )
        for name, candidates in zip(names_list, candidate_lists):
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
                    is_speaker=bool(candidate.get("is_speaker")),
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
                    is_speaker=False,
                )
        self.logger.info(
            "entity candidate resolution done | group_id=%s, episode_uuid=%s, unique_entities=%s",
            group_id,
            episode_uuid,
            len(existing_by_name),
        )

        existing_by_normalized_name = {
            self._normalize_text(name): entity
            for name, entity in existing_by_name.items()
        }
        forced_speaker_names = {
            self._normalize_text(name)
            for name in [forced_speaker_name]
            if name
        }
        details = await self._generate_entity_details(
            group_id=group_id,
            context=context,
            entities=list(existing_by_name.values()),
            forced_speaker_names=forced_speaker_names,
        )

        name_embeddings = await self.llm.embed([entity.name for entity in details.entities])
        summary_embeddings = await self.llm.embed([entity.summary for entity in details.entities])
        async def prepare_and_upsert_entity(entity, name_embedding, summary_embedding):
            normalized_name = self._normalize_text(entity.name)
            seed = existing_by_normalized_name.get(normalized_name)
            record = EntityRecord(
                uuid=seed.uuid if seed is not None else make_uuid("entity"),
                group_id=group_id,
                name=entity.name,
                summary=entity.summary,
                tag=entity.tag,
                episode_idx=list(seed.episode_idx) if seed is not None else [],
                source_ids=list(seed.source_ids) if seed is not None else [],
                is_speaker=seed.is_speaker if seed is not None else False,
            )
            if normalized_name in forced_speaker_names:
                record.is_speaker = True
            if episode_uuid not in record.episode_idx:
                record.episode_idx.append(episode_uuid)
            if episode.source_id not in record.source_ids:
                record.source_ids.append(episode.source_id)
            await self.store.upsert_entity(record, name_embedding, summary_embedding)
            await self.store.connect_entity_to_episode(record.uuid, episode_uuid, group_id)
            return record

        deduped = await self._gather_limited(
            [
                prepare_and_upsert_entity(entity, name_embedding, summary_embedding)
                for entity, name_embedding, summary_embedding in zip(details.entities, name_embeddings, summary_embeddings)
            ],
            label="upsert_entity_and_connect",
        )
        self.logger.info(
            "entity extraction done | group_id=%s, episode_uuid=%s, deduped_entities=%s",
            group_id,
            episode_uuid,
            len(deduped),
        )
        return deduped

    async def _extract_edges(self, group_id: str, context: str, entities: list[EntityRecord]) -> None:
        if len(entities) < 2:
            self.logger.info("edge extraction skipped | group_id=%s, entity_count=%s", group_id, len(entities))
            return
        self.logger.info("edge extraction start | group_id=%s, entity_count=%s", group_id, len(entities))
        entity_payload = [
            {"name": entity.name, "summary": entity.summary, "tag": entity.tag}
            for entity in entities
        ]
        edges = await self._generate_edges(
            group_id=group_id,
            context=context,
            entities=entities,
        )
        for _ in range(self.config.max_reflection_rounds):
            try:
                reflection = await self.llm.complete_json(
                    MinimalEdgeExtractionPayload,
                    [
                        {"role": "system", "content": EDGE_REFLECTION_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "group_id": group_id,
                                    "available_entities": entity_payload,
                                    "context": context,
                                    "existing_edges": [edge.model_dump(mode="json") for edge in edges],
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    stage="base_graph_ingestion",
                    operation="edge_reflection",
                    max_completion_tokens=self.config.edge_reflection_max_completion_tokens,
                )
            except ValueError as exc:
                self.logger.warning(
                    "edge reflection skipped after parse failure | group_id=%s, error=%s",
                    group_id,
                    exc,
                )
                break
            seen = {
                (
                    self._normalize_text(edge.source_entity_name),
                    self._normalize_text(edge.target_entity_name),
                    self._normalize_text(edge.fact),
                )
                for edge in edges
            }
            for edge in reflection.edges:
                key = (
                    self._normalize_text(edge.source_entity_name),
                    self._normalize_text(edge.target_entity_name),
                    self._normalize_text(edge.fact),
                )
                if key not in seen:
                    seen.add(key)
                    edges.append(edge)

        if not edges:
            self.logger.info("edge extraction produced no usable edges | group_id=%s, entity_count=%s", group_id, len(entities))
            return

        entities_by_name = await self.store.fetch_entities_by_name(group_id, [entity.name for entity in entities])
        fact_embeddings = await self.llm.embed([edge.fact for edge in edges])
        edge_jobs = []
        for edge, fact_embedding in zip(edges, fact_embeddings):
            source = entities_by_name.get(edge.source_entity_name)
            target = entities_by_name.get(edge.target_entity_name)
            if not source or not target or source["uuid"] == target["uuid"]:
                continue
            edge_record = EdgeRecord(
                uuid=make_uuid("fact"),
                group_id=group_id,
                source_entity_name=edge.source_entity_name,
                target_entity_name=edge.target_entity_name,
                fact=edge.fact,
                valid_at=edge.valid_at,
                invalid_at=None,
            )
            edge_jobs.append(
                {
                    "edge": edge_record,
                    "fact_embedding": fact_embedding,
                    "source_uuid": source["uuid"],
                    "target_uuid": target["uuid"],
                }
            )

        candidate_lists = await self._gather_limited(
            [
                self.store.search_edge_dedup_candidates(
                    group_id,
                    job["edge"].fact,
                    job["fact_embedding"],
                    job["source_uuid"],
                    job["target_uuid"],
                )
                for job in edge_jobs
            ],
            label="search_edge_dedup_candidates",
        )

        async def finalize_and_upsert_edge(job: dict, existing_candidates: list[dict]) -> bool:
            edge = job["edge"]
            existing = self._choose_edge_match(edge.fact, existing_candidates)
            if existing is not None:
                edge.uuid = existing["uuid"]
            await self.store.upsert_edge(
                edge,
                job["fact_embedding"],
                job["source_uuid"],
                job["target_uuid"],
            )
            return True

        upsert_results = await self._gather_limited(
            [
                finalize_and_upsert_edge(job, existing_candidates)
                for job, existing_candidates in zip(edge_jobs, candidate_lists)
            ],
            label="upsert_edge",
        )
        upserted_edges = sum(1 for item in upsert_results if item)
        self.logger.info("edge extraction done | group_id=%s, generated_edges=%s, upserted_edges=%s", group_id, len(edges), upserted_edges)
