from __future__ import annotations

import json
import re
from typing import Any

from .config import BuildConfig
from .logging_utils import get_logger
from .models import CategoryRecord, EdgeRecord, EntityRecord, EpisodeInput


class Neo4jGraphStore:
    def __init__(self, config: BuildConfig):
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:
            raise RuntimeError(
                "The neo4j package is required to use the graph builder. "
                "Install it with: python -m pip install -r requirements-mnemis-build.txt"
            ) from exc
        self.config = config
        self.logger = get_logger("neo4j")
        self.driver = AsyncGraphDatabase.driver(
            config.neo4j_url,
            auth=(config.neo4j_user, config.neo4j_password),
        )

    async def close(self) -> None:
        await self.driver.close()

    async def execute(self, cypher: str, **parameters: Any):
        kwargs = dict(parameters)
        if self.config.neo4j_database:
            kwargs["database_"] = self.config.neo4j_database
        return await self.driver.execute_query(cypher, **kwargs)

    def _sanitize_fulltext_query(self, text: str) -> str:
        tokens = re.findall(r"[\w-]+", text or "", flags=re.UNICODE)
        return " ".join(tokens) or "none"

    def _merge_ranked_rows(
        self,
        *ranked_lists: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for rows in ranked_lists:
            for rank, row in enumerate(rows, start=1):
                uuid = row["uuid"]
                current = merged.setdefault(uuid, {"rrf_score": 0.0})
                current["rrf_score"] += 1.0 / (self.config.rrf_k + rank)
                for key, value in row.items():
                    if value is not None or key not in current:
                        current[key] = value
        return sorted(
            merged.values(),
            key=lambda item: (
                item.get("rrf_score", 0.0),
                item.get("similarity_score", 0.0),
                item.get("fulltext_score", 0.0),
            ),
            reverse=True,
        )[:limit]

    async def ensure_indexes(self) -> None:
        self.logger.info("ensure indexes start")
        queries = [
            "CREATE CONSTRAINT entity_uuid IF NOT EXISTS FOR (n:Entity) REQUIRE n.uuid IS UNIQUE",
            "CREATE CONSTRAINT episode_uuid IF NOT EXISTS FOR (n:Episodic) REQUIRE n.uuid IS UNIQUE",
            "CREATE CONSTRAINT category_uuid IF NOT EXISTS FOR (n:Category) REQUIRE n.uuid IS UNIQUE",
            "CREATE CONSTRAINT fact_uuid IF NOT EXISTS FOR ()-[r:RELATES_TO]-() REQUIRE r.uuid IS UNIQUE",
            "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS FOR (n:Entity) ON EACH [n.name]",
            "CREATE FULLTEXT INDEX entity_text_ft IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.summary]",
            "CREATE FULLTEXT INDEX episode_content_ft IF NOT EXISTS FOR (n:Episodic) ON EACH [n.content]",
            "CREATE FULLTEXT INDEX edge_fact_ft IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON EACH [r.fact]",
            "CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)",
            "CREATE INDEX category_group_layer IF NOT EXISTS FOR (n:Category) ON (n.group_id, n.layer)",
            "CREATE INDEX episode_group_id IF NOT EXISTS FOR (n:Episodic) ON (n.group_id)",
        ]
        for query in queries:
            await self.execute(query)
        self.logger.info("ensure indexes done | query_count=%s", len(queries))

    async def clear_group(self, group_id: str) -> None:
        self.logger.info("clear group start | group_id=%s", group_id)
        await self.execute(
            """
            MATCH (n)
            WHERE n.group_id = $group_id
            DETACH DELETE n
            """,
            group_id=group_id,
        )
        self.logger.info("clear group done | group_id=%s", group_id)

    async def fetch_recent_episodes(
        self,
        group_id: str,
        limit: int,
        *,
        exclude_source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.logger.info(
            "fetch recent episodes | group_id=%s, limit=%s, exclude_source_id=%s",
            group_id,
            limit,
            exclude_source_id,
        )
        result = await self.execute(
            """
            MATCH (e:Episodic {group_id: $group_id})
            WHERE $exclude_source_id IS NULL OR e.source_id <> $exclude_source_id
            RETURN e.uuid AS uuid, e.content AS content, e.valid_at AS valid_at, e.source_id AS source_id
            ORDER BY e.valid_at DESC
            LIMIT $limit
            """,
            group_id=group_id,
            limit=limit,
            exclude_source_id=exclude_source_id,
        )
        return [dict(record) for record in result.records]

    async def fetch_completed_episode_source_ids(self, group_id: str) -> list[str]:
        self.logger.info("fetch completed episode source ids | group_id=%s", group_id)
        result = await self.execute(
            """
            MATCH (e:Episodic {group_id: $group_id})
            WHERE coalesce(e.ingestion_complete, false)
            RETURN e.source_id AS source_id
            ORDER BY e.valid_at ASC, e.source_id ASC
            """,
            group_id=group_id,
        )
        return [str(record["source_id"]) for record in result.records if record.get("source_id")]

    async def search_entity_dedup_candidates(
        self,
        group_id: str,
        name: str,
        embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        self.logger.info("entity dedup search | group_id=%s, name=%s, limit=%s", group_id, name, limit)
        query = self._sanitize_fulltext_query(name)
        fulltext_result = await self.execute(
            """
            CALL db.index.fulltext.queryNodes('entity_name_ft', $query) YIELD node, score
            WHERE node.group_id = $group_id
            RETURN node.uuid AS uuid, node.name AS name, node.summary AS summary, node.tag AS tag,
                   coalesce(node.is_speaker, false) AS is_speaker,
                   node.episode_idx AS episode_idx, node.source_ids AS source_ids, score AS fulltext_score
            ORDER BY fulltext_score DESC
            LIMIT $limit
            """,
            group_id=group_id,
            query=query,
            limit=limit,
        )
        vector_result = await self.execute(
            """
            MATCH (n:Entity {group_id: $group_id})
            WHERE n.name_embedding IS NOT NULL
            WITH n,
                 reduce(dot = 0.0, i IN range(0, size($embedding) - 1) | dot + ($embedding[i] * n.name_embedding[i])) AS dot,
                 sqrt(reduce(a = 0.0, x IN $embedding | a + x * x)) AS q_norm,
                 sqrt(reduce(b = 0.0, x IN n.name_embedding | b + x * x)) AS n_norm
            WITH n, CASE WHEN q_norm = 0 OR n_norm = 0 THEN 0.0 ELSE dot / (q_norm * n_norm) END AS similarity_score
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary, n.tag AS tag,
                   coalesce(n.is_speaker, false) AS is_speaker,
                   n.episode_idx AS episode_idx, n.source_ids AS source_ids, similarity_score
            ORDER BY similarity_score DESC
            LIMIT $limit
            """,
            group_id=group_id,
            embedding=embedding,
            limit=limit,
        )
        return self._merge_ranked_rows(
            [dict(record) for record in fulltext_result.records],
            [dict(record) for record in vector_result.records],
            limit=limit,
        )

    async def search_entities(
        self,
        group_id: str,
        query_text: str,
        embedding: list[float],
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query = self._sanitize_fulltext_query(query_text)
        fulltext_result = await self.execute(
            """
            CALL db.index.fulltext.queryNodes('entity_text_ft', $query) YIELD node, score
            WHERE node.group_id = $group_id
            RETURN node.uuid AS uuid, node.name AS name, node.summary AS summary, node.tag AS tag,
                   coalesce(node.is_speaker, false) AS is_speaker,
                   node.episode_idx AS episode_idx, node.source_ids AS source_ids, score AS fulltext_score
            ORDER BY fulltext_score DESC
            LIMIT $limit
            """,
            group_id=group_id,
            query=query,
            limit=limit,
        )
        vector_result = await self.execute(
            """
            MATCH (n:Entity {group_id: $group_id})
            WHERE n.summary_embedding IS NOT NULL
            WITH n,
                 reduce(dot = 0.0, i IN range(0, size($embedding) - 1) | dot + ($embedding[i] * n.summary_embedding[i])) AS dot,
                 sqrt(reduce(a = 0.0, x IN $embedding | a + x * x)) AS q_norm,
                 sqrt(reduce(b = 0.0, x IN n.summary_embedding | b + x * x)) AS n_norm
            WITH n, CASE WHEN q_norm = 0 OR n_norm = 0 THEN 0.0 ELSE dot / (q_norm * n_norm) END AS similarity_score
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary, n.tag AS tag,
                   coalesce(n.is_speaker, false) AS is_speaker,
                   n.episode_idx AS episode_idx, n.source_ids AS source_ids, similarity_score
            ORDER BY similarity_score DESC
            LIMIT $limit
            """,
            group_id=group_id,
            embedding=embedding,
            limit=limit,
        )
        return self._merge_ranked_rows(
            [dict(record) for record in fulltext_result.records],
            [dict(record) for record in vector_result.records],
            limit=limit,
        )

    async def expand_entities_for_retrieval(
        self,
        group_id: str,
        entity_uuids: list[str],
        limit: int = 50,
    ) -> dict[str, list[dict[str, Any]]]:
        if not entity_uuids:
            return {"episodes": [], "edges": [], "nodes": []}
        self.logger.info(
            "expand entities for retrieval | group_id=%s, entity_count=%s, limit=%s",
            group_id,
            len(entity_uuids),
            limit,
        )
        episode_result = await self.execute(
            """
            MATCH (seed:Entity)-[:MENTIONS]->(episode:Episodic)
            WHERE seed.group_id = $group_id AND seed.uuid IN $entity_uuids
            WITH episode,
                 collect(DISTINCT seed.uuid) AS matched_entity_uuids,
                 collect(DISTINCT seed.name) AS matched_entity_names,
                 count(DISTINCT seed) AS matched_entity_count
            RETURN episode.uuid AS uuid,
                   episode.content AS content,
                   episode.valid_at AS valid_at,
                   episode.source_id AS source_id,
                   matched_entity_uuids,
                   matched_entity_names,
                   matched_entity_count
            ORDER BY matched_entity_count DESC, episode.valid_at DESC
            LIMIT $limit
            """,
            group_id=group_id,
            entity_uuids=entity_uuids,
            limit=limit,
        )
        edge_result = await self.execute(
            """
            MATCH (seed:Entity)-[rel:RELATES_TO]-(neighbor:Entity)
            WHERE seed.group_id = $group_id AND seed.uuid IN $entity_uuids
            WITH rel, neighbor,
                 startNode(rel) AS source,
                 endNode(rel) AS target,
                 properties(rel) AS rel_props,
                 collect(DISTINCT seed.uuid) AS matched_entity_uuids,
                 collect(DISTINCT seed.name) AS matched_entity_names,
                 count(DISTINCT seed) AS matched_entity_count
            RETURN rel.uuid AS uuid,
                   rel.fact AS fact,
                   rel.valid_at AS valid_at,
                   rel_props['invalid_at'] AS invalid_at,
                   source.uuid AS source_uuid,
                   source.name AS source_name,
                   target.uuid AS target_uuid,
                   target.name AS target_name,
                   neighbor.uuid AS entity_uuid,
                   neighbor.name AS name,
                   neighbor.tag AS tag,
                   neighbor.summary AS summary,
                   matched_entity_uuids,
                   matched_entity_names,
                   matched_entity_count
            ORDER BY matched_entity_count DESC, rel.valid_at DESC
            LIMIT $limit
            """,
            group_id=group_id,
            entity_uuids=entity_uuids,
            limit=limit,
        )

        episodes = [dict(record) for record in episode_result.records]
        edges: dict[str, dict[str, Any]] = {}
        nodes: dict[str, dict[str, Any]] = {}
        for record in edge_result.records:
            neighbor = dict(record)
            edges[neighbor["uuid"]] = {
                "uuid": neighbor["uuid"],
                "fact": neighbor["fact"],
                "valid_at": neighbor["valid_at"],
                "invalid_at": neighbor["invalid_at"],
                "source_uuid": neighbor["source_uuid"],
                "source_name": neighbor["source_name"],
                "target_uuid": neighbor["target_uuid"],
                "target_name": neighbor["target_name"],
                "matched_entity_uuids": neighbor["matched_entity_uuids"],
                "matched_entity_names": neighbor["matched_entity_names"],
                "matched_entity_count": neighbor["matched_entity_count"],
            }
            nodes[neighbor["entity_uuid"]] = {
                "uuid": neighbor["entity_uuid"],
                "name": neighbor["name"],
                "tag": neighbor["tag"],
                "summary": neighbor["summary"],
                "matched_entity_uuids": neighbor["matched_entity_uuids"],
                "matched_entity_names": neighbor["matched_entity_names"],
                "matched_entity_count": neighbor["matched_entity_count"],
            }
        return {
            "episodes": episodes,
            "edges": list(edges.values()),
            "nodes": list(nodes.values()),
        }

    async def search_episodes(
        self,
        group_id: str,
        query_text: str,
        embedding: list[float],
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query = self._sanitize_fulltext_query(query_text)
        fulltext_result = await self.execute(
            """
            CALL db.index.fulltext.queryNodes('episode_content_ft', $query) YIELD node, score
            WHERE node.group_id = $group_id
            RETURN node.uuid AS uuid, node.content AS content, node.valid_at AS valid_at,
                   node.source_id AS source_id, score AS fulltext_score
            ORDER BY fulltext_score DESC
            LIMIT $limit
            """,
            group_id=group_id,
            query=query,
            limit=limit,
        )
        vector_result = await self.execute(
            """
            MATCH (e:Episodic {group_id: $group_id})
            WHERE e.episode_embedding IS NOT NULL
            WITH e,
                 reduce(dot = 0.0, i IN range(0, size($embedding) - 1) | dot + ($embedding[i] * e.episode_embedding[i])) AS dot,
                 sqrt(reduce(a = 0.0, x IN $embedding | a + x * x)) AS q_norm,
                 sqrt(reduce(b = 0.0, x IN e.episode_embedding | b + x * x)) AS n_norm
            WITH e, CASE WHEN q_norm = 0 OR n_norm = 0 THEN 0.0 ELSE dot / (q_norm * n_norm) END AS similarity_score
            RETURN e.uuid AS uuid, e.content AS content, e.valid_at AS valid_at,
                   e.source_id AS source_id, similarity_score
            ORDER BY similarity_score DESC
            LIMIT $limit
            """,
            group_id=group_id,
            embedding=embedding,
            limit=limit,
        )
        return self._merge_ranked_rows(
            [dict(record) for record in fulltext_result.records],
            [dict(record) for record in vector_result.records],
            limit=limit,
        )

    async def search_edge_dedup_candidates(
        self,
        group_id: str,
        fact: str,
        embedding: list[float],
        source_uuid: str,
        target_uuid: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        self.logger.info(
            "edge dedup search | group_id=%s, source_uuid=%s, target_uuid=%s, limit=%s",
            group_id,
            source_uuid,
            target_uuid,
            limit,
        )
        query = self._sanitize_fulltext_query(fact)
        fulltext_result = await self.execute(
            """
            CALL db.index.fulltext.queryRelationships('edge_fact_ft', $query) YIELD relationship, score
            MATCH (a:Entity)-[relationship]->(b:Entity)
            WHERE relationship.group_id = $group_id
              AND a.uuid = $source_uuid
              AND b.uuid = $target_uuid
            WITH relationship, score, properties(relationship) AS rel_props
            RETURN relationship.uuid AS uuid, relationship.fact AS fact,
                   relationship.valid_at AS valid_at, rel_props['invalid_at'] AS invalid_at,
                   score AS fulltext_score
            ORDER BY fulltext_score DESC
            LIMIT $limit
            """,
            group_id=group_id,
            query=query,
            source_uuid=source_uuid,
            target_uuid=target_uuid,
            limit=limit,
        )
        vector_result = await self.execute(
            """
            MATCH (:Entity {uuid: $source_uuid})-[r:RELATES_TO]->(:Entity {uuid: $target_uuid})
            WHERE r.group_id = $group_id
              AND r.fact_embedding IS NOT NULL
            WITH r,
                 reduce(dot = 0.0, i IN range(0, size($embedding) - 1) | dot + ($embedding[i] * r.fact_embedding[i])) AS dot,
                 sqrt(reduce(a = 0.0, x IN $embedding | a + x * x)) AS q_norm,
                 sqrt(reduce(b = 0.0, x IN r.fact_embedding | b + x * x)) AS n_norm
            WITH r, CASE WHEN q_norm = 0 OR n_norm = 0 THEN 0.0 ELSE dot / (q_norm * n_norm) END AS similarity_score,
                 properties(r) AS rel_props
            RETURN r.uuid AS uuid, r.fact AS fact, r.valid_at AS valid_at, rel_props['invalid_at'] AS invalid_at, similarity_score
            ORDER BY similarity_score DESC
            LIMIT $limit
            """,
            group_id=group_id,
            embedding=embedding,
            source_uuid=source_uuid,
            target_uuid=target_uuid,
            limit=limit,
        )
        return self._merge_ranked_rows(
            [dict(record) for record in fulltext_result.records],
            [dict(record) for record in vector_result.records],
            limit=limit,
        )

    async def search_edges(
        self,
        group_id: str,
        query_text: str,
        embedding: list[float],
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query = self._sanitize_fulltext_query(query_text)
        fulltext_result = await self.execute(
            """
            CALL db.index.fulltext.queryRelationships('edge_fact_ft', $query) YIELD relationship, score
            MATCH (a:Entity)-[relationship]->(b:Entity)
            WHERE relationship.group_id = $group_id
            WITH a, b, relationship, score, properties(relationship) AS rel_props
            RETURN relationship.uuid AS uuid, relationship.fact AS fact,
                   relationship.valid_at AS valid_at, rel_props['invalid_at'] AS invalid_at,
                   a.uuid AS source_uuid, a.name AS source_name,
                   b.uuid AS target_uuid, b.name AS target_name,
                   score AS fulltext_score
            ORDER BY fulltext_score DESC
            LIMIT $limit
            """,
            group_id=group_id,
            query=query,
            limit=limit,
        )
        vector_result = await self.execute(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            WHERE r.group_id = $group_id
              AND r.fact_embedding IS NOT NULL
            WITH a, b, r,
                 reduce(dot = 0.0, i IN range(0, size($embedding) - 1) | dot + ($embedding[i] * r.fact_embedding[i])) AS dot,
                 sqrt(reduce(a_norm = 0.0, x IN $embedding | a_norm + x * x)) AS q_norm,
                 sqrt(reduce(b_norm = 0.0, x IN r.fact_embedding | b_norm + x * x)) AS n_norm
            WITH a, b, r, CASE WHEN q_norm = 0 OR n_norm = 0 THEN 0.0 ELSE dot / (q_norm * n_norm) END AS similarity_score,
                 properties(r) AS rel_props
            RETURN r.uuid AS uuid, r.fact AS fact, r.valid_at AS valid_at, rel_props['invalid_at'] AS invalid_at,
                   a.uuid AS source_uuid, a.name AS source_name,
                   b.uuid AS target_uuid, b.name AS target_name,
                   similarity_score
            ORDER BY similarity_score DESC
            LIMIT $limit
            """,
            group_id=group_id,
            embedding=embedding,
            limit=limit,
        )
        return self._merge_ranked_rows(
            [dict(record) for record in fulltext_result.records],
            [dict(record) for record in vector_result.records],
            limit=limit,
        )

    async def upsert_episode(self, group_id: str, episode_uuid: str, episode: EpisodeInput, embedding: list[float]) -> str:
        self.logger.info("upsert episode | group_id=%s, episode_uuid=%s, source_id=%s", group_id, episode_uuid, episode.source_id)
        metadata_json = json.dumps(
            {key: value for key, value in episode.metadata.items() if value is not None},
            ensure_ascii=False,
            sort_keys=True,
        )
        result = await self.execute(
            """
            MERGE (e:Episodic {group_id: $group_id, source_id: $source_id})
            ON CREATE SET e.uuid = $uuid
            SET e.uuid = coalesce(e.uuid, $uuid),
                e.group_id = $group_id,
                e.speaker = $speaker,
                e.content = $content,
                e.valid_at = datetime($valid_at),
                e.source_id = $source_id,
                e.metadata_json = $metadata_json,
                e.episode_embedding = $embedding,
                e.ingestion_complete = coalesce(e.ingestion_complete, false)
            RETURN e.uuid AS uuid
            """,
            uuid=episode_uuid,
            group_id=group_id,
            speaker=episode.speaker,
            content=episode.content,
            valid_at=episode.valid_at.isoformat(),
            source_id=episode.source_id,
            metadata_json=metadata_json,
            embedding=embedding,
        )
        if not result.records:
            return episode_uuid
        return str(result.records[0]["uuid"])

    async def mark_episode_ingested(self, group_id: str, source_id: str) -> None:
        self.logger.info("mark episode ingested | group_id=%s, source_id=%s", group_id, source_id)
        await self.execute(
            """
            MATCH (e:Episodic {group_id: $group_id, source_id: $source_id})
            SET e.ingestion_complete = true
            """,
            group_id=group_id,
            source_id=source_id,
        )

    async def upsert_entity(self, entity: EntityRecord, name_embedding: list[float], summary_embedding: list[float]) -> None:
        self.logger.info("upsert entity | group_id=%s, entity_uuid=%s, name=%s", entity.group_id, entity.uuid, entity.name)
        await self.execute(
            """
            MERGE (n:Entity {uuid: $uuid})
            SET n.group_id = $group_id,
                n.name = $name,
                n.summary = $summary,
                n.tag = $tag,
                n.is_speaker = $is_speaker,
                n.episode_idx = $episode_idx,
                n.source_ids = $source_ids,
                n.name_embedding = $name_embedding,
                n.summary_embedding = $summary_embedding
            """,
            uuid=entity.uuid,
            group_id=entity.group_id,
            name=entity.name,
            summary=entity.summary,
            tag=entity.tag,
            is_speaker=entity.is_speaker,
            episode_idx=entity.episode_idx,
            source_ids=entity.source_ids,
            name_embedding=name_embedding,
            summary_embedding=summary_embedding,
        )

    async def connect_entity_to_episode(self, entity_uuid: str, episode_uuid: str, group_id: str) -> None:
        self.logger.info("connect entity to episode | group_id=%s, entity_uuid=%s, episode_uuid=%s", group_id, entity_uuid, episode_uuid)
        await self.execute(
            """
            MATCH (n:Entity {uuid: $entity_uuid, group_id: $group_id})
            MATCH (e:Episodic {uuid: $episode_uuid, group_id: $group_id})
            MERGE (n)-[:MENTIONS]->(e)
            """,
            entity_uuid=entity_uuid,
            episode_uuid=episode_uuid,
            group_id=group_id,
        )

    async def fetch_entities_by_name(self, group_id: str, names: list[str]) -> dict[str, dict[str, Any]]:
        self.logger.info("fetch entities by name | group_id=%s, name_count=%s", group_id, len(names))
        result = await self.execute(
            """
            MATCH (n:Entity {group_id: $group_id})
            WHERE n.name IN $names
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary, n.tag AS tag,
                   coalesce(n.is_speaker, false) AS is_speaker
            """,
            group_id=group_id,
            names=names,
        )
        return {record["name"]: dict(record) for record in result.records}

    async def upsert_edge(self, edge: EdgeRecord, fact_embedding: list[float], source_uuid: str, target_uuid: str) -> None:
        self.logger.info(
            "upsert edge | group_id=%s, edge_uuid=%s, source_uuid=%s, target_uuid=%s",
            edge.group_id,
            edge.uuid,
            source_uuid,
            target_uuid,
        )
        await self.execute(
            """
            MATCH (a:Entity {uuid: $source_uuid})
            MATCH (b:Entity {uuid: $target_uuid})
            MERGE (a)-[r:RELATES_TO {uuid: $uuid}]->(b)
            SET r.group_id = $group_id,
                r.fact = $fact,
                r.valid_at = CASE WHEN $valid_at IS NULL THEN NULL ELSE datetime($valid_at) END,
                r.invalid_at = CASE WHEN $invalid_at IS NULL THEN NULL ELSE datetime($invalid_at) END,
                r.fact_embedding = $fact_embedding
            """,
            uuid=edge.uuid,
            group_id=edge.group_id,
            fact=edge.fact,
            valid_at=edge.valid_at.isoformat() if edge.valid_at else None,
            invalid_at=edge.invalid_at.isoformat() if edge.invalid_at else None,
            fact_embedding=fact_embedding,
            source_uuid=source_uuid,
            target_uuid=target_uuid,
        )

    async def clear_hierarchy(self, group_id: str) -> None:
        self.logger.info("clear hierarchy start | group_id=%s", group_id)
        await self.execute(
            """
            MATCH (c:Category {group_id: $group_id})
            DETACH DELETE c
            """,
            group_id=group_id,
        )
        self.logger.info("clear hierarchy done | group_id=%s", group_id)

    async def fetch_layer_zero_nodes(self, group_id: str) -> list[dict[str, Any]]:
        self.logger.info("fetch layer zero nodes | group_id=%s", group_id)
        result = await self.execute(
            """
            MATCH (n:Entity {group_id: $group_id})
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary, n.tag AS tag,
                   coalesce(n.is_speaker, false) AS is_speaker
            ORDER BY n.name
            """,
            group_id=group_id,
        )
        return [dict(record) for record in result.records]

    async def upsert_category(self, category: CategoryRecord, summary_embedding: list[float]) -> None:
        self.logger.info(
            "upsert category | group_id=%s, category_uuid=%s, layer=%s, name=%s",
            category.group_id,
            category.uuid,
            category.layer,
            category.name,
        )
        labels = f":Category:Category_{category.layer}"
        await self.execute(
            f"""
            MERGE (c{labels} {{uuid: $uuid}})
            SET c.group_id = $group_id,
                c.name = $name,
                c.summary = $summary,
                c.tag = $tag,
                c.layer = $layer,
                c.summary_embedding = $summary_embedding
            """,
            uuid=category.uuid,
            group_id=category.group_id,
            name=category.name,
            summary=category.summary,
            tag=category.tag,
            layer=category.layer,
            summary_embedding=summary_embedding,
        )

    async def connect_category(self, parent_uuid: str, child_uuid: str, group_id: str) -> None:
        self.logger.info("connect category | group_id=%s, parent_uuid=%s, child_uuid=%s", group_id, parent_uuid, child_uuid)
        await self.execute(
            """
            MATCH (p {uuid: $parent_uuid, group_id: $group_id})
            MATCH (c {uuid: $child_uuid, group_id: $group_id})
            MERGE (p)-[:CATEGORIZES]->(c)
            """,
            parent_uuid=parent_uuid,
            child_uuid=child_uuid,
            group_id=group_id,
        )

    async def fetch_max_layer(self, group_id: str) -> int:
        result = await self.execute(
            """
            MATCH (n:Category {group_id: $group_id})
            RETURN max(n.layer) AS max_layer
            """,
            group_id=group_id,
        )
        record = result.records[0]
        return record["max_layer"] if record["max_layer"] is not None else 0

    async def fetch_nodes_by_layer(self, group_id: str, layer: int) -> list[dict[str, Any]]:
        label = f"Category_{layer}" if layer > 0 else "Entity"
        result = await self.execute(
            f"""
            MATCH (n:{label})
            WHERE n.group_id = $group_id
            RETURN n.uuid AS uuid, n.name AS name, n.tag AS tag, n.summary AS summary, n.layer AS layer
            ORDER BY n.name
            """,
            group_id=group_id,
        )
        return [dict(record) for record in result.records]

    async def fetch_child_nodes(self, group_id: str, parent_uuids: list[str]) -> list[dict[str, Any]]:
        if not parent_uuids:
            return []
        result = await self.execute(
            """
            MATCH (parent:Category)-[:CATEGORIZES]->(child:Category|Entity)
            WHERE parent.group_id = $group_id AND parent.uuid IN $parent_uuids
            RETURN DISTINCT child.uuid AS uuid, child.name AS name, child.tag AS tag,
                            child.summary AS summary, child.layer AS layer
            """,
            group_id=group_id,
            parent_uuids=parent_uuids,
        )
        return [dict(record) for record in result.records]

    async def fetch_all_descendants(self, group_id: str, parent_uuids: list[str]) -> list[dict[str, Any]]:
        if not parent_uuids:
            return []
        result = await self.execute(
            """
            MATCH (parent:Category|Entity)-[:CATEGORIZES*1..]->(child:Category|Entity)
            WHERE parent.group_id = $group_id AND parent.uuid IN $parent_uuids
            RETURN DISTINCT child.uuid AS uuid, child.name AS name, child.tag AS tag,
                            child.summary AS summary, child.layer AS layer
            """,
            group_id=group_id,
            parent_uuids=parent_uuids,
        )
        return [dict(record) for record in result.records]

    async def fetch_descendant_entities(self, group_id: str, parent_uuids: list[str]) -> list[dict[str, Any]]:
        if not parent_uuids:
            return []
        result = await self.execute(
            """
            MATCH (parent:Category)-[:CATEGORIZES*1..]->(child:Entity)
            WHERE parent.group_id = $group_id AND parent.uuid IN $parent_uuids
            RETURN DISTINCT child.uuid AS uuid, child.name AS name, child.tag AS tag,
                            child.summary AS summary, child.layer AS layer
            """,
            group_id=group_id,
            parent_uuids=parent_uuids,
        )
        return [dict(record) for record in result.records]

    async def fetch_one_hop_neighbors(self, group_id: str, node_uuids: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not node_uuids:
            return {"episodes": [], "edges": [], "nodes": []}

        episode_result = await self.execute(
            """
            MATCH (n)-[:MENTIONS]-(m:Episodic)
            WHERE n.group_id = $group_id AND n.uuid IN $node_uuids
            RETURN DISTINCT m.uuid AS uuid, m.content AS content, m.valid_at AS valid_at, m.source_id AS source_id
            """,
            group_id=group_id,
            node_uuids=node_uuids,
        )
        edge_result = await self.execute(
            """
            MATCH (n)-[r:RELATES_TO]-(m:Entity)
            WHERE n.group_id = $group_id AND n.uuid IN $node_uuids
            WITH DISTINCT r, m, properties(r) AS rel_props
            RETURN DISTINCT r.uuid AS uuid, r.fact AS fact, r.valid_at AS valid_at, rel_props['invalid_at'] AS invalid_at,
                            m.uuid AS entity_uuid, m.name AS name, m.tag AS tag, m.summary AS summary
            """,
            group_id=group_id,
            node_uuids=node_uuids,
        )

        episodes = [dict(record) for record in episode_result.records]
        edges: dict[str, dict[str, Any]] = {}
        nodes: dict[str, dict[str, Any]] = {}
        for record in edge_result.records:
            neighbor = dict(record)
            edges[neighbor["uuid"]] = {
                "uuid": neighbor["uuid"],
                "fact": neighbor["fact"],
                "valid_at": neighbor["valid_at"],
                "invalid_at": neighbor["invalid_at"],
            }
            nodes[neighbor["entity_uuid"]] = {
                "uuid": neighbor["entity_uuid"],
                "name": neighbor["name"],
                "tag": neighbor["tag"],
                "summary": neighbor["summary"],
            }
        return {
            "episodes": episodes,
            "edges": list(edges.values()),
            "nodes": list(nodes.values()),
        }
