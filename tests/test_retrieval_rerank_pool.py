import asyncio
from collections.abc import Iterable

import pytest

import mnemis_build.retrieval as retrieval_module
from mnemis_build.config import BuildConfig
from mnemis_build.retrieval import MnemisRetriever
from mnemis_build.reranker import RerankBackendStatus, RerankResponse


def _build_config() -> BuildConfig:
    return BuildConfig(
        neo4j_url="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        neo4j_database=None,
        llm_api_key="test-key",
        llm_base_url=None,
        embedding_api_key="test-key",
        embedding_base_url=None,
        llm_model="gpt-4.1-mini",
        small_llm_model="gpt-4.1-mini",
        rerank_mode="auto",
        reranker_api_key=None,
        reranker_base_url=None,
        reranker_model=None,
        rerank_allow_llm_fallback=True,
        embedding_model="text-embedding-3-small",
        embedding_dim=128,
        max_coroutines=4,
        recent_episode_window=6,
        max_reflection_rounds=1,
        force_base_speaker_entity=True,
        speaker_hierarchy_mode="paper_v2",
        min_children_per_category=2,
        max_hierarchy_layers=4,
        max_categories_per_call=0,
        hierarchy_assignment_batch_size=96,
        category_detail_batch_size=48,
        entity_name_max_completion_tokens=192,
        entity_reflection_max_completion_tokens=192,
        entity_detail_max_completion_tokens=768,
        edge_extraction_max_completion_tokens=768,
        edge_reflection_max_completion_tokens=512,
        episode_top_k=2,
        entity_top_k=2,
        edge_top_k=2,
        retrieval_candidate_limit=6,
        rrf_k=60,
    )


class _FakeLLM:
    async def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeReranker:
    def __init__(self, scores_by_uuid: dict[str, float]) -> None:
        self.scores_by_uuid = scores_by_uuid
        self.last_status: RerankBackendStatus | None = None

    async def rerank(self, *, query: str, item_type: str, candidates) -> RerankResponse:
        status = RerankBackendStatus(
            mode="test_reranker",
            backend="unit_test",
            model="fake",
            requested_mode="auto",
        )
        self.last_status = status
        return RerankResponse(
            scores={candidate.uuid: self.scores_by_uuid.get(candidate.uuid, 0.0) for candidate in candidates},
            status=status,
        )


class _FakeStore:
    def __init__(self) -> None:
        self.entity_candidates = [
            {"uuid": "entity_1", "name": "Noise One", "summary": "irrelevant", "tag": ["noise"], "rrf_score": 0.91},
            {"uuid": "entity_2", "name": "Noise Two", "summary": "irrelevant", "tag": ["noise"], "rrf_score": 0.83},
            {"uuid": "entity_gold", "name": "Gold Entity", "summary": "the true answer", "tag": ["gold"], "rrf_score": 0.44},
            {"uuid": "entity_4", "name": "Noise Four", "summary": "irrelevant", "tag": ["noise"], "rrf_score": 0.31},
        ]
        self.expansions = {
            "entity_1": {
                "episodes": [{"uuid": "episode_1", "content": "noise one", "valid_at": "2023-06-01", "source_id": "D0:1", "matched_entity_uuids": ["entity_1"], "matched_entity_count": 1}],
                "edges": [{"uuid": "edge_1", "fact": "noise edge one", "matched_entity_uuids": ["entity_1"], "matched_entity_count": 1}],
                "nodes": [],
            },
            "entity_2": {
                "episodes": [{"uuid": "episode_2", "content": "noise two", "valid_at": "2023-06-02", "source_id": "D0:2", "matched_entity_uuids": ["entity_2"], "matched_entity_count": 1}],
                "edges": [{"uuid": "edge_2", "fact": "noise edge two", "matched_entity_uuids": ["entity_2"], "matched_entity_count": 1}],
                "nodes": [],
            },
            "entity_gold": {
                "episodes": [{"uuid": "episode_gold", "content": "gold evidence", "valid_at": "2023-06-03", "source_id": "D9:9", "matched_entity_uuids": ["entity_gold"], "matched_entity_count": 1}],
                "edges": [{"uuid": "edge_gold", "fact": "gold edge", "matched_entity_uuids": ["entity_gold"], "matched_entity_count": 1}],
                "nodes": [],
            },
            "entity_4": {
                "episodes": [{"uuid": "episode_4", "content": "noise four", "valid_at": "2023-06-04", "source_id": "D0:4", "matched_entity_uuids": ["entity_4"], "matched_entity_count": 1}],
                "edges": [{"uuid": "edge_4", "fact": "noise edge four", "matched_entity_uuids": ["entity_4"], "matched_entity_count": 1}],
                "nodes": [],
            },
        }

    async def ensure_indexes(self) -> None:
        return None

    async def search_entities(self, group_id: str, query: str, embedding: list[float], limit: int = 20):
        return list(self.entity_candidates[:limit])

    async def expand_entities_for_retrieval(self, group_id: str, entity_uuids: list[str], limit: int = 50):
        episodes = []
        edges = []
        nodes = []
        for entity_uuid in entity_uuids:
            payload = self.expansions[entity_uuid]
            episodes.extend(payload["episodes"])
            edges.extend(payload["edges"])
            nodes.extend(payload["nodes"])
        return {"episodes": episodes, "edges": edges, "nodes": nodes}

    async def search_episodes(self, group_id: str, query: str, embedding: list[float], limit: int = 20):
        return []

    async def fetch_max_layer(self, group_id: str) -> int:
        return 0

    async def fetch_one_hop_neighbors(self, group_id: str, node_uuids: list[str]):
        return {"episodes": [], "edges": [], "nodes": []}


def _build_retriever(monkeypatch: pytest.MonkeyPatch, reranker: _FakeReranker) -> MnemisRetriever:
    monkeypatch.setattr(retrieval_module, "build_reranker", lambda _llm, _config: reranker)
    return MnemisRetriever(store=_FakeStore(), llm=_FakeLLM(), config=_build_config())


async def _run_pre_fix_retrieve(retriever: MnemisRetriever, query: str, group_id: str) -> dict[str, object]:
    query_embedding = (await retriever.llm.embed([query]))[0]
    candidate_limit = retriever.config.retrieval_candidate_limit
    entity_candidates = await retriever.store.search_entities(group_id, query, query_embedding, limit=candidate_limit)
    entity_candidates = entity_candidates[: retriever.config.entity_top_k]
    candidate_scores = {item["uuid"]: item.get("rrf_score", 0.0) for item in entity_candidates}
    expanded = await retriever.store.expand_entities_for_retrieval(
        group_id,
        [item["uuid"] for item in entity_candidates],
        limit=candidate_limit,
    )
    expanded_episode_scores = {
        item["uuid"]: max([candidate_scores.get(uuid, 0.0) for uuid in item.get("matched_entity_uuids", [])], default=0.0)
        for item in expanded["episodes"]
    }
    expanded_edge_scores = {
        item["uuid"]: max([candidate_scores.get(uuid, 0.0) for uuid in item.get("matched_entity_uuids", [])], default=0.0)
        for item in expanded["edges"]
    }
    expanded_node_scores = {
        item["uuid"]: max([candidate_scores.get(uuid, 0.0) for uuid in item.get("matched_entity_uuids", [])], default=0.0)
        for item in expanded["nodes"]
    }
    direct_episode_hits = await retriever.store.search_episodes(group_id, query, query_embedding, limit=candidate_limit)
    nodes = retriever._annotate_route(entity_candidates, branch="entity_candidates")
    episodes = retriever._merge_items_by_uuid(
        retriever._annotate_route(expanded["episodes"], branch="entity_expansion", score_lookup=expanded_episode_scores),
        retriever._annotate_route(direct_episode_hits, branch="direct_episode", fallback_score=0.0),
    )
    edges = retriever._annotate_route(expanded["edges"], branch="entity_expansion", score_lookup=expanded_edge_scores)
    expanded_nodes = retriever._annotate_route(expanded["nodes"], branch="entity_expansion", score_lookup=expanded_node_scores)
    nodes = retriever._merge_items_by_uuid(nodes, expanded_nodes)
    system1 = {
        "episodes": retriever._format_items(retriever._sort_candidate_pool(episodes, limit=candidate_limit)),
        "nodes": retriever._format_items(retriever._sort_candidate_pool(nodes, limit=candidate_limit)),
        "edges": retriever._format_items(retriever._sort_candidate_pool(edges, limit=candidate_limit)),
    }
    system2 = await retriever._system2_retrieve(query, group_id)
    merged = retriever._merge_route_items(system1, system2)
    episodes_final, _ = await retriever._rerank_items(query, "episodes", merged["episodes"], retriever.config.episode_top_k)
    nodes_final, _ = await retriever._rerank_items(query, "nodes", merged["nodes"], retriever.config.entity_top_k)
    edges_final, _ = await retriever._rerank_items(query, "edges", merged["edges"], retriever.config.edge_top_k)
    return {
        "system1": system1,
        "final": {"episodes": episodes_final, "nodes": nodes_final, "edges": edges_final},
    }


def test_retrieve_reranks_from_full_raw_candidate_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = _FakeReranker(
        {
            "episode_gold": 10.0,
            "episode_1": 0.4,
            "episode_2": 0.3,
            "episode_4": 0.2,
            "entity_gold": 9.0,
            "entity_1": 0.4,
            "entity_2": 0.3,
            "entity_4": 0.2,
            "edge_gold": 8.0,
            "edge_1": 0.4,
            "edge_2": 0.3,
            "edge_4": 0.2,
        }
    )
    retriever = _build_retriever(monkeypatch, reranker)

    before = asyncio.run(_run_pre_fix_retrieve(retriever, "gold query", "locomo_user_0"))
    after = asyncio.run(retriever.retrieve("gold query", "locomo_user_0"))

    assert [item["uuid"] for item in before["system1"]["nodes"]] == ["entity_1", "entity_2"]
    assert "episode_gold" not in [item["uuid"] for item in before["final"]["episodes"]]
    assert "entity_gold" not in [item["uuid"] for item in before["final"]["nodes"]]

    assert [item["uuid"] for item in after["system1"]["nodes"]] == ["entity_1", "entity_2", "entity_gold", "entity_4"]
    assert after["final"]["episodes"][0]["uuid"] == "episode_gold"
    assert after["final"]["nodes"][0]["uuid"] == "entity_gold"
    assert after["final"]["edges"][0]["uuid"] == "edge_gold"
    assert after["counts"]["merged"]["episodes"] == {
        "system1_count": 4,
        "system2_count": 0,
        "raw_count": 4,
        "merged_unique_count": 4,
    }
    assert after["counts"]["final"]["episodes"] == {"reranked_count": 4, "final_count": 2}
