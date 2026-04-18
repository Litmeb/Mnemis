import asyncio
import logging
from types import SimpleNamespace

import pytest

import mnemis_build.retrieval as retrieval_module
from mnemis_build.config import BuildConfig
from mnemis_build.retrieval import MnemisRetriever


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
    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeStore:
    def __init__(self) -> None:
        self.direct_episode_hits = {
            "relationship status": [
                {
                    "uuid": "episode_noise_relationship",
                    "content": "Caroline discussed a friend's relationship.",
                    "valid_at": "2023-05-01",
                    "source_id": "D0:1",
                    "rrf_score": 0.12,
                    "fulltext_score": 0.9,
                }
            ],
            "where did caroline move from 4 years ago": [
                {
                    "uuid": "episode_noise_move",
                    "content": "Caroline mentioned moving boxes four years ago.",
                    "valid_at": "2023-05-11",
                    "source_id": "D0:2",
                    "rrf_score": 0.14,
                    "fulltext_score": 0.88,
                }
            ],
        }
        self.entity_hits = {
            "relationship status": [
                {
                    "uuid": "entity_relationship_status",
                    "name": "Caroline Relationship Status",
                    "summary": "Caroline is single.",
                    "tag": ["relationship", "single"],
                    "rrf_score": 0.91,
                    "fulltext_score": 2.1,
                },
                {
                    "uuid": "entity_relationship_misc",
                    "name": "Relationship Small Talk",
                    "summary": "General relationship chatter.",
                    "tag": ["relationship"],
                    "rrf_score": 0.25,
                    "fulltext_score": 1.2,
                },
            ],
            "where did caroline move from 4 years ago": [
                {
                    "uuid": "entity_move_origin",
                    "name": "Caroline Move Origin",
                    "summary": "Caroline moved from Sweden 4 years ago.",
                    "tag": ["move", "sweden"],
                    "rrf_score": 0.95,
                    "fulltext_score": 2.3,
                },
                {
                    "uuid": "entity_move_misc",
                    "name": "Moving Day",
                    "summary": "Packing and unpacking logistics.",
                    "tag": ["move"],
                    "rrf_score": 0.22,
                    "fulltext_score": 1.1,
                },
            ],
        }
        self.expansions = {
            "entity_relationship_status": {
                "episodes": [
                    {
                        "uuid": "episode_rel_gold_1",
                        "content": "Caroline is single.",
                        "valid_at": "2023-06-01",
                        "source_id": "D3:13",
                        "matched_entity_uuids": ["entity_relationship_status"],
                        "matched_entity_names": ["Caroline Relationship Status"],
                        "matched_entity_count": 1,
                    },
                    {
                        "uuid": "episode_rel_gold_2",
                        "content": "Caroline confirmed she is single.",
                        "valid_at": "2023-05-20",
                        "source_id": "D2:14",
                        "matched_entity_uuids": ["entity_relationship_status"],
                        "matched_entity_names": ["Caroline Relationship Status"],
                        "matched_entity_count": 1,
                    },
                ],
                "edges": [
                    {
                        "uuid": "edge_rel_gold",
                        "fact": "Caroline is single.",
                        "valid_at": None,
                        "invalid_at": None,
                        "source_uuid": "entity_caroline",
                        "source_name": "Caroline",
                        "target_uuid": "entity_relationship_status",
                        "target_name": "Caroline Relationship Status",
                        "matched_entity_uuids": ["entity_relationship_status"],
                        "matched_entity_names": ["Caroline Relationship Status"],
                        "matched_entity_count": 1,
                    }
                ],
                "nodes": [],
            },
            "entity_move_origin": {
                "episodes": [
                    {
                        "uuid": "episode_move_gold_1",
                        "content": "Caroline talked about Sweden.",
                        "valid_at": "2023-06-01",
                        "source_id": "D3:13",
                        "matched_entity_uuids": ["entity_move_origin"],
                        "matched_entity_names": ["Caroline Move Origin"],
                        "matched_entity_count": 1,
                    },
                    {
                        "uuid": "episode_move_gold_2",
                        "content": "Caroline moved from Sweden 4 years ago.",
                        "valid_at": "2023-06-15",
                        "source_id": "D4:3",
                        "matched_entity_uuids": ["entity_move_origin"],
                        "matched_entity_names": ["Caroline Move Origin"],
                        "matched_entity_count": 1,
                    },
                ],
                "edges": [
                    {
                        "uuid": "edge_move_gold",
                        "fact": "Caroline moved from Sweden.",
                        "valid_at": None,
                        "invalid_at": None,
                        "source_uuid": "entity_caroline",
                        "source_name": "Caroline",
                        "target_uuid": "entity_move_origin",
                        "target_name": "Caroline Move Origin",
                        "matched_entity_uuids": ["entity_move_origin"],
                        "matched_entity_names": ["Caroline Move Origin"],
                        "matched_entity_count": 1,
                    }
                ],
                "nodes": [],
            },
        }

    async def search_entities(self, group_id: str, query: str, embedding: list[float], limit: int = 20):
        return list(self.entity_hits.get(query.lower(), []))

    async def expand_entities_for_retrieval(self, group_id: str, entity_uuids: list[str], limit: int = 50):
        episodes = []
        edges = []
        nodes = []
        for entity_uuid in entity_uuids:
            payload = self.expansions.get(entity_uuid, {})
            episodes.extend(payload.get("episodes", []))
            edges.extend(payload.get("edges", []))
            nodes.extend(payload.get("nodes", []))
        return {"episodes": episodes, "edges": edges, "nodes": nodes}

    async def search_episodes(self, group_id: str, query: str, embedding: list[float], limit: int = 20):
        return list(self.direct_episode_hits.get(query.lower(), []))

    async def search_edges(self, group_id: str, query: str, embedding: list[float], limit: int = 20):
        return []


def _build_retriever(monkeypatch: pytest.MonkeyPatch, store: _FakeStore) -> MnemisRetriever:
    monkeypatch.setattr(
        retrieval_module,
        "build_reranker",
        lambda _llm, _config: SimpleNamespace(last_status=None),
    )
    return MnemisRetriever(store=store, llm=_FakeLLM(), config=_build_config())


async def _run_pre_fix_system1(retriever: MnemisRetriever, query: str, group_id: str):
    query_embedding = (await retriever.llm.embed([query]))[0]
    candidate_limit = retriever.config.retrieval_candidate_limit
    episodes = await retriever.store.search_episodes(group_id, query, query_embedding, limit=candidate_limit)
    nodes = await retriever.store.search_entities(group_id, query, query_embedding, limit=candidate_limit)
    edges = await retriever.store.search_edges(group_id, query, query_embedding, limit=candidate_limit)
    return {
        "episodes": retriever._format_items(episodes[: retriever.config.episode_top_k]),
        "nodes": retriever._format_items(nodes[: retriever.config.entity_top_k]),
        "edges": retriever._format_items(edges[: retriever.config.edge_top_k]),
    }


def _episode_hit_count(payload: dict[str, list[dict[str, object]]], gold_evidence: list[str]) -> int:
    source_ids = {str(item.get("source_id")) for item in payload["episodes"]}
    return len(source_ids.intersection(gold_evidence))


@pytest.mark.parametrize(
    ("query", "gold_evidence"),
    [
        ("relationship status", ["D3:13", "D2:14"]),
        ("where did caroline move from 4 years ago", ["D3:13", "D4:3"]),
    ],
)
def test_system1_entity_first_expansion_improves_gold_evidence_hits(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    gold_evidence: list[str],
) -> None:
    before = asyncio.run(_run_pre_fix_system1(_build_retriever(monkeypatch, _FakeStore()), query, "locomo_user_0"))
    after = asyncio.run(_build_retriever(monkeypatch, _FakeStore())._system1_retrieve(query, "locomo_user_0"))

    before_hits = _episode_hit_count(before, gold_evidence)
    after_hits = _episode_hit_count(after, gold_evidence)

    assert before_hits == 0
    assert after_hits == len(gold_evidence)
    assert any(item.get("system1_branch") == "entity_expansion" for item in after["episodes"])
    assert any(item.get("system1_branch") == "entity_candidates" for item in after["nodes"])


def test_system1_entity_first_logs_candidate_and_expansion_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _build_retriever(monkeypatch, _FakeStore())
    retriever.logger.setLevel(logging.INFO)
    messages: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = _ListHandler()
    retriever.logger.addHandler(handler)

    try:
        result = asyncio.run(retriever._system1_retrieve("relationship status", "locomo_user_0"))
    finally:
        retriever.logger.removeHandler(handler)

    assert [item["source_id"] for item in result["episodes"][:2]] == ["D3:13", "D2:14"]
    combined = "\n".join(messages)
    assert "system1 entity-first" in combined
    assert "raw_entity_candidates=2" in combined
    assert "raw_episodes=3" in combined
    assert "merged_unique_episodes=3" in combined
    assert "episode_pool=3" in combined
    assert "expanded_episode_sources=['D3:13', 'D2:14']" in combined
    assert "raw_edges=1" in combined
    assert "merged_unique_edges=1" in combined
