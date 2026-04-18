import asyncio
from types import SimpleNamespace

import pytest

import mnemis_build.retrieval as retrieval_module
from mnemis_build.config import BuildConfig
from mnemis_build.models import NodeSelectionList
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
        episode_top_k=10,
        entity_top_k=20,
        edge_top_k=20,
        retrieval_candidate_limit=50,
        rrf_k=60,
    )


class _ScriptedLLM:
    def __init__(self, selections_by_query: dict[str, list[NodeSelectionList]]) -> None:
        self.selections_by_query = {query: list(items) for query, items in selections_by_query.items()}

    async def complete_json(self, model, messages, **kwargs):
        prompt = messages[0]["content"]
        for query, queue in self.selections_by_query.items():
            if query in prompt:
                if not queue:
                    raise AssertionError(f"No scripted selection left for query: {query}")
                return queue.pop(0)
        raise AssertionError(f"Unexpected prompt: {prompt}")


class _FakeStore:
    def __init__(self) -> None:
        self.fetch_one_hop_neighbors_calls: list[list[str]] = []
        self.layer2_nodes = [
            {"uuid": "cat_personal_life", "name": "Personal Life", "tag": ["identity"], "layer": 2},
            {"uuid": "cat_public_events", "name": "Public Events", "tag": ["events"], "layer": 2},
            {"uuid": "cat_personal_history", "name": "Personal History", "tag": ["history"], "layer": 2},
        ]
        self.children = {
            "cat_personal_life": [
                {"uuid": "cat_relationships", "name": "Relationships", "tag": ["partner"], "layer": 1},
            ],
            "cat_public_events": [
                {"uuid": "cat_school_speech", "name": "School Speech", "tag": ["speech"], "layer": 1},
            ],
            "cat_personal_history": [
                {"uuid": "cat_moves", "name": "Moves", "tag": ["relocation"], "layer": 1},
            ],
            "cat_relationships": [
                {
                    "uuid": "entity_relationship_status",
                    "name": "Caroline Relationship Status",
                    "tag": ["single"],
                    "summary": "Caroline is single.",
                    "layer": 0,
                }
            ],
            "cat_school_speech": [
                {
                    "uuid": "entity_school_speech",
                    "name": "Caroline School Speech",
                    "tag": ["school", "speech"],
                    "summary": "Caroline gave a speech at a school the week before 9 June 2023.",
                    "layer": 0,
                }
            ],
            "cat_moves": [
                {
                    "uuid": "entity_move_origin",
                    "name": "Caroline Move Origin",
                    "tag": ["sweden"],
                    "summary": "Caroline moved from Sweden 4 years ago.",
                    "layer": 0,
                }
            ],
        }
        self.entity_neighbors = {
            "entity_relationship_status": {
                "episodes": [
                    {"uuid": "episode_d3_13", "content": "Caroline is single.", "valid_at": "2023-06-01", "source_id": "D3:13"},
                    {"uuid": "episode_d2_14", "content": "Caroline confirmed she is single.", "valid_at": "2023-05-20", "source_id": "D2:14"},
                ],
                "edges": [{"uuid": "edge_relationship_status", "fact": "Caroline is single.", "valid_at": None, "invalid_at": None}],
                "nodes": [],
            },
            "entity_school_speech": {
                "episodes": [
                    {
                        "uuid": "episode_d3_1",
                        "content": "Caroline gave a speech at a school the week before 9 June 2023.",
                        "valid_at": "2023-06-09",
                        "source_id": "D3:1",
                    }
                ],
                "edges": [{"uuid": "edge_school_speech", "fact": "Caroline gave a speech at a school.", "valid_at": None, "invalid_at": None}],
                "nodes": [],
            },
            "entity_move_origin": {
                "episodes": [
                    {"uuid": "episode_d3_13_move", "content": "Caroline talked about Sweden.", "valid_at": "2023-06-01", "source_id": "D3:13"},
                    {"uuid": "episode_d4_3", "content": "Caroline moved from Sweden 4 years ago.", "valid_at": "2023-06-15", "source_id": "D4:3"},
                ],
                "edges": [{"uuid": "edge_move_origin", "fact": "Caroline moved from Sweden.", "valid_at": None, "invalid_at": None}],
                "nodes": [],
            },
        }

    async def fetch_max_layer(self, group_id: str) -> int:
        return 2

    async def fetch_nodes_by_layer(self, group_id: str, layer: int) -> list[dict[str, object]]:
        assert layer == 2
        return list(self.layer2_nodes)

    async def fetch_child_nodes(self, group_id: str, parent_uuids: list[str]) -> list[dict[str, object]]:
        children: list[dict[str, object]] = []
        for parent_uuid in parent_uuids:
            children.extend(self.children.get(parent_uuid, []))
        return children

    async def fetch_all_descendants(self, group_id: str, parent_uuids: list[str]) -> list[dict[str, object]]:
        descendants: list[dict[str, object]] = []
        for parent_uuid in parent_uuids:
            descendants.extend(self.children.get(parent_uuid, []))
        return descendants

    async def fetch_descendant_entities(self, group_id: str, parent_uuids: list[str]) -> list[dict[str, object]]:
        descendants: list[dict[str, object]] = []
        for parent_uuid in parent_uuids:
            descendants.extend(self.children.get(parent_uuid, []))
        return [node for node in descendants if node.get("layer") in (None, 0)]

    async def fetch_one_hop_neighbors(self, group_id: str, node_uuids: list[str]) -> dict[str, list[dict[str, object]]]:
        self.fetch_one_hop_neighbors_calls.append(list(node_uuids))
        episodes: list[dict[str, object]] = []
        edges: list[dict[str, object]] = []
        nodes: list[dict[str, object]] = []
        for node_uuid in node_uuids:
            neighbor_payload = self.entity_neighbors.get(node_uuid)
            if not neighbor_payload:
                continue
            episodes.extend(neighbor_payload["episodes"])
            edges.extend(neighbor_payload["edges"])
            nodes.extend(neighbor_payload["nodes"])
        return {"episodes": episodes, "edges": edges, "nodes": nodes}


def _build_retriever(
    monkeypatch: pytest.MonkeyPatch,
    store: _FakeStore,
    llm: _ScriptedLLM,
) -> MnemisRetriever:
    monkeypatch.setattr(
        retrieval_module,
        "build_reranker",
        lambda _llm, _config: SimpleNamespace(last_status=None),
    )
    return MnemisRetriever(store=store, llm=llm, config=_build_config())


def _selection(name: str, uuid: str, *, get_all_children: bool = False) -> NodeSelectionList:
    return NodeSelectionList.model_validate(
        [{"name": name, "uuid": uuid, "get_all_children": get_all_children}]
    )


async def _run_pre_fix_system2(
    retriever: MnemisRetriever,
    query: str,
    group_id: str,
) -> dict[str, list[dict[str, object]]]:
    max_layer = await retriever.store.fetch_max_layer(group_id)
    previous_layer_nodes: list[dict[str, object]] = []
    selected_nodes: dict[str, dict[str, object]] = {}

    for layer in range(max_layer, 0, -1):
        if layer == max_layer:
            current_layer_nodes = await retriever.store.fetch_nodes_by_layer(group_id, layer)
        elif previous_layer_nodes:
            current_layer_nodes = await retriever.store.fetch_child_nodes(
                group_id,
                [node["uuid"] for node in previous_layer_nodes],
            )
        else:
            break
        selected, shortcuts = await retriever._layer_selection(query, current_layer_nodes)
        for node in selected:
            selected_nodes[node["uuid"]] = node
        descendants = await retriever.store.fetch_all_descendants(
            group_id,
            [node["uuid"] for node in shortcuts],
        )
        for node in descendants:
            selected_nodes[node["uuid"]] = node
        previous_layer_nodes = selected

    neighbors = await retriever.store.fetch_one_hop_neighbors(group_id, list(selected_nodes))
    selected_nodes.update({node["uuid"]: node for node in neighbors["nodes"]})
    return {
        "episodes": retriever._format_items(neighbors["episodes"]),
        "edges": retriever._format_items(neighbors["edges"]),
        "nodes": retriever._format_items(list(selected_nodes.values())),
    }


def _episode_hit_count(payload: dict[str, list[dict[str, object]]], gold_evidence: list[str]) -> int:
    source_ids = {str(item.get("source_id")) for item in payload["episodes"]}
    return len(source_ids.intersection(gold_evidence))


def test_system2_neighbor_expansion_uses_entity_leaf_nodes_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "What is Caroline's relationship status?"
    store = _FakeStore()
    llm = _ScriptedLLM(
        {
            query: [
                _selection("Personal Life", "cat_personal_life"),
                _selection("Relationships", "cat_relationships"),
            ]
        }
    )
    retriever = _build_retriever(monkeypatch, store, llm)

    result = asyncio.run(retriever._system2_retrieve(query, "locomo_user_0"))

    assert store.fetch_one_hop_neighbors_calls == [["entity_relationship_status"]]
    assert [item["uuid"] for item in result["nodes"]] == ["entity_relationship_status"]
    assert [item["source_id"] for item in result["episodes"]] == ["D3:13", "D2:14"]


def test_system2_locomo_queries_improve_gold_evidence_hits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    queries = {
        "What is Caroline's relationship status?": {
            "selections": [
                _selection("Personal Life", "cat_personal_life"),
                _selection("Relationships", "cat_relationships"),
            ],
            "gold_evidence": ["D3:13", "D2:14"],
        },
        "When did Caroline give a speech at a school?": {
            "selections": [
                _selection("Public Events", "cat_public_events"),
                _selection("School Speech", "cat_school_speech"),
            ],
            "gold_evidence": ["D3:1"],
        },
        "Where did Caroline move from 4 years ago?": {
            "selections": [
                _selection("Personal History", "cat_personal_history"),
                _selection("Moves", "cat_moves"),
            ],
            "gold_evidence": ["D3:13", "D4:3"],
        },
    }

    before_after: dict[str, tuple[int, int]] = {}
    for query, payload in queries.items():
        before_store = _FakeStore()
        before_retriever = _build_retriever(
            monkeypatch,
            before_store,
            _ScriptedLLM({query: list(payload["selections"])}),
        )
        before = asyncio.run(_run_pre_fix_system2(before_retriever, query, "locomo_user_0"))

        after_store = _FakeStore()
        after_retriever = _build_retriever(
            monkeypatch,
            after_store,
            _ScriptedLLM({query: list(payload["selections"])}),
        )
        after = asyncio.run(after_retriever._system2_retrieve(query, "locomo_user_0"))

        before_hits = _episode_hit_count(before, payload["gold_evidence"])
        after_hits = _episode_hit_count(after, payload["gold_evidence"])
        before_after[query] = (before_hits, after_hits)
        print(f"{query} | before_hits={before_hits} | after_hits={after_hits}")

        assert len(before["episodes"]) == 0
        assert len(after["episodes"]) > 0
        assert before_hits == 0
        assert after_hits > 0

    captured = capsys.readouterr()
    assert "relationship status" in captured.out
    assert "speech at a school" in captured.out
    assert "move from 4 years ago" in captured.out
    assert before_after["What is Caroline's relationship status?"] == (0, 2)
    assert before_after["When did Caroline give a speech at a school?"] == (0, 1)
    assert before_after["Where did Caroline move from 4 years ago?"] == (0, 2)
