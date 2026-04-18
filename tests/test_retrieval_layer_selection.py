import asyncio
from types import SimpleNamespace

import pytest

import mnemis_build.retrieval as retrieval_module
from global_selection.prompts import NODE_SELECTION_PROMPT_TEMPLATE
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


class _FakeLLM:
    def __init__(self, response: NodeSelectionList | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def complete_json(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": list(messages), "kwargs": kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _build_retriever(monkeypatch: pytest.MonkeyPatch, llm: _FakeLLM) -> MnemisRetriever:
    monkeypatch.setattr(
        retrieval_module,
        "build_reranker",
        lambda _llm, _config: SimpleNamespace(last_status=None),
    )
    return MnemisRetriever(store=SimpleNamespace(), llm=llm, config=_build_config())


def test_node_selection_prompt_requires_strict_json() -> None:
    assert "Return strict JSON only." in NODE_SELECTION_PROMPT_TEMPLATE
    assert '"required": ["selections"]' in NODE_SELECTION_PROMPT_TEMPLATE
    assert '"additionalProperties": false' in NODE_SELECTION_PROMPT_TEMPLATE


def test_layer_selection_relationship_status_returns_non_empty_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _FakeLLM(
        NodeSelectionList.model_validate(
            [
                {
                    "name": "Relationships",
                    "uuid": "cat_relationships",
                    "get_all_children": True,
                }
            ]
        )
    )
    retriever = _build_retriever(monkeypatch, llm)
    layer_4_nodes = [
        {"uuid": "cat_profile", "name": "Personal Profile", "tag": ["identity"], "layer": 4},
        {"uuid": "cat_relationships", "name": "Relationships", "tag": ["partner", "dating"], "layer": 4},
        {"uuid": "cat_work", "name": "Work", "tag": ["career"], "layer": 4},
    ]

    selected, shortcuts = asyncio.run(
        retriever._layer_selection("What is my relationship status?", layer_4_nodes)
    )

    assert not selected
    assert shortcuts
    assert shortcuts[0]["uuid"] == "cat_relationships"
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "single JSON object with exactly one top-level key" in prompt
    assert "What is my relationship status?" in prompt


def test_layer_selection_propagates_non_json_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _build_retriever(
        monkeypatch,
        _FakeLLM(ValueError("Could not obtain valid JSON after 3 attempts for NodeSelectionList.")),
    )

    with pytest.raises(ValueError, match="valid JSON"):
        asyncio.run(
            retriever._layer_selection(
                "What is my relationship status?",
                [{"uuid": "cat_relationships", "name": "Relationships", "tag": ["partner"], "layer": 4}],
            )
        )
