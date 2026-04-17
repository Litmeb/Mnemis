import asyncio
from types import MethodType

from mnemis_build.base_graph import BaseGraphBuilder
from mnemis_build.config import BuildConfig
from mnemis_build.hierarchical_graph import HierarchicalGraphBuilder
from mnemis_build.models import CategoryAssignmentPayload, EpisodeInput, IndexedNode


def _build_config(*, speaker_hierarchy_mode: str = "paper_v2", force_base_speaker_entity: bool = True) -> BuildConfig:
    return BuildConfig(
        neo4j_url="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        neo4j_database=None,
        llm_api_key="test-key",
        llm_base_url=None,
        llm_model="gpt-4.1-mini",
        small_llm_model="gpt-4.1-mini",
        rerank_mode="auto",
        reranker_api_key=None,
        reranker_base_url=None,
        reranker_model=None,
        rerank_allow_llm_fallback=True,
        embedding_model="text-embedding-3-small",
        embedding_dim=128,
        recent_episode_window=6,
        max_reflection_rounds=1,
        force_base_speaker_entity=force_base_speaker_entity,
        speaker_hierarchy_mode=speaker_hierarchy_mode,
        min_children_per_category=2,
        max_hierarchy_layers=4,
        max_categories_per_call=0,
        episode_top_k=10,
        entity_top_k=20,
        edge_top_k=20,
        retrieval_candidate_limit=50,
        rrf_k=60,
    )


def _build_hierarchy_builder(config: BuildConfig) -> HierarchicalGraphBuilder:
    builder = object.__new__(HierarchicalGraphBuilder)
    builder.config = config
    builder.store = None
    builder.llm = None
    return builder


def test_paper_v2_mode_reserves_marked_speaker_nodes() -> None:
    builder = _build_hierarchy_builder(_build_config(speaker_hierarchy_mode="paper_v2"))
    nodes = [
        IndexedNode(index=0, uuid="speaker-1", name="Alice", summary="Current episode speaker.", is_speaker=True),
        IndexedNode(index=1, uuid="entity-1", name="Peanuts", summary="Allergy trigger.", tag=["food"]),
    ]

    assignable, speaker_nodes = builder._partition_nodes_for_assignment(1, nodes)

    assert [node.name for node in assignable] == ["Peanuts"]
    assert [node.name for node in speaker_nodes] == ["Alice"]


def test_appendix_prompt_mode_only_reserves_literal_prompt_nodes() -> None:
    builder = _build_hierarchy_builder(_build_config(speaker_hierarchy_mode="appendix_prompt"))
    nodes = [
        IndexedNode(index=0, uuid="speaker-1", name="Alice", summary="Marked speaker entity.", is_speaker=True),
        IndexedNode(index=1, uuid="speaker-2", name="user", summary="Literal user node."),
    ]

    assignable, speaker_nodes = builder._partition_nodes_for_assignment(1, nodes)

    assert [node.name for node in assignable] == ["Alice"]
    assert [node.name for node in speaker_nodes] == ["user"]


def test_disabled_mode_keeps_all_nodes_assignable() -> None:
    builder = _build_hierarchy_builder(_build_config(speaker_hierarchy_mode="disabled"))
    nodes = [
        IndexedNode(index=0, uuid="speaker-1", name="Alice", summary="Marked speaker entity.", is_speaker=True),
        IndexedNode(index=1, uuid="speaker-2", name="user", summary="Literal user node."),
    ]

    assignable, speaker_nodes = builder._partition_nodes_for_assignment(1, nodes)

    assert [node.name for node in assignable] == ["Alice", "user"]
    assert speaker_nodes == []


def test_materialize_categories_injects_one_reserved_speaker_category() -> None:
    builder = _build_hierarchy_builder(_build_config(speaker_hierarchy_mode="paper_v2"))

    async def _fake_generate_category_details(self, layer, grouped):
        return {}

    builder._generate_category_details = MethodType(_fake_generate_category_details, builder)
    assignable_nodes = [
        IndexedNode(index=1, uuid="entity-1", name="Peanuts", summary="Allergy trigger.", tag=["food"]),
    ]
    speaker_nodes = [
        IndexedNode(index=0, uuid="speaker-1", name="Alice", summary="Current episode speaker.", is_speaker=True),
    ]
    assignments = CategoryAssignmentPayload.model_validate(
        [{"category": "Food", "indexes": [1]}]
    )

    categories = asyncio.run(
        builder._materialize_categories("group-1", 1, assignable_nodes, speaker_nodes, assignments)
    )

    by_name = {category.name: category for category in categories}
    assert set(by_name) == {"Food", "Speaker"}
    assert by_name["Food"].child_uuids == ["entity-1"]
    assert by_name["Speaker"].child_uuids == ["speaker-1"]


def test_base_graph_forced_speaker_name_can_be_disabled() -> None:
    builder = object.__new__(BaseGraphBuilder)
    builder.config = _build_config(force_base_speaker_entity=False)

    episode = EpisodeInput(
        speaker="Alice",
        content="Alice mentioned peanuts.",
        valid_at="2026-01-01T00:00:00",
        source_id="ep-1",
    )

    assert builder._forced_speaker_name(episode) is None
