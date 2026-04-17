import asyncio
from types import MethodType

from mnemis_build.config import BuildConfig
from mnemis_build.hierarchical_graph import HierarchicalGraphBuilder, MaterializedLayer
from mnemis_build.models import CategoryAssignmentPayload, CategoryRecord, IndexedNode


def _build_config(
    *,
    speaker_hierarchy_mode: str = "paper_v2",
    min_children_per_category: int = 2,
) -> BuildConfig:
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
        force_base_speaker_entity=True,
        speaker_hierarchy_mode=speaker_hierarchy_mode,
        min_children_per_category=min_children_per_category,
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


class _FakeStore:
    def __init__(self) -> None:
        self.upserted: list[CategoryRecord] = []
        self.connected: list[tuple[str, str, str]] = []

    async def clear_hierarchy(self, group_id: str) -> None:
        self.cleared_group_id = group_id

    async def fetch_layer_zero_nodes(self, group_id: str) -> list[dict[str, object]]:
        return [
            {"uuid": "entity-0", "name": "Apple", "summary": "Fruit.", "tag": []},
            {"uuid": "entity-1", "name": "Banana", "summary": "Fruit.", "tag": []},
            {"uuid": "entity-2", "name": "Car", "summary": "Vehicle.", "tag": []},
        ]

    async def upsert_category(self, category: CategoryRecord, embedding: list[float]) -> None:
        self.upserted.append(category)

    async def connect_category(self, parent_uuid: str, child_uuid: str, group_id: str) -> None:
        self.connected.append((parent_uuid, child_uuid, group_id))


class _FakeLLM:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def test_materialize_categories_only_promotes_truly_uncovered_nodes() -> None:
    builder = _build_hierarchy_builder(_build_config())

    async def _fake_generate_category_details(self, layer, grouped):
        return {}

    builder._generate_category_details = MethodType(_fake_generate_category_details, builder)
    assignable_nodes = [
        IndexedNode(index=0, uuid="entity-0", name="Apple", summary="Fruit."),
        IndexedNode(index=1, uuid="entity-1", name="Banana", summary="Fruit."),
        IndexedNode(index=2, uuid="entity-2", name="Car", summary="Vehicle."),
    ]
    assignments = CategoryAssignmentPayload.model_validate(
        [
            {"category": "Singleton Fruit", "indexes": [0]},
            {"category": "Fruit", "indexes": [0, 1]},
        ]
    )

    materialized = asyncio.run(
        builder._materialize_categories("group-1", 1, assignable_nodes, [], assignments)
    )

    by_name = {category.name: category for category in materialized.categories}
    assert set(by_name) == {"Fruit", "Car"}
    assert by_name["Fruit"].child_uuids == ["entity-0", "entity-1"]
    assert by_name["Car"].child_uuids == ["entity-2"]
    assert materialized.promoted_child_uuids == {"entity-2"}


def test_passes_compression_rejects_non_promoted_singletons() -> None:
    builder = _build_hierarchy_builder(_build_config())
    current_nodes = [
        IndexedNode(index=0, uuid="entity-0", name="Apple", summary="Fruit."),
        IndexedNode(index=1, uuid="entity-1", name="Banana", summary="Fruit."),
    ]
    materialized = MaterializedLayer(
        categories=[
            CategoryRecord(
                uuid="category-1",
                group_id="group-1",
                name="Fruit",
                summary="Fruit group.",
                tag=[],
                layer=1,
                child_uuids=["entity-0"],
            )
        ],
        natural_child_uuids={"entity-0"},
        promoted_child_uuids=set(),
    )

    assert builder._passes_compression(1, current_nodes, materialized) is False


def test_promoted_singleton_avoids_name_collision_with_natural_group() -> None:
    builder = _build_hierarchy_builder(_build_config())

    async def _fake_generate_category_details(self, layer, grouped):
        return {}

    builder._generate_category_details = MethodType(_fake_generate_category_details, builder)
    assignable_nodes = [
        IndexedNode(index=0, uuid="entity-0", name="Fruit", summary="A fruit node."),
        IndexedNode(index=1, uuid="entity-1", name="Apple", summary="Fruit."),
        IndexedNode(index=2, uuid="entity-2", name="Banana", summary="Fruit."),
    ]
    assignments = CategoryAssignmentPayload.model_validate(
        [{"category": "Fruit", "indexes": [1, 2]}]
    )

    materialized = asyncio.run(
        builder._materialize_categories("group-1", 1, assignable_nodes, [], assignments)
    )

    by_name = {category.name: category for category in materialized.categories}
    assert set(by_name) == {"Fruit", "Fruit (Promoted)"}
    assert by_name["Fruit"].child_uuids == ["entity-1", "entity-2"]
    assert by_name["Fruit (Promoted)"].child_uuids == ["entity-0"]


def test_passes_compression_rejects_layer_two_node_count_increase() -> None:
    builder = _build_hierarchy_builder(_build_config())
    current_nodes = [
        IndexedNode(index=0, uuid="category-a", name="Fruit", summary="Fruit group.", layer=1),
        IndexedNode(index=1, uuid="category-b", name="Vehicle", summary="Vehicle group.", layer=1),
    ]
    materialized = MaterializedLayer(
        categories=[
            CategoryRecord(
                uuid="upper-1",
                group_id="group-1",
                name="Consumer Concepts",
                summary="Consumer concepts.",
                tag=[],
                layer=2,
                child_uuids=["category-a", "category-b"],
            ),
            CategoryRecord(
                uuid="upper-2",
                group_id="group-1",
                name="Daily Topics",
                summary="Daily topics.",
                tag=[],
                layer=2,
                child_uuids=["category-a", "category-b"],
            ),
            CategoryRecord(
                uuid="upper-3",
                group_id="group-1",
                name="Fruit",
                summary="Directly promoted fruit node.",
                tag=[],
                layer=2,
                child_uuids=["category-a"],
            ),
        ],
        natural_child_uuids={"category-a", "category-b"},
        promoted_child_uuids={"category-a"},
    )

    assert builder._passes_compression(2, current_nodes, materialized) is False


def test_rebuild_stops_when_layer_two_breaks_node_count_rule() -> None:
    builder = HierarchicalGraphBuilder(_FakeStore(), _FakeLLM(), _build_config())

    async def _fake_generate_category_details(self, layer, grouped):
        return {}

    async def _fake_extract_categories(self, layer, nodes, existing_categories, speaker_policy_note=None):
        if layer == 1:
            return CategoryAssignmentPayload.model_validate(
                [{"category": "Fruit", "indexes": [0, 1]}]
            )
        return CategoryAssignmentPayload.model_validate(
            [
                {"category": "Consumer Concepts", "indexes": [0, 1]},
                {"category": "Daily Topics", "indexes": [0, 1]},
                {"category": "General Knowledge", "indexes": [0, 1]},
            ]
        )

    builder._generate_category_details = MethodType(_fake_generate_category_details, builder)
    builder._extract_categories = MethodType(_fake_extract_categories, builder)

    created = asyncio.run(builder.rebuild("group-1"))

    assert [category.name for category in created] == ["Fruit", "Car"]
    assert [category.name for category in builder.store.upserted] == ["Fruit", "Car"]
