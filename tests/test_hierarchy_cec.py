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
        speaker_hierarchy_mode=speaker_hierarchy_mode,
        min_children_per_category=min_children_per_category,
        max_hierarchy_layers=4,
        max_categories_per_call=0,
        hierarchy_assignment_batch_size=3,
        category_detail_batch_size=2,
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


def test_extract_categories_batches_large_node_sets() -> None:
    class _BatchingLLM:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def complete_json(self, model, messages, **kwargs):
            self.calls.append(messages[1]["content"])
            payload = messages[1]["content"]
            indexes: list[int] = []
            for line in payload.splitlines():
                line = line.strip()
                if not line or ". " not in line:
                    continue
                prefix = line.split(". ", 1)[0]
                if prefix.isdigit():
                    indexes.append(int(prefix))
            return CategoryAssignmentPayload.model_validate(
                [{"category": f"Batch {len(self.calls)}", "indexes": indexes}]
            )

    builder = HierarchicalGraphBuilder(None, _BatchingLLM(), _build_config())
    nodes = [
        IndexedNode(index=i, uuid=f"entity-{i}", name=f"Node {i}", summary=f"Summary {i}.")
        for i in range(7)
    ]

    payload = asyncio.run(builder._extract_categories(1, nodes, {}, None))

    assert len(builder.llm.calls) == 3
    assert [assignment.indexes for assignment in payload.assignments] == [
        [0, 1, 2],
        [3, 4, 5],
        [6],
    ]


def test_generate_category_details_batches_large_category_sets() -> None:
    class _DetailBatchingLLM:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def complete_json(self, model, messages, **kwargs):
            self.calls.append(messages[1]["content"])
            categories: list[dict[str, object]] = []
            for line in messages[1]["content"].splitlines():
                if not line.startswith("Category: "):
                    continue
                name = line.removeprefix("Category: ").strip()
                categories.append(
                    {
                        "name": name,
                        "summary": f"{name} groups related members in this batch for compact hierarchy summaries.",
                        "tag": ["alpha", "beta", "gamma"],
                    }
                )
            return model.model_validate({"categories": categories})

    builder = HierarchicalGraphBuilder(None, _DetailBatchingLLM(), _build_config())
    grouped = {
        "Fruit": [IndexedNode(index=0, uuid="a", name="Apple", summary="Fruit.")],
        "Vehicles": [IndexedNode(index=1, uuid="b", name="Car", summary="Vehicle.")],
        "Pets": [IndexedNode(index=2, uuid="c", name="Dog", summary="Pet.")],
        "Cities": [IndexedNode(index=3, uuid="d", name="Paris", summary="City.")],
        "Music": [IndexedNode(index=4, uuid="e", name="Jazz", summary="Genre.")],
    }

    details = asyncio.run(builder._generate_category_details(1, grouped))

    assert len(builder.llm.calls) == 3
    assert set(details) == {"Fruit", "Vehicles", "Pets", "Cities", "Music"}
    assert details["Fruit"]["tag"] == ["alpha", "beta"]


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
