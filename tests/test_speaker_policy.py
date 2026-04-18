import asyncio
from types import MethodType

from mnemis_build.base_graph import BaseGraphBuilder
from mnemis_build.config import BuildConfig
from mnemis_build.hierarchical_graph import HierarchicalGraphBuilder
from mnemis_build.logging_utils import get_logger
from mnemis_build.models import CategoryAssignmentPayload, EpisodeInput, IndexedNode, MinimalEdgeExtractionPayload, MinimalEntityExtractionPayload


def _build_config(*, speaker_hierarchy_mode: str = "paper_v2", force_base_speaker_entity: bool = True) -> BuildConfig:
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
        force_base_speaker_entity=force_base_speaker_entity,
        speaker_hierarchy_mode=speaker_hierarchy_mode,
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

    materialized = asyncio.run(
        builder._materialize_categories("group-1", 1, assignable_nodes, speaker_nodes, assignments)
    )
    categories = materialized.categories

    by_name = {category.name: category for category in categories}
    assert set(by_name) == {"Peanuts", "Speaker"}
    assert by_name["Peanuts"].child_uuids == ["entity-1"]
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


def test_base_graph_build_processes_episodes_in_input_order() -> None:
    class _FakeStore:
        def __init__(self) -> None:
            self.source_ids: list[str] = []
            self.marked_source_ids: list[str] = []

        async def ensure_indexes(self) -> None:
            return None

        async def fetch_recent_episodes(self, group_id, limit, exclude_source_id=None):
            return []

        async def upsert_episode(self, group_id, episode_uuid, episode, embedding):
            self.source_ids.append(episode.source_id)
            return episode_uuid

        async def mark_episode_ingested(self, group_id, source_id) -> None:
            self.marked_source_ids.append(source_id)

    builder = object.__new__(BaseGraphBuilder)
    builder.config = _build_config()
    builder.store = _FakeStore()
    builder.llm = None
    builder.logger = get_logger("test.base_graph")

    class _FakeLLM:
        async def embed(self, texts):
            return [[0.0] for _ in texts]

    builder.llm = _FakeLLM()

    async def _fake_extract_entities(self, group_id, episode_uuid, episode, context):
        return []

    async def _fake_extract_edges(self, group_id, context, entities) -> None:
        return None

    builder._extract_entities = MethodType(_fake_extract_entities, builder)
    builder._extract_edges = MethodType(_fake_extract_edges, builder)

    episodes = [
        EpisodeInput(
            speaker="Alice",
            content="turn 0",
            valid_at="2026-01-01T00:00:00",
            source_id="ep-1",
        ),
        EpisodeInput(
            speaker="Alice",
            content="turn 1",
            valid_at="2026-01-01T00:00:00",
            source_id="ep-2",
        ),
        EpisodeInput(
            speaker="Bob",
            content="turn 2",
            valid_at="2026-01-02T00:00:00",
            source_id="ep-3",
        ),
    ]

    asyncio.run(builder.build("group-1", episodes))

    assert builder.store.source_ids == ["ep-1", "ep-2", "ep-3"]
    assert builder.store.marked_source_ids == ["ep-1", "ep-2", "ep-3"]


def test_base_graph_build_reports_turn_progress() -> None:
    class _FakeStore:
        async def ensure_indexes(self) -> None:
            return None

        async def fetch_recent_episodes(self, group_id, limit, exclude_source_id=None):
            return []

        async def upsert_episode(self, group_id, episode_uuid, episode, embedding):
            return episode_uuid

    class _FakeLLM:
        async def embed(self, texts):
            return [[0.0] for _ in texts]

    builder = object.__new__(BaseGraphBuilder)
    builder.config = _build_config()
    builder.store = _FakeStore()
    builder.llm = _FakeLLM()
    builder.logger = get_logger("test.base_graph")

    async def _fake_extract_entities(self, group_id, episode_uuid, episode, context):
        return []

    async def _fake_extract_edges(self, group_id, context, entities) -> None:
        return None

    builder._extract_entities = MethodType(_fake_extract_entities, builder)
    builder._extract_edges = MethodType(_fake_extract_edges, builder)

    progress_events: list[tuple[int, int, str]] = []

    async def _progress_callback(completed_count, total_count, episode) -> None:
        progress_events.append((completed_count, total_count, episode.source_id))

    episodes = [
        EpisodeInput(speaker="Alice", content="turn 0", valid_at="2026-01-01T00:00:00", source_id="ep-1"),
        EpisodeInput(speaker="Alice", content="turn 1", valid_at="2026-01-01T00:00:00", source_id="ep-2"),
    ]

    asyncio.run(builder.build("group-1", episodes, progress_callback=_progress_callback))

    assert progress_events == [(1, 2, "ep-1"), (2, 2, "ep-2")]


def test_base_graph_build_can_resume_from_start_index() -> None:
    class _FakeStore:
        def __init__(self) -> None:
            self.source_ids: list[str] = []

        async def ensure_indexes(self) -> None:
            return None

        async def fetch_recent_episodes(self, group_id, limit, exclude_source_id=None):
            return []

        async def upsert_episode(self, group_id, episode_uuid, episode, embedding):
            self.source_ids.append(episode.source_id)
            return episode_uuid

    class _FakeLLM:
        async def embed(self, texts):
            return [[0.0] for _ in texts]

    builder = object.__new__(BaseGraphBuilder)
    builder.config = _build_config()
    builder.store = _FakeStore()
    builder.llm = _FakeLLM()
    builder.logger = get_logger("test.base_graph")

    async def _fake_extract_entities(self, group_id, episode_uuid, episode, context):
        return []

    async def _fake_extract_edges(self, group_id, context, entities) -> None:
        return None

    builder._extract_entities = MethodType(_fake_extract_entities, builder)
    builder._extract_edges = MethodType(_fake_extract_edges, builder)

    episodes = [
        EpisodeInput(speaker="Alice", content="turn 0", valid_at="2026-01-01T00:00:00", source_id="ep-1"),
        EpisodeInput(speaker="Alice", content="turn 1", valid_at="2026-01-01T00:00:00", source_id="ep-2"),
        EpisodeInput(speaker="Bob", content="turn 2", valid_at="2026-01-02T00:00:00", source_id="ep-3"),
    ]

    asyncio.run(builder.build("group-1", episodes, start_index=2))

    assert builder.store.source_ids == ["ep-3"]


def test_edge_reflection_parse_failure_is_skipped() -> None:
    class _FakeStore:
        async def fetch_entities_by_name(self, group_id, names):
            return {
                "Alice": {"uuid": "entity-alice", "name": "Alice"},
                "Bob": {"uuid": "entity-bob", "name": "Bob"},
            }

        async def search_edge_dedup_candidates(self, group_id, fact, fact_embedding, source_uuid, target_uuid):
            return []

        async def upsert_edge(self, edge, fact_embedding, source_uuid, target_uuid) -> None:
            return None

    class _FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, model, messages, **kwargs):
            from mnemis_build.models import MinimalEdgeExtractionPayload

            self.calls += 1
            if self.calls == 1:
                return MinimalEdgeExtractionPayload.model_validate(
                    {"edges": [{"source_entity_name": "Alice", "target_entity_name": "Bob", "fact": "Alice knows Bob"}]}
                )
            raise ValueError("mock parse failure")

        async def embed(self, texts):
            return [[0.0] for _ in texts]

    builder = object.__new__(BaseGraphBuilder)
    builder.config = _build_config()
    builder.store = _FakeStore()
    builder.llm = _FakeLLM()
    builder.logger = get_logger("test.base_graph")

    entities = [
        IndexedNode(index=0, uuid="entity-alice", name="Alice", summary="Person."),
        IndexedNode(index=1, uuid="entity-bob", name="Bob", summary="Person."),
    ]

    asyncio.run(builder._extract_edges("group-1", "Alice talked to Bob.", entities))


def test_entity_detail_generation_is_chunked_to_fit_token_budget() -> None:
    class _FakeLLM:
        def __init__(self) -> None:
            self.detail_batch_sizes: list[int] = []

        async def complete_json(self, model, messages, **kwargs):
            if model.__name__ == "EntityNameExtraction":
                return model.model_validate({"names": [f"Entity {i}" for i in range(7)]})
            payload = __import__("json").loads(messages[1]["content"])
            self.detail_batch_sizes.append(len(payload["entities"]))
            return MinimalEntityExtractionPayload.model_validate(
                {
                    "entities": [
                        {
                            "name": item["name"],
                            "summary": f"Summary for {item['name']}",
                            "tag": ["tag"],
                        }
                        for item in payload["entities"]
                    ]
                }
            )

        async def embed(self, texts):
            return [[0.0] for _ in texts]

    class _FakeStore:
        async def search_entity_dedup_candidates(self, group_id, name, embedding):
            return []

        async def upsert_entity(self, record, name_embedding, summary_embedding) -> None:
            return None

        async def connect_entity_to_episode(self, entity_uuid, episode_uuid, group_id) -> None:
            return None

    builder = object.__new__(BaseGraphBuilder)
    builder.config = _build_config()
    builder.config.entity_detail_max_completion_tokens = 384
    builder.store = _FakeStore()
    builder.llm = _FakeLLM()
    builder.logger = get_logger("test.base_graph")

    episode = EpisodeInput(
        speaker="Alice",
        content="Conversation about several durable entities.",
        valid_at="2026-01-01T00:00:00",
        source_id="ep-1",
    )

    entities = asyncio.run(builder._extract_entities("group-1", "episode-1", episode, "context"))

    assert len(entities) == 8
    assert builder.llm.detail_batch_sizes == [3, 3, 2]


def test_entity_detail_generation_falls_back_when_json_is_truncated() -> None:
    class _FakeLLM:
        def __init__(self) -> None:
            self.detail_calls = 0

        async def complete_json(self, model, messages, **kwargs):
            if model.__name__ == "EntityNameExtraction":
                return model.model_validate({"names": ["Project Atlas", "Alice"]})
            self.detail_calls += 1
            raise ValueError("Could not obtain valid JSON after 3 attempts")

        async def embed(self, texts):
            return [[0.0] for _ in texts]

    class _FakeStore:
        def __init__(self) -> None:
            self.records = []

        async def search_entity_dedup_candidates(self, group_id, name, embedding):
            return []

        async def upsert_entity(self, record, name_embedding, summary_embedding) -> None:
            self.records.append(record)

        async def connect_entity_to_episode(self, entity_uuid, episode_uuid, group_id) -> None:
            return None

    builder = object.__new__(BaseGraphBuilder)
    builder.config = _build_config()
    builder.store = _FakeStore()
    builder.llm = _FakeLLM()
    builder.logger = get_logger("test.base_graph")

    episode = EpisodeInput(
        speaker="Alice",
        content="Alice mentioned Project Atlas.",
        valid_at="2026-01-01T00:00:00",
        source_id="ep-1",
    )

    entities = asyncio.run(builder._extract_entities("group-1", "episode-1", episode, "context"))

    by_name = {entity.name: entity for entity in entities}
    assert set(by_name) == {"Project Atlas", "Alice"}
    assert "Project Atlas" in by_name["Project Atlas"].summary
    assert by_name["Alice"].is_speaker is True


def test_edge_extraction_is_chunked_to_fit_token_budget() -> None:
    class _FakeLLM:
        def __init__(self) -> None:
            self.edge_batch_sizes: list[int] = []

        async def complete_json(self, model, messages, **kwargs):
            payload = __import__("json").loads(messages[1]["content"])
            if kwargs.get("operation") == "edge_reflection":
                return MinimalEdgeExtractionPayload.model_validate({"edges": []})
            self.edge_batch_sizes.append(len(payload["entities"]))
            entities = payload["entities"]
            if len(entities) < 2:
                return MinimalEdgeExtractionPayload.model_validate({"edges": []})
            return MinimalEdgeExtractionPayload.model_validate(
                {
                    "edges": [
                        {
                            "source_entity_name": entities[0]["name"],
                            "target_entity_name": entities[1]["name"],
                            "fact": f"{entities[0]['name']} relates to {entities[1]['name']}",
                        }
                    ]
                }
            )

        async def embed(self, texts):
            return [[0.0] for _ in texts]

    class _FakeStore:
        async def fetch_entities_by_name(self, group_id, names):
            return {name: {"uuid": f"uuid-{name}", "name": name} for name in names}

        async def search_edge_dedup_candidates(self, group_id, fact, fact_embedding, source_uuid, target_uuid):
            return []

        async def upsert_edge(self, edge, fact_embedding, source_uuid, target_uuid) -> None:
            return None

    builder = object.__new__(BaseGraphBuilder)
    builder.config = _build_config()
    builder.config.edge_extraction_max_completion_tokens = 384
    builder.store = _FakeStore()
    builder.llm = _FakeLLM()
    builder.logger = get_logger("test.base_graph")

    entities = [
        IndexedNode(index=i, uuid=f"entity-{i}", name=f"Entity {i}", summary="Person.")
        for i in range(7)
    ]

    asyncio.run(builder._extract_edges("group-1", "context", entities))

    assert builder.llm.edge_batch_sizes == [3, 3, 1]


def test_edge_extraction_falls_back_when_json_is_truncated() -> None:
    class _FakeLLM:
        async def complete_json(self, model, messages, **kwargs):
            if kwargs.get("operation") == "edge_reflection":
                return MinimalEdgeExtractionPayload.model_validate({"edges": []})
            raise ValueError("Could not obtain valid JSON after 3 attempts")

        async def embed(self, texts):
            return [[0.0] for _ in texts]

    class _FakeStore:
        def __init__(self) -> None:
            self.upsert_calls = 0

        async def fetch_entities_by_name(self, group_id, names):
            return {name: {"uuid": f"uuid-{name}", "name": name} for name in names}

        async def search_edge_dedup_candidates(self, group_id, fact, fact_embedding, source_uuid, target_uuid):
            return []

        async def upsert_edge(self, edge, fact_embedding, source_uuid, target_uuid) -> None:
            self.upsert_calls += 1

    builder = object.__new__(BaseGraphBuilder)
    builder.config = _build_config()
    builder.store = _FakeStore()
    builder.llm = _FakeLLM()
    builder.logger = get_logger("test.base_graph")

    entities = [
        IndexedNode(index=0, uuid="entity-a", name="Alice", summary="Person."),
        IndexedNode(index=1, uuid="entity-b", name="Bob", summary="Person."),
        IndexedNode(index=2, uuid="entity-c", name="Carol", summary="Person."),
    ]

    asyncio.run(builder._extract_edges("group-1", "context", entities))

    assert builder.store.upsert_calls == 0
