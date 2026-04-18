import asyncio
from types import SimpleNamespace

import mnemis_build.llm as llm_module
from mnemis_build.config import BuildConfig
from mnemis_build.llm import OpenAILLMClient
from mnemis_build.logging_utils import get_logger


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


def _embedding_response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=vector) for vector in vectors],
        usage=None,
    )


class _FakeEmbeddingsAPI:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses) -> None:
        self.embeddings = _FakeEmbeddingsAPI(responses)


def _build_client(responses) -> OpenAILLMClient:
    client = object.__new__(OpenAILLMClient)
    client.config = _build_config()
    client.recorder = None
    client.client = _FakeClient(responses)
    client.embedding_client = client.client
    client.logger = get_logger("test.llm")
    return client


def test_embed_retries_transient_empty_embedding_response(monkeypatch) -> None:
    async def _skip_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(llm_module.asyncio, "sleep", _skip_sleep)
    client = _build_client(
        [
            ValueError("No embedding data received"),
            _embedding_response([[1.0], [2.0]]),
        ]
    )

    vectors = asyncio.run(client.embed(["alpha", "beta"]))

    assert vectors == [[1.0], [2.0]]
    assert len(client.client.embeddings.calls) == 2


def test_embed_splits_large_requests_into_batches() -> None:
    client = _build_client(
        [
            _embedding_response([[float(i)] for i in range(32)]),
            _embedding_response([[32.0]]),
        ]
    )

    vectors = asyncio.run(client.embed([f"text-{i}" for i in range(33)]))

    assert len(vectors) == 33
    assert [len(call["input"]) for call in client.client.embeddings.calls] == [32, 1]
    assert all(call["dimensions"] == 128 for call in client.client.embeddings.calls)


def test_complete_json_passes_max_completion_tokens() -> None:
    client = object.__new__(OpenAILLMClient)
    client.config = _build_config()
    client.recorder = None
    client.logger = get_logger("test.llm")
    captured: list[dict] = []

    async def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"assignments":[{"category":"Research Labs","indexes":[0,1]}]}'),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )

    from mnemis_build.models import CategoryAssignmentPayload

    payload = asyncio.run(
        client.complete_json(
            CategoryAssignmentPayload,
            [{"role": "user", "content": "Return JSON."}],
            max_completion_tokens=123,
        )
    )

    assert payload.assignments[0].category == "Research Labs"
    assert captured[0]["max_completion_tokens"] == 123
