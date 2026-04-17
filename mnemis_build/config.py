from __future__ import annotations

import os
from dataclasses import dataclass


def _pick_env(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


def _pick_bool_env(*keys: str, default: bool) -> bool:
    value = _pick_env(*keys)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_rerank_mode(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized in {"auto", "true_reranker", "llm_scoring"}:
        return normalized
    return "auto"


def _normalize_speaker_hierarchy_mode(value: str | None) -> str:
    normalized = (value or "paper_v2").strip().lower().replace("-", "_")
    aliases = {
        "paper": "paper_v2",
        "paperv2": "paper_v2",
        "appendix": "appendix_prompt",
        "appendix_v2": "appendix_prompt",
        "off": "disabled",
        "none": "disabled",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"paper_v2", "appendix_prompt", "disabled"}:
        return normalized
    return "paper_v2"


@dataclass(slots=True)
class BuildConfig:
    neo4j_url: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str | None
    llm_api_key: str
    llm_base_url: str | None
    llm_model: str
    small_llm_model: str
    rerank_mode: str
    reranker_api_key: str | None
    reranker_base_url: str | None
    reranker_model: str | None
    rerank_allow_llm_fallback: bool
    embedding_model: str
    embedding_dim: int
    recent_episode_window: int
    max_reflection_rounds: int
    force_base_speaker_entity: bool
    speaker_hierarchy_mode: str
    min_children_per_category: int
    max_hierarchy_layers: int
    max_categories_per_call: int
    episode_top_k: int
    entity_top_k: int
    edge_top_k: int
    retrieval_candidate_limit: int
    rrf_k: int

    @classmethod
    def from_env(cls) -> "BuildConfig":
        neo4j_user = _pick_env("MNEMIS_NEO4J_USER", "NEO4J_USERNAME")
        neo4j_password = _pick_env("MNEMIS_NEO4J_PASSWORD", "NEO4J_PASSWORD")
        llm_api_key = _pick_env("MNEMIS_OPENAI_API_KEY", "MNEMIS_API_KEY")
        if not neo4j_user or not neo4j_password:
            raise RuntimeError(
                "Missing Neo4j credentials. Set MNEMIS_NEO4J_USER/MNEMIS_NEO4J_PASSWORD "
                "or NEO4J_USERNAME/NEO4J_PASSWORD."
            )
        if not llm_api_key:
            raise RuntimeError(
                "Missing API key. Set MNEMIS_OPENAI_API_KEY or MNEMIS_API_KEY."
            )

        return cls(
            neo4j_url=_pick_env("MNEMIS_NEO4J_URL", "NEO4J_URI", default="bolt://localhost:7687") or "bolt://localhost:7687",
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=_pick_env("MNEMIS_NEO4J_DATABASE", "NEO4J_DATABASE"),
            llm_api_key=llm_api_key,
            llm_base_url=_pick_env("MNEMIS_OPENAI_BASE_URL", "MNEMIS_BASE_URL"),
            llm_model=_pick_env("MNEMIS_OPENAI_MODEL", "MNEMIS_MODEL", default="gpt-4.1-mini") or "gpt-4.1-mini",
            small_llm_model=_pick_env("MNEMIS_OPENAI_SMALL_MODEL", "MNEMIS_SMALL_MODEL", default="gpt-4.1-mini") or "gpt-4.1-mini",
            rerank_mode=_normalize_rerank_mode(_pick_env("MNEMIS_RERANK_MODE", default="auto")),
            reranker_api_key=_pick_env("MNEMIS_RERANKER_API_KEY", "MNEMIS_OPENAI_API_KEY", "MNEMIS_API_KEY"),
            reranker_base_url=_pick_env("MNEMIS_RERANKER_BASE_URL", "MNEMIS_OPENAI_BASE_URL", "MNEMIS_BASE_URL"),
            reranker_model=_pick_env("MNEMIS_RERANKER_MODEL"),
            rerank_allow_llm_fallback=_pick_bool_env("MNEMIS_RERANK_ALLOW_LLM_FALLBACK", default=True),
            embedding_model=_pick_env("MNEMIS_EMBEDDING_MODEL", default="Qwen/Qwen3-Embedding-0.6B") or "Qwen/Qwen3-Embedding-0.6B",
            embedding_dim=int(_pick_env("EMBEDDING_DIM", default="128") or "128"),
            recent_episode_window=int(_pick_env("MNEMIS_RECENT_EPISODE_WINDOW", default="6") or "6"),
            max_reflection_rounds=int(_pick_env("MNEMIS_MAX_REFLECTION_ROUNDS", default="1") or "1"),
            force_base_speaker_entity=_pick_bool_env("MNEMIS_FORCE_BASE_SPEAKER_ENTITY", default=True),
            speaker_hierarchy_mode=_normalize_speaker_hierarchy_mode(
                _pick_env("MNEMIS_SPEAKER_HIERARCHY_MODE", default="paper_v2")
            ),
            min_children_per_category=int(_pick_env("MNEMIS_MIN_CHILDREN_PER_CATEGORY", default="2") or "2"),
            max_hierarchy_layers=int(_pick_env("MNEMIS_MAX_HIERARCHY_LAYERS", default="4") or "4"),
            max_categories_per_call=int(_pick_env("MNEMIS_MAX_CATEGORIES_PER_CALL", default="0") or "0"),
            episode_top_k=int(_pick_env("MNEMIS_EPISODE_TOP_K", default="10") or "10"),
            entity_top_k=int(_pick_env("MNEMIS_ENTITY_TOP_K", default="20") or "20"),
            edge_top_k=int(_pick_env("MNEMIS_EDGE_TOP_K", default="20") or "20"),
            retrieval_candidate_limit=int(_pick_env("MNEMIS_RETRIEVAL_CANDIDATE_LIMIT", default="50") or "50"),
            rrf_k=int(_pick_env("MNEMIS_RRF_K", default="60") or "60"),
        )
