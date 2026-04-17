# Mnemis Build Pipeline

This repository originally exposed only the global selector. The new `mnemis_build` package adds a paper-aligned graph construction flow for:

- base graph ingestion
- hierarchical graph rebuilding
- local Neo4j storage
- local retrieval / answer pipeline

## What Matches The Paper

- Base graph uses incremental episode ingestion
- Entity extraction follows: extract -> reflection -> de-dup -> summarize/tag
- Edge extraction follows: extract -> reflection -> de-dup
- Speaker is forcibly extracted as an entity
- Hierarchical graph is built bottom-up from layer 0 entities
- Category summaries and tags are generated from child members
- Hierarchical prompt follows the appendix constraints:
  - minimal abstraction
  - many-to-many mapping
  - no category names with `and`
  - no leftover nodes
  - `Speaker` rule for `user` / first-person references
- Hierarchy stops when compression constraints fail or max layer is reached
- Retrieval includes:
  - System-1 search with full-text + embedding fusion
  - System-2 top-down traversal using the released selector prompt
  - final per-type reranking and context assembly

## What Is Reconstructed

The paper does not publish the full base-graph prompts. Those parts are reconstructed from the method section and aligned to the released graph schema:

- `Entity`
- `Episodic`
- `RELATES_TO`
- `MENTIONS`
- `Category`
- `Category_{layer}`
- `CATEGORIZES`

## Environment

Set local Neo4j and model variables before running:

```powershell
$env:MNEMIS_NEO4J_URL="bolt://localhost:7687"
$env:MNEMIS_NEO4J_USER="neo4j"
$env:MNEMIS_NEO4J_PASSWORD="your-password"
$env:MNEMIS_OPENAI_API_KEY="your-key"
$env:MNEMIS_OPENAI_MODEL="gpt-4.1-mini"
$env:MNEMIS_OPENAI_SMALL_MODEL="gpt-4.1-mini"
$env:MNEMIS_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"
$env:EMBEDDING_DIM="128"
```

Optional knobs:

```powershell
$env:MNEMIS_MIN_CHILDREN_PER_CATEGORY="2"
$env:MNEMIS_MAX_HIERARCHY_LAYERS="4"
$env:MNEMIS_RECENT_EPISODE_WINDOW="6"
$env:MNEMIS_MAX_REFLECTION_ROUNDS="1"
$env:MNEMIS_EPISODE_TOP_K="10"
$env:MNEMIS_ENTITY_TOP_K="20"
$env:MNEMIS_EDGE_TOP_K="20"
$env:MNEMIS_RETRIEVAL_CANDIDATE_LIMIT="50"
$env:MNEMIS_RRF_K="60"
```

## Install

```powershell
python -m pip install -r requirements-mnemis-build.txt
```

## Run

Rebuild one LoCoMo user into local Neo4j:

```powershell
python .\build_mnemis_graph.py rebuild-locomo --group-id locomo_user_0 --user-index 0
```

Retrieve memory for one query:

```powershell
python .\build_mnemis_graph.py retrieve --group-id locomo_user_0 --query "Which cities did Dave travel to in 2023?"
```

Answer one query with the full local pipeline:

```powershell
python .\build_mnemis_graph.py answer --group-id locomo_user_0 --query "Which cities did Dave travel to in 2023?"
```

The resulting graph is directly compatible with `global_selection/global_selector.py`:

- categories are labeled as both `Category` and `Category_{layer}`
- hierarchy edges use `CATEGORIZES`
- entity-to-episode edges use `MENTIONS`
