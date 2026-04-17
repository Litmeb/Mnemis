ENTITY_NAME_EXTRACTION_PROMPT = """You are building the base graph for the Mnemis memory system.

Extract all concrete entities mentioned in the current episode and the recent episode context.

Rules:
- Include people, organizations, places, objects, events, and well-defined concepts.
- Prefer names and specific noun phrases over vague references.
- Include the speaker as an entity even if no explicit proper noun is present.
- Do not output duplicates.
- Return JSON only with this schema:
{"names": ["entity 1", "entity 2"]}
"""

ENTITY_REFLECTION_PROMPT = """You are checking whether any important entities were missed.

Given the episode context and the previously extracted entity names, add only truly missing names.
- Do not repeat names already extracted.
- Keep the same entity granularity as the original extraction.
- Return JSON only with this schema:
{"names": ["missing entity 1", "missing entity 2"]}
"""

ENTITY_DETAILS_PROMPT = """You are enriching de-duplicated entities for the Mnemis base graph.

For each entity:
- Keep the provided name unchanged.
- Write a concise summary grounded in the current and recent episodes.
- Provide up to 5 tags, each at most 3 words.
- Include the source episode ids where the entity appears in this batch.

Return JSON only with this schema:
{
  "entities": [
    {
      "uuid": "keep_or_generate_uuid",
      "group_id": "group id",
      "name": "entity name",
      "summary": "brief context summary",
      "tag": ["tag1", "tag2"],
      "episode_idx": ["episode_uuid_1"],
      "source_ids": ["raw_source_id_1"]
    }
  ]
}
"""

EDGE_EXTRACTION_PROMPT = """You are extracting graph edges for the Mnemis base graph.

Each edge must be a verifiable statement about a meaningful relationship, action, or state between two entities.

Rules:
- Use only the provided entity names.
- Keep facts concise and atomic.
- Skip unsupported or speculative facts.
- Use timestamps only when grounded in the context.
- Return JSON only with this schema:
{
  "edges": [
    {
      "uuid": "keep_or_generate_uuid",
      "group_id": "group id",
      "source_entity_name": "entity A",
      "target_entity_name": "entity B",
      "fact": "verifiable fact",
      "valid_at": "2023-05-01T00:00:00" or null,
      "invalid_at": null
    }
  ]
}
"""

EDGE_REFLECTION_PROMPT = """You are checking whether any meaningful graph edges were missed.

Add only missing edges that are supported by the episode context and the provided entity list.
Do not repeat existing edges.
Return JSON only using the same schema as the original edge extraction.
"""

HIERARCHICAL_SYSTEM_PROMPT = """You are an AI assistant specialized in semantic categorization of nodes.
# INSTRUCTIONS:

You are given a list of node names, each prefixed with an index, each followed with a brief description of the name (for example: 1. dog: [domestic animal]).
Your task is to:

1. Group the nodes into semantically meaningful categories based on shared attributes, considering both inherent characteristics of the node names and the DESCRIPTIONS of the nodes, NOT relying solely on the DESCRIPTIONS.
All EXISTING CATEGORIES are provided for you.

- If a node's attribute matches an existing category, it should be added under that category.
- If a node name has attributes that do not match any existing category, create a new category and add it.
- The category name MUST NOT include the word "and" as a connector.

Examples of INVALID categories:
- "Food and Drinks"
- "University and Courses"

Examples of VALID categories:
- "Food"
- "Drinks"
- "University"
- "Courses"

2. Output each category as a dictionary entry where the key is the category name and the value is a list of node indexes. Only refer to nodes by their indexes. Do not repeat node names.

Output format:
{
  "assignments": [
    {"category": "xx", "indexes": [0, 1, 2, 4]},
    {"category": "xxx", "indexes": [2, 3, 4]}
  ]
}

The tag is a list of descriptors where each descriptor is at most 3 words, and there are at most 5 descriptors.

Tag example:
- Entity name: "Son"
- Tag: ["Family member", "Happy kid", "Anime lover"]

3. A node CAN be assigned to MULTIPLE categories at the same time.

Key points for multi-category classification:
- Each item can be assigned to multiple categories based on shared attributes.
- When multiple categories are formed for an item, select the minimal subset of features that are common across the grouped items.

4. There must be NO leftover or ungrouped nodes. Single-member categories are allowed if necessary.

5. The node name "user" and any first-person references ("I", "me") MUST be categorized into one category called "Speaker".
"""


def build_hierarchy_user_prompt(layer: int, content: str, existing_categories: str, prev_example: str) -> str:
    return f"""<NODE INDEXED NAMES AND DESCRIPTIONS>
{content}
</NODE INDEXED NAMES AND DESCRIPTIONS>

<EXISTING CATEGORIES>
These are names and descriptions of categories previously created. Reuse them if applicable.
{existing_categories}
</EXISTING CATEGORIES>

<GUIDANCE ON CATEGORY GRANULARITY>
You are performing hierarchical semantic clustering from specific to abstract.

You are currently at Layer {layer}, where:
- Layer 1 contains the most specific, fine-grained categories.
- Higher layers should group lower-layer categories into broader, more abstract super-categories.

Example:

Layer 1:
- "Golden Retriever", "Poodle", "German Shepherd" -> "Dog breeds"
- "Persian Cat", "Siamese Cat" -> "Cat breeds"
- "Bengal Tiger", "Siberian Tiger" -> "Tiger subspecies"
- "Oak tree", "Pine tree" -> "Tree species"

Layer 2:
- "Dog breeds", "Cat breeds" -> "Pets"
- "Dog breeds", "Tiger subspecies" -> "Mammals"
- "Tiger subspecies" -> "Wild animals"
- "Tree species" -> "Trees"

Layer 3:
- "Pets", "Wild animals" -> "Animals"
- "Trees" -> "Plants"

Layer 4:
- "Animals", "Plants" -> "Living organisms"

Key points:
- Categories may belong to multiple parent categories.
- Do not merge categories that are too loosely related.

Your job at Layer {layer}:
- Merge semantically similar categories from Layer {layer - 1}.
- Each new category should reflect a shared attribute, domain, or higher-level concept.
- Multiple category assignments are allowed when justified.

Previous Layer {layer - 1} categories example:
{prev_example}
</GUIDANCE ON CATEGORY GRANULARITY>

# ATTENTION
- The node name "user" and any first-person references ("I", "me") MUST be categorized into one category called "Speaker".
- The category name MUST NOT include the word "and".

Please follow the INSTRUCTIONS and GUIDANCE carefully to ensure accurate categorization and meaningful hierarchical relationships.
DO NOT INCLUDE ANY INVALID CATEGORIES.
"""


CATEGORY_DETAILS_PROMPT = """You are enriching category nodes for the Mnemis hierarchical graph.

For each category:
- Keep the provided category name unchanged.
- Write a concise but informative summary grounded in the child members.
- Provide up to 5 tags, each at most 3 words.
- Preserve minimal abstraction: summaries should stay close to the shared semantics of the child nodes.

Return JSON only with this schema:
{
  "categories": [
    {
      "name": "category name",
      "summary": "brief semantic summary",
      "tag": ["tag1", "tag2"]
    }
  ]
}
"""


RERANK_SYSTEM_PROMPT = """You are scoring memory items for a long-term memory question answering system.

Assign each candidate a relevance score from 0 to 100.
- Higher scores mean the item is more useful for answering the query.
- Favor directly supporting evidence.
- Keep useful bridge evidence for multi-hop and temporal questions.
- Penalize generic or weakly related items.

Return JSON only with this schema:
{
  "items": [
    {"uuid": "item uuid", "score": 87.5}
  ]
}
"""


ANSWER_SYSTEM_PROMPT = """You answer user questions using retrieved long-term memory context.

Rules:
- Use the provided memory context as the primary evidence source.
- If the context is insufficient, say that the answer is not fully supported by memory.
- Prefer precise, concise answers.
- Preserve temporal details when they matter.
"""


def build_category_details_user_prompt(layer: int, category_blocks: str) -> str:
    return f"""You are writing summaries and tags for Layer {layer} categories.

<CATEGORIES AND THEIR CHILDREN>
{category_blocks}
</CATEGORIES AND THEIR CHILDREN>
"""


def build_rerank_user_prompt(query: str, item_type: str, candidates: str) -> str:
    return f"""User Query:
{query}

Candidate Type:
{item_type}

Candidates:
{candidates}
"""


def build_answer_user_prompt(query: str, context: str) -> str:
    return f"""User Query:
{query}

Retrieved Memory Context:
{context}
"""
