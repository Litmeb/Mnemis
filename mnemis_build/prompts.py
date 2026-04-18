ENTITY_NAME_EXTRACTION_PROMPT = """You are building the base graph for the Mnemis memory system.

Extract concrete, durable entities mentioned in the current episode and the recent episode context.

Rules:
- Include people, organizations, places, named events, named works, products, tools, pets, vehicles, courses, jobs, hobbies, and other specific noun phrases that are useful graph nodes.
- Prefer names and specific noun phrases over vague references.
- Include a common-noun concept only when it is central and persistent in the episode, such as a recurring hobby, job title, diagnosis, class, or project.
- Exclude standalone emotions, virtues, broad themes, generic qualities, and vague abstractions such as "empathy", "understanding", "courage", "support", or "mental health" unless they refer to a specific named program, organization, event, or diagnosis.
- Exclude details that are only adjectives, sentiments, or non-reusable descriptors.
- Do not output duplicates.
- Return JSON only with this schema:
{"names": ["entity 1", "entity 2"]}
"""

ENTITY_REFLECTION_PROMPT = """You are checking whether any important entities were missed.

Given the episode context and the previously extracted entity names, add only truly missing names.
- Do not repeat names already extracted.
- Keep the same entity granularity as the original extraction.
- Apply the same filtering rules: prefer durable concrete nodes and skip vague abstractions unless they clearly denote a specific recurring entity.
- Return JSON only with this schema:
{"names": ["missing entity 1", "missing entity 2"]}
"""

ENTITY_DETAILS_PROMPT = """You are enriching de-duplicated entities for the Mnemis base graph.

For each entity:
- Keep the provided name unchanged.
- Write exactly one concise sentence grounded in the current and recent episodes.
- Keep each summary under 30 words.
- Provide 1 to 3 short tags, each at most 3 words.
- Do not add entities that were not provided.

Return JSON only with this schema:
{
  "entities": [
    {
      "name": "entity name",
      "summary": "brief context summary",
      "tag": ["tag1", "tag2"]
    }
  ]
}
"""

EDGE_EXTRACTION_PROMPT = """You are extracting graph edges for the Mnemis base graph.

Each edge must be a verifiable statement about a meaningful relationship, action, or state between two entities.

Rules:
- Use only the provided entity names.
- Return at most 6 edges.
- Keep facts concise and atomic, with each fact under 20 words.
- Skip unsupported or speculative facts.
- Use timestamps only when grounded in the context.
- Do not emit duplicate edges or paraphrases of the same edge.
- Prefer the most important, durable relationships over exhaustive coverage.
- Return JSON only with this schema:
{
  "edges": [
    {
      "source_entity_name": "entity A",
      "target_entity_name": "entity B",
      "fact": "verifiable fact",
      "valid_at": "2023-05-01T00:00:00" or null
    }
  ]
}
"""

EDGE_REFLECTION_PROMPT = """You are checking whether any meaningful graph edges were missed.

The input JSON contains reference data such as available entity descriptions, episode context, and existing edges.

Add only missing edges that are supported by the episode context and the provided entity list.
Do not repeat existing edges.
Do not echo or rewrite the input.
Do not return `group_id`, `context`, `available_entities`, or `existing_edges`.
Return a JSON object with exactly one top-level key, `edges`, using the same edge-item schema as the original edge extraction.
If no additional edges are needed, return `{"edges": []}`.
"""

HIERARCHICAL_SYSTEM_PROMPT = """You are an AI assistant specialized in semantic categorization of nodes.

You are given a list of node names, each prefixed with an index, and each followed with a brief description.

# INSTRUCTIONS:

1. Group the nodes into semantically meaningful categories based on shared attributes, considering both
the inherent characteristics of the node names and the DESCRIPTIONS of the nodes, not relying solely on
the DESCRIPTIONS.

All EXISTING CATEGORIES are provided for you.
- If a node's attributes match an existing category, add it under that category.
- If a node does not fit any existing category, create a new category and add it.
- The category name MUST NOT include the word "and" as a connector.

Examples of INVALID categories:
- "Food and Drinks"
- "University and Courses"

Examples of VALID categories:
- "Food"
- "Drinks"
- "University"
- "Courses"

2. Output category assignments using only node indexes. Do not repeat node names.

3. A node CAN be assigned to MULTIPLE categories at the same time.
- Use multiple category assignments only when the node clearly belongs in more than one strong semantic group.
- When multiple categories are formed for a node, select the minimal subset of features common across
the grouped nodes.

4. Prefer high-quality semantic groupings over exhaustive coverage.
- Do not create weak or forced categories just to cover every leftover node.
- Avoid singleton categories unless they are strongly justified.

Return JSON only.

Preferred output format:
[
  {"category": "xx", "indexes": [0, 1, 2, 4]},
  {"category": "xxx", "indexes": [2, 3, 4]}
]

If your serving stack requires a top-level JSON object, this wrapper is also acceptable:
{
  "assignments": [
    {"category": "xx", "indexes": [0, 1, 2, 4]},
    {"category": "xxx", "indexes": [2, 3, 4]}
  ]
}
"""


def build_hierarchy_user_prompt(
    layer: int,
    content: str,
    existing_categories: str,
    prev_example: str,
    speaker_policy_note: str | None = None,
    batch_note: str | None = None,
) -> str:
    speaker_policy = ""
    if speaker_policy_note:
        speaker_policy = f"""
<SPEAKER POLICY>
{speaker_policy_note}
</SPEAKER POLICY>

"""
    batch_policy = ""
    if batch_note:
        batch_policy = f"""
<BATCH POLICY>
{batch_note}
</BATCH POLICY>

"""
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

Key points:
- Categories may belong to multiple parent categories.
- Do not merge categories that are too loosely related.
- Prefer the minimal shared semantic feature that cleanly explains a grouping.

Your job at Layer {layer}:
- Merge semantically similar categories from Layer {layer - 1}.
- Each new category should reflect a shared attribute, domain, or higher-level concept.
- Multiple category assignments are allowed when justified.

Previous Layer {layer - 1} categories example:
{prev_example}
</GUIDANCE ON CATEGORY GRANULARITY>

{speaker_policy}{batch_policy}# ATTENTION
- Cover the strongest semantic groups in this batch. Uncovered leftovers can be promoted automatically later.
- The category name MUST NOT include the word "and".
- Prefer fewer, stronger categories over many thin categories.

Please follow the INSTRUCTIONS and GUIDANCE carefully to ensure accurate categorization and meaningful hierarchical relationships.
DO NOT INCLUDE ANY INVALID CATEGORIES.
Return JSON only. Prefer the top-level list format unless your API requires an object wrapper.
"""


CATEGORY_DETAILS_PROMPT = """You are enriching category nodes for the Mnemis hierarchical graph.

For each category:
- Keep the provided category name unchanged.
- Write a concise but informative summary grounded in the child members.
- Keep the summary between 12 and 16 words when possible.
- Provide at most 2 tags, each at most 3 words.
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


def build_category_details_user_prompt(layer: int, category_blocks: str, batch_note: str | None = None) -> str:
    batch_section = ""
    if batch_note:
        batch_section = f"""

<BATCH POLICY>
{batch_note}
</BATCH POLICY>"""
    return f"""You are writing summaries and tags for Layer {layer} categories.{batch_section}

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
