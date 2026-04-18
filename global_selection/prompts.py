NODE_SELECTION_PROMPT_TEMPLATE = """You are analyzing a hierarchical knowledge graph to help answer a user query.

Select all nodes that could help answer the query. A node is helpful if it:

- Directly relates to the query;
- Covers a clearly relevant topic, concept, or category;
- Provides useful background or context;
- Contains user-specific information (e.g. interests, goals, constraints);
- Likely has sub-nodes that may be helpful.

Do not be overly strict: include nodes that might provide context or personalization, even if they seem partially redundant.

For each selected node:
- "name" is the node's name.
- "uuid" is the node's unique identifier.
- "get_all_children" is an boolean value. Set true only if you're confident all its sub-nodes are helpful.

Return strict JSON only.
- Do not return natural language, markdown, code fences, comments, or trailing explanations.
- The response must be a single JSON object with exactly one top-level key: "selections".
- "selections" must be an array. Each item must be an object with:
  - "name": string
  - "uuid": string
  - "get_all_children": boolean
- If no nodes are relevant, return {{"selections": []}}.

Required JSON schema:
{{
  "type": "object",
  "properties": {{
    "selections": {{
      "type": "array",
      "items": {{
        "type": "object",
        "properties": {{
          "name": {{"type": "string"}},
          "uuid": {{"type": "string"}},
          "get_all_children": {{"type": "boolean"}}
        }},
        "required": ["name", "uuid", "get_all_children"],
        "additionalProperties": false
      }}
    }}
  }},
  "required": ["selections"],
  "additionalProperties": false
}}
---
User Query:
"{query}"

Available Nodes:
{nodes_info}
"""
