from mnemis_build.llm import OpenAILLMClient
from mnemis_build.models import CategoryAssignmentPayload


def test_category_assignments_accept_top_level_list() -> None:
    payload = CategoryAssignmentPayload.model_validate(
        [
            {"category": "Research Labs", "indexes": [0, 1]},
            {"category": "AI Organizations", "indexes": [1, 2]},
        ]
    )

    assert [assignment.category for assignment in payload.assignments] == [
        "Research Labs",
        "AI Organizations",
    ]
    assert payload.assignments[1].indexes == [1, 2]


def test_category_assignments_accept_wrapper_object() -> None:
    payload = CategoryAssignmentPayload.model_validate(
        {
            "assignments": [
                {"category": "Research Labs", "indexes": [0, 1]},
                {"category": "AI Organizations", "indexes": [1, 2]},
            ]
        }
    )

    assert len(payload.assignments) == 2
    assert payload.assignments[0].indexes == [0, 1]


def test_llm_json_parser_accepts_embedded_array_or_object() -> None:
    llm = object.__new__(OpenAILLMClient)

    wrapped = llm._parse_json_content(
        '```json\n{"assignments":[{"category":"Research Labs","indexes":[0,1]}]}\n```'
    )
    listed = llm._parse_json_content(
        '```json\n[{"category":"Research Labs","indexes":[0,1]}]\n```'
    )

    assert isinstance(wrapped, dict)
    assert isinstance(listed, list)
