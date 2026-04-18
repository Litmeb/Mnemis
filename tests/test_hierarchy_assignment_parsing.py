import asyncio
from types import SimpleNamespace

from mnemis_build.llm import OpenAILLMClient
from mnemis_build.logging_utils import get_logger
from mnemis_build.models import CategoryAssignmentPayload, NodeSelectionList


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


def test_llm_assignment_parser_accepts_list_or_wrapper_end_to_end() -> None:
    llm = object.__new__(OpenAILLMClient)

    wrapped = llm.parse_json_response(
        CategoryAssignmentPayload,
        '```json\n{"assignments":[{"category":"Research Labs","indexes":[0,1]}]}\n```',
    )
    listed = llm.parse_json_response(
        CategoryAssignmentPayload,
        '```json\n[{"category":"Research Labs","indexes":[0,1]}]\n```',
    )

    assert wrapped.assignments[0].category == "Research Labs"
    assert listed.assignments[0].category == "Research Labs"
    assert wrapped.assignments[0].indexes == listed.assignments[0].indexes == [0, 1]


def test_complete_json_retries_after_truncated_response() -> None:
    llm = object.__new__(OpenAILLMClient)
    llm.config = SimpleNamespace(llm_model="test-model", small_llm_model="test-model")
    llm.recorder = None
    llm.logger = get_logger("test.llm")

    responses = iter(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"assignments":[{"category":"Research Labs"'),
                        finish_reason="length",
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"assignments":[{"category":"Research Labs","indexes":[0,1]}]}'
                        ),
                        finish_reason="stop",
                    )
                ]
            ),
        ]
    )

    async def create(**_: object):
        return next(responses)

    llm.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )

    payload = asyncio.run(
        llm.complete_json(
            CategoryAssignmentPayload,
            [{"role": "user", "content": "Return JSON."}],
        )
    )

    assert payload.assignments[0].category == "Research Labs"
    assert payload.assignments[0].indexes == [0, 1]


def test_complete_json_surfaces_finish_reason_after_retry_exhaustion() -> None:
    llm = object.__new__(OpenAILLMClient)
    llm.config = SimpleNamespace(llm_model="test-model", small_llm_model="test-model")
    llm.recorder = None
    llm.logger = get_logger("test.llm")

    async def create(**_: object):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"assignments":[{"category":"Research Labs"'),
                    finish_reason="length",
                )
            ]
        )

    llm.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )

    try:
        asyncio.run(
            llm.complete_json(
                CategoryAssignmentPayload,
                [{"role": "user", "content": "Return JSON."}],
            )
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected complete_json to raise ValueError")

    assert "finish_reason='length'" in message
    assert "Response prefix" in message


def test_node_selection_parser_accepts_top_level_list_or_object() -> None:
    payload_from_list = NodeSelectionList.model_validate(
        [
            {
                "name": "Relationships",
                "uuid": "cat_relationships",
                "get_all_children": True,
            }
        ]
    )
    payload_from_object = NodeSelectionList.model_validate(
        {
            "selections": [
                {
                    "name": "Relationships",
                    "uuid": "cat_relationships",
                    "get_all_children": False,
                }
            ]
        }
    )

    assert payload_from_list.selections[0].uuid == "cat_relationships"
    assert payload_from_object.selections[0].name == "Relationships"


def test_complete_json_retries_after_natural_language_response() -> None:
    llm = object.__new__(OpenAILLMClient)
    llm.config = SimpleNamespace(llm_model="test-model", small_llm_model="test-model")
    llm.recorder = None
    llm.logger = get_logger("test.llm")

    responses = iter(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="I would select the Relationships node."),
                        finish_reason="stop",
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"selections":[{"name":"Relationships","uuid":"cat_relationships","get_all_children":true}]}'
                        ),
                        finish_reason="stop",
                    )
                ]
            ),
        ]
    )

    async def create(**_: object):
        return next(responses)

    llm.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )

    payload = asyncio.run(
        llm.complete_json(
            NodeSelectionList,
            [{"role": "user", "content": "Return strict JSON selections only."}],
        )
    )

    assert payload.selections[0].name == "Relationships"
