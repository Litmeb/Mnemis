import json
from pathlib import Path

from evaluate_locomo import _load_eval_questions


def test_load_eval_questions_reads_selected_users(tmp_path: Path) -> None:
    data_path = tmp_path / "locomo.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "qa": [
                        {
                            "question": "When did Alice travel?",
                            "answer": "May 2023",
                            "evidence": ["D1:1"],
                            "category": 2,
                        }
                    ]
                },
                {
                    "qa": [
                        {
                            "question": "Where does Bob live?",
                            "answer": "Seattle",
                            "evidence": ["D2:4"],
                            "category": 1,
                        }
                    ]
                },
            ]
        ),
        encoding="utf-8",
    )

    questions = _load_eval_questions(
        data_path,
        group_id_prefix="locomo_user",
        selected_user_indexes=[1],
    )

    assert len(questions) == 1
    assert questions[0].user_index == 1
    assert questions[0].group_id == "locomo_user_1"
    assert questions[0].question == "Where does Bob live?"
    assert questions[0].reference_answer == "Seattle"
