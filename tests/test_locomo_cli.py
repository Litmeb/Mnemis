import json
from pathlib import Path

import pytest

from mnemis_build.cli import (
    _compute_resume_start_index,
    _load_resume_log_progress,
    _parse_user_index_list,
    _resolve_resume_start_index,
    _resolve_user_indexes,
    build_parser,
)
from mnemis_build.loaders import count_locomo_users


def test_count_locomo_users_reads_dataset_length(tmp_path: Path) -> None:
    data_path = tmp_path / "locomo.json"
    data_path.write_text(json.dumps([{"conversation": {}}, {"conversation": {}}, {"conversation": {}}]), encoding="utf-8")

    assert count_locomo_users(data_path) == 3


def test_build_parser_accepts_rebuild_locomo_all_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "rebuild-locomo-all",
            "--data",
            "data/locomo.json",
            "--group-id-prefix",
            "locomo_user",
            "--max-concurrent-users",
            "6",
        ]
    )

    assert args.command == "rebuild-locomo-all"
    assert args.group_id_prefix == "locomo_user"
    assert args.max_concurrent_users == 6


def test_build_parser_accepts_rebuild_locomo_all_user_index_filter() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "rebuild-locomo-all",
            "--user-index",
            "1,3,7",
        ]
    )

    assert args.command == "rebuild-locomo-all"
    assert args.user_indexes == [1, 3, 7]


def test_build_parser_accepts_reuse_existing_base_for_rebuild_locomo() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "rebuild-locomo",
            "--group-id",
            "locomo_user_0",
            "--reuse-existing-base",
        ]
    )

    assert args.command == "rebuild-locomo"
    assert args.reuse_existing_base is True


def test_build_parser_accepts_reuse_existing_base_for_rebuild_locomo_all() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "rebuild-locomo-all",
            "--reuse-existing-base",
        ]
    )

    assert args.command == "rebuild-locomo-all"
    assert args.reuse_existing_base is True


def test_build_parser_accepts_resume_flags() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "rebuild-locomo",
            "--group-id",
            "locomo_user_0",
            "--resume",
            "--resume-log",
            "results/logs/20260417_175031/mnemis_build.log",
        ]
    )

    assert args.command == "rebuild-locomo"
    assert args.resume is True
    assert str(args.resume_log).endswith("mnemis_build.log")


def test_compute_resume_start_index_uses_contiguous_prefix() -> None:
    episodes = [
        type("Episode", (), {"source_id": "ep-1"})(),
        type("Episode", (), {"source_id": "ep-2"})(),
        type("Episode", (), {"source_id": "ep-3"})(),
    ]

    assert _compute_resume_start_index(episodes, {"ep-1", "ep-3"}) == 1


def test_compute_resume_start_index_can_fall_back_to_last_completed_source_id() -> None:
    episodes = [
        type("Episode", (), {"source_id": "ep-1"})(),
        type("Episode", (), {"source_id": "ep-2"})(),
        type("Episode", (), {"source_id": "ep-3"})(),
    ]

    assert _compute_resume_start_index(
        episodes,
        {"ep-3"},
        resume_completed_source_id="ep-3",
    ) == 3


def test_parse_user_index_list_dedupes_and_preserves_order() -> None:
    assert _parse_user_index_list("3,1,3,7") == [3, 1, 7]


def test_parse_user_index_list_rejects_invalid_items() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["rebuild-locomo-all", "--user-index", "1,two,3"])


def test_resolve_user_indexes_defaults_to_all_users() -> None:
    assert _resolve_user_indexes(None, total_users=3) == [0, 1, 2]


def test_resolve_user_indexes_raises_for_out_of_range_values() -> None:
    with pytest.raises(RuntimeError, match="out of range"):
        _resolve_user_indexes([1, 4], total_users=3)


def test_load_resume_log_progress_extracts_last_completed_source_id(tmp_path: Path) -> None:
    log_path = tmp_path / "mnemis_build.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-04-17 18:26:21.267 [INFO] mnemis_build.base_graph - turn done | group_id=locomo_user_0, turn=114/419, source_id=locomo_user_0:D1:114, entity_count=3",
                "2026-04-17 18:26:35.703 [INFO] mnemis_build.base_graph - turn done | group_id=locomo_user_2, turn=96/663, source_id=locomo_user_2:D13:18, entity_count=9",
                "2026-04-17 18:26:40.000 [INFO] mnemis_build.base_graph - turn done | group_id=locomo_user_0, turn=115/419, source_id=locomo_user_0:D1:115, entity_count=2",
            ]
        ),
        encoding="utf-8",
    )

    progress = _load_resume_log_progress(log_path)

    assert progress == {
        "locomo_user_0": "locomo_user_0:D1:115",
        "locomo_user_2": "locomo_user_2:D13:18",
    }


def test_resolve_resume_start_index_raises_without_resume_evidence() -> None:
    episodes = [
        type("Episode", (), {"source_id": "ep-1"})(),
        type("Episode", (), {"source_id": "ep-2"})(),
    ]

    with pytest.raises(RuntimeError, match="no completed-turn markers"):
        _resolve_resume_start_index(episodes, set(), group_id="locomo_user_0")


def test_resolve_resume_start_index_raises_when_resume_point_cannot_be_mapped() -> None:
    episodes = [
        type("Episode", (), {"source_id": "ep-1"})(),
        type("Episode", (), {"source_id": "ep-2"})(),
    ]

    with pytest.raises(RuntimeError, match="could not be mapped"):
        _resolve_resume_start_index(
            episodes,
            {"missing-ep"},
            group_id="locomo_user_0",
            resume_completed_source_id="missing-ep",
        )
