import json
from pathlib import Path

from mnemis_build.cli import build_parser
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
