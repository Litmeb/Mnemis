from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .models import EpisodeInput


def _parse_locomo_datetime(raw_dt: str | None) -> datetime:
    if not raw_dt:
        return datetime.utcnow()

    normalized = " ".join(raw_dt.strip().replace(".", "").split())
    formats = (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
        "%Y-%m-%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported LoCoMo datetime format: {raw_dt}")


def load_locomo_episodes(file_path: str | Path, *, user_index: int, group_id: str) -> list[EpisodeInput]:
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    user = data[user_index]
    conversation = user["conversation"]
    episodes: list[EpisodeInput] = []
    for key in sorted(k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")):
        session_no = key.split("_")[1]
        raw_dt = conversation.get(f"session_{session_no}_date_time")
        valid_at = _parse_locomo_datetime(raw_dt)
        turns = conversation[key]
        for turn_index, turn in enumerate(turns):
            content = turn.get("text", "").strip()
            if not content:
                continue
            episodes.append(
                EpisodeInput(
                    speaker=turn.get("speaker", "unknown"),
                    content=content,
                    valid_at=valid_at,
                    source_id=f"{group_id}:{turn.get('dia_id', key)}",
                    metadata={
                        "query": turn.get("query"),
                        "blip_caption": turn.get("blip_caption"),
                        "img_url": turn.get("img_url"),
                        "session_id": key,
                        "turn_index": turn_index,
                    },
                )
            )
    return episodes


def count_locomo_users(file_path: str | Path) -> int:
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    return len(data)
