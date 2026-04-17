from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned.strip("_") or "run"


@dataclass(slots=True)
class InstrumentationEvent:
    event_type: str
    stage: str
    operation: str
    runtime_seconds: float
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_utc_now_iso)


class InstrumentationRecorder:
    def __init__(self, run_name: str):
        self.run_name = run_name
        self.created_at = _utc_now_iso()
        self._events: list[InstrumentationEvent] = []

    @property
    def events(self) -> list[InstrumentationEvent]:
        return list(self._events)

    def record_llm_call(
        self,
        *,
        stage: str,
        operation: str,
        runtime_seconds: float,
        model: str | None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        total = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        self._events.append(
            InstrumentationEvent(
                event_type="llm_call",
                stage=stage,
                operation=operation,
                runtime_seconds=runtime_seconds,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total,
                metadata=metadata or {},
            )
        )

    def record_stage_runtime(
        self,
        *,
        stage: str,
        operation: str,
        runtime_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._events.append(
            InstrumentationEvent(
                event_type="stage_runtime",
                stage=stage,
                operation=operation,
                runtime_seconds=runtime_seconds,
                metadata=metadata or {},
            )
        )

    @contextmanager
    def stage_timer(
        self,
        stage: str,
        operation: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.record_stage_runtime(
                stage=stage,
                operation=operation,
                runtime_seconds=perf_counter() - start,
                metadata=metadata,
            )

    def build_report(self) -> dict[str, Any]:
        stage_map: dict[tuple[str, str], dict[str, Any]] = {}
        for event in self._events:
            key = (event.stage, event.operation)
            summary = stage_map.setdefault(
                key,
                {
                    "stage": event.stage,
                    "operation": event.operation,
                    "wall_clock_runtime_seconds": 0.0,
                    "llm_runtime_seconds": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "models": [],
                    "llm_call_count": 0,
                    "stage_runtime_count": 0,
                },
            )
            if event.event_type == "stage_runtime":
                summary["wall_clock_runtime_seconds"] += event.runtime_seconds
                summary["stage_runtime_count"] += 1
            elif event.event_type == "llm_call":
                summary["llm_runtime_seconds"] += event.runtime_seconds
                summary["prompt_tokens"] += event.prompt_tokens
                summary["completion_tokens"] += event.completion_tokens
                summary["total_tokens"] += event.total_tokens
                summary["llm_call_count"] += 1
                if event.model and event.model not in summary["models"]:
                    summary["models"].append(event.model)
                if summary["wall_clock_runtime_seconds"] <= 0:
                    summary["wall_clock_runtime_seconds"] += event.runtime_seconds

        summaries = sorted(
            stage_map.values(),
            key=lambda item: (item["stage"], item["operation"]),
        )
        return {
            "run_name": self.run_name,
            "created_at": self.created_at,
            "generated_at": _utc_now_iso(),
            "stage_summaries": summaries,
            "events": [asdict(event) for event in self._events],
        }

    def write_reports(self, output_dir: str | Path, *, stem: str | None = None) -> dict[str, str]:
        report = self.build_report()
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        file_stem = _safe_slug(stem or self.run_name)

        json_path = base_dir / f"{file_stem}.json"
        summary_csv_path = base_dir / f"{file_stem}_summary.csv"
        events_csv_path = base_dir / f"{file_stem}_events.csv"

        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "stage",
                    "operation",
                    "wall_clock_runtime_seconds",
                    "llm_runtime_seconds",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "llm_call_count",
                    "stage_runtime_count",
                    "models",
                ],
            )
            writer.writeheader()
            for row in report["stage_summaries"]:
                writer.writerow({**row, "models": "|".join(row["models"])})

        with events_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "event_type",
                    "stage",
                    "operation",
                    "runtime_seconds",
                    "model",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "started_at",
                    "metadata",
                ],
            )
            writer.writeheader()
            for row in report["events"]:
                writer.writerow({**row, "metadata": json.dumps(row["metadata"], ensure_ascii=False, sort_keys=True)})

        return {
            "json": str(json_path),
            "summary_csv": str(summary_csv_path),
            "events_csv": str(events_csv_path),
        }
