from pathlib import Path

from mnemis_build.instrumentation import InstrumentationRecorder


def test_instrumentation_recorder_builds_stage_summary_and_writes_reports(tmp_path: Path) -> None:
    recorder = InstrumentationRecorder("unit_test_run")
    recorder.record_llm_call(
        stage="base_graph_ingestion",
        operation="entity_name_extraction",
        runtime_seconds=1.25,
        model="gpt-4.1-mini",
        prompt_tokens=100,
        completion_tokens=25,
    )
    recorder.record_stage_runtime(
        stage="base_graph_ingestion",
        operation="build",
        runtime_seconds=3.5,
    )

    report = recorder.build_report()

    assert report["run_name"] == "unit_test_run"
    by_operation = {
        (row["stage"], row["operation"]): row
        for row in report["stage_summaries"]
    }
    assert by_operation[("base_graph_ingestion", "entity_name_extraction")]["total_tokens"] == 125
    assert by_operation[("base_graph_ingestion", "build")]["wall_clock_runtime_seconds"] == 3.5

    paths = recorder.write_reports(tmp_path)
    assert Path(paths["json"]).exists()
    assert Path(paths["summary_csv"]).exists()
    assert Path(paths["events_csv"]).exists()
