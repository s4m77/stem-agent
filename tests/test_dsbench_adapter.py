from __future__ import annotations

from pathlib import Path

from stemds.cli import main
from stemds.datasets.dsbench import DSBenchAdapter, inspect_dsbench
from stemds.tasks import load_tasks_jsonl

FIXTURE_ROOT = Path("tests/fixtures/fake_dsbench")


def test_inspect_dsbench_handles_fake_fixture() -> None:
    inspection = inspect_dsbench(FIXTURE_ROOT)

    assert inspection.exists
    assert inspection.data_analysis_records == 1
    assert inspection.data_modeling_records == 1
    assert inspection.simple_subset_records == 1
    assert inspection.conversion_supported
    assert "simple_tasks.jsonl" in inspection.metadata_files


def test_dsbench_inspection_renders_markdown() -> None:
    markdown = inspect_dsbench(FIXTURE_ROOT).to_markdown()

    assert "# DSBench Inspection" in markdown
    assert "Simple convertible" in markdown
    assert "sales_total" in markdown


def test_dsbench_adapter_converts_simple_tabular_fixture() -> None:
    tasks, skipped = DSBenchAdapter(FIXTURE_ROOT).convert_simple_tabular_subset()

    assert skipped == []
    assert len(tasks) == 1
    assert tasks[0].task_id == "dsbench_sales_total"
    assert tasks[0].answer == 30
    assert tasks[0].answer_type == "number"
    assert Path(tasks[0].dataset_path).exists()


def test_dsbench_adapter_handles_unsupported_layout_gracefully(tmp_path) -> None:
    tasks, skipped = DSBenchAdapter(tmp_path).convert_simple_tabular_subset()

    assert tasks == []
    assert skipped
    assert "No simple tabular metadata" in skipped[0]


def test_inspect_dsbench_cli_writes_report(tmp_path) -> None:
    report_path = tmp_path / "dsbench_inspection.md"

    exit_code = main(["inspect-dsbench", "--root", str(FIXTURE_ROOT), "--out", str(report_path)])

    assert exit_code == 0
    assert report_path.exists()
    assert "DSBench Inspection" in report_path.read_text(encoding="utf-8")


def test_convert_dsbench_subset_cli_writes_jsonl(tmp_path) -> None:
    output_path = tmp_path / "dsbench_subset.jsonl"

    exit_code = main(
        [
            "convert-dsbench-subset",
            "--root",
            str(FIXTURE_ROOT),
            "--out",
            str(output_path),
            "--limit",
            "10",
        ]
    )

    assert exit_code == 0
    tasks = load_tasks_jsonl(output_path)
    assert len(tasks) == 1
    assert tasks[0].task_id == "dsbench_sales_total"
