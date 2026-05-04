from __future__ import annotations

import json
from pathlib import Path

from stemds.cli import main
from stemds.datasets.dabench import DABenchAdapter
from stemds.tasks import load_tasks_jsonl

FIXTURE_ROOT = Path("tests/fixtures/fake_dabench")


def test_dabench_adapter_loads_fake_tasks() -> None:
    tasks = DABenchAdapter(FIXTURE_ROOT).load_tasks()

    assert len(tasks) == 2
    assert tasks[0].task_id == "dabench_1"
    assert Path(tasks[0].dataset_path).exists()
    assert tasks[0].answer == "West"
    assert tasks[0].answer_type == "string"
    assert tasks[0].metadata["original_id"] == 1
    assert "summary_statistics" in tasks[0].tags
    assert tasks[1].answer == 300.5
    assert tasks[1].answer_type == "number"
    assert tasks[1].tolerance == 1e-6


def test_convert_dataset_command_writes_jsonl(tmp_path) -> None:
    output_path = tmp_path / "dabench_tasks.jsonl"

    exit_code = main(
        [
            "convert-dataset",
            "--adapter",
            "dabench",
            "--root",
            str(FIXTURE_ROOT),
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    tasks = load_tasks_jsonl(output_path)
    assert len(tasks) == 2
    assert Path(tasks[0].dataset_path).exists()


def test_split_data_command_is_deterministic(tmp_path) -> None:
    data_path = tmp_path / "dabench_tasks.jsonl"
    DABenchAdapter(FIXTURE_ROOT).write_jsonl(DABenchAdapter(FIXTURE_ROOT).load_tasks(), data_path)

    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    args = [
        "split-data",
        "--data",
        str(data_path),
        "--train-frac",
        "0.5",
        "--val-frac",
        "0.25",
        "--seed",
        "42",
    ]

    assert main([*args, "--out-dir", str(first_out)]) == 0
    assert main([*args, "--out-dir", str(second_out)]) == 0

    for split_name in ["train", "val", "test"]:
        first_file = first_out / f"dabench_{split_name}.jsonl"
        second_file = second_out / f"dabench_{split_name}.jsonl"
        assert first_file.exists()
        assert second_file.exists()
        assert first_file.read_text(encoding="utf-8") == second_file.read_text(encoding="utf-8")


def test_sample_data_command_prints_jsonl(tmp_path, capsys) -> None:
    data_path = tmp_path / "dabench_tasks.jsonl"
    DABenchAdapter(FIXTURE_ROOT).write_jsonl(DABenchAdapter(FIXTURE_ROOT).load_tasks(), data_path)

    exit_code = main(["sample-data", "--data", str(data_path), "--n", "1"])

    assert exit_code == 0
    captured = capsys.readouterr()
    sample = json.loads(captured.out)
    assert sample["task_id"] == "dabench_1"
