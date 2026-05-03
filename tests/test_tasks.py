from __future__ import annotations

from pathlib import Path

from stemds.tasks import DataAnalysisTask, load_tasks_jsonl


def test_data_analysis_task_validates_answer_type() -> None:
    task = DataAnalysisTask(
        task_id="example",
        dataset_path="data/toy_csvs/sales.csv",
        question="How many rows?",
        answer=12,
        answer_type="number",
        tags=["count"],
    )

    assert task.domain == "data_analysis"


def test_load_tasks_jsonl() -> None:
    tasks = load_tasks_jsonl(Path("data/toy_analysis_train.jsonl"))

    assert len(tasks) == 10
    assert tasks[0].task_id == "sales_001"

