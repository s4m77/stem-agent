from __future__ import annotations

from stemds.agents.baseline import DummyBaselineAgent, extract_python_code
from stemds.metrics import compare_answers
from stemds.sandbox import PythonSandbox
from stemds.tasks import DataAnalysisTask


def test_dummy_baseline_solves_known_task() -> None:
    task = DataAnalysisTask(
        task_id="sales_004",
        dataset_path="data/toy_csvs/sales.csv",
        question="Which region has the lowest total revenue?",
        answer="South",
        answer_type="string",
        tags=["groupby"],
    )

    output = DummyBaselineAgent().solve(task)
    result = PythonSandbox(timeout_sec=15).run(output.code, task.dataset_path)

    assert result.status == "success"
    assert compare_answers(task.answer, result.extracted_answer, task.answer_type)


def test_extract_python_code_from_fenced_response() -> None:
    response = "```python\nprint('FINAL_ANSWER: West')\n```"

    assert extract_python_code(response) == "print('FINAL_ANSWER: West')\n"
