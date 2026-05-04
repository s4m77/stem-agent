from __future__ import annotations

import json

from stemds.analysis.failures import FailureCategory, analyze_run, categorize_failure, compare_runs


def test_categorize_failure_execution_and_multi_answer() -> None:
    assert (
        categorize_failure({"correct": False, "sandbox_status": "runtime_error"})
        == FailureCategory.EXECUTION_ERROR.value
    )
    assert (
        categorize_failure(
            {
                "correct": False,
                "sandbox_status": "pass",
                "expected_answer": "@a[1]\n@b[2]",
                "predicted_answer": "@b[2]\n@a[1]",
            }
        )
        == FailureCategory.UNSUPPORTED_MULTI_ANSWER.value
    )


def test_analyze_run_handles_minimal_run_json(tmp_path) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "metrics": {"total_tasks": 2, "answer_accuracy": 0.5, "execution_success_rate": 0.5},
                "results": [
                    {"task_id": "a", "correct": True, "sandbox_status": "pass"},
                    {
                        "task_id": "b",
                        "correct": False,
                        "sandbox_status": "runtime_error",
                        "tags": ["summary_statistics"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = analyze_run(run_path)

    assert report.total_tasks == 2
    assert report.total_failures == 1
    assert report.failures_by_category[FailureCategory.EXECUTION_ERROR.value] == 1
    assert report.failures_by_tag["summary_statistics"] == 1


def test_compare_runs_reports_improved_regressed_and_skill_usage(tmp_path) -> None:
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(
        json.dumps(
            {
                "metrics": {"answer_accuracy": 0.5, "execution_success_rate": 1.0, "invalid_code_rate": 0.0, "composite_score": 0.5},
                "results": [
                    {"task_id": "a", "correct": False, "tags": ["summary_statistics"]},
                    {"task_id": "b", "correct": True, "tags": ["summary_statistics"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    b_path.write_text(
        json.dumps(
            {
                "metrics": {"answer_accuracy": 0.5, "execution_success_rate": 0.5, "invalid_code_rate": 0.5, "composite_score": 0.35},
                "results": [
                    {
                        "task_id": "a",
                        "correct": True,
                        "tags": ["summary_statistics"],
                        "metadata": {"selected_skill_ids": ["summary_statistics_skill"]},
                    },
                    {
                        "task_id": "b",
                        "correct": False,
                        "tags": ["summary_statistics"],
                        "metadata": {"selected_skill_ids": ["summary_statistics_skill"]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    comparison = compare_runs(a_path, b_path)

    assert comparison["deltas"]["answer_accuracy"] == 0.0
    assert comparison["tasks_improved"] == ["a"]
    assert comparison["tasks_regressed"] == ["b"]
    assert comparison["selected_skill_usage_counts"]["summary_statistics_skill"] == 2

