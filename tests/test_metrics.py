from __future__ import annotations

from stemds.metrics import TaskEvalResult, aggregate_metrics, compare_answers


def test_compare_answers_by_type() -> None:
    assert compare_answers("West", " west ", "string")
    assert compare_answers(True, "yes", "boolean")
    assert compare_answers(25.0, "25%", "number", tolerance=0.01)
    assert not compare_answers(25.0, "24.8", "number", tolerance=0.01)


def test_aggregate_metrics() -> None:
    results = [
        TaskEvalResult("a", 1, "1", True, False, "success", 0, 0.1),
        TaskEvalResult("b", 2, None, False, True, "runtime_error", 1, 0.3),
    ]

    metrics = aggregate_metrics(results)

    assert metrics["total_tasks"] == 2
    assert metrics["answer_accuracy"] == 0.5
    assert metrics["execution_success_rate"] == 0.5
    assert metrics["invalid_code_rate"] == 0.5

