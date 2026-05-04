from __future__ import annotations

from stemds.metrics import TaskEvalResult, aggregate_metrics, compare_answers, parse_dabench_pairs


def test_compare_answers_by_type() -> None:
    assert compare_answers("West", " west ", "string")
    assert compare_answers(True, "yes", "boolean")
    assert compare_answers(25.0, "25%", "number", tolerance=0.01)
    assert not compare_answers(25.0, "24.8", "number", tolerance=0.01)


def test_parse_dabench_pairs() -> None:
    assert parse_dabench_pairs("@mean_acceleration[15.49]\n@std_acceleration[2.68]") == {
        "mean_acceleration": "15.49",
        "std_acceleration": "2.68",
    }


def test_compare_dabench_pairs_order_insensitive() -> None:
    expected = "@std_acceleration[2.68]\n@mean_acceleration[15.49]"
    predicted = "@mean_acceleration[15.49]\n@std_acceleration[2.68]"

    assert compare_answers(expected, predicted, "string", metadata={"raw_answer_type": "multi"})


def test_compare_dabench_pairs_missing_pair_is_incorrect() -> None:
    expected = "@std_acceleration[2.68]\n@mean_acceleration[15.49]"
    predicted = "@mean_acceleration[15.49]"

    assert not compare_answers(expected, predicted, "string", metadata={"raw_answer_type": "multi"})


def test_compare_dabench_pairs_numeric_tolerance() -> None:
    expected = "@std_acceleration[2.680000]\n@mean_acceleration[15.490000]"
    within = "@mean_acceleration[15.4900004]\n@std_acceleration[2.6800004]"
    beyond = "@mean_acceleration[15.4902]\n@std_acceleration[2.680000]"

    assert compare_answers(expected, within, "string", tolerance=1e-6, metadata={"raw_answer_type": "multi"})
    assert not compare_answers(expected, beyond, "string", tolerance=1e-6, metadata={"raw_answer_type": "multi"})


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


def test_aggregate_metrics_includes_subquestion_accuracy() -> None:
    results = [
        TaskEvalResult(
            "a",
            "@x[1]\n@y[2]",
            "@x[1]\n@y[3]",
            False,
            False,
            "pass",
            1,
            0.1,
            metadata={"subquestion_total": 2, "subquestion_correct": 1},
        )
    ]

    metrics = aggregate_metrics(results)

    assert metrics["subquestion_total"] == 2
    assert metrics["subquestion_correct"] == 1
    assert metrics["subquestion_accuracy"] == 0.5
