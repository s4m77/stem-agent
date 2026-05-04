"""Failure analysis and run comparison helpers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class FailureCategory(StrEnum):
    EXECUTION_ERROR = "execution_error"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    MISSING_ANSWER_VARIABLE = "missing_answer_variable"
    ANSWER_FORMAT_MISMATCH = "answer_format_mismatch"
    NUMERIC_TOLERANCE_ISSUE = "numeric_tolerance_issue"
    WRONG_CALCULATION = "wrong_calculation"
    UNSUPPORTED_MULTI_ANSWER = "unsupported_multi_answer"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class FailureRecord:
    task_id: str
    tags: list[str]
    question: str | None
    expected_answer: Any | None
    predicted_answer: Any | None
    status: str
    correct: bool
    error_message: str | None
    stdout: str
    stderr: str
    generated_code: str
    category: str
    llm_calls: int
    duration_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FailureAnalysisReport:
    total_tasks: int
    total_failures: int
    failures_by_category: dict[str, int]
    failures_by_tag: dict[str, int]
    execution_success_rate: float
    accuracy: float
    top_examples_by_category: dict[str, list[dict[str, Any]]]
    recommendations: list[str]
    failures: list[FailureRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failures"] = [failure.to_dict() for failure in self.failures]
        return payload


def categorize_failure(result_or_trace: dict[str, Any]) -> str:
    if bool(result_or_trace.get("correct")):
        return FailureCategory.UNKNOWN.value

    status = str(result_or_trace.get("sandbox_status") or result_or_trace.get("status") or "").lower()
    expected = result_or_trace.get("expected_answer")
    predicted = result_or_trace.get("predicted_answer")
    stderr = str(result_or_trace.get("stderr") or "")
    error_message = str(result_or_trace.get("error_message") or "")
    combined_error = f"{stderr}\n{error_message}".lower()

    if status == "syntax_error":
        return FailureCategory.SYNTAX_ERROR.value
    if status == "timeout":
        return FailureCategory.TIMEOUT.value
    if "did not assign answer" in combined_error or "name 'answer'" in combined_error:
        return FailureCategory.MISSING_ANSWER_VARIABLE.value
    if status in {"runtime_error", "unsafe_code"}:
        return FailureCategory.EXECUTION_ERROR.value
    if status in {"success", "pass"} and predicted is None:
        return FailureCategory.MISSING_ANSWER_VARIABLE.value
    if _is_multi_answer(expected):
        return FailureCategory.UNSUPPORTED_MULTI_ANSWER.value
    if _is_numeric_like(expected) and _is_numeric_like(predicted):
        expected_float = _to_float(expected)
        predicted_float = _to_float(predicted)
        if expected_float is not None and predicted_float is not None:
            diff = abs(expected_float - predicted_float)
            if diff <= max(0.01, abs(expected_float) * 0.001):
                return FailureCategory.NUMERIC_TOLERANCE_ISSUE.value
        return FailureCategory.WRONG_CALCULATION.value
    if _looks_like_format_mismatch(expected, predicted):
        return FailureCategory.ANSWER_FORMAT_MISMATCH.value
    if status in {"success", "pass"}:
        return FailureCategory.WRONG_CALCULATION.value
    return FailureCategory.UNKNOWN.value


def analyze_run(results_path: Path) -> FailureAnalysisReport:
    payload = _read_json(results_path)
    results = list(payload.get("results", []))
    metrics = payload.get("metrics", {})
    failures = [_failure_record(result) for result in results if not bool(result.get("correct"))]
    failures_by_category = Counter(failure.category for failure in failures)
    failures_by_tag: Counter[str] = Counter()
    for failure in failures:
        failures_by_tag.update(failure.tags)

    top_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for failure in failures:
        if len(top_examples[failure.category]) < 3:
            top_examples[failure.category].append(
                {
                    "task_id": failure.task_id,
                    "status": failure.status,
                    "expected_answer": failure.expected_answer,
                    "predicted_answer": failure.predicted_answer,
                    "error_message": failure.error_message,
                }
            )

    total_tasks = int(metrics.get("total_tasks") or len(results))
    accuracy = float(metrics.get("answer_accuracy") or _rate(result.get("correct") for result in results))
    execution_success_rate = float(
        metrics.get("execution_success_rate")
        or _rate((result.get("sandbox_status") in {"success", "pass"}) for result in results)
    )
    return FailureAnalysisReport(
        total_tasks=total_tasks,
        total_failures=len(failures),
        failures_by_category=dict(sorted(failures_by_category.items())),
        failures_by_tag=dict(sorted(failures_by_tag.items())),
        execution_success_rate=execution_success_rate,
        accuracy=accuracy,
        top_examples_by_category=dict(top_examples),
        recommendations=_recommendations(failures_by_category, failures_by_tag),
        failures=failures,
    )


def render_failure_analysis_markdown(report: FailureAnalysisReport) -> str:
    lines = [
        "# Failure Analysis",
        "",
        f"- total_tasks: {report.total_tasks}",
        f"- total_failures: {report.total_failures}",
        f"- accuracy: {report.accuracy:.3f}",
        f"- execution_success_rate: {report.execution_success_rate:.3f}",
        "",
        "## Failures By Category",
    ]
    lines.extend(f"- {category}: {count}" for category, count in report.failures_by_category.items())
    lines.append("")
    lines.append("## Failures By Tag")
    lines.extend(f"- {tag}: {count}" for tag, count in report.failures_by_tag.items())
    lines.append("")
    lines.append("## Recommendations")
    lines.extend(f"- {recommendation}" for recommendation in report.recommendations)
    return "\n".join(lines).strip() + "\n"


def compare_runs(a_path: Path, b_path: Path) -> dict[str, Any]:
    a_payload = _read_json(a_path)
    b_payload = _read_json(b_path)
    a_metrics = dict(a_payload.get("metrics", {}))
    b_metrics = dict(b_payload.get("metrics", {}))
    a_results = {result.get("task_id"): result for result in a_payload.get("results", [])}
    b_results = {result.get("task_id"): result for result in b_payload.get("results", [])}
    common_ids = sorted(task_id for task_id in a_results if task_id in b_results)

    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []
    for task_id in common_ids:
        a_correct = bool(a_results[task_id].get("correct"))
        b_correct = bool(b_results[task_id].get("correct"))
        if not a_correct and b_correct:
            improved.append(task_id)
        elif a_correct and not b_correct:
            regressed.append(task_id)
        else:
            unchanged.append(task_id)

    return {
        "a": str(a_path),
        "b": str(b_path),
        "metrics_a": a_metrics,
        "metrics_b": b_metrics,
        "deltas": _metric_deltas(a_metrics, b_metrics),
        "per_tag_deltas": _per_tag_deltas(a_payload.get("results", []), b_payload.get("results", [])),
        "tasks_improved": improved,
        "tasks_regressed": regressed,
        "tasks_unchanged": unchanged,
        "selected_skill_usage_counts": _selected_skill_usage_counts(b_payload.get("results", [])),
    }


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = ["# Run Comparison", "", "## Metric Deltas"]
    lines.extend(f"- {metric}: {value}" for metric, value in comparison["deltas"].items())
    lines.append("")
    lines.append("## Task Outcomes")
    lines.append(f"- improved: {len(comparison['tasks_improved'])}")
    lines.append(f"- regressed: {len(comparison['tasks_regressed'])}")
    lines.append(f"- unchanged: {len(comparison['tasks_unchanged'])}")
    lines.append("")
    lines.append("## Selected Skill Usage")
    usage = comparison.get("selected_skill_usage_counts", {})
    if usage:
        lines.extend(f"- {skill_id}: {count}" for skill_id, count in usage.items())
    else:
        lines.append("- none")
    return "\n".join(lines).strip() + "\n"


def _failure_record(result: dict[str, Any]) -> FailureRecord:
    category = categorize_failure(result)
    return FailureRecord(
        task_id=str(result.get("task_id") or "unknown"),
        tags=list(result.get("tags") or []),
        question=result.get("question"),
        expected_answer=result.get("expected_answer"),
        predicted_answer=result.get("predicted_answer"),
        status=str(result.get("sandbox_status") or result.get("status") or "unknown"),
        correct=bool(result.get("correct")),
        error_message=result.get("error_message"),
        stdout=str(result.get("stdout") or ""),
        stderr=str(result.get("stderr") or ""),
        generated_code=str(result.get("generated_code") or ""),
        category=category,
        llm_calls=int(result.get("llm_calls") or 0),
        duration_sec=float(result.get("duration_sec") or 0.0),
        metadata=dict(result.get("metadata") or {}),
    )


def _recommendations(category_counts: Counter[str], tag_counts: Counter[str]) -> list[str]:
    recommendations: list[str] = []
    if category_counts[FailureCategory.EXECUTION_ERROR.value]:
        recommendations.append("Add robust CSV loading and execution-safety instructions.")
    if category_counts[FailureCategory.MISSING_ANSWER_VARIABLE.value]:
        recommendations.append("Emphasize assigning the final result to ANSWER.")
    if category_counts[FailureCategory.UNSUPPORTED_MULTI_ANSWER.value] or category_counts[
        FailureCategory.ANSWER_FORMAT_MISMATCH.value
    ]:
        recommendations.append("Add explicit multi-answer formatting guidance and structured comparison.")
    if category_counts[FailureCategory.NUMERIC_TOLERANCE_ISSUE.value]:
        recommendations.append("Add numeric answer formatting and rounding guidance.")
    for tag, _count in tag_counts.most_common(3):
        recommendations.append(f"Retrieve or create skills for frequent failure tag: {tag}.")
    if not recommendations:
        recommendations.append("No dominant failure pattern detected.")
    return recommendations


def _metric_deltas(a_metrics: dict[str, Any], b_metrics: dict[str, Any]) -> dict[str, float]:
    keys = ["answer_accuracy", "execution_success_rate", "invalid_code_rate", "composite_score"]
    return {key: float(b_metrics.get(key, 0.0)) - float(a_metrics.get(key, 0.0)) for key in keys}


def _per_tag_deltas(a_results: list[dict[str, Any]], b_results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    a_by_id = {result.get("task_id"): result for result in a_results}
    b_by_id = {result.get("task_id"): result for result in b_results}
    tag_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"a_correct": 0, "b_correct": 0, "total": 0})
    for task_id, a_result in a_by_id.items():
        b_result = b_by_id.get(task_id)
        if b_result is None:
            continue
        tags = set(a_result.get("tags") or b_result.get("tags") or [])
        for tag in tags:
            tag_counts[tag]["a_correct"] += int(bool(a_result.get("correct")))
            tag_counts[tag]["b_correct"] += int(bool(b_result.get("correct")))
            tag_counts[tag]["total"] += 1
    return {
        tag: {
            "accuracy_delta": (counts["b_correct"] / counts["total"]) - (counts["a_correct"] / counts["total"]),
            "total": counts["total"],
        }
        for tag, counts in sorted(tag_counts.items())
        if counts["total"]
    }


def _selected_skill_usage_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for result in results:
        metadata = result.get("metadata") or {}
        counter.update(metadata.get("selected_skill_ids") or [])
    return dict(sorted(counter.items()))


def _is_multi_answer(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.count("@") > 1 and value.count("[") > 1


def _looks_like_format_mismatch(expected: Any, predicted: Any) -> bool:
    if not isinstance(expected, str) or predicted is None:
        return False
    expected_text = expected.lower().replace(" ", "")
    predicted_text = str(predicted).lower().replace(" ", "")
    return "@" in expected_text and "[" in expected_text and any(part in predicted_text for part in expected_text.split("\n"))


def _is_numeric_like(value: Any) -> bool:
    return _to_float(value) is not None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).strip().replace(",", "").rstrip("%"))
    except ValueError:
        return None


def _rate(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(bool(item) for item in items) / len(items)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
