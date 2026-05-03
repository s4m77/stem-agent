"""Answer checking and aggregate evaluation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class TaskEvalResult:
    task_id: str
    expected_answer: str | float | int | bool
    predicted_answer: str | None
    correct: bool
    invalid_code: bool
    sandbox_status: str
    llm_calls: int
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_answers(
    expected: str | float | int | bool,
    predicted: str | None,
    answer_type: str,
    tolerance: float | None = None,
) -> bool:
    if predicted is None:
        return False
    if answer_type == "string":
        return _normalize_string(expected) == _normalize_string(predicted)
    if answer_type == "boolean":
        return _parse_bool(expected) == _parse_bool(predicted)
    if answer_type == "number":
        expected_float = _parse_float(expected)
        predicted_float = _parse_float(predicted)
        if expected_float is None or predicted_float is None:
            return False
        return abs(expected_float - predicted_float) <= (tolerance if tolerance is not None else 1e-6)
    raise ValueError(f"Unsupported answer_type: {answer_type}")


def aggregate_metrics(results: list[TaskEvalResult]) -> dict[str, float | int]:
    total = len(results)
    if total == 0:
        return {
            "total_tasks": 0,
            "answer_accuracy": 0.0,
            "execution_success_rate": 0.0,
            "invalid_code_rate": 0.0,
            "avg_llm_calls": 0.0,
            "avg_runtime_sec": 0.0,
            "composite_score": 0.0,
        }

    answer_accuracy = sum(result.correct for result in results) / total
    execution_success_rate = sum(result.sandbox_status == "success" for result in results) / total
    invalid_code_rate = sum(result.invalid_code for result in results) / total
    avg_llm_calls = sum(result.llm_calls for result in results) / total
    avg_runtime_sec = sum(result.duration_sec for result in results) / total
    composite_score = answer_accuracy - 0.25 * invalid_code_rate - 0.02 * avg_llm_calls
    return {
        "total_tasks": total,
        "answer_accuracy": answer_accuracy,
        "execution_success_rate": execution_success_rate,
        "invalid_code_rate": invalid_code_rate,
        "avg_llm_calls": avg_llm_calls,
        "avg_runtime_sec": avg_runtime_sec,
        "composite_score": composite_score,
    }


def _normalize_string(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = _normalize_string(value)
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0"}:
        return False
    return None


def _parse_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError:
        return None

