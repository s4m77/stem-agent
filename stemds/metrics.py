"""Answer checking and aggregate evaluation metrics."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

DABENCH_PAIR_PATTERN = re.compile(r"@([A-Za-z0-9_]+)\[([^\]]*)\]")


@dataclass(slots=True)
class TaskEvalResult:
    task_id: str
    expected_answer: str | float | int | bool
    predicted_answer: Any | None
    correct: bool
    invalid_code: bool
    sandbox_status: str
    llm_calls: int
    duration_sec: float
    tags: list[str] = None  # type: ignore[assignment]
    question: str | None = None
    error_message: str | None = None
    stdout: str = ""
    stderr: str = ""
    generated_code: str = ""
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_answers(
    expected: str | float | int | bool,
    predicted: Any | None,
    answer_type: str,
    tolerance: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    if predicted is None:
        return False
    if _should_compare_dabench_pairs(expected, metadata):
        return compare_dabench_pairs(expected, predicted, tolerance=tolerance)
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


def parse_dabench_pairs(text: str) -> dict[str, str]:
    return {name: value.strip() for name, value in DABENCH_PAIR_PATTERN.findall(text)}


def compare_dabench_pairs(expected: Any, predicted: Any, tolerance: float | None = None) -> bool:
    expected_pairs = _pairs_from_value(expected)
    predicted_pairs = _pairs_from_value(predicted)
    if not expected_pairs:
        return False
    if set(expected_pairs) != set(predicted_pairs):
        return False
    pair_tolerance = tolerance if tolerance is not None else 1e-6
    for name, expected_value in expected_pairs.items():
        predicted_value = predicted_pairs[name]
        expected_float = _parse_float(expected_value)
        predicted_float = _parse_float(predicted_value)
        if expected_float is not None and predicted_float is not None:
            if abs(expected_float - predicted_float) > pair_tolerance:
                return False
        elif _normalize_string(expected_value) != _normalize_string(predicted_value):
            return False
    return True


def dabench_pair_counts(expected: Any, predicted: Any, tolerance: float | None = None) -> tuple[int, int]:
    expected_pairs = _pairs_from_value(expected)
    predicted_pairs = _pairs_from_value(predicted)
    if not expected_pairs:
        return 0, 0
    pair_tolerance = tolerance if tolerance is not None else 1e-6
    correct = 0
    for name, expected_value in expected_pairs.items():
        if name not in predicted_pairs:
            continue
        predicted_value = predicted_pairs[name]
        expected_float = _parse_float(expected_value)
        predicted_float = _parse_float(predicted_value)
        if expected_float is not None and predicted_float is not None:
            correct += int(abs(expected_float - predicted_float) <= pair_tolerance)
        else:
            correct += int(_normalize_string(expected_value) == _normalize_string(predicted_value))
    return len(expected_pairs), correct


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
            "subquestion_total": 0,
            "subquestion_correct": 0,
            "subquestion_accuracy": 0.0,
        }

    answer_accuracy = sum(result.correct for result in results) / total
    execution_success_rate = sum(result.sandbox_status in {"success", "pass"} for result in results) / total
    invalid_code_rate = sum(result.invalid_code for result in results) / total
    avg_llm_calls = sum(result.llm_calls for result in results) / total
    avg_runtime_sec = sum(result.duration_sec for result in results) / total
    composite_score = answer_accuracy - 0.25 * invalid_code_rate - 0.02 * avg_llm_calls
    subquestion_total = sum(int(result.metadata.get("subquestion_total", 0)) for result in results)
    subquestion_correct = sum(int(result.metadata.get("subquestion_correct", 0)) for result in results)
    subquestion_accuracy = subquestion_correct / subquestion_total if subquestion_total else 0.0
    return {
        "total_tasks": total,
        "answer_accuracy": answer_accuracy,
        "execution_success_rate": execution_success_rate,
        "invalid_code_rate": invalid_code_rate,
        "avg_llm_calls": avg_llm_calls,
        "avg_runtime_sec": avg_runtime_sec,
        "composite_score": composite_score,
        "subquestion_total": subquestion_total,
        "subquestion_correct": subquestion_correct,
        "subquestion_accuracy": subquestion_accuracy,
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


def _should_compare_dabench_pairs(expected: Any, metadata: dict[str, Any] | None) -> bool:
    if metadata and metadata.get("raw_answer_type") == "multi":
        return True
    return isinstance(expected, str) and bool(parse_dabench_pairs(expected))


def _pairs_from_value(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    if isinstance(value, list):
        pairs: dict[str, str] = {}
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs[str(item[0])] = str(item[1])
        if pairs:
            return pairs
    if isinstance(value, str):
        return parse_dabench_pairs(value)
    return {}
