"""Metrics and dummy baselines for ML-engineering tasks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from stemds.ml.datasets import load_sklearn_dataset
from stemds.ml.tasks import MLEngineeringTask

LOWER_IS_BETTER = {"rmse", "mae"}


@dataclass(slots=True)
class MLTaskEvalResult:
    task_id: str
    dataset_name: str
    metric: str
    problem_type: str
    score: float | None
    baseline_score: float | None
    min_score: float | None
    valid: bool
    status: str
    llm_calls: int
    duration_sec: float
    model: str | None = None
    tags: list[str] = field(default_factory=list)
    error_message: str | None = None
    stdout: str = ""
    stderr: str = ""
    generated_code: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_dummy_baseline_score(task: MLEngineeringTask, seed: int = 42) -> float:
    dataframe, target_column = load_sklearn_dataset(task.dataset_name)
    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]
    stratify = target if task.problem_type == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=seed,
        stratify=stratify,
    )
    model = (
        DummyClassifier(strategy="most_frequent")
        if task.problem_type == "classification"
        else DummyRegressor(strategy="mean")
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return compute_metric(task.metric, y_test, predictions)


def ensure_baseline_score(task: MLEngineeringTask, seed: int = 42) -> MLEngineeringTask:
    if task.baseline_score is None:
        task.baseline_score = compute_dummy_baseline_score(task, seed=seed)
    return task


def compute_metric(metric: str, y_true: Any, y_pred: Any) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "rmse":
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    raise ValueError(f"Unsupported ML metric: {metric}")


def metric_higher_is_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


def normalized_improvement(score: float | None, baseline_score: float | None, metric: str) -> float | None:
    if score is None or baseline_score is None:
        return None
    if metric_higher_is_better(metric):
        return score - baseline_score
    return baseline_score - score


def score_beats_baseline(score: float | None, baseline_score: float | None, metric: str) -> bool:
    improvement = normalized_improvement(score, baseline_score, metric)
    return improvement is not None and improvement > 0


def score_meets_minimum(score: float | None, min_score: float | None, metric: str) -> bool:
    if score is None or min_score is None:
        return False
    return score >= min_score if metric_higher_is_better(metric) else score <= min_score


def aggregate_ml_metrics(results: list[MLTaskEvalResult]) -> dict[str, float | int]:
    total = len(results)
    if total == 0:
        return {
            "total_tasks": 0,
            "valid_run_rate": 0.0,
            "avg_score": 0.0,
            "avg_score_delta_vs_baseline": 0.0,
            "tasks_above_baseline": 0,
            "tasks_above_min_score": 0,
            "invalid_rate": 0.0,
            "avg_llm_calls": 0.0,
            "composite": 0.0,
        }
    valid_results = [result for result in results if result.valid and result.score is not None]
    improvements = [
        improvement
        for result in valid_results
        if (improvement := normalized_improvement(result.score, result.baseline_score, result.metric)) is not None
    ]
    avg_score = sum(float(result.score) for result in valid_results) / len(valid_results) if valid_results else 0.0
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
    invalid_rate = 1.0 - (len(valid_results) / total)
    avg_llm_calls = sum(result.llm_calls for result in results) / total
    composite = avg_improvement - 0.25 * invalid_rate - 0.02 * avg_llm_calls
    return {
        "total_tasks": total,
        "valid_run_rate": len(valid_results) / total,
        "avg_score": avg_score,
        "avg_score_delta_vs_baseline": avg_improvement,
        "tasks_above_baseline": sum(
            score_beats_baseline(result.score, result.baseline_score, result.metric) for result in valid_results
        ),
        "tasks_above_min_score": sum(
            score_meets_minimum(result.score, result.min_score, result.metric) for result in valid_results
        ),
        "invalid_rate": invalid_rate,
        "avg_llm_calls": avg_llm_calls,
        "composite": composite,
    }
