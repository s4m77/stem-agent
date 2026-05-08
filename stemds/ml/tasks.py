"""ML-engineering task schema and JSONL helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ProblemType = Literal["classification", "regression"]
MLMetric = Literal["accuracy", "f1_macro", "rmse", "mae", "r2"]

PROBLEM_TYPES = {"classification", "regression"}
ML_METRICS = {"accuracy", "f1_macro", "rmse", "mae", "r2"}


@dataclass(slots=True)
class MLEngineeringTask:
    task_id: str
    dataset_name: str
    target_name: str
    problem_type: ProblemType
    metric: MLMetric
    domain: str = "ml_engineering"
    min_score: float | None = None
    baseline_score: float | None = None
    tags: list[str] = field(default_factory=list)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain != "ml_engineering":
            raise ValueError(f"Unsupported ML domain: {self.domain}")
        if self.problem_type not in PROBLEM_TYPES:
            raise ValueError(f"problem_type must be one of: {sorted(PROBLEM_TYPES)}")
        if self.metric not in ML_METRICS:
            raise ValueError(f"metric must be one of: {sorted(ML_METRICS)}")
        if self.problem_type == "classification" and self.metric not in {"accuracy", "f1_macro"}:
            raise ValueError("classification tasks require accuracy or f1_macro")
        if self.problem_type == "regression" and self.metric not in {"rmse", "mae", "r2"}:
            raise ValueError("regression tasks require rmse, mae, or r2")
        if not all(isinstance(tag, str) for tag in self.tags):
            raise TypeError("tags must be a list of strings")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MLEngineeringTask":
        return cls(
            task_id=str(payload["task_id"]),
            domain=str(payload.get("domain", "ml_engineering")),
            dataset_name=str(payload["dataset_name"]),
            target_name=str(payload["target_name"]),
            problem_type=payload["problem_type"],
            metric=payload["metric"],
            min_score=payload.get("min_score"),
            baseline_score=payload.get("baseline_score"),
            tags=list(payload.get("tags", [])),
            description=str(payload.get("description", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ml_tasks_jsonl(path: str | Path) -> list[MLEngineeringTask]:
    tasks: list[MLEngineeringTask] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                tasks.append(MLEngineeringTask.from_dict(json.loads(stripped)))
            except Exception as exc:
                raise ValueError(f"Invalid ML task JSONL at line {line_number}: {exc}") from exc
    return tasks


def save_ml_tasks_jsonl(tasks: list[MLEngineeringTask], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), sort_keys=True) + "\n")
