"""Task schemas and JSONL loading utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ANSWER_TYPES = {"string", "number", "boolean"}


@dataclass(slots=True)
class DataAnalysisTask:
    task_id: str
    dataset_path: str
    question: str
    answer: str | float | int | bool
    answer_type: str
    tags: list[str] = field(default_factory=list)
    domain: str = "data_analysis"
    tolerance: float | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain != "data_analysis":
            raise ValueError(f"Unsupported domain: {self.domain}")
        if self.answer_type not in ANSWER_TYPES:
            allowed = ", ".join(sorted(ANSWER_TYPES))
            raise ValueError(f"answer_type must be one of: {allowed}")
        if not isinstance(self.tags, list):
            raise TypeError("tags must be a list of strings")
        if not all(isinstance(tag, str) for tag in self.tags):
            raise TypeError("tags must be a list of strings")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataAnalysisTask":
        return cls(
            task_id=str(payload["task_id"]),
            domain=str(payload.get("domain", "data_analysis")),
            dataset_path=str(payload["dataset_path"]),
            question=str(payload["question"]),
            answer=payload["answer"],
            answer_type=str(payload["answer_type"]),
            tolerance=payload.get("tolerance"),
            tags=list(payload.get("tags", [])),
            notes=payload.get("notes"),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_tasks_jsonl(path: str | Path) -> list[DataAnalysisTask]:
    tasks: list[DataAnalysisTask] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                tasks.append(DataAnalysisTask.from_dict(payload))
            except Exception as exc:
                raise ValueError(f"Invalid task JSONL at line {line_number}: {exc}") from exc
    return tasks


def save_tasks_jsonl(tasks: list[DataAnalysisTask], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), sort_keys=True) + "\n")
