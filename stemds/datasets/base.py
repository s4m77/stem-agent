"""Shared dataset adapter interface."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from stemds.tasks import DataAnalysisTask, save_tasks_jsonl


class DatasetAdapter(Protocol):
    name: str

    def load_tasks(self) -> list[DataAnalysisTask]:
        ...

    def write_jsonl(self, tasks: list[DataAnalysisTask], output_path: Path) -> None:
        ...


class JSONLWriterMixin:
    def write_jsonl(self, tasks: list[DataAnalysisTask], output_path: Path) -> None:
        save_tasks_jsonl(tasks, output_path)

