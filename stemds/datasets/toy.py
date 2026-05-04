"""Adapter for existing StemDS JSONL task files."""

from __future__ import annotations

from pathlib import Path

from stemds.datasets.base import JSONLWriterMixin
from stemds.tasks import DataAnalysisTask, load_tasks_jsonl


class ToyJSONLAdapter(JSONLWriterMixin):
    name = "toy"

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def load_tasks(self) -> list[DataAnalysisTask]:
        return load_tasks_jsonl(self.data_path)

