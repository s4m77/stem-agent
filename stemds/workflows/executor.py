"""Workflow executor placeholder."""

from __future__ import annotations

from stemds.workflows.base import WorkflowConfig


class WorkflowExecutor:
    # TODO: WorkflowSearcher will evaluate candidate workflow configs.
    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config

    def describe(self) -> str:
        return self.config.name

