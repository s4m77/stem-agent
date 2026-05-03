"""Base data-analysis agent interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from stemds.tasks import DataAnalysisTask


@dataclass(slots=True)
class AgentOutput:
    code: str
    raw_response: str | None
    llm_calls: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDataAnalysisAgent(Protocol):
    def solve(self, task: DataAnalysisTask) -> AgentOutput:
        ...

