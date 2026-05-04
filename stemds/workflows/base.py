"""Workflow configuration primitives."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WorkflowSpec:
    workflow_id: str
    name: str
    description: str
    prompt_strategy: str
    max_repair_attempts: int = 0
    uses_schema_summary: bool = True
    uses_plan: bool = False
    uses_answer_check: bool = False
    uses_repair_loop: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkflowSpec":
        return cls(
            workflow_id=str(payload["workflow_id"]),
            name=str(payload["name"]),
            description=str(payload["description"]),
            prompt_strategy=str(payload["prompt_strategy"]),
            max_repair_attempts=int(payload.get("max_repair_attempts", 0)),
            uses_schema_summary=bool(payload.get("uses_schema_summary", True)),
            uses_plan=bool(payload.get("uses_plan", False)),
            uses_answer_check=bool(payload.get("uses_answer_check", False)),
            uses_repair_loop=bool(payload.get("uses_repair_loop", False)),
            metadata=dict(payload.get("metadata", {})),
        )

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "WorkflowSpec":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(slots=True)
class WorkflowResult:
    workflow_id: str
    validation_run_path: str
    metrics: dict[str, Any]
    accepted: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkflowResult":
        return cls(
            workflow_id=str(payload["workflow_id"]),
            validation_run_path=str(payload["validation_run_path"]),
            metrics=dict(payload.get("metrics", {})),
            accepted=bool(payload.get("accepted", False)),
            reason=str(payload.get("reason", "")),
            metadata=dict(payload.get("metadata", {})),
        )
