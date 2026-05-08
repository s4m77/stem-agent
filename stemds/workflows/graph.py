"""Safe generated-workflow DSL for StemDS."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WorkflowLimits:
    max_llm_calls: int = 3
    max_repairs: int = 1
    timeout_sec: int = 30

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "WorkflowLimits":
        payload = payload or {}
        return cls(
            max_llm_calls=int(payload.get("max_llm_calls", 3)),
            max_repairs=int(payload.get("max_repairs", 1)),
            timeout_sec=int(payload.get("timeout_sec", 30)),
        )


@dataclass(slots=True)
class WorkflowNode:
    id: str
    type: str
    prompt_strategy: str | None = None
    condition: str | None = None
    inputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkflowNode":
        return cls(
            id=str(payload["id"]),
            type=str(payload["type"]),
            prompt_strategy=payload.get("prompt_strategy"),
            condition=payload.get("condition"),
            inputs=list(payload.get("inputs", [])),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class GeneratedWorkflowSpec:
    workflow_id: str
    name: str
    description: str
    nodes: list[WorkflowNode]
    edges: list[tuple[str, str]]
    limits: WorkflowLimits
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["edges"] = [[source, target] for source, target in self.edges]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GeneratedWorkflowSpec":
        return cls(
            workflow_id=str(payload["workflow_id"]),
            name=str(payload.get("name", payload["workflow_id"])),
            description=str(payload.get("description", "")),
            nodes=[WorkflowNode.from_dict(item) for item in payload.get("nodes", [])],
            edges=[_edge_from_payload(item) for item in payload.get("edges", [])],
            limits=WorkflowLimits.from_dict(payload.get("limits")),
            metadata=dict(payload.get("metadata", {})),
        )

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "GeneratedWorkflowSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _edge_from_payload(item: Any) -> tuple[str, str]:
    if isinstance(item, dict):
        return str(item["from"]), str(item["to"])
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return str(item[0]), str(item[1])
    raise ValueError(f"Invalid workflow edge: {item!r}")
