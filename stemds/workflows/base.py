"""Workflow configuration primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class WorkflowConfig:
    name: str
    skill_names: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

