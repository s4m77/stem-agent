"""Skill metadata classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PromptSkill:
    skill_id: str
    name: str
    description: str
    applies_to_tags: list[str]
    applies_to_failure_categories: list[str]
    prompt_instructions: str
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptSkill":
        return cls(
            skill_id=str(payload["skill_id"]),
            name=str(payload["name"]),
            description=str(payload["description"]),
            applies_to_tags=[str(item) for item in payload.get("applies_to_tags", [])],
            applies_to_failure_categories=[str(item) for item in payload.get("applies_to_failure_categories", [])],
            prompt_instructions=str(payload["prompt_instructions"]),
            priority=int(payload.get("priority", 0)),
            metadata=dict(payload.get("metadata", {})),
        )
