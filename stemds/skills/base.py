"""Skill metadata classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    kind: str = "generic"
    version: str = "0.1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PromptSkill(Skill):
    prompt: str = ""
    kind: str = "prompt"


@dataclass(slots=True)
class PythonSkill(Skill):
    code: str = ""
    kind: str = "python"

