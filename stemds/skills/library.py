"""Simple in-memory skill library with JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path

from stemds.skills.base import PromptSkill, PythonSkill, Skill


class SkillLibrary:
    # TODO: SkillLibrary will store accepted PromptSkills/PythonSkills with validation evidence.
    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._skills = list(skills or [])

    @property
    def skills(self) -> list[Skill]:
        return list(self._skills)

    def add(self, skill: Skill) -> None:
        self._skills.append(skill)

    def search(self, tags: list[str]) -> list[Skill]:
        requested = set(tags)
        return [skill for skill in self._skills if requested.intersection(skill.tags)]

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump([skill.to_dict() for skill in self._skills], handle, indent=2, sort_keys=True)

    @classmethod
    def load_json(cls, path: str | Path) -> "SkillLibrary":
        with Path(path).open("r", encoding="utf-8") as handle:
            payloads = json.load(handle)
        return cls([_skill_from_dict(payload) for payload in payloads])


def _skill_from_dict(payload: dict[str, object]) -> Skill:
    kind = payload.get("kind")
    if kind == "prompt":
        return PromptSkill(**payload)  # type: ignore[arg-type]
    if kind == "python":
        return PythonSkill(**payload)  # type: ignore[arg-type]
    return Skill(**payload)  # type: ignore[arg-type]

