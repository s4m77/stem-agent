"""Simple PromptSkill library with JSON persistence and keyword retrieval."""

from __future__ import annotations

import json
from pathlib import Path

from stemds.skills.base import PromptSkill


class SkillLibrary:
    def __init__(self, skills: list[PromptSkill] | None = None) -> None:
        self._skills = list(skills or [])

    @property
    def skills(self) -> list[PromptSkill]:
        return list(self._skills)

    def add_skill(self, skill: PromptSkill) -> None:
        self._skills.append(skill)

    def add(self, skill: PromptSkill) -> None:
        self.add_skill(skill)

    def retrieve(
        self,
        task_tags: list[str],
        failure_categories: list[str] | None = None,
        k: int = 5,
    ) -> list[PromptSkill]:
        tag_set = {_normalize_key(tag) for tag in task_tags}
        category_set = {_normalize_key(category) for category in (failure_categories or [])}
        scored: list[tuple[int, PromptSkill]] = []
        for skill in self._skills:
            skill_tags = {_normalize_key(tag) for tag in skill.applies_to_tags}
            skill_categories = {_normalize_key(category) for category in skill.applies_to_failure_categories}
            is_global = not skill_tags and not skill_categories
            overlap = len(tag_set.intersection(skill_tags)) + len(category_set.intersection(skill_categories))
            if overlap == 0 and not is_global:
                continue
            scored.append((overlap * 100 + skill.priority, skill))
        scored.sort(key=lambda item: (-item[0], item[1].skill_id))
        return [skill for _score, skill in scored[:k]]

    def search(self, tags: list[str]) -> list[PromptSkill]:
        return self.retrieve(tags, k=len(self._skills))

    def save_to_dir(self, path: str | Path) -> None:
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        for skill in self._skills:
            skill_path = output_dir / f"{skill.skill_id}.json"
            skill_path.write_text(json.dumps(skill.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_from_dir(cls, path: str | Path) -> "SkillLibrary":
        input_dir = Path(path)
        if not input_dir.exists():
            raise ValueError(f"Skill directory does not exist: {input_dir}")
        skills = [
            PromptSkill.from_dict(json.loads(skill_path.read_text(encoding="utf-8")))
            for skill_path in sorted(input_dir.glob("*.json"))
        ]
        return cls(skills)

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([skill.to_dict() for skill in self._skills], indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "SkillLibrary":
        payloads = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([PromptSkill.from_dict(payload) for payload in payloads])


def _normalize_key(value: str) -> str:
    return str(value).lower().strip().replace(" ", "_").replace("-", "_")
