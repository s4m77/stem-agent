"""Trace models for constrained stem development."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from stemds.skills.base import PromptSkill

CandidateStatus = Literal["proposed", "accepted", "rejected", "invalid"]


@dataclass(slots=True)
class CandidateSkillRecord:
    skill: PromptSkill
    source_failure_categories: list[str]
    source_tags: list[str]
    proposal_prompt: str
    raw_response: str
    status: CandidateStatus
    reason: str
    baseline_score: float | None
    trial_score: float | None
    score_delta: float | None
    validation_result_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["skill"] = self.skill.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateSkillRecord":
        return cls(
            skill=PromptSkill.from_dict(payload["skill"]),
            source_failure_categories=list(payload.get("source_failure_categories", [])),
            source_tags=list(payload.get("source_tags", [])),
            proposal_prompt=str(payload.get("proposal_prompt", "")),
            raw_response=str(payload.get("raw_response", "")),
            status=payload.get("status", "proposed"),
            reason=str(payload.get("reason", "")),
            baseline_score=payload.get("baseline_score"),
            trial_score=payload.get("trial_score"),
            score_delta=payload.get("score_delta"),
            validation_result_path=payload.get("validation_result_path"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class StemDevelopmentTrace:
    run_id: str
    created_at: str
    train_data: str
    val_data: str
    model: str
    baseline_run_path: str
    baseline_analysis_path: str
    accepted_skills: list[CandidateSkillRecord]
    rejected_skills: list[CandidateSkillRecord]
    final_library_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["accepted_skills"] = [record.to_dict() for record in self.accepted_skills]
        payload["rejected_skills"] = [record.to_dict() for record in self.rejected_skills]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StemDevelopmentTrace":
        return cls(
            run_id=str(payload["run_id"]),
            created_at=str(payload["created_at"]),
            train_data=str(payload["train_data"]),
            val_data=str(payload["val_data"]),
            model=str(payload["model"]),
            baseline_run_path=str(payload["baseline_run_path"]),
            baseline_analysis_path=str(payload["baseline_analysis_path"]),
            accepted_skills=[CandidateSkillRecord.from_dict(item) for item in payload.get("accepted_skills", [])],
            rejected_skills=[CandidateSkillRecord.from_dict(item) for item in payload.get("rejected_skills", [])],
            final_library_path=str(payload["final_library_path"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "StemDevelopmentTrace":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

