"""LLM-backed PromptSkill proposal from failure reports."""

from __future__ import annotations

import json
import re
from typing import Any

from stemds.analysis.failures import FailureAnalysisReport
from stemds.llm import BaseLLMClient
from stemds.skills.base import PromptSkill
from stemds.skills.library import SkillLibrary
from stemds.stem.traces import CandidateSkillRecord


class CandidateSkillProposer:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        model: str,
        max_candidates: int = 3,
        temperature: float = 0.2,
        seed: int | None = 42,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.max_candidates = max_candidates
        self.temperature = temperature
        self.seed = seed

    def propose(
        self,
        report: FailureAnalysisReport,
        existing_library: SkillLibrary,
    ) -> list[CandidateSkillRecord]:
        prompt = self._build_prompt(report, existing_library)
        raw_response = self.llm_client.generate_text(
            prompt,
            model=self.model,
            temperature=self.temperature,
            seed=self.seed,
        )
        source_categories = list(report.failures_by_category.keys())
        source_tags = list(report.failures_by_tag.keys())
        try:
            payload = _extract_json_payload(raw_response)
            items = _candidate_items(payload)
        except Exception as exc:
            return [
                _invalid_record(
                    {"skill_id": "invalid_proposal", "name": "Invalid Proposal"},
                    prompt,
                    raw_response,
                    source_categories,
                    source_tags,
                    f"Malformed proposal response: {exc}",
                )
            ]
        existing_ids = {skill.skill_id for skill in existing_library.skills}
        existing_names = {skill.name.lower().strip() for skill in existing_library.skills}
        records: list[CandidateSkillRecord] = []
        seen_ids: set[str] = set()
        for item in items:
            try:
                skill = PromptSkill.from_dict(item)
            except Exception as exc:
                records.append(
                    _invalid_record(
                        item,
                        prompt,
                        raw_response,
                        source_categories,
                        source_tags,
                        f"Malformed skill proposal: {exc}",
                    )
                )
                continue
            if skill.skill_id in existing_ids or skill.name.lower().strip() in existing_names:
                records.append(
                    _record_for_skill(
                        skill,
                        prompt,
                        raw_response,
                        source_categories,
                        source_tags,
                        status="invalid",
                        reason="Duplicate of existing skill.",
                    )
                )
                continue
            if skill.skill_id in seen_ids:
                records.append(
                    _record_for_skill(
                        skill,
                        prompt,
                        raw_response,
                        source_categories,
                        source_tags,
                        status="invalid",
                        reason="Duplicate candidate skill id.",
                    )
                )
                continue
            seen_ids.add(skill.skill_id)
            records.append(
                _record_for_skill(
                    skill,
                    prompt,
                    raw_response,
                    source_categories,
                    source_tags,
                    status="proposed",
                    reason="Candidate parsed successfully.",
                )
            )
            if len([record for record in records if record.status == "proposed"]) >= self.max_candidates:
                break
        return records

    def _build_prompt(self, report: FailureAnalysisReport, existing_library: SkillLibrary) -> str:
        examples = []
        for category, category_examples in report.top_examples_by_category.items():
            for example in category_examples[:2]:
                examples.append(
                    {
                        "category": category,
                        "task_id": example.get("task_id"),
                        "expected_answer": example.get("expected_answer"),
                        "predicted_answer": example.get("predicted_answer"),
                        "status": example.get("status"),
                    }
                )
        existing = [{"skill_id": skill.skill_id, "name": skill.name} for skill in existing_library.skills]
        return f"""You propose reusable PromptSkills for a Python/pandas data-analysis agent.

Return JSON only. Do not include markdown.
Propose at most {self.max_candidates} skills.

Each skill must have:
- skill_id
- name
- description
- applies_to_tags
- applies_to_failure_categories
- prompt_instructions
- priority

Existing skills to avoid duplicating:
{json.dumps(existing, indent=2)}

Failure categories:
{json.dumps(report.failures_by_category, indent=2)}

Failure tags:
{json.dumps(report.failures_by_tag, indent=2)}

Example failures:
{json.dumps(examples, indent=2)}

Return shape:
{{"skills": [{{...}}]}}
"""


def _extract_json_payload(raw_response: str) -> Any:
    stripped = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    return json.loads(stripped)


def _candidate_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("skills"), list):
        return [item for item in payload["skills"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Skill proposal must be a list or an object with a skills list")


def _record_for_skill(
    skill: PromptSkill,
    prompt: str,
    raw_response: str,
    source_categories: list[str],
    source_tags: list[str],
    status: str,
    reason: str,
) -> CandidateSkillRecord:
    return CandidateSkillRecord(
        skill=skill,
        source_failure_categories=source_categories,
        source_tags=source_tags,
        proposal_prompt=prompt,
        raw_response=raw_response,
        status=status,  # type: ignore[arg-type]
        reason=reason,
        baseline_score=None,
        trial_score=None,
        score_delta=None,
        validation_result_path=None,
        metadata={},
    )


def _invalid_record(
    item: dict[str, Any],
    prompt: str,
    raw_response: str,
    source_categories: list[str],
    source_tags: list[str],
    reason: str,
) -> CandidateSkillRecord:
    placeholder = PromptSkill(
        skill_id=str(item.get("skill_id", "invalid_skill")),
        name=str(item.get("name", "Invalid Skill")),
        description=str(item.get("description", "")),
        applies_to_tags=list(item.get("applies_to_tags", [])),
        applies_to_failure_categories=list(item.get("applies_to_failure_categories", [])),
        prompt_instructions=str(item.get("prompt_instructions", "")),
        priority=int(item.get("priority", 0)) if str(item.get("priority", "0")).isdigit() else 0,
        metadata={"raw_proposal": item},
    )
    return _record_for_skill(
        placeholder,
        prompt,
        raw_response,
        source_categories,
        source_tags,
        status="invalid",
        reason=reason,
    )
