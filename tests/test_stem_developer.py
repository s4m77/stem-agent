from __future__ import annotations

import json
from pathlib import Path

from stemds.analysis.failures import FailureAnalysisReport
from stemds.llm import MockLLMClient
from stemds.skills.base import PromptSkill
from stemds.skills.library import SkillLibrary
from stemds.stem.proposer import CandidateSkillProposer
from stemds.stem.traces import CandidateSkillRecord, StemDevelopmentTrace
from stemds.stem.validator import SkillValidator


def _report() -> FailureAnalysisReport:
    return FailureAnalysisReport(
        total_tasks=2,
        total_failures=1,
        failures_by_category={"execution_error": 1},
        failures_by_tag={"summary_statistics": 1},
        execution_success_rate=0.5,
        accuracy=0.5,
        top_examples_by_category={
            "execution_error": [
                {
                    "task_id": "a",
                    "status": "runtime_error",
                    "expected_answer": 1,
                    "predicted_answer": None,
                    "error_message": "boom",
                }
            ]
        },
        recommendations=["Improve CSV loading."],
    )


def _skill(skill_id: str = "candidate_summary_precision") -> PromptSkill:
    return PromptSkill(
        skill_id=skill_id,
        name="Candidate Summary Precision",
        description="Compute summary statistics carefully.",
        applies_to_tags=["summary_statistics"],
        applies_to_failure_categories=["execution_error"],
        prompt_instructions="Use CSV_PATH and assign a plain scalar to ANSWER.",
        priority=10,
    )


def test_candidate_skill_proposer_parses_valid_json() -> None:
    llm = MockLLMClient(json.dumps({"skills": [_skill().to_dict()]}))

    records = CandidateSkillProposer(llm, model="mock", max_candidates=3).propose(_report(), SkillLibrary())

    assert len(records) == 1
    assert records[0].status == "proposed"
    assert records[0].skill.skill_id == "candidate_summary_precision"


def test_candidate_skill_proposer_rejects_malformed_json() -> None:
    llm = MockLLMClient("not json")

    records = CandidateSkillProposer(llm, model="mock", max_candidates=3).propose(_report(), SkillLibrary())

    assert len(records) == 1
    assert records[0].status == "invalid"
    assert "Malformed proposal response" in records[0].reason


def test_skill_validator_accepts_improving_candidate(tmp_path) -> None:
    calls = []

    def evaluate_fn(library, data_path: Path, model: str, output_path: Path, limit: int | None, seed: int | None):
        calls.append(len(library.skills))
        score = 0.10 if len(library.skills) == 0 else 0.151
        return {"metrics": {"composite_score": score, "invalid_code_rate": 0.1}}

    result = SkillValidator(
        llm_client=None,
        model="mock",
        val_data=tmp_path / "val.jsonl",
        out_dir=tmp_path,
        evaluate_fn=evaluate_fn,
    ).validate(_skill(), SkillLibrary())

    assert calls == [0, 1]
    assert result.accepted
    assert result.score_delta > 0.05


def test_skill_validator_rejects_regressing_candidate(tmp_path) -> None:
    def evaluate_fn(library, data_path: Path, model: str, output_path: Path, limit: int | None, seed: int | None):
        score = 0.20 if len(library.skills) == 0 else 0.18
        return {"metrics": {"composite_score": score, "invalid_code_rate": 0.1}}

    result = SkillValidator(
        llm_client=None,
        model="mock",
        val_data=tmp_path / "val.jsonl",
        out_dir=tmp_path,
        evaluate_fn=evaluate_fn,
    ).validate(_skill(), SkillLibrary())

    assert not result.accepted
    assert "Rejected" in result.reason


def test_skill_validator_reuses_current_library_score_until_acceptance(tmp_path) -> None:
    calls = []

    def evaluate_fn(library, data_path: Path, model: str, output_path: Path, limit: int | None, seed: int | None):
        calls.append((tuple(skill.skill_id for skill in library.skills), output_path.name))
        score = 0.10 if len(library.skills) == 0 else 0.10
        return {"metrics": {"composite_score": score, "invalid_code_rate": 0.1}}

    validator = SkillValidator(
        llm_client=None,
        model="mock",
        val_data=tmp_path / "val.jsonl",
        out_dir=tmp_path,
        evaluate_fn=evaluate_fn,
    )

    validator.validate(_skill("candidate_a"), SkillLibrary())
    validator.validate(_skill("candidate_b"), SkillLibrary())

    current_calls = [call for call in calls if call[0] == ()]
    assert len(current_calls) == 1


def test_stem_development_trace_serializes(tmp_path) -> None:
    record = CandidateSkillRecord(
        skill=_skill(),
        source_failure_categories=["execution_error"],
        source_tags=["summary_statistics"],
        proposal_prompt="prompt",
        raw_response="{}",
        status="accepted",
        reason="good",
        baseline_score=0.1,
        trial_score=0.2,
        score_delta=0.1,
        validation_result_path="candidate.json",
        metadata={"x": 1},
    )
    trace = StemDevelopmentTrace(
        run_id="dev_test",
        created_at="2026-05-04T00:00:00+00:00",
        train_data="train.jsonl",
        val_data="val.jsonl",
        model="mock",
        baseline_run_path="baseline.json",
        baseline_analysis_path="analysis.json",
        accepted_skills=[record],
        rejected_skills=[],
        final_library_path="accepted_skills",
        metadata={"baseline": 0.1},
    )
    output_path = tmp_path / "trace.json"

    trace.save_json(output_path)
    loaded = StemDevelopmentTrace.load_json(output_path)

    assert loaded.run_id == "dev_test"
    assert loaded.accepted_skills[0].skill.skill_id == "candidate_summary_precision"
