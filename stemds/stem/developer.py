"""Constrained stem-development loop for PromptSkills."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from stemds.analysis.failures import analyze_run, render_failure_analysis_markdown
from stemds.llm import BaseLLMClient
from stemds.skills.library import SkillLibrary
from stemds.stem.proposer import CandidateSkillProposer
from stemds.stem.traces import CandidateSkillRecord, StemDevelopmentTrace
from stemds.stem.validator import SkillValidator


@dataclass(slots=True)
class StemDevelopmentResult:
    trace: StemDevelopmentTrace
    baseline_metrics: dict
    proposed_records: list[CandidateSkillRecord]

    def summary_dict(self) -> dict:
        return {
            "baseline_composite": self.baseline_metrics.get("composite_score"),
            "proposed_count": len(self.proposed_records),
            "accepted_count": len(self.trace.accepted_skills),
            "accepted_skill_ids": [record.skill.skill_id for record in self.trace.accepted_skills],
            "rejected": [
                {"skill_id": record.skill.skill_id, "reason": record.reason}
                for record in self.trace.rejected_skills
            ],
            "final_library_path": self.trace.final_library_path,
        }


class StemDeveloper:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        model: str,
        train_data: Path,
        val_data: Path,
        out_dir: Path,
        max_candidates: int = 3,
        val_limit: int | None = None,
        seed: int | None = 42,
        min_delta: float = 0.05,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.out_dir = out_dir
        self.max_candidates = max_candidates
        self.val_limit = val_limit
        self.seed = seed
        self.min_delta = min_delta

    def develop(self) -> StemDevelopmentResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        run_id = self.out_dir.name
        baseline_run_path = self.out_dir / "train_baseline.json"
        baseline_analysis_path = self.out_dir / "train_analysis.json"
        baseline_analysis_md_path = self.out_dir / "train_analysis.md"
        proposed_skills_path = self.out_dir / "proposed_skills.json"
        accepted_skills_dir = self.out_dir / "accepted_skills"
        rejected_skills_path = self.out_dir / "rejected_skills.json"
        trace_path = self.out_dir / "development_trace.json"
        summary_path = self.out_dir / "development_summary.md"

        validator = SkillValidator(
            llm_client=self.llm_client,
            model=self.model,
            val_data=self.val_data,
            out_dir=self.out_dir,
            val_limit=self.val_limit,
            seed=self.seed,
            min_delta=self.min_delta,
        )
        empty_library = SkillLibrary()
        if baseline_run_path.exists():
            baseline_payload = json.loads(baseline_run_path.read_text(encoding="utf-8"))
        else:
            baseline_payload = validator.evaluate_fn(
                empty_library,
                self.train_data,
                self.model,
                baseline_run_path,
                None,
                self.seed,
            )
        report = analyze_run(baseline_run_path)
        baseline_analysis_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        baseline_analysis_md_path.write_text(render_failure_analysis_markdown(report), encoding="utf-8")

        proposer = CandidateSkillProposer(
            llm_client=self.llm_client,
            model=self.model,
            max_candidates=self.max_candidates,
            seed=self.seed,
        )
        current_library = SkillLibrary()
        if proposed_skills_path.exists():
            proposed_records = [
                CandidateSkillRecord.from_dict(item)
                for item in json.loads(proposed_skills_path.read_text(encoding="utf-8"))
            ]
        else:
            proposed_records = proposer.propose(report, current_library)
            proposed_skills_path.write_text(
                json.dumps([record.to_dict() for record in proposed_records], indent=2, sort_keys=True),
                encoding="utf-8",
            )

        accepted: list[CandidateSkillRecord] = []
        rejected: list[CandidateSkillRecord] = []
        for record in proposed_records:
            if record.status != "proposed":
                record.status = "invalid"
                rejected.append(record)
                continue
            validation = validator.validate(record.skill, current_library)
            record.baseline_score = validation.baseline_score
            record.trial_score = validation.trial_score
            record.score_delta = validation.score_delta
            record.validation_result_path = validation.trial_result_path
            record.reason = validation.reason
            record.metadata.update(
                {
                    "baseline_invalid_code_rate": validation.baseline_invalid_code_rate,
                    "trial_invalid_code_rate": validation.trial_invalid_code_rate,
                    "baseline_result_path": validation.baseline_result_path,
                }
            )
            if validation.accepted:
                record.status = "accepted"
                current_library.add_skill(record.skill)
                validator.invalidate_current_cache()
                accepted.append(record)
            else:
                record.status = "rejected"
                rejected.append(record)

        current_library.save_to_dir(accepted_skills_dir)
        rejected_skills_path.write_text(
            json.dumps([record.to_dict() for record in rejected], indent=2, sort_keys=True),
            encoding="utf-8",
        )

        trace = StemDevelopmentTrace(
            run_id=run_id,
            created_at=datetime.now(UTC).isoformat(),
            train_data=str(self.train_data),
            val_data=str(self.val_data),
            model=self.model,
            baseline_run_path=str(baseline_run_path),
            baseline_analysis_path=str(baseline_analysis_path),
            accepted_skills=accepted,
            rejected_skills=rejected,
            final_library_path=str(accepted_skills_dir),
            metadata={
                "max_candidates": self.max_candidates,
                "val_limit": self.val_limit,
                "seed": self.seed,
                "min_delta": self.min_delta,
                "baseline_metrics": baseline_payload["metrics"],
            },
        )
        trace.save_json(trace_path)
        result = StemDevelopmentResult(
            trace=trace,
            baseline_metrics=baseline_payload["metrics"],
            proposed_records=proposed_records,
        )
        summary_path.write_text(render_development_summary(result), encoding="utf-8")
        return result


def render_development_summary(result: StemDevelopmentResult) -> str:
    summary = result.summary_dict()
    lines = [
        "# Stem Development Summary",
        "",
        f"- baseline_composite: {summary['baseline_composite']}",
        f"- proposed_count: {summary['proposed_count']}",
        f"- accepted_count: {summary['accepted_count']}",
        f"- final_library_path: {summary['final_library_path']}",
        "",
        "## Accepted Skills",
    ]
    accepted_ids = summary["accepted_skill_ids"]
    lines.extend(f"- {skill_id}" for skill_id in accepted_ids) if accepted_ids else lines.append("- none")
    lines.append("")
    lines.append("## Rejected Skills")
    rejected = summary["rejected"]
    if rejected:
        lines.extend(f"- {item['skill_id']}: {item['reason']}" for item in rejected)
    else:
        lines.append("- none")
    return "\n".join(lines).strip() + "\n"
