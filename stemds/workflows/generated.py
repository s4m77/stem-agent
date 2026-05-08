"""Generative workflow search over safe workflow DSL graphs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from stemds.analysis.failures import FailureAnalysisReport, analyze_run, render_failure_analysis_markdown
from stemds.llm import BaseLLMClient
from stemds.metrics import aggregate_metrics
from stemds.sandbox import PythonSandbox
from stemds.tasks import load_tasks_jsonl
from stemds.workflows.graph import GeneratedWorkflowSpec, WorkflowLimits, WorkflowNode
from stemds.workflows.graph_executor import GeneratedWorkflowExecutor
from stemds.workflows.graph_proposer import GeneratedWorkflowProposal, GeneratedWorkflowProposer


GeneratedWorkflowEvaluationFunction = Callable[
    [GeneratedWorkflowSpec, Path, str, Path, int | None, int | None],
    dict[str, Any],
]
GeneratedWorkflowProposalFunction = Callable[[FailureAnalysisReport], list[GeneratedWorkflowProposal]]


@dataclass(slots=True)
class GeneratedWorkflowSearchEntry:
    workflow_id: str
    validation_run_path: str
    metrics: dict[str, Any]
    accepted: bool = False
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeneratedWorkflowSearchOutcome:
    train_baseline_path: str
    train_analysis_path: str
    baseline_result: GeneratedWorkflowSearchEntry
    results: list[GeneratedWorkflowSearchEntry]
    proposals: list[GeneratedWorkflowProposal]
    frozen_workflow: GeneratedWorkflowSpec
    differentiated: bool
    baseline_composite: float
    selected_composite: float
    min_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_baseline_path": self.train_baseline_path,
            "train_analysis_path": self.train_analysis_path,
            "baseline_result": self.baseline_result.to_dict(),
            "results": [result.to_dict() for result in self.results],
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "frozen_workflow": self.frozen_workflow.to_dict(),
            "differentiated": self.differentiated,
            "baseline_composite": self.baseline_composite,
            "selected_composite": self.selected_composite,
            "min_delta": self.min_delta,
            "proposal_count": len(self.proposals),
            "valid_proposal_count": sum(1 for proposal in self.proposals if proposal.status == "valid"),
        }

    def summary_dict(self) -> dict[str, Any]:
        return {
            "proposal_count": len(self.proposals),
            "valid_proposal_count": sum(1 for proposal in self.proposals if proposal.status == "valid"),
            "baseline_composite": self.baseline_composite,
            "selected_composite": self.selected_composite,
            "frozen_workflow_id": self.frozen_workflow.workflow_id,
            "differentiated": self.differentiated,
        }


class GeneratedWorkflowSearcher:
    def __init__(
        self,
        llm_client: BaseLLMClient | None,
        model: str,
        train_data: Path,
        val_data: Path,
        out_dir: Path,
        max_candidates: int = 3,
        val_limit: int | None = 38,
        seed: int | None = 42,
        min_delta: float = 0.03,
        evaluate_fn: GeneratedWorkflowEvaluationFunction | None = None,
        propose_fn: GeneratedWorkflowProposalFunction | None = None,
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
        self.evaluate_fn = evaluate_fn or self._evaluate_workflow
        self.propose_fn = propose_fn

    def search(self) -> GeneratedWorkflowSearchOutcome:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        train_baseline_path = self.out_dir / "train_baseline.json"
        train_analysis_path = self.out_dir / "train_analysis.json"
        train_analysis_md_path = self.out_dir / "train_analysis.md"
        raw_proposals_path = self.out_dir / "proposed_workflows_raw.json"
        validated_proposals_path = self.out_dir / "proposed_workflows_validated.json"
        search_results_path = self.out_dir / "generated_workflow_search_results.json"
        frozen_path = self.out_dir / "frozen_generated_workflow.json"
        summary_path = self.out_dir / "generated_workflow_summary.md"

        direct_workflow = direct_code_generated_workflow()
        self._load_or_evaluate(direct_workflow, self.train_data, train_baseline_path, limit=None)
        train_report = analyze_run(train_baseline_path)
        train_analysis_path.write_text(json.dumps(train_report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        train_analysis_md_path.write_text(render_failure_analysis_markdown(train_report), encoding="utf-8")

        proposals = self._load_or_propose(
            train_report=train_report,
            raw_proposals_path=raw_proposals_path,
            validated_proposals_path=validated_proposals_path,
        )

        baseline_val_path = self.out_dir / "generated_workflow_direct_code_val.json"
        baseline_payload = self._load_or_evaluate(
            direct_workflow,
            self.val_data,
            baseline_val_path,
            limit=self.val_limit,
        )
        baseline_result = GeneratedWorkflowSearchEntry(
            workflow_id=direct_workflow.workflow_id,
            validation_run_path=str(baseline_val_path),
            metrics=dict(baseline_payload.get("metrics", {})),
            accepted=False,
            reason="Direct-code validation baseline.",
            metadata={"workflow": direct_workflow.to_dict()},
        )

        results = [baseline_result]
        for proposal in proposals:
            if proposal.status != "valid" or proposal.workflow is None:
                continue
            workflow = proposal.workflow
            run_path = self.out_dir / f"generated_workflow_{_safe_filename(workflow.workflow_id)}_val.json"
            payload = self._load_or_evaluate(workflow, self.val_data, run_path, limit=self.val_limit)
            results.append(
                GeneratedWorkflowSearchEntry(
                    workflow_id=workflow.workflow_id,
                    validation_run_path=str(run_path),
                    metrics=dict(payload.get("metrics", {})),
                    accepted=False,
                    reason="Generated workflow evaluated on validation set.",
                    metadata={
                        "workflow": workflow.to_dict(),
                        "proposal_validation": proposal.validation.to_dict(),
                        "proposal_reason": proposal.reason,
                    },
                )
            )

        baseline_composite = _composite(baseline_result)
        best_result = max(results, key=_composite)
        workflow_by_id = {
            direct_workflow.workflow_id: direct_workflow,
            **{
                proposal.workflow.workflow_id: proposal.workflow
                for proposal in proposals
                if proposal.status == "valid" and proposal.workflow is not None
            },
        }
        selected_composite = _composite(best_result)
        differentiated = (
            best_result.workflow_id != direct_workflow.workflow_id
            and selected_composite > baseline_composite + self.min_delta
        )
        if differentiated:
            best_result.accepted = True
            best_result.reason = (
                f"Accepted: composite {selected_composite:.3f} exceeded direct_code "
                f"{baseline_composite:.3f} by more than min_delta {self.min_delta:.3f}."
            )
            frozen_workflow = workflow_by_id[best_result.workflow_id]
        else:
            baseline_result.accepted = True
            baseline_result.reason = "Frozen direct_code: no generated workflow exceeded baseline by min_delta."
            frozen_workflow = direct_workflow
            selected_composite = baseline_composite

        outcome = GeneratedWorkflowSearchOutcome(
            train_baseline_path=str(train_baseline_path),
            train_analysis_path=str(train_analysis_path),
            baseline_result=baseline_result,
            results=results,
            proposals=proposals,
            frozen_workflow=frozen_workflow,
            differentiated=differentiated,
            baseline_composite=baseline_composite,
            selected_composite=selected_composite,
            min_delta=self.min_delta,
        )
        search_results_path.write_text(json.dumps(outcome.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        frozen_path.write_text(json.dumps(frozen_workflow.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        summary = render_generated_workflow_summary(outcome)
        summary_path.write_text(summary, encoding="utf-8")
        reports_summary_path = Path("reports/generative_workflow_search_summary.md")
        reports_summary_path.parent.mkdir(parents=True, exist_ok=True)
        reports_summary_path.write_text(summary, encoding="utf-8")
        return outcome

    def _load_or_propose(
        self,
        train_report: FailureAnalysisReport,
        raw_proposals_path: Path,
        validated_proposals_path: Path,
    ) -> list[GeneratedWorkflowProposal]:
        if validated_proposals_path.exists():
            return [
                GeneratedWorkflowProposal.from_dict(item)
                for item in json.loads(validated_proposals_path.read_text(encoding="utf-8"))
            ]
        if self.propose_fn is not None:
            proposals = self.propose_fn(train_report)
        else:
            if self.llm_client is None:
                raise ValueError("llm_client is required when no custom propose_fn is provided")
            proposer = GeneratedWorkflowProposer(
                llm_client=self.llm_client,
                model=self.model,
                max_candidates=self.max_candidates,
                seed=self.seed,
            )
            proposals = proposer.propose(train_report)
        raw_proposals_path.write_text(
            json.dumps(
                {
                    "proposals": [
                        {
                            "status": proposal.status,
                            "reason": proposal.reason,
                            "raw_item": proposal.raw_item,
                            "raw_response": proposal.raw_response,
                            "proposal_prompt": proposal.proposal_prompt,
                        }
                        for proposal in proposals
                    ]
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        validated_proposals_path.write_text(
            json.dumps([proposal.to_dict() for proposal in proposals], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return proposals

    def _load_or_evaluate(
        self,
        workflow: GeneratedWorkflowSpec,
        data_path: Path,
        output_path: Path,
        limit: int | None,
    ) -> dict[str, Any]:
        if output_path.exists():
            return json.loads(output_path.read_text(encoding="utf-8"))
        payload = self.evaluate_fn(workflow, data_path, self.model, output_path, limit, self.seed)
        if not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def _evaluate_workflow(
        self,
        workflow: GeneratedWorkflowSpec,
        data_path: Path,
        model: str,
        output_path: Path,
        limit: int | None,
        seed: int | None,
    ) -> dict[str, Any]:
        if self.llm_client is None:
            raise ValueError("llm_client is required when no custom evaluate_fn is provided")
        from stemds.cli import _evaluate_task, _progress_tasks, _select_tasks

        tasks = _select_tasks(load_tasks_jsonl(data_path), limit=limit, seed=seed)
        sandbox = PythonSandbox(timeout_sec=workflow.limits.timeout_sec)
        agent = GeneratedWorkflowExecutor(
            workflow=workflow,
            model=model,
            llm_client=self.llm_client,
            seed=seed,
            sandbox=sandbox,
        )
        results = [
            _evaluate_task(task, agent, sandbox)
            for task in _progress_tasks(tasks, desc=f"generated-workflow:{workflow.workflow_id}")
        ]
        payload = {
            "workflow": workflow.to_dict(),
            "results": [result.to_dict() for result in results],
            "metrics": aggregate_metrics(results),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload


def direct_code_generated_workflow() -> GeneratedWorkflowSpec:
    return GeneratedWorkflowSpec(
        workflow_id="direct_code",
        name="Direct Code",
        description="Generated-workflow representation of the one-shot pandas code baseline.",
        nodes=[
            WorkflowNode(id="schema", type="schema_summary"),
            WorkflowNode(id="code", type="llm_code", prompt_strategy="direct_code"),
            WorkflowNode(id="execute", type="python_execute"),
            WorkflowNode(id="stop", type="stop"),
        ],
        edges=[("schema", "code"), ("code", "execute"), ("execute", "stop")],
        limits=WorkflowLimits(max_llm_calls=1, max_repairs=0, timeout_sec=30),
        metadata={"source": "generated_workflow_baseline"},
    )


def render_generated_workflow_summary(outcome: GeneratedWorkflowSearchOutcome) -> str:
    lines = [
        "# Generative Workflow Search Summary",
        "",
        "This extension lets StemDS propose workflow graphs from failure analysis instead of only selecting from a human-authored menu.",
        "The generated graphs are constrained by a safe DSL and accepted only if validation performance improves.",
        "",
        "## Proposal Summary",
        "",
        f"- Generated proposals: `{len(outcome.proposals)}`",
        f"- Structurally valid proposals: `{sum(1 for proposal in outcome.proposals if proposal.status == 'valid')}`",
        f"- Frozen workflow: `{outcome.frozen_workflow.workflow_id}`",
        f"- Differentiated: `{outcome.differentiated}`",
        f"- Baseline composite: `{outcome.baseline_composite:.6f}`",
        f"- Selected composite: `{outcome.selected_composite:.6f}`",
        f"- Min delta: `{outcome.min_delta:.6f}`",
        "",
        "## Generated Candidates",
        "",
    ]
    if outcome.proposals:
        for proposal in outcome.proposals:
            workflow_id = proposal.workflow.workflow_id if proposal.workflow else "invalid"
            errors = "; ".join(proposal.validation.errors) or "none"
            lines.append(f"- `{workflow_id}`: {proposal.status}; {proposal.reason}; errors: {errors}")
    else:
        lines.append("- none")

    lines.extend(["", "## Validation Results", "", "| workflow_id | accuracy | composite | execution_success | invalid_code_rate | selected |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for result in outcome.results:
        metrics = result.metrics
        lines.append(
            f"| `{result.workflow_id}` | "
            f"{float(metrics.get('answer_accuracy', 0.0)):.3f} | "
            f"{float(metrics.get('composite_score', 0.0)):.3f} | "
            f"{float(metrics.get('execution_success_rate', 0.0)):.3f} | "
            f"{float(metrics.get('invalid_code_rate', 0.0)):.3f} | "
            f"{'yes' if result.accepted else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Comparison With Human-Authored Workflow Search",
            "",
            "Human-authored workflow search remains the current validated DABench headline path. "
            "This generative layer is intended to test whether the stem loop can create safe workflow architectures, "
            "not just choose from a predefined grid.",
            "",
            "## Limitations",
            "",
            "- v0 linearizes validated graphs rather than implementing a fully general graph engine.",
            "- `answer_normalize` and `llm_answer_check` are reserved no-op nodes in this pass.",
            "- Generated workflows are still constrained to fixed primitives and bounded repair loops.",
            "- Results may vary because OpenAI API seeding is not fully deterministic.",
            "",
        ]
    )
    return "\n".join(lines)


def _composite(result: GeneratedWorkflowSearchEntry) -> float:
    return float(result.metrics.get("composite_score", 0.0))


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "workflow"
