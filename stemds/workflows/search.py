"""Validation-set workflow search for StemDS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from stemds.llm import BaseLLMClient
from stemds.metrics import aggregate_metrics
from stemds.sandbox import PythonSandbox
from stemds.tasks import load_tasks_jsonl
from stemds.workflows.base import WorkflowResult, WorkflowSpec
from stemds.workflows.candidates import candidate_workflows, direct_code_workflow
from stemds.workflows.executor import WorkflowAnalysisAgent


WorkflowEvaluationFunction = Callable[[WorkflowSpec, Path, str, Path, int | None, int | None], dict]


@dataclass(slots=True)
class WorkflowSearchOutcome:
    baseline_result: WorkflowResult
    results: list[WorkflowResult]
    frozen_workflow: WorkflowSpec
    differentiated: bool
    baseline_composite: float
    selected_composite: float
    min_delta: float

    def to_dict(self) -> dict:
        return {
            "baseline_result": self.baseline_result.to_dict(),
            "results": [result.to_dict() for result in self.results],
            "frozen_workflow": self.frozen_workflow.to_dict(),
            "differentiated": self.differentiated,
            "baseline_composite": self.baseline_composite,
            "selected_composite": self.selected_composite,
            "min_delta": self.min_delta,
        }


class WorkflowSearcher:
    def __init__(
        self,
        llm_client: BaseLLMClient | None,
        model: str,
        val_data: Path,
        out_dir: Path,
        workflows: list[WorkflowSpec] | None = None,
        val_limit: int | None = 38,
        seed: int | None = 42,
        min_delta: float = 0.03,
        evaluate_fn: WorkflowEvaluationFunction | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.val_data = val_data
        self.out_dir = out_dir
        self.workflows = workflows or candidate_workflows()
        self.val_limit = val_limit
        self.seed = seed
        self.min_delta = min_delta
        self.evaluate_fn = evaluate_fn or self._evaluate_workflow

    def search(self) -> WorkflowSearchOutcome:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        workflows = _dedupe_workflows(self.workflows)
        direct = next((workflow for workflow in workflows if workflow.workflow_id == "direct_code"), None)
        if direct is None:
            direct = direct_code_workflow()
            workflows.insert(0, direct)

        results: list[WorkflowResult] = []
        for workflow in workflows:
            run_path = self.out_dir / f"workflow_{workflow.workflow_id}_val.json"
            if run_path.exists():
                payload = json.loads(run_path.read_text(encoding="utf-8"))
            else:
                payload = self.evaluate_fn(workflow, self.val_data, self.model, run_path, self.val_limit, self.seed)
            metrics = dict(payload.get("metrics", {}))
            results.append(
                WorkflowResult(
                    workflow_id=workflow.workflow_id,
                    validation_run_path=str(run_path),
                    metrics=metrics,
                    accepted=False,
                    reason="Evaluated on validation set.",
                    metadata={"workflow": workflow.to_dict()},
                )
            )

        baseline_result = next(result for result in results if result.workflow_id == "direct_code")
        baseline_composite = _composite(baseline_result)
        best_result = max(results, key=_composite)
        selected_workflow = next(workflow for workflow in workflows if workflow.workflow_id == best_result.workflow_id)
        selected_composite = _composite(best_result)
        differentiated = (
            best_result.workflow_id != "direct_code"
            and selected_composite > baseline_composite + self.min_delta
        )

        if differentiated:
            best_result.accepted = True
            best_result.reason = (
                f"Accepted: composite {selected_composite:.3f} exceeded direct_code "
                f"{baseline_composite:.3f} by more than min_delta {self.min_delta:.3f}."
            )
            frozen_workflow = selected_workflow
        else:
            baseline_result.accepted = True
            baseline_result.reason = "Frozen direct_code: no workflow exceeded baseline by min_delta."
            frozen_workflow = direct
            selected_composite = baseline_composite

        outcome = WorkflowSearchOutcome(
            baseline_result=baseline_result,
            results=results,
            frozen_workflow=frozen_workflow,
            differentiated=differentiated,
            baseline_composite=baseline_composite,
            selected_composite=selected_composite,
            min_delta=self.min_delta,
        )
        self._write_outputs(outcome)
        return outcome

    def _evaluate_workflow(
        self,
        workflow: WorkflowSpec,
        data_path: Path,
        model: str,
        output_path: Path,
        limit: int | None,
        seed: int | None,
    ) -> dict:
        if self.llm_client is None:
            raise ValueError("llm_client is required when no custom evaluate_fn is provided")
        from stemds.cli import _evaluate_task, _progress_tasks, _select_tasks

        tasks = _select_tasks(load_tasks_jsonl(data_path), limit=limit, seed=seed)
        sandbox = PythonSandbox()
        agent = WorkflowAnalysisAgent(workflow=workflow, model=model, llm_client=self.llm_client, seed=seed, sandbox=sandbox)
        results = [
            _evaluate_task(task, agent, sandbox)
            for task in _progress_tasks(tasks, desc=f"workflow:{workflow.workflow_id}")
        ]
        payload = {
            "workflow": workflow.to_dict(),
            "results": [result.to_dict() for result in results],
            "metrics": aggregate_metrics(results),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def _write_outputs(self, outcome: WorkflowSearchOutcome) -> None:
        results_path = self.out_dir / "workflow_search_results.json"
        summary_path = self.out_dir / "workflow_search_summary.md"
        frozen_path = self.out_dir / "frozen_workflow.json"
        results_path.write_text(json.dumps(outcome.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        frozen_path.write_text(
            json.dumps(outcome.frozen_workflow.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary_path.write_text(render_workflow_search_summary(outcome), encoding="utf-8")


def render_workflow_search_summary(outcome: WorkflowSearchOutcome) -> str:
    lines = [
        "# Workflow Search Summary",
        "",
        f"- Baseline workflow: `{outcome.baseline_result.workflow_id}`",
        f"- Baseline composite: `{outcome.baseline_composite:.6f}`",
        f"- Frozen workflow: `{outcome.frozen_workflow.workflow_id}`",
        f"- Frozen composite: `{outcome.selected_composite:.6f}`",
        f"- Min delta: `{outcome.min_delta:.6f}`",
        f"- Differentiated: `{outcome.differentiated}`",
        "",
        "## Validation Results",
        "",
    ]
    for result in sorted(outcome.results, key=lambda item: item.workflow_id):
        metrics = result.metrics
        lines.append(
            f"- `{result.workflow_id}`: composite `{float(metrics.get('composite_score', 0.0)):.6f}`, "
            f"accuracy `{float(metrics.get('answer_accuracy', 0.0)):.6f}`, "
            f"invalid code `{float(metrics.get('invalid_code_rate', 0.0)):.6f}`"
        )
    lines.append("")
    return "\n".join(lines)


def _dedupe_workflows(workflows: list[WorkflowSpec]) -> list[WorkflowSpec]:
    seen: set[str] = set()
    unique: list[WorkflowSpec] = []
    for workflow in workflows:
        if workflow.workflow_id in seen:
            raise ValueError(f"Duplicate workflow_id: {workflow.workflow_id}")
        seen.add(workflow.workflow_id)
        unique.append(workflow)
    return unique


def _composite(result: WorkflowResult) -> float:
    return float(result.metrics.get("composite_score", 0.0))
