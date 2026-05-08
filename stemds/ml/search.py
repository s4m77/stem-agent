"""Workflow search for the mini ML-engineering domain."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from stemds.llm import BaseLLMClient
from stemds.ml.agents import DummyMLAgent, OpenAIMLEngineeringAgent
from stemds.ml.metrics import MLTaskEvalResult, aggregate_ml_metrics, ensure_baseline_score
from stemds.ml.sandbox import MLEngineeringSandbox, MLSandboxResult
from stemds.ml.tasks import MLEngineeringTask
from stemds.ml.workflows import MLWorkflowSpec, candidate_ml_workflows, direct_ml_workflow

MLEvaluationFunction = Callable[[MLWorkflowSpec, list[MLEngineeringTask], str, Path, int | None, int | None], dict]


@dataclass(slots=True)
class MLWorkflowSearchOutcome:
    results: list[dict]
    frozen_workflow: MLWorkflowSpec
    differentiated: bool
    baseline_composite: float
    selected_composite: float
    min_delta: float

    def to_dict(self) -> dict:
        return {
            "results": self.results,
            "frozen_workflow": self.frozen_workflow.to_dict(),
            "differentiated": self.differentiated,
            "baseline_composite": self.baseline_composite,
            "selected_composite": self.selected_composite,
            "min_delta": self.min_delta,
        }


class MLWorkflowSearcher:
    def __init__(
        self,
        tasks: list[MLEngineeringTask],
        llm_client: BaseLLMClient | None,
        model: str,
        out_dir: Path,
        workflows: list[MLWorkflowSpec] | None = None,
        limit: int | None = None,
        seed: int | None = 42,
        min_delta: float = 0.03,
        evaluate_fn: MLEvaluationFunction | None = None,
    ) -> None:
        self.tasks = tasks
        self.llm_client = llm_client
        self.model = model
        self.out_dir = out_dir
        self.workflows = workflows or candidate_ml_workflows()
        self.limit = limit
        self.seed = seed
        self.min_delta = min_delta
        self.evaluate_fn = evaluate_fn or self._evaluate_workflow

    def search(self) -> MLWorkflowSearchOutcome:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        workflows = _dedupe(self.workflows)
        direct = next((workflow for workflow in workflows if workflow.workflow_id == "ml_direct"), None)
        if direct is None:
            direct = direct_ml_workflow()
            workflows.insert(0, direct)

        results: list[dict] = []
        selected_tasks = select_ml_tasks(self.tasks, limit=self.limit, seed=self.seed)
        for workflow in workflows:
            run_path = self.out_dir / f"ml_workflow_{workflow.workflow_id}.json"
            if run_path.exists():
                payload = json.loads(run_path.read_text(encoding="utf-8"))
            else:
                payload = self.evaluate_fn(workflow, selected_tasks, self.model, run_path, self.limit, self.seed)
            metrics = dict(payload.get("metrics", {}))
            results.append(
                {
                    "workflow_id": workflow.workflow_id,
                    "run_path": str(run_path),
                    "metrics": metrics,
                    "accepted": False,
                    "reason": "Evaluated on ML tasks.",
                    "workflow": workflow.to_dict(),
                }
            )

        baseline = next(result for result in results if result["workflow_id"] == "ml_direct")
        baseline_composite = float(baseline["metrics"].get("composite", 0.0))
        best = max(results, key=lambda result: float(result["metrics"].get("composite", 0.0)))
        best_composite = float(best["metrics"].get("composite", 0.0))
        differentiated = best["workflow_id"] != "ml_direct" and best_composite > baseline_composite + self.min_delta
        if differentiated:
            best["accepted"] = True
            best["reason"] = (
                f"Accepted: composite {best_composite:.3f} exceeded ml_direct "
                f"{baseline_composite:.3f} by more than min_delta {self.min_delta:.3f}."
            )
            frozen = next(workflow for workflow in workflows if workflow.workflow_id == best["workflow_id"])
            selected_composite = best_composite
        else:
            baseline["accepted"] = True
            baseline["reason"] = "Frozen ml_direct: no workflow exceeded baseline by min_delta."
            frozen = direct
            selected_composite = baseline_composite

        outcome = MLWorkflowSearchOutcome(
            results=results,
            frozen_workflow=frozen,
            differentiated=differentiated,
            baseline_composite=baseline_composite,
            selected_composite=selected_composite,
            min_delta=self.min_delta,
        )
        self._write_outputs(outcome)
        return outcome

    def _evaluate_workflow(
        self,
        workflow: MLWorkflowSpec,
        tasks: list[MLEngineeringTask],
        model: str,
        output_path: Path,
        limit: int | None,
        seed: int | None,
    ) -> dict:
        if self.llm_client is None:
            raise ValueError("llm_client is required when no custom evaluate_fn is provided")
        results = evaluate_ml_workflow_tasks(
            tasks=tasks,
            workflow=workflow,
            model=model,
            llm_client=self.llm_client,
            seed=seed,
        )
        payload = {
            "workflow": workflow.to_dict(),
            "results": [result.to_dict() for result in results],
            "metrics": aggregate_ml_metrics(results),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def _write_outputs(self, outcome: MLWorkflowSearchOutcome) -> None:
        (self.out_dir / "ml_workflow_search_results.json").write_text(
            json.dumps(outcome.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        outcome.frozen_workflow.save_json(self.out_dir / "frozen_ml_workflow.json")
        (self.out_dir / "ml_workflow_search_summary.md").write_text(
            render_ml_workflow_search_summary(outcome),
            encoding="utf-8",
        )


def evaluate_ml_workflow_tasks(
    tasks: list[MLEngineeringTask],
    workflow: MLWorkflowSpec,
    model: str,
    llm_client: BaseLLMClient,
    seed: int | None = 42,
) -> list[MLTaskEvalResult]:
    sandbox = MLEngineeringSandbox()
    agent = OpenAIMLEngineeringAgent(model=model, llm_client=llm_client, workflow=workflow, seed=seed)
    results: list[MLTaskEvalResult] = []
    for task in tasks:
        ensure_baseline_score(task, seed=seed or 42)
        output = agent.solve(task)
        code = output.code
        llm_calls = output.llm_calls
        sandbox_result = sandbox.run(code, task)
        repair_attempts = 0
        while (
            workflow.uses_repair_loop
            and repair_attempts < workflow.max_repair_attempts
            and sandbox_result.status != "success"
        ):
            repair_attempts += 1
            repair_output = agent.repair(task, code, sandbox_result, repair_attempts)
            code = repair_output.code
            llm_calls += repair_output.llm_calls
            sandbox_result = sandbox.run(code, task)
        sandbox_result.llm_calls = llm_calls
        results.append(ml_eval_result_from_sandbox(task, sandbox_result, code, model=model))
    return results


def evaluate_dummy_ml_tasks(tasks: list[MLEngineeringTask], seed: int | None = 42) -> list[MLTaskEvalResult]:
    sandbox = MLEngineeringSandbox()
    agent = DummyMLAgent()
    results: list[MLTaskEvalResult] = []
    for task in tasks:
        ensure_baseline_score(task, seed=seed or 42)
        output = agent.solve(task)
        sandbox_result = sandbox.run(output.code, task)
        sandbox_result.llm_calls = output.llm_calls
        results.append(ml_eval_result_from_sandbox(task, sandbox_result, output.code, model="dummy"))
    return results


def ml_eval_result_from_sandbox(
    task: MLEngineeringTask,
    sandbox_result: MLSandboxResult,
    generated_code: str,
    model: str | None,
) -> MLTaskEvalResult:
    return MLTaskEvalResult(
        task_id=task.task_id,
        dataset_name=task.dataset_name,
        metric=task.metric,
        problem_type=task.problem_type,
        score=sandbox_result.score,
        baseline_score=task.baseline_score,
        min_score=task.min_score,
        valid=sandbox_result.status == "success",
        status=sandbox_result.status,
        llm_calls=sandbox_result.llm_calls,
        duration_sec=sandbox_result.duration_sec,
        model=model,
        tags=task.tags,
        error_message=sandbox_result.stderr.strip() or None,
        stdout=sandbox_result.stdout,
        stderr=sandbox_result.stderr,
        generated_code=generated_code,
        metadata={"sandbox": sandbox_result.metadata},
    )


def render_ml_workflow_search_summary(outcome: MLWorkflowSearchOutcome) -> str:
    lines = [
        "# ML Workflow Search Summary",
        "",
        f"- Baseline composite: `{outcome.baseline_composite:.6f}`",
        f"- Frozen workflow: `{outcome.frozen_workflow.workflow_id}`",
        f"- Frozen composite: `{outcome.selected_composite:.6f}`",
        f"- Min delta: `{outcome.min_delta:.6f}`",
        f"- Differentiated: `{outcome.differentiated}`",
        "",
        "## Validation Results",
        "",
    ]
    for result in sorted(outcome.results, key=lambda item: item["workflow_id"]):
        metrics = result["metrics"]
        lines.append(
            f"- `{result['workflow_id']}`: composite `{float(metrics.get('composite', 0.0)):.6f}`, "
            f"valid run rate `{float(metrics.get('valid_run_rate', 0.0)):.6f}`, "
            f"avg improvement `{float(metrics.get('avg_score_delta_vs_baseline', 0.0)):.6f}`"
        )
    return "\n".join(lines) + "\n"


def select_ml_tasks(tasks: list[MLEngineeringTask], limit: int | None, seed: int | None) -> list[MLEngineeringTask]:
    if limit is None or limit >= len(tasks):
        return list(tasks)
    if limit < 1:
        raise ValueError("--limit must be a positive integer")
    indexed = list(enumerate(tasks))
    random.Random(seed).shuffle(indexed)
    return [task for _index, task in sorted(indexed[:limit], key=lambda item: item[0])]


def _dedupe(workflows: list[MLWorkflowSpec]) -> list[MLWorkflowSpec]:
    seen: set[str] = set()
    unique: list[MLWorkflowSpec] = []
    for workflow in workflows:
        if workflow.workflow_id in seen:
            raise ValueError(f"Duplicate ML workflow_id: {workflow.workflow_id}")
        seen.add(workflow.workflow_id)
        unique.append(workflow)
    return unique
