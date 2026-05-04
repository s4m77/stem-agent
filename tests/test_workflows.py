from __future__ import annotations

import json
from pathlib import Path

from stemds.llm import MockLLMClient
from stemds.tasks import DataAnalysisTask
from stemds.workflows.base import WorkflowSpec
from stemds.workflows.candidates import candidate_workflows
from stemds.workflows.executor import WorkflowAnalysisAgent
from stemds.workflows.search import WorkflowSearcher


def _task(dataset_path: Path) -> DataAnalysisTask:
    return DataAnalysisTask(
        task_id="workflow_test",
        dataset_path=str(dataset_path),
        question="What is the sum of value?",
        answer=3,
        answer_type="number",
        tolerance=1e-6,
        tags=["summary_statistics"],
    )


def test_workflow_spec_serializes(tmp_path) -> None:
    workflow = WorkflowSpec(
        workflow_id="x",
        name="Example",
        description="desc",
        prompt_strategy="direct_code",
        max_repair_attempts=1,
        uses_repair_loop=True,
        metadata={"a": 1},
    )
    path = tmp_path / "workflow.json"

    workflow.save_json(path)
    loaded = WorkflowSpec.load_json(path)

    assert loaded.workflow_id == "x"
    assert loaded.uses_repair_loop
    assert loaded.metadata == {"a": 1}


def test_candidate_workflows_are_unique_and_valid() -> None:
    workflows = candidate_workflows()
    ids = [workflow.workflow_id for workflow in workflows]

    assert len(ids) == len(set(ids))
    assert "direct_code" in ids
    assert all(workflow.name for workflow in workflows)
    assert all(workflow.prompt_strategy for workflow in workflows)


def test_workflow_agent_prompt_changes_by_workflow(tmp_path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("value\n1\n2\n", encoding="utf-8")
    direct_llm = MockLLMClient("ANSWER = 3")
    plan_llm = MockLLMClient("ANSWER = 3")
    direct = WorkflowSpec("direct_code", "Direct", "desc", "direct_code")
    plan = WorkflowSpec("plan_then_code", "Plan", "desc", "plan_then_code", uses_plan=True)

    WorkflowAnalysisAgent(direct, model="mock", llm_client=direct_llm).solve(_task(dataset))
    WorkflowAnalysisAgent(plan, model="mock", llm_client=plan_llm).solve(_task(dataset))

    assert "Start with a short plan" not in direct_llm.prompts[0]
    assert "Start with a short plan" in plan_llm.prompts[0]


def test_repair_workflow_calls_mock_llm_twice_when_first_code_fails(tmp_path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("value\n1\n2\n", encoding="utf-8")
    workflow = WorkflowSpec(
        workflow_id="code_then_repair",
        name="Repair",
        description="desc",
        prompt_strategy="direct_code",
        max_repair_attempts=1,
        uses_repair_loop=True,
    )
    llm = MockLLMClient(["raise RuntimeError('boom')", "ANSWER = 3"])

    output = WorkflowAnalysisAgent(workflow, model="mock", llm_client=llm).solve(_task(dataset))

    assert output.llm_calls == 2
    assert len(llm.prompts) == 2
    assert "Previous code" in llm.prompts[1]
    assert output.code == "ANSWER = 3\n"
    assert output.metadata["repair_attempts"] == 1


def test_workflow_searcher_selects_improved_workflow(tmp_path) -> None:
    direct = WorkflowSpec("direct_code", "Direct", "desc", "direct_code")
    improved = WorkflowSpec("schema_then_code", "Schema", "desc", "schema_then_code")

    def evaluate_fn(workflow, data_path: Path, model: str, output_path: Path, limit: int | None, seed: int | None):
        score = 0.20 if workflow.workflow_id == "direct_code" else 0.25
        payload = {"metrics": {"composite_score": score, "answer_accuracy": score, "invalid_code_rate": 0.0}}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    outcome = WorkflowSearcher(
        llm_client=None,
        model="mock",
        val_data=tmp_path / "val.jsonl",
        out_dir=tmp_path,
        workflows=[direct, improved],
        min_delta=0.03,
        evaluate_fn=evaluate_fn,
    ).search()

    assert outcome.differentiated
    assert outcome.frozen_workflow.workflow_id == "schema_then_code"
    assert (tmp_path / "frozen_workflow.json").exists()


def test_workflow_searcher_freezes_direct_when_no_candidate_improves(tmp_path) -> None:
    direct = WorkflowSpec("direct_code", "Direct", "desc", "direct_code")
    weak = WorkflowSpec("schema_then_code", "Schema", "desc", "schema_then_code")

    def evaluate_fn(workflow, data_path: Path, model: str, output_path: Path, limit: int | None, seed: int | None):
        score = 0.20 if workflow.workflow_id == "direct_code" else 0.21
        payload = {"metrics": {"composite_score": score, "answer_accuracy": score, "invalid_code_rate": 0.0}}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    outcome = WorkflowSearcher(
        llm_client=None,
        model="mock",
        val_data=tmp_path / "val.jsonl",
        out_dir=tmp_path,
        workflows=[direct, weak],
        min_delta=0.03,
        evaluate_fn=evaluate_fn,
    ).search()

    assert not outcome.differentiated
    assert outcome.frozen_workflow.workflow_id == "direct_code"
