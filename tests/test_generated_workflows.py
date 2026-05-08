from __future__ import annotations

import json
from pathlib import Path

from stemds.analysis.failures import FailureAnalysisReport
from stemds.llm import MockLLMClient
from stemds.tasks import DataAnalysisTask
from stemds.workflows.generated import GeneratedWorkflowSearcher, direct_code_generated_workflow
from stemds.workflows.graph import GeneratedWorkflowSpec, WorkflowLimits, WorkflowNode
from stemds.workflows.graph_executor import GeneratedWorkflowExecutor
from stemds.workflows.graph_proposer import GeneratedWorkflowProposal, GeneratedWorkflowProposer
from stemds.workflows.graph_validator import validate_generated_workflow


def _valid_workflow(workflow_id: str = "plan_code_execute") -> GeneratedWorkflowSpec:
    return GeneratedWorkflowSpec(
        workflow_id=workflow_id,
        name="Plan Code Execute",
        description="Plan, write code, execute, stop.",
        nodes=[
            WorkflowNode(id="schema", type="schema_summary"),
            WorkflowNode(id="plan", type="llm_plan", prompt_strategy="concise_plan"),
            WorkflowNode(id="code", type="llm_code", prompt_strategy="plan_then_code"),
            WorkflowNode(id="execute", type="python_execute"),
            WorkflowNode(id="stop", type="stop"),
        ],
        edges=[("schema", "plan"), ("plan", "code"), ("code", "execute"), ("execute", "stop")],
        limits=WorkflowLimits(max_llm_calls=2, max_repairs=0, timeout_sec=30),
        metadata={"test": True},
    )


def _repair_workflow(workflow_id: str = "repair_graph") -> GeneratedWorkflowSpec:
    return GeneratedWorkflowSpec(
        workflow_id=workflow_id,
        name="Repair Graph",
        description="Code, execute, repair once if needed.",
        nodes=[
            WorkflowNode(id="schema", type="schema_summary"),
            WorkflowNode(id="code", type="llm_code"),
            WorkflowNode(id="execute", type="python_execute"),
            WorkflowNode(id="repair", type="llm_repair"),
            WorkflowNode(id="stop", type="stop"),
        ],
        edges=[
            ("schema", "code"),
            ("code", "execute"),
            ("execute", "repair"),
            ("repair", "execute"),
            ("execute", "stop"),
        ],
        limits=WorkflowLimits(max_llm_calls=2, max_repairs=1, timeout_sec=30),
    )


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
                    "expected_answer": 3,
                    "predicted_answer": None,
                    "error_message": "boom",
                }
            ]
        },
        recommendations=["Use repair."],
    )


def _task(dataset_path: Path) -> DataAnalysisTask:
    return DataAnalysisTask(
        task_id="generated_workflow_test",
        dataset_path=str(dataset_path),
        question="What is the sum of value?",
        answer=3,
        answer_type="number",
        tolerance=1e-6,
        tags=["summary_statistics"],
    )


def test_generated_workflow_spec_serializes(tmp_path) -> None:
    path = tmp_path / "workflow.json"
    workflow = _valid_workflow()

    workflow.save_json(path)
    loaded = GeneratedWorkflowSpec.load_json(path)

    assert loaded.workflow_id == "plan_code_execute"
    assert loaded.edges == [("schema", "plan"), ("plan", "code"), ("code", "execute"), ("execute", "stop")]
    assert loaded.limits.max_llm_calls == 2


def test_validate_generated_workflow_accepts_valid_graph() -> None:
    result = validate_generated_workflow(_valid_workflow())

    assert result.valid
    assert result.errors == []


def test_validate_generated_workflow_rejects_unknown_node_type() -> None:
    workflow = _valid_workflow()
    workflow.nodes[0] = WorkflowNode(id="schema", type="unknown")

    result = validate_generated_workflow(workflow)

    assert not result.valid
    assert any("unsupported node type" in error for error in result.errors)


def test_validate_generated_workflow_rejects_unbounded_cycle() -> None:
    workflow = GeneratedWorkflowSpec(
        workflow_id="bad_cycle",
        name="Bad Cycle",
        description="Invalid arbitrary cycle.",
        nodes=[
            WorkflowNode(id="code", type="llm_code"),
            WorkflowNode(id="execute", type="python_execute"),
            WorkflowNode(id="stop", type="stop"),
        ],
        edges=[("code", "execute"), ("execute", "code"), ("execute", "stop")],
        limits=WorkflowLimits(max_llm_calls=3, max_repairs=0, timeout_sec=30),
    )

    result = validate_generated_workflow(workflow)

    assert not result.valid
    assert any("unsupported cycle" in error for error in result.errors)


def test_validate_generated_workflow_allows_bounded_repair_cycle() -> None:
    result = validate_generated_workflow(_repair_workflow())

    assert result.valid


def test_graph_proposer_parses_valid_mock_json() -> None:
    workflow = _valid_workflow("generated_plan")
    llm = MockLLMClient(json.dumps({"workflows": [workflow.to_dict()]}))

    proposals = GeneratedWorkflowProposer(llm, model="mock", max_candidates=3).propose(_report())

    assert len(proposals) == 1
    assert proposals[0].status == "valid"
    assert proposals[0].workflow is not None
    assert proposals[0].workflow.workflow_id == "generated_plan"


def test_graph_proposer_records_invalid_json() -> None:
    llm = MockLLMClient("not json")

    proposals = GeneratedWorkflowProposer(llm, model="mock", max_candidates=3).propose(_report())

    assert len(proposals) == 1
    assert proposals[0].status == "invalid"
    assert "Malformed workflow proposal response" in proposals[0].reason


def test_graph_executor_runs_plan_code_execute_workflow(tmp_path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("value\n1\n2\n", encoding="utf-8")
    llm = MockLLMClient(["Sum the value column.", "import pandas as pd\ndf = pd.read_csv(CSV_PATH)\nANSWER = int(df['value'].sum())"])
    agent = GeneratedWorkflowExecutor(_valid_workflow(), model="mock", llm_client=llm)

    output = agent.solve(_task(dataset))

    assert output.llm_calls == 2
    assert "ANSWER" in output.code
    assert output.metadata["workflow_id"] == "plan_code_execute"
    assert any(item["type"] == "python_execute" and item["status"] == "pass" for item in output.metadata["node_trace"])


def test_generated_workflow_searcher_selects_improved_mocked_workflow(tmp_path) -> None:
    improved = _valid_workflow("generated_improved")

    def propose_fn(_report):
        return [
            GeneratedWorkflowProposal(
                workflow=improved,
                status="valid",
                reason="ok",
                validation=validate_generated_workflow(improved),
                proposal_prompt="prompt",
                raw_response="{}",
                raw_item=improved.to_dict(),
            )
        ]

    def evaluate_fn(workflow, data_path: Path, model: str, output_path: Path, limit: int | None, seed: int | None):
        score = 0.10 if workflow.workflow_id == "direct_code" else 0.15
        return {"results": [], "metrics": {"composite_score": score, "answer_accuracy": score, "invalid_code_rate": 0.0}}

    outcome = GeneratedWorkflowSearcher(
        llm_client=None,
        model="mock",
        train_data=tmp_path / "train.jsonl",
        val_data=tmp_path / "val.jsonl",
        out_dir=tmp_path,
        min_delta=0.03,
        evaluate_fn=evaluate_fn,
        propose_fn=propose_fn,
    ).search()

    assert outcome.differentiated
    assert outcome.frozen_workflow.workflow_id == "generated_improved"
    assert (tmp_path / "frozen_generated_workflow.json").exists()


def test_generated_workflow_searcher_freezes_direct_when_no_improvement(tmp_path) -> None:
    weak = _valid_workflow("generated_weak")

    def propose_fn(_report):
        return [
            GeneratedWorkflowProposal(
                workflow=weak,
                status="valid",
                reason="ok",
                validation=validate_generated_workflow(weak),
                proposal_prompt="prompt",
                raw_response="{}",
                raw_item=weak.to_dict(),
            )
        ]

    def evaluate_fn(workflow, data_path: Path, model: str, output_path: Path, limit: int | None, seed: int | None):
        score = 0.10 if workflow.workflow_id == "direct_code" else 0.11
        return {"results": [], "metrics": {"composite_score": score, "answer_accuracy": score, "invalid_code_rate": 0.0}}

    outcome = GeneratedWorkflowSearcher(
        llm_client=None,
        model="mock",
        train_data=tmp_path / "train.jsonl",
        val_data=tmp_path / "val.jsonl",
        out_dir=tmp_path,
        min_delta=0.03,
        evaluate_fn=evaluate_fn,
        propose_fn=propose_fn,
    ).search()

    assert not outcome.differentiated
    assert outcome.frozen_workflow.workflow_id == direct_code_generated_workflow().workflow_id
