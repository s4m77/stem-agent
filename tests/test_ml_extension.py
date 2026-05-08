from __future__ import annotations

import json
from pathlib import Path

from stemds.llm import MockLLMClient
from stemds.ml.agents import OpenAIMLEngineeringAgent
from stemds.ml.datasets import create_builtin_ml_tasks, load_sklearn_dataset
from stemds.ml.metrics import (
    MLTaskEvalResult,
    aggregate_ml_metrics,
    compute_dummy_baseline_score,
)
from stemds.ml.sandbox import MLEngineeringSandbox
from stemds.ml.search import MLWorkflowSearcher
from stemds.ml.tasks import MLEngineeringTask
from stemds.ml.workflows import MLWorkflowSpec


def test_sklearn_dataset_loader_supports_required_datasets() -> None:
    for dataset_name in ["iris", "wine", "breast_cancer", "diabetes"]:
        dataframe, target_column = load_sklearn_dataset(dataset_name)

        assert target_column in dataframe.columns
        assert len(dataframe) > 0
        assert dataframe.shape[1] > 1


def test_create_builtin_ml_tasks_returns_valid_tasks() -> None:
    tasks = create_builtin_ml_tasks()

    assert len(tasks) >= 7
    assert {task.task_id for task in tasks} >= {"iris_accuracy", "diabetes_rmse"}
    assert all(task.domain == "ml_engineering" for task in tasks)


def test_dummy_baseline_scores_are_computed() -> None:
    classification = MLEngineeringTask(
        task_id="iris_accuracy",
        dataset_name="iris",
        target_name="target",
        problem_type="classification",
        metric="accuracy",
    )
    regression = MLEngineeringTask(
        task_id="diabetes_rmse",
        dataset_name="diabetes",
        target_name="target",
        problem_type="regression",
        metric="rmse",
    )

    assert compute_dummy_baseline_score(classification) >= 0.0
    assert compute_dummy_baseline_score(regression) > 0.0


def test_ml_sandbox_executes_simple_valid_result() -> None:
    task = MLEngineeringTask(
        task_id="iris_accuracy",
        dataset_name="iris",
        target_name="target",
        problem_type="classification",
        metric="accuracy",
    )
    code = 'RESULT = {"score": 0.5, "metric": METRIC, "model": "constant"}\n'

    result = MLEngineeringSandbox(timeout_sec=5).run(code, task)

    assert result.status == "success"
    assert result.score == 0.5
    assert result.metric == "accuracy"


def test_ml_sandbox_rejects_missing_result() -> None:
    task = MLEngineeringTask(
        task_id="iris_accuracy",
        dataset_name="iris",
        target_name="target",
        problem_type="classification",
        metric="accuracy",
    )

    result = MLEngineeringSandbox(timeout_sec=5).run("x = 1\n", task)

    assert result.status == "invalid_result"
    assert "RESULT" in result.stderr


def test_ml_metrics_handle_higher_and_lower_better() -> None:
    results = [
        MLTaskEvalResult(
            task_id="classification",
            dataset_name="iris",
            metric="accuracy",
            problem_type="classification",
            score=0.8,
            baseline_score=0.5,
            min_score=0.7,
            valid=True,
            status="success",
            llm_calls=1,
            duration_sec=0.1,
        ),
        MLTaskEvalResult(
            task_id="regression",
            dataset_name="diabetes",
            metric="rmse",
            problem_type="regression",
            score=50.0,
            baseline_score=60.0,
            min_score=55.0,
            valid=True,
            status="success",
            llm_calls=1,
            duration_sec=0.1,
        ),
    ]

    metrics = aggregate_ml_metrics(results)

    assert metrics["tasks_above_baseline"] == 2
    assert metrics["tasks_above_min_score"] == 2
    assert metrics["avg_score_delta_vs_baseline"] > 0


def test_ml_workflow_spec_serializes(tmp_path) -> None:
    workflow = MLWorkflowSpec(
        workflow_id="x",
        name="Example",
        description="desc",
        prompt_strategy="preprocess_pipeline",
        uses_preprocessing=True,
    )
    path = tmp_path / "workflow.json"

    workflow.save_json(path)
    loaded = MLWorkflowSpec.load_json(path)

    assert loaded.workflow_id == "x"
    assert loaded.uses_preprocessing


def test_ml_workflow_search_selects_improved_workflow(tmp_path) -> None:
    task = MLEngineeringTask(
        task_id="iris_accuracy",
        dataset_name="iris",
        target_name="target",
        problem_type="classification",
        metric="accuracy",
        baseline_score=0.3,
    )
    direct = MLWorkflowSpec("ml_direct", "Direct", "desc", "ml_direct")
    improved = MLWorkflowSpec("compare_models", "Compare", "desc", "compare_models")

    def evaluate_fn(workflow, tasks, model: str, output_path: Path, limit: int | None, seed: int | None):
        score = 0.1 if workflow.workflow_id == "ml_direct" else 0.2
        payload = {"metrics": {"composite": score, "valid_run_rate": 1.0, "avg_score_delta_vs_baseline": score}}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    outcome = MLWorkflowSearcher(
        tasks=[task],
        llm_client=None,
        model="mock",
        out_dir=tmp_path,
        workflows=[direct, improved],
        min_delta=0.03,
        evaluate_fn=evaluate_fn,
    ).search()

    assert outcome.differentiated
    assert outcome.frozen_workflow.workflow_id == "compare_models"


def test_openai_ml_agent_prompt_includes_contract_terms() -> None:
    task = MLEngineeringTask(
        task_id="iris_accuracy",
        dataset_name="iris",
        target_name="target",
        problem_type="classification",
        metric="accuracy",
    )
    llm = MockLLMClient('RESULT = {"score": 0.5, "metric": METRIC, "model": "test"}')

    OpenAIMLEngineeringAgent(model="mock", llm_client=llm).solve(task)

    prompt = llm.prompts[0]
    assert "TRAIN_CSV_PATH" in prompt
    assert "TEST_CSV_PATH" in prompt
    assert "TARGET_COLUMN" in prompt
    assert "RESULT" in prompt
