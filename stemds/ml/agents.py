"""ML-engineering baseline agents."""

from __future__ import annotations

from stemds.agents.base import AgentOutput
from stemds.agents.baseline import extract_python_code
from stemds.llm import BaseLLMClient
from stemds.ml.datasets import load_sklearn_dataset
from stemds.ml.sandbox import MLSandboxResult
from stemds.ml.tasks import MLEngineeringTask
from stemds.ml.workflows import MLWorkflowSpec, direct_ml_workflow


class OpenAIMLEngineeringAgent:
    def __init__(
        self,
        model: str,
        llm_client: BaseLLMClient,
        workflow: MLWorkflowSpec | None = None,
        seed: int | None = 42,
    ) -> None:
        self.model = model
        self.llm_client = llm_client
        self.workflow = workflow or direct_ml_workflow()
        self.seed = seed

    def solve(self, task: MLEngineeringTask) -> AgentOutput:
        prompt = self.build_prompt(task)
        raw_response = self.llm_client.generate_text(prompt, model=self.model, temperature=0.0, seed=self.seed)
        code = extract_python_code(raw_response)
        return AgentOutput(
            code=code,
            raw_response=raw_response,
            llm_calls=1,
            metadata={
                "agent": "openai_ml_engineering",
                "model": self.model,
                "workflow_id": self.workflow.workflow_id,
                "prompt": prompt,
                "seed": self.seed,
                "llm_api_path": getattr(self.llm_client, "last_api_path", None),
                "llm_seed_ignored": getattr(self.llm_client, "last_seed_ignored", False),
            },
        )

    def repair(self, task: MLEngineeringTask, code: str, result: MLSandboxResult, attempt_number: int) -> AgentOutput:
        prompt = self.build_repair_prompt(task, code, result, attempt_number)
        raw_response = self.llm_client.generate_text(prompt, model=self.model, temperature=0.0, seed=self.seed)
        repaired_code = extract_python_code(raw_response)
        return AgentOutput(
            code=repaired_code,
            raw_response=raw_response,
            llm_calls=1,
            metadata={
                "agent": "openai_ml_engineering",
                "model": self.model,
                "workflow_id": self.workflow.workflow_id,
                "prompt": prompt,
                "repair_attempt": attempt_number,
                "seed": self.seed,
                "llm_api_path": getattr(self.llm_client, "last_api_path", None),
                "llm_seed_ignored": getattr(self.llm_client, "last_seed_ignored", False),
            },
        )

    def build_prompt(self, task: MLEngineeringTask) -> str:
        dataframe, target_column = load_sklearn_dataset(task.dataset_name)
        dtypes = "\n".join(f"- {column}: {dtype}" for column, dtype in dataframe.dtypes.items())
        sample = dataframe.head(5).to_csv(index=False).strip()
        return f"""You are an ML engineer writing fast, deterministic sklearn code.

Task: {task.description or task.task_id}
Dataset name: {task.dataset_name}
Problem type: {task.problem_type}
Metric: {task.metric}
Target column: {target_column}
Workflow: {self.workflow.name}

Train data is available at TRAIN_CSV_PATH.
Test data is available at TEST_CSV_PATH.
The target column name is TARGET_COLUMN.
The metric name is METRIC.
The problem type is PROBLEM_TYPE.

Columns and dtypes:
{dtypes}

Sample rows as CSV:
{sample}

Workflow instructions:
{_workflow_instructions(self.workflow, task)}

Requirements:
- write Python code only
- use sklearn, pandas, and numpy only
- read train/test data with pd.read_csv(TRAIN_CSV_PATH) and pd.read_csv(TEST_CSV_PATH)
- set random_state=42 where applicable
- keep code simple and fast
- do not download data
- do not use network
- do not print prose
- assign final output to RESULT exactly as:
  RESULT = {{"score": <float>, "metric": METRIC, "model": "<short model description>"}}
"""

    def build_repair_prompt(
        self,
        task: MLEngineeringTask,
        code: str,
        result: MLSandboxResult,
        attempt_number: int,
    ) -> str:
        return f"""The previous ML-engineering code failed or produced an invalid RESULT.

Task: {task.description or task.task_id}
Dataset: {task.dataset_name}
Metric: {task.metric}
Problem type: {task.problem_type}

Previous code:
```python
{code}
```

Status: {result.status}
Stdout:
{result.stdout}

Stderr:
{result.stderr}

Repair attempt: {attempt_number}

Return repaired Python code only. Use TRAIN_CSV_PATH, TEST_CSV_PATH, TARGET_COLUMN, METRIC, and PROBLEM_TYPE.
Assign RESULT = {{"score": <float>, "metric": METRIC, "model": "<short model description>"}}.
"""


class DummyMLAgent:
    def solve(self, task: MLEngineeringTask) -> AgentOutput:
        strategy = "most_frequent" if task.problem_type == "classification" else "mean"
        estimator = (
            f"DummyClassifier(strategy='{strategy}')"
            if task.problem_type == "classification"
            else f"DummyRegressor(strategy='{strategy}')"
        )
        score_expr = _metric_score_expression(task.metric)
        imports = (
            "from sklearn.dummy import DummyClassifier, DummyRegressor\n"
            "from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score\n"
        )
        code = f"""
import math
import pandas as pd
{imports}

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)
X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]
X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN]
model = {estimator}
model.fit(X_train, y_train)
pred = model.predict(X_test)
score = {score_expr}
RESULT = {{"score": float(score), "metric": METRIC, "model": "{estimator}"}}
""".strip() + "\n"
        return AgentOutput(code=code, raw_response=None, llm_calls=0, metadata={"agent": "dummy_ml"})


class MockMLAgent:
    def __init__(self, responses: list[str] | str) -> None:
        self._responses = [responses] if isinstance(responses, str) else list(responses)
        self.prompts: list[str] = []

    def solve(self, task: MLEngineeringTask) -> AgentOutput:
        response = self._responses.pop(0)
        return AgentOutput(code=extract_python_code(response), raw_response=response, llm_calls=1, metadata={"agent": "mock_ml"})


def _workflow_instructions(workflow: MLWorkflowSpec, task: MLEngineeringTask) -> str:
    if workflow.prompt_strategy == "preprocess_pipeline":
        return (
            "Use a sklearn Pipeline. Include SimpleImputer and StandardScaler where appropriate. "
            "For classification, LogisticRegression or RandomForestClassifier are acceptable. "
            "For regression, Ridge or RandomForestRegressor are acceptable."
        )
    if workflow.prompt_strategy == "compare_models":
        return "Try two simple fast sklearn models and report the better test metric."
    if workflow.prompt_strategy == "preprocess_compare":
        return (
            "Use preprocessing with SimpleImputer and scaling where appropriate, try two simple fast models, "
            "and report the better test metric."
        )
    return "Train one simple robust sklearn model and report the requested test metric."


def _metric_score_expression(metric: str) -> str:
    if metric == "accuracy":
        return "accuracy_score(y_test, pred)"
    if metric == "f1_macro":
        return "f1_score(y_test, pred, average='macro', zero_division=0)"
    if metric == "rmse":
        return "math.sqrt(mean_squared_error(y_test, pred))"
    if metric == "mae":
        return "mean_absolute_error(y_test, pred)"
    if metric == "r2":
        return "r2_score(y_test, pred)"
    raise ValueError(f"Unsupported metric: {metric}")
