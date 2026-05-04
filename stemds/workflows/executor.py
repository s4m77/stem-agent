"""Workflow-aware data-analysis agent."""

from __future__ import annotations

from stemds.agents.base import AgentOutput
from stemds.agents.baseline import extract_python_code, inspect_csv
from stemds.llm import BaseLLMClient
from stemds.sandbox import PythonSandbox, SandboxResult
from stemds.tasks import DataAnalysisTask
from stemds.workflows.base import WorkflowSpec


class WorkflowAnalysisAgent:
    """LLM-backed analyst whose prompt/control flow is governed by a WorkflowSpec."""

    def __init__(
        self,
        workflow: WorkflowSpec,
        model: str,
        llm_client: BaseLLMClient,
        seed: int | None = 42,
        sandbox: PythonSandbox | None = None,
    ) -> None:
        self.workflow = workflow
        self.model = model
        self.llm_client = llm_client
        self.seed = seed
        self.sandbox = sandbox or PythonSandbox()

    def solve(self, task: DataAnalysisTask) -> AgentOutput:
        profile = inspect_csv(task.dataset_path)
        prompts: list[str] = []
        raw_responses: list[str] = []
        repair_attempts = 0

        prompt = self.build_prompt(task, profile)
        prompts.append(prompt)
        raw_response = self.llm_client.generate_text(prompt, model=self.model, temperature=0.0, seed=self.seed)
        raw_responses.append(raw_response)
        code = extract_python_code(raw_response)

        repair_results: list[dict[str, object]] = []
        for attempt in range(self.workflow.max_repair_attempts if self.workflow.uses_repair_loop else 0):
            sandbox_result = self.sandbox.run_generated_analysis(code, task.dataset_path)
            repair_results.append(_sandbox_result_metadata(sandbox_result))
            if sandbox_result.status == "pass" and sandbox_result.answer is not None:
                break
            repair_attempts += 1
            repair_prompt = self.build_repair_prompt(task, code, sandbox_result, attempt + 1)
            prompts.append(repair_prompt)
            repaired_response = self.llm_client.generate_text(
                repair_prompt,
                model=self.model,
                temperature=0.0,
                seed=self.seed,
            )
            raw_responses.append(repaired_response)
            code = extract_python_code(repaired_response)

        return AgentOutput(
            code=code,
            raw_response=raw_responses[-1] if raw_responses else None,
            llm_calls=len(raw_responses),
            metadata={
                "agent": "workflow_openai",
                "answer_contract": "ANSWER",
                "workflow_id": self.workflow.workflow_id,
                "workflow": self.workflow.to_dict(),
                "model": self.model,
                "prompt": prompts[-1] if prompts else "",
                "prompts": prompts,
                "raw_responses": raw_responses,
                "repair_attempts": repair_attempts,
                "repair_results": repair_results,
                "final_generated_code": code,
                "csv_shape": profile["shape"],
                "csv_columns": profile["columns"],
                "seed": self.seed,
                "llm_api_path": getattr(self.llm_client, "last_api_path", None),
                "llm_seed_ignored": getattr(self.llm_client, "last_seed_ignored", False),
            },
        )

    def build_prompt(self, task: DataAnalysisTask, profile: dict[str, object]) -> str:
        common_context = f"""You are a careful data analyst writing Python pandas code.

The dataset is available at the variable CSV_PATH.
Question: {task.question}
Expected answer type: {task.answer_type}
Task tags: {", ".join(task.tags) if task.tags else "none"}
"""
        schema_context = ""
        if self.workflow.uses_schema_summary:
            schema_context = f"""
Dataset shape: {profile["shape"]}
Columns and dtypes:
{profile["columns_text"]}

Sample rows as CSV:
{profile["sample_csv"]}
"""
        strategy_text = _strategy_instructions(self.workflow)
        return f"""{common_context}{schema_context}
Workflow strategy: {self.workflow.name}
{strategy_text}

Requirements:
- write Python code only
- use pandas
- read the dataset with pd.read_csv(CSV_PATH)
- assign the final answer to a variable named ANSWER
- do not print prose
- do not make plots
- do not read external files
- do not use network
- keep code simple and deterministic
- if the answer is numeric, assign a plain int or float
- if the answer is categorical or string, assign a string
"""

    def build_repair_prompt(
        self,
        task: DataAnalysisTask,
        code: str,
        sandbox_result: SandboxResult,
        attempt_number: int,
    ) -> str:
        return f"""The previous Python pandas code failed for this data-analysis task.

Question: {task.question}
Expected answer type: {task.answer_type}
Dataset is available at CSV_PATH.

Previous code:
```python
{code}
```

Execution status: {sandbox_result.status}
Stdout:
{sandbox_result.stdout}

Stderr:
{sandbox_result.stderr}

Repair attempt: {attempt_number}

Return repaired Python code only.
Requirements:
- use pandas
- read the dataset with pd.read_csv(CSV_PATH)
- assign the final answer to ANSWER
- do not print prose
- do not use external files or network
"""


def _strategy_instructions(workflow: WorkflowSpec) -> str:
    if workflow.prompt_strategy == "schema_then_code":
        return """Pay close attention to the dataset schema, column names, dtypes, and sample rows before writing code.
Use the exact column names from the schema. Prefer explicit conversions for dates, currencies, and numeric strings."""
    if workflow.prompt_strategy == "plan_then_code":
        return """Start with a short plan as Python comments only.
Then write executable pandas code. Keep the plan concise and verify each derived quantity in code."""
    if workflow.prompt_strategy == "strict_answer_contract":
        return """Use the strict DABench answer contract.
If the question asks for multiple named values, assign ANSWER to a deterministic string with one @name[value] pair per requested value.
Use stable names from the question where possible. Numeric values should be plain ints/floats or numeric text without units.
For scalar answers, assign only the requested scalar value."""
    return "Solve directly with simple pandas code."


def _sandbox_result_metadata(result: SandboxResult) -> dict[str, object]:
    return {
        "status": result.status,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_sec": result.duration_sec,
        "answer": result.answer,
        "extracted_answer": result.extracted_answer,
    }
