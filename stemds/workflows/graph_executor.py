"""Executor for safe generated workflow graphs."""

from __future__ import annotations

from typing import Any

from stemds.agents.base import AgentOutput
from stemds.agents.baseline import extract_python_code, inspect_csv
from stemds.llm import BaseLLMClient
from stemds.sandbox import PythonSandbox, SandboxResult
from stemds.tasks import DataAnalysisTask
from stemds.workflows.graph import GeneratedWorkflowSpec, WorkflowNode
from stemds.workflows.graph_validator import validate_generated_workflow


class GeneratedWorkflowExecutor:
    """Execute a validated workflow graph using fixed safe primitives."""

    def __init__(
        self,
        workflow: GeneratedWorkflowSpec,
        model: str,
        llm_client: BaseLLMClient,
        seed: int | None = 42,
        sandbox: PythonSandbox | None = None,
    ) -> None:
        validation = validate_generated_workflow(workflow)
        if not validation.valid:
            raise ValueError(f"Invalid generated workflow: {validation.errors}")
        self.workflow = workflow
        self.model = model
        self.llm_client = llm_client
        self.seed = seed
        self.sandbox = sandbox or PythonSandbox(timeout_sec=workflow.limits.timeout_sec)

    def solve(self, task: DataAnalysisTask) -> AgentOutput:
        node_trace: list[dict[str, Any]] = []
        prompts: list[str] = []
        raw_responses: list[str] = []
        profile: dict[str, object] | None = None
        plan: str | None = None
        code = ""
        last_execution: SandboxResult | None = None
        repair_attempts = 0
        stopped = False

        for node in self.workflow.nodes:
            if node.type == "schema_summary":
                profile = inspect_csv(task.dataset_path)
                node_trace.append({"node_id": node.id, "type": node.type, "status": "ok", "shape": profile["shape"]})
            elif node.type == "llm_plan":
                prompt = self._build_plan_prompt(task, profile, node)
                response = self._call_llm(prompt, prompts, raw_responses, node_trace, node)
                plan = response.strip() if response is not None else None
            elif node.type == "llm_code":
                prompt = self._build_code_prompt(task, profile, plan, node)
                response = self._call_llm(prompt, prompts, raw_responses, node_trace, node)
                code = extract_python_code(response or "")
            elif node.type == "python_execute":
                last_execution = self._execute_code(code, task, node, node_trace)
                if _execution_failed(last_execution) and self._has_repair_node():
                    code, last_execution, repair_attempts = self._run_repair_loop(
                        task=task,
                        code=code,
                        last_execution=last_execution,
                        prompts=prompts,
                        raw_responses=raw_responses,
                        node_trace=node_trace,
                        repair_attempts=repair_attempts,
                    )
            elif node.type == "llm_repair":
                node_trace.append(
                    {
                        "node_id": node.id,
                        "type": node.type,
                        "status": "handled_by_python_execute" if last_execution is not None else "skipped_no_execution",
                    }
                )
            elif node.type in {"answer_normalize", "llm_answer_check"}:
                node_trace.append({"node_id": node.id, "type": node.type, "status": "noop_v0"})
            elif node.type == "stop":
                node_trace.append({"node_id": node.id, "type": node.type, "status": "ok"})
                stopped = True
                break

        if not stopped:
            node_trace.append({"node_id": "__implicit_stop__", "type": "stop", "status": "implicit"})

        return AgentOutput(
            code=code,
            raw_response=raw_responses[-1] if raw_responses else None,
            llm_calls=len(raw_responses),
            metadata={
                "agent": "generated_workflow_openai",
                "answer_contract": "ANSWER",
                "workflow_id": self.workflow.workflow_id,
                "workflow": self.workflow.to_dict(),
                "model": self.model,
                "node_trace": node_trace,
                "prompts": prompts,
                "raw_responses": raw_responses,
                "llm_calls": len(raw_responses),
                "repair_attempts": repair_attempts,
                "execution_statuses": [
                    item.get("status")
                    for item in node_trace
                    if item.get("type") in {"python_execute", "python_execute_repair"}
                ],
                "final_code": code,
                "seed": self.seed,
                "llm_api_path": getattr(self.llm_client, "last_api_path", None),
                "llm_seed_ignored": getattr(self.llm_client, "last_seed_ignored", False),
            },
        )

    def _call_llm(
        self,
        prompt: str,
        prompts: list[str],
        raw_responses: list[str],
        node_trace: list[dict[str, Any]],
        node: WorkflowNode,
    ) -> str | None:
        if len(raw_responses) >= self.workflow.limits.max_llm_calls:
            node_trace.append({"node_id": node.id, "type": node.type, "status": "skipped_llm_budget_exhausted"})
            return None
        prompts.append(prompt)
        response = self.llm_client.generate_text(
            prompt,
            model=self.model,
            temperature=0.0,
            seed=self.seed,
        )
        raw_responses.append(response)
        node_trace.append({"node_id": node.id, "type": node.type, "status": "ok"})
        return response

    def _execute_code(
        self,
        code: str,
        task: DataAnalysisTask,
        node: WorkflowNode,
        node_trace: list[dict[str, Any]],
        trace_type: str = "python_execute",
    ) -> SandboxResult:
        result = self.sandbox.run_generated_analysis(code, task.dataset_path)
        node_trace.append(
            {
                "node_id": node.id,
                "type": trace_type,
                "status": result.status,
                "answer_present": result.answer is not None,
                "stderr": result.stderr,
                "stdout": result.stdout,
                "duration_sec": result.duration_sec,
            }
        )
        return result

    def _run_repair_loop(
        self,
        task: DataAnalysisTask,
        code: str,
        last_execution: SandboxResult,
        prompts: list[str],
        raw_responses: list[str],
        node_trace: list[dict[str, Any]],
        repair_attempts: int,
    ) -> tuple[str, SandboxResult, int]:
        repair_node = next(node for node in self.workflow.nodes if node.type == "llm_repair")
        execute_node = next(node for node in self.workflow.nodes if node.type == "python_execute")
        current_code = code
        current_execution = last_execution
        while (
            _execution_failed(current_execution)
            and repair_attempts < self.workflow.limits.max_repairs
            and len(raw_responses) < self.workflow.limits.max_llm_calls
        ):
            repair_attempts += 1
            prompt = self._build_repair_prompt(task, current_code, current_execution, repair_node, repair_attempts)
            response = self._call_llm(prompt, prompts, raw_responses, node_trace, repair_node)
            if response is None:
                break
            current_code = extract_python_code(response)
            current_execution = self._execute_code(
                current_code,
                task,
                execute_node,
                node_trace,
                trace_type="python_execute_repair",
            )
        return current_code, current_execution, repair_attempts

    def _has_repair_node(self) -> bool:
        return any(node.type == "llm_repair" for node in self.workflow.nodes) and self.workflow.limits.max_repairs > 0

    def _build_plan_prompt(
        self,
        task: DataAnalysisTask,
        profile: dict[str, object] | None,
        node: WorkflowNode,
    ) -> str:
        return f"""You are planning a Python/pandas data-analysis solution.

Question: {task.question}
Expected answer type: {task.answer_type}
Task tags: {", ".join(task.tags) if task.tags else "none"}
Dataset is available at CSV_PATH.

{_schema_text(profile)}

Return a concise plan only. Do not write executable code yet.
Prompt strategy: {node.prompt_strategy or "plan"}
"""

    def _build_code_prompt(
        self,
        task: DataAnalysisTask,
        profile: dict[str, object] | None,
        plan: str | None,
        node: WorkflowNode,
    ) -> str:
        plan_text = f"\nPlan to follow:\n{plan}\n" if plan else ""
        strategy = _code_strategy_instructions(node.prompt_strategy)
        return f"""You are a careful data analyst writing Python pandas code.

The dataset is available at the variable CSV_PATH.
Question: {task.question}
Expected answer type: {task.answer_type}
Task tags: {", ".join(task.tags) if task.tags else "none"}

{_schema_text(profile)}
{plan_text}
Workflow prompt strategy: {node.prompt_strategy or "direct_code"}
{strategy}

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
- if multiple named values are requested, use a deterministic @name[value] string
"""

    def _build_repair_prompt(
        self,
        task: DataAnalysisTask,
        code: str,
        result: SandboxResult,
        node: WorkflowNode,
        attempt_number: int,
    ) -> str:
        return f"""The previous Python/pandas code failed or did not assign ANSWER.

Question: {task.question}
Expected answer type: {task.answer_type}
Dataset is available at CSV_PATH.
Repair attempt: {attempt_number}
Prompt strategy: {node.prompt_strategy or "repair"}

Previous code:
```python
{code}
```

Execution status: {result.status}
Stdout:
{result.stdout}

Stderr:
{result.stderr}

Return repaired Python code only.
Requirements:
- read the dataset with pd.read_csv(CSV_PATH)
- assign final answer to ANSWER
- do not use external files or network
- keep code simple and deterministic
"""


def _schema_text(profile: dict[str, object] | None) -> str:
    if profile is None:
        return "No schema summary node has run yet."
    return f"""Dataset shape: {profile["shape"]}
Columns and dtypes:
{profile["columns_text"]}

Sample rows as CSV:
{profile["sample_csv"]}"""


def _code_strategy_instructions(prompt_strategy: str | None) -> str:
    if prompt_strategy == "plan_then_code":
        return "Follow the plan. You may include the plan as short Python comments before executable code."
    if prompt_strategy == "strict_answer_contract":
        return (
            "Use the strict answer contract. For multi-answer DABench questions, assign ANSWER to @name[value] "
            "pairs in deterministic order."
        )
    if prompt_strategy == "schema_then_code":
        return "Use exact column names from the schema and convert dates/numeric strings explicitly when needed."
    if prompt_strategy == "repair_first":
        return "Prioritize code that is easy to repair and inspect if execution fails."
    return "Solve directly with simple pandas code."


def _execution_failed(result: SandboxResult | None) -> bool:
    return result is None or result.status != "pass" or result.answer is None
