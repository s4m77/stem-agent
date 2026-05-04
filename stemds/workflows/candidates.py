"""Finite workflow candidates for Workflow Search v0."""

from __future__ import annotations

from stemds.workflows.base import WorkflowSpec


def direct_code_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        workflow_id="direct_code",
        name="Direct Code",
        description="Current generic baseline: prompt once for pandas code assigning ANSWER.",
        prompt_strategy="direct_code",
    )


def candidate_workflows() -> list[WorkflowSpec]:
    return [
        direct_code_workflow(),
        WorkflowSpec(
            workflow_id="schema_then_code",
            name="Schema Then Code",
            description="Emphasize schema, dtypes, shape, and sample rows before generating code.",
            prompt_strategy="schema_then_code",
            metadata={"differentiation_axis": "schema_emphasis"},
        ),
        WorkflowSpec(
            workflow_id="plan_then_code",
            name="Plan Then Code",
            description="Ask for a short plan in comments before the pandas code.",
            prompt_strategy="plan_then_code",
            uses_plan=True,
            metadata={"differentiation_axis": "planning"},
        ),
        WorkflowSpec(
            workflow_id="strict_answer_contract",
            name="Strict Answer Contract",
            description="Emphasize scalar numeric/string answers and DABench @name[value] multi-answer format.",
            prompt_strategy="strict_answer_contract",
            uses_answer_check=True,
            metadata={"differentiation_axis": "answer_contract"},
        ),
        WorkflowSpec(
            workflow_id="code_then_repair",
            name="Code Then Repair",
            description="Generate code, execute once, and ask for one repair if execution fails or ANSWER is missing.",
            prompt_strategy="direct_code",
            max_repair_attempts=1,
            uses_repair_loop=True,
            metadata={"differentiation_axis": "repair"},
        ),
        WorkflowSpec(
            workflow_id="plan_code_repair",
            name="Plan Code Repair",
            description="Plan in comments, generate code, and ask for one repair if execution fails or ANSWER is missing.",
            prompt_strategy="plan_then_code",
            max_repair_attempts=1,
            uses_plan=True,
            uses_repair_loop=True,
            metadata={"differentiation_axis": "planning_repair"},
        ),
    ]


CANDIDATE_WORKFLOWS = candidate_workflows()
