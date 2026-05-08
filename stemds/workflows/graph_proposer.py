"""LLM-backed proposal of constrained generated workflow graphs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from stemds.analysis.failures import FailureAnalysisReport
from stemds.llm import BaseLLMClient
from stemds.workflows.graph import GeneratedWorkflowSpec
from stemds.workflows.graph_validator import ALLOWED_NODE_TYPES, ValidationResult, validate_generated_workflow


@dataclass(slots=True)
class GeneratedWorkflowProposal:
    workflow: GeneratedWorkflowSpec | None
    status: str
    reason: str
    validation: ValidationResult
    proposal_prompt: str
    raw_response: str
    raw_item: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow.to_dict() if self.workflow is not None else None,
            "status": self.status,
            "reason": self.reason,
            "validation": self.validation.to_dict(),
            "proposal_prompt": self.proposal_prompt,
            "raw_response": self.raw_response,
            "raw_item": self.raw_item,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GeneratedWorkflowProposal":
        workflow_payload = payload.get("workflow")
        validation_payload = payload.get("validation", {})
        return cls(
            workflow=GeneratedWorkflowSpec.from_dict(workflow_payload) if workflow_payload else None,
            status=str(payload.get("status", "invalid")),
            reason=str(payload.get("reason", "")),
            validation=ValidationResult(
                valid=bool(validation_payload.get("valid", False)),
                errors=list(validation_payload.get("errors", [])),
                warnings=list(validation_payload.get("warnings", [])),
            ),
            proposal_prompt=str(payload.get("proposal_prompt", "")),
            raw_response=str(payload.get("raw_response", "")),
            raw_item=payload.get("raw_item"),
        )


class GeneratedWorkflowProposer:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        model: str,
        max_candidates: int = 3,
        temperature: float = 0.2,
        seed: int | None = 42,
        primitive_descriptions: dict[str, str] | None = None,
        reserved_workflow_ids: set[str] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.max_candidates = max_candidates
        self.temperature = temperature
        self.seed = seed
        self.primitive_descriptions = primitive_descriptions or default_primitive_descriptions()
        self.reserved_workflow_ids = reserved_workflow_ids or {"direct_code"}

    def propose(
        self,
        report: FailureAnalysisReport,
        previous_rejected_workflows: list[dict[str, Any]] | None = None,
    ) -> list[GeneratedWorkflowProposal]:
        prompt = self._build_prompt(report, previous_rejected_workflows or [])
        raw_response = self.llm_client.generate_text(
            prompt,
            model=self.model,
            temperature=self.temperature,
            seed=self.seed,
        )
        try:
            payload = _extract_json_payload(raw_response)
            items = _workflow_items(payload)
        except Exception as exc:
            return [
                GeneratedWorkflowProposal(
                    workflow=None,
                    status="invalid",
                    reason=f"Malformed workflow proposal response: {exc}",
                    validation=ValidationResult(valid=False, errors=[str(exc)], warnings=[]),
                    proposal_prompt=prompt,
                    raw_response=raw_response,
                    raw_item=None,
                )
            ]

        proposals: list[GeneratedWorkflowProposal] = []
        seen_ids: set[str] = set()
        valid_count = 0
        for item in items:
            try:
                workflow = GeneratedWorkflowSpec.from_dict(item)
            except Exception as exc:
                proposals.append(
                    GeneratedWorkflowProposal(
                        workflow=None,
                        status="invalid",
                        reason=f"Malformed workflow spec: {exc}",
                        validation=ValidationResult(valid=False, errors=[str(exc)], warnings=[]),
                        proposal_prompt=prompt,
                        raw_response=raw_response,
                        raw_item=item,
                    )
                )
                continue

            validation = validate_generated_workflow(workflow)
            reason = "Workflow graph parsed and validated."
            status = "valid" if validation.valid else "invalid"
            if workflow.workflow_id in self.reserved_workflow_ids:
                validation.errors.append(f"workflow_id is reserved: {workflow.workflow_id}")
                validation.valid = False
                status = "invalid"
                reason = "Duplicate or reserved workflow id."
            elif workflow.workflow_id in seen_ids:
                validation.errors.append(f"duplicate generated workflow id: {workflow.workflow_id}")
                validation.valid = False
                status = "invalid"
                reason = "Duplicate generated workflow id."
            elif not validation.valid:
                reason = "Workflow failed structural validation."
            else:
                valid_count += 1
                seen_ids.add(workflow.workflow_id)

            proposals.append(
                GeneratedWorkflowProposal(
                    workflow=workflow,
                    status=status,
                    reason=reason,
                    validation=validation,
                    proposal_prompt=prompt,
                    raw_response=raw_response,
                    raw_item=item,
                )
            )
            if valid_count >= self.max_candidates:
                break
        return proposals

    def _build_prompt(
        self,
        report: FailureAnalysisReport,
        previous_rejected_workflows: list[dict[str, Any]],
    ) -> str:
        examples = []
        for category, category_examples in report.top_examples_by_category.items():
            for example in category_examples[:2]:
                examples.append(
                    {
                        "category": category,
                        "task_id": example.get("task_id"),
                        "status": example.get("status"),
                        "expected_answer": example.get("expected_answer"),
                        "predicted_answer": example.get("predicted_answer"),
                        "error_message": example.get("error_message"),
                    }
                )
        return f"""You generate safe workflow graphs for a Python/pandas data-analysis agent.

Return JSON only. Do not include markdown.
Propose at most {self.max_candidates} candidate workflows.

The system will validate and execute only this DSL. You must not propose arbitrary Python orchestrator code.

Allowed node types and meanings:
{json.dumps(self.primitive_descriptions, indent=2)}

Budget constraints:
- max_llm_calls must be <= 5
- max_repairs must be <= 2
- no unbounded loops
- the only allowed cycle is python_execute -> llm_repair -> python_execute, bounded by max_repairs

Failure categories from training:
{json.dumps(report.failures_by_category, indent=2)}

Common failure tags:
{json.dumps(report.failures_by_tag, indent=2)}

Example failures:
{json.dumps(examples, indent=2)}

Previous rejected workflow summaries:
{json.dumps(previous_rejected_workflows, indent=2)}

Minimal valid workflow example:
{json.dumps(_minimal_valid_workflow_example(), indent=2)}

Each workflow must include:
- workflow_id
- name
- description
- nodes
- edges
- limits
- metadata

Return shape:
{{"workflows": [{{...}}]}}
"""


def default_primitive_descriptions() -> dict[str, str]:
    return {
        "schema_summary": "Inspect CSV shape, dtypes, columns, and sample rows.",
        "llm_plan": "Ask the model for a short task-specific analysis plan.",
        "llm_code": "Ask the model for pandas code assigning final answer to ANSWER.",
        "python_execute": "Run generated code in the existing sandbox and capture ANSWER.",
        "llm_repair": "Ask the model to repair failed code using stdout/stderr/status.",
        "answer_normalize": "No-op/TODO in v0; reserved for deterministic post-processing.",
        "llm_answer_check": "No-op/TODO in v0; reserved for bounded answer checking.",
        "stop": "Terminate workflow and return final generated code.",
    }


def _extract_json_payload(raw_response: str) -> Any:
    stripped = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    return json.loads(stripped)


def _workflow_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("workflows"), list):
        return [item for item in payload["workflows"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Workflow proposal must be a list or an object with a workflows list")


def _minimal_valid_workflow_example() -> dict[str, Any]:
    return {
        "workflow_id": "example_plan_code_execute",
        "name": "Example Plan Code Execute",
        "description": "Inspect schema, plan, generate code, execute, then stop.",
        "nodes": [
            {"id": "schema", "type": "schema_summary"},
            {"id": "plan", "type": "llm_plan", "prompt_strategy": "concise_plan"},
            {"id": "code", "type": "llm_code", "prompt_strategy": "plan_then_code"},
            {"id": "execute", "type": "python_execute"},
            {"id": "stop", "type": "stop"},
        ],
        "edges": [["schema", "plan"], ["plan", "code"], ["code", "execute"], ["execute", "stop"]],
        "limits": {"max_llm_calls": 2, "max_repairs": 0, "timeout_sec": 30},
        "metadata": {"generated_reason": "minimal valid example"},
    }
