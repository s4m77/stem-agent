"""Candidate workflows for the mini ML-engineering domain."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class MLWorkflowSpec:
    workflow_id: str
    name: str
    description: str
    prompt_strategy: str
    uses_preprocessing: bool = False
    compares_models: bool = False
    uses_repair_loop: bool = False
    max_repair_attempts: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "MLWorkflowSpec":
        return cls(
            workflow_id=str(payload["workflow_id"]),
            name=str(payload["name"]),
            description=str(payload["description"]),
            prompt_strategy=str(payload["prompt_strategy"]),
            uses_preprocessing=bool(payload.get("uses_preprocessing", False)),
            compares_models=bool(payload.get("compares_models", False)),
            uses_repair_loop=bool(payload.get("uses_repair_loop", False)),
            max_repair_attempts=int(payload.get("max_repair_attempts", 0)),
        )

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "MLWorkflowSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def direct_ml_workflow() -> MLWorkflowSpec:
    return MLWorkflowSpec(
        workflow_id="ml_direct",
        name="ML Direct",
        description="One-shot sklearn code generation.",
        prompt_strategy="ml_direct",
    )


def candidate_ml_workflows() -> list[MLWorkflowSpec]:
    return [
        direct_ml_workflow(),
        MLWorkflowSpec(
            workflow_id="preprocess_pipeline",
            name="Preprocess Pipeline",
            description="Use imputation/scaling where appropriate and a simple sklearn pipeline.",
            prompt_strategy="preprocess_pipeline",
            uses_preprocessing=True,
        ),
        MLWorkflowSpec(
            workflow_id="compare_models",
            name="Compare Models",
            description="Try two simple fast sklearn models and report the better test metric.",
            prompt_strategy="compare_models",
            compares_models=True,
        ),
        MLWorkflowSpec(
            workflow_id="ml_code_then_repair",
            name="ML Code Then Repair",
            description="Generate code and repair once if execution fails or RESULT is invalid.",
            prompt_strategy="ml_direct",
            uses_repair_loop=True,
            max_repair_attempts=1,
        ),
        MLWorkflowSpec(
            workflow_id="preprocess_compare_repair",
            name="Preprocess Compare Repair",
            description="Use preprocessing, compare simple models, and repair once if needed.",
            prompt_strategy="preprocess_compare",
            uses_preprocessing=True,
            compares_models=True,
            uses_repair_loop=True,
            max_repair_attempts=1,
        ),
    ]
