"""PromptSkill validation by ablation on held-out tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from stemds.agents.baseline import OpenAIGenericAnalysisAgent, SkillAugmentedAnalysisAgent
from stemds.llm import BaseLLMClient
from stemds.metrics import TaskEvalResult, aggregate_metrics
from stemds.sandbox import PythonSandbox
from stemds.skills.base import PromptSkill
from stemds.skills.library import SkillLibrary
from stemds.tasks import load_tasks_jsonl


@dataclass(slots=True)
class SkillValidationResult:
    accepted: bool
    reason: str
    baseline_score: float
    trial_score: float
    score_delta: float
    baseline_invalid_code_rate: float
    trial_invalid_code_rate: float
    baseline_result_path: str
    trial_result_path: str


EvaluationFunction = Callable[[SkillLibrary, Path, str, Path, int | None, int | None], dict]


class SkillValidator:
    def __init__(
        self,
        llm_client: BaseLLMClient | None,
        model: str,
        val_data: Path,
        out_dir: Path,
        val_limit: int | None = None,
        min_delta: float = 0.05,
        max_invalid_rate_increase: float = 0.10,
        seed: int | None = 42,
        evaluate_fn: EvaluationFunction | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.val_data = val_data
        self.out_dir = out_dir
        self.val_limit = val_limit
        self.min_delta = min_delta
        self.max_invalid_rate_increase = max_invalid_rate_increase
        self.seed = seed
        self.evaluate_fn = evaluate_fn or self._evaluate_library
        self._current_cache: dict[tuple[str, ...], tuple[dict, Path]] = {}

    def validate(self, candidate: PromptSkill, current_library: SkillLibrary) -> SkillValidationResult:
        trial_path = self.out_dir / f"candidate_{candidate.skill_id}_val.json"
        baseline_payload, current_path = self._current_library_payload(current_library)
        trial_library = SkillLibrary([*current_library.skills, candidate])
        if trial_path.exists():
            trial_payload = json.loads(trial_path.read_text(encoding="utf-8"))
        else:
            trial_payload = self.evaluate_fn(
                trial_library,
                self.val_data,
                self.model,
                trial_path,
                self.val_limit,
                self.seed,
            )
        baseline_metrics = baseline_payload["metrics"]
        trial_metrics = trial_payload["metrics"]
        baseline_score = float(baseline_metrics.get("composite_score", 0.0))
        trial_score = float(trial_metrics.get("composite_score", 0.0))
        score_delta = trial_score - baseline_score
        baseline_invalid = float(baseline_metrics.get("invalid_code_rate", 0.0))
        trial_invalid = float(trial_metrics.get("invalid_code_rate", 0.0))
        invalid_delta = trial_invalid - baseline_invalid
        accepted = score_delta > self.min_delta and invalid_delta <= self.max_invalid_rate_increase
        if accepted:
            reason = f"Accepted: composite improved by {score_delta:.3f}."
        elif score_delta <= self.min_delta:
            reason = f"Rejected: composite delta {score_delta:.3f} did not exceed min_delta {self.min_delta:.3f}."
        else:
            reason = (
                f"Rejected: invalid_code_rate increased by {invalid_delta:.3f}, "
                f"above tolerance {self.max_invalid_rate_increase:.3f}."
            )
        return SkillValidationResult(
            accepted=accepted,
            reason=reason,
            baseline_score=baseline_score,
            trial_score=trial_score,
            score_delta=score_delta,
            baseline_invalid_code_rate=baseline_invalid,
            trial_invalid_code_rate=trial_invalid,
            baseline_result_path=str(current_path),
            trial_result_path=str(trial_path),
        )

    def invalidate_current_cache(self) -> None:
        self._current_cache.clear()

    def _current_library_payload(self, current_library: SkillLibrary) -> tuple[dict, Path]:
        key = tuple(skill.skill_id for skill in current_library.skills)
        if key not in self._current_cache:
            current_path = self.out_dir / f"current_library_{_library_key_name(key)}_val.json"
            if current_path.exists():
                payload = json.loads(current_path.read_text(encoding="utf-8"))
            else:
                payload = self.evaluate_fn(
                    current_library,
                    self.val_data,
                    self.model,
                    current_path,
                    self.val_limit,
                    self.seed,
                )
            self._current_cache[key] = (payload, current_path)
        return self._current_cache[key]

    def _evaluate_library(
        self,
        library: SkillLibrary,
        data_path: Path,
        model: str,
        output_path: Path,
        limit: int | None,
        seed: int | None,
    ) -> dict:
        if self.llm_client is None:
            raise ValueError("llm_client is required when no custom evaluate_fn is provided")
        tasks = load_tasks_jsonl(data_path)
        from stemds.cli import _progress_tasks, _select_tasks

        tasks = _select_tasks(tasks, limit=limit, seed=seed)
        agent = (
            SkillAugmentedAnalysisAgent(model=model, llm_client=self.llm_client, skill_library=library, seed=seed)
            if library.skills
            else OpenAIGenericAnalysisAgent(model=model, llm_client=self.llm_client, seed=seed)
        )
        sandbox = PythonSandbox()
        results = [
            _evaluate_task_local(task, agent, sandbox)
            for task in _progress_tasks(tasks, desc=f"evaluate:{output_path.stem}")
        ]
        payload = {"results": [result.to_dict() for result in results], "metrics": aggregate_metrics(results)}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload


def _evaluate_task_local(task, agent, sandbox: PythonSandbox) -> TaskEvalResult:
    from stemds.cli import _evaluate_task

    return _evaluate_task(task, agent, sandbox)


def _library_key_name(key: tuple[str, ...]) -> str:
    if not key:
        return "empty"
    return "_".join(key)
