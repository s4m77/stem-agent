"""Command-line interface for StemDS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from stemds.agents.baseline import DummyBaselineAgent, OpenAIBaselineAgent
from stemds.config import DEFAULT_MODEL, DEFAULT_SANDBOX_TIMEOUT_SEC
from stemds.llm import OpenAIResponsesClient
from stemds.metrics import TaskEvalResult, aggregate_metrics, compare_answers
from stemds.sandbox import PythonSandbox
from stemds.tasks import DataAnalysisTask, load_tasks_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stemds")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-data", help="Validate a JSONL task file.")
    validate_parser.add_argument("--data", required=True, help="Path to task JSONL file.")
    validate_parser.set_defaults(func=_cmd_validate_data)

    baseline_parser = subparsers.add_parser("run-baseline", help="Run a baseline agent.")
    baseline_parser.add_argument("--data", required=True, help="Path to task JSONL file.")
    baseline_parser.add_argument("--agent", choices=["dummy", "openai"], default="dummy")
    baseline_parser.add_argument("--model", default=DEFAULT_MODEL)
    baseline_parser.add_argument("--out", required=True, help="Path to output JSON file.")
    baseline_parser.add_argument("--timeout", type=float, default=DEFAULT_SANDBOX_TIMEOUT_SEC)
    baseline_parser.set_defaults(func=_cmd_run_baseline)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def _cmd_validate_data(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    tasks = load_tasks_jsonl(data_path)
    missing_datasets = [task.dataset_path for task in tasks if not Path(task.dataset_path).exists()]
    if missing_datasets:
        for dataset_path in missing_datasets:
            print(f"Missing dataset: {dataset_path}")
        return 1
    print(f"Validated {len(tasks)} tasks from {data_path}")
    return 0


def _cmd_run_baseline(args: argparse.Namespace) -> int:
    tasks = load_tasks_jsonl(args.data)
    agent = _create_agent(args)
    sandbox = PythonSandbox(timeout_sec=args.timeout)
    results = [_evaluate_task(task, agent, sandbox) for task in tasks]
    payload = {
        "results": [result.to_dict() for result in results],
        "metrics": aggregate_metrics(results),
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    return 0


def _create_agent(args: argparse.Namespace) -> DummyBaselineAgent | OpenAIBaselineAgent:
    if args.agent == "dummy":
        return DummyBaselineAgent()
    client = OpenAIResponsesClient()
    return OpenAIBaselineAgent(model=args.model, llm_client=client)


def _evaluate_task(
    task: DataAnalysisTask,
    agent: DummyBaselineAgent | OpenAIBaselineAgent,
    sandbox: PythonSandbox,
) -> TaskEvalResult:
    agent_output = agent.solve(task)
    sandbox_result = sandbox.run(agent_output.code, task.dataset_path)
    predicted_answer = sandbox_result.extracted_answer
    invalid_code = sandbox_result.status != "success" or predicted_answer is None
    correct = compare_answers(task.answer, predicted_answer, task.answer_type, task.tolerance)
    return TaskEvalResult(
        task_id=task.task_id,
        expected_answer=task.answer,
        predicted_answer=predicted_answer,
        correct=correct,
        invalid_code=invalid_code,
        sandbox_status=sandbox_result.status,
        llm_calls=agent_output.llm_calls,
        duration_sec=sandbox_result.duration_sec,
    )


if __name__ == "__main__":
    raise SystemExit(main())

