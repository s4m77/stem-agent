"""Command-line interface for StemDS."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from tqdm import tqdm

from stemds.agents.base import BaseDataAnalysisAgent
from stemds.agents.baseline import DummyBaselineAgent, OpenAIGenericAnalysisAgent, SkillAugmentedAnalysisAgent
from stemds.analysis.failures import (
    analyze_run,
    compare_runs,
    render_comparison_markdown,
    render_failure_analysis_markdown,
)
from stemds.config import DEFAULT_MODEL, DEFAULT_SANDBOX_TIMEOUT_SEC
from stemds.datasets.dabench import DABenchAdapter
from stemds.llm import LLMClientError, OpenAIClient
from stemds.metrics import TaskEvalResult, aggregate_metrics, compare_answers, dabench_pair_counts
from stemds.reporting.report import write_experiment_report
from stemds.sandbox import PythonSandbox
from stemds.skills.library import SkillLibrary
from stemds.skills.seed import create_seed_prompt_skills
from stemds.stem.developer import StemDeveloper
from stemds.tasks import DataAnalysisTask, load_tasks_jsonl, save_tasks_jsonl
from stemds.workflows.base import WorkflowSpec
from stemds.workflows.executor import WorkflowAnalysisAgent
from stemds.workflows.search import WorkflowSearcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stemds")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-data", help="Validate a JSONL task file.")
    validate_parser.add_argument("--data", required=True, help="Path to task JSONL file.")
    validate_parser.set_defaults(func=_cmd_validate_data)

    baseline_parser = subparsers.add_parser("run-baseline", help="Run a baseline agent.")
    baseline_parser.add_argument("--data", required=True, help="Path to task JSONL file.")
    baseline_parser.add_argument("--agent", choices=["dummy", "openai", "skill_openai"], default="dummy")
    baseline_parser.add_argument("--model", default=DEFAULT_MODEL)
    baseline_parser.add_argument("--skills", default=None, help="Skill directory for --agent skill_openai.")
    baseline_parser.add_argument("--out", required=True, help="Path to output JSON file.")
    baseline_parser.add_argument("--timeout", type=float, default=DEFAULT_SANDBOX_TIMEOUT_SEC)
    baseline_parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of tasks to run.")
    baseline_parser.add_argument("--seed", type=int, default=42)
    baseline_parser.set_defaults(func=_cmd_run_baseline)

    inspect_parser = subparsers.add_parser("inspect-dataset", help="Inspect an external dataset adapter.")
    inspect_parser.add_argument("--adapter", choices=["dabench"], required=True)
    inspect_parser.add_argument("--root", required=True, help="External dataset root directory.")
    inspect_parser.set_defaults(func=_cmd_inspect_dataset)

    convert_parser = subparsers.add_parser("convert-dataset", help="Convert an external dataset to StemDS JSONL.")
    convert_parser.add_argument("--adapter", choices=["dabench"], required=True)
    convert_parser.add_argument("--root", required=True, help="External dataset root directory.")
    convert_parser.add_argument("--out", required=True, help="Output StemDS JSONL path.")
    convert_parser.set_defaults(func=_cmd_convert_dataset)

    split_parser = subparsers.add_parser("split-data", help="Split a StemDS JSONL dataset.")
    split_parser.add_argument("--data", required=True)
    split_parser.add_argument("--out-dir", required=True)
    split_parser.add_argument("--train-frac", type=float, required=True)
    split_parser.add_argument("--val-frac", type=float, required=True)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.set_defaults(func=_cmd_split_data)

    sample_parser = subparsers.add_parser("sample-data", help="Print sample StemDS JSONL tasks.")
    sample_parser.add_argument("--data", required=True)
    sample_parser.add_argument("--n", type=int, default=5)
    sample_parser.set_defaults(func=_cmd_sample_data)

    init_skills_parser = subparsers.add_parser("init-seed-skills", help="Write hand-authored seed skills.")
    init_skills_parser.add_argument("--out", required=True)
    init_skills_parser.set_defaults(func=_cmd_init_seed_skills)

    analyze_parser = subparsers.add_parser("analyze-run", help="Analyze baseline run failures.")
    analyze_parser.add_argument("--run", required=True)
    analyze_parser.add_argument("--out", required=True)
    analyze_parser.add_argument("--markdown-out", default=None)
    analyze_parser.set_defaults(func=_cmd_analyze_run)

    compare_parser = subparsers.add_parser("compare-runs", help="Compare two run JSON files.")
    compare_parser.add_argument("--a", required=True)
    compare_parser.add_argument("--b", required=True)
    compare_parser.add_argument("--out", required=True)
    compare_parser.add_argument("--markdown-out", default=None)
    compare_parser.set_defaults(func=_cmd_compare_runs)

    develop_parser = subparsers.add_parser("develop", help="Run constrained StemDeveloper v0.")
    develop_parser.add_argument("--train", required=True)
    develop_parser.add_argument("--val", required=True)
    develop_parser.add_argument("--model", default=DEFAULT_MODEL)
    develop_parser.add_argument("--out-dir", required=True)
    develop_parser.add_argument("--max-candidates", type=int, default=3)
    develop_parser.add_argument("--val-limit", type=int, default=None)
    develop_parser.add_argument("--seed", type=int, default=42)
    develop_parser.add_argument("--min-delta", type=float, default=0.05)
    develop_parser.set_defaults(func=_cmd_develop)

    eval_developed_parser = subparsers.add_parser("evaluate-developed", help="Evaluate an accepted skill library.")
    eval_developed_parser.add_argument("--data", required=True)
    eval_developed_parser.add_argument("--skills", required=True)
    eval_developed_parser.add_argument("--model", default=DEFAULT_MODEL)
    eval_developed_parser.add_argument("--limit", type=int, default=None)
    eval_developed_parser.add_argument("--seed", type=int, default=42)
    eval_developed_parser.add_argument("--out", required=True)
    eval_developed_parser.set_defaults(func=_cmd_evaluate_developed)

    search_workflows_parser = subparsers.add_parser("search-workflows", help="Search validation-set workflow candidates.")
    search_workflows_parser.add_argument("--val", required=True)
    search_workflows_parser.add_argument("--model", default=DEFAULT_MODEL)
    search_workflows_parser.add_argument("--out-dir", required=True)
    search_workflows_parser.add_argument("--val-limit", type=int, default=38)
    search_workflows_parser.add_argument("--seed", type=int, default=42)
    search_workflows_parser.add_argument("--min-delta", type=float, default=0.03)
    search_workflows_parser.set_defaults(func=_cmd_search_workflows)

    eval_workflow_parser = subparsers.add_parser("evaluate-workflow", help="Evaluate a frozen workflow.")
    eval_workflow_parser.add_argument("--data", required=True)
    eval_workflow_parser.add_argument("--workflow", required=True)
    eval_workflow_parser.add_argument("--model", default=DEFAULT_MODEL)
    eval_workflow_parser.add_argument("--limit", type=int, default=None)
    eval_workflow_parser.add_argument("--seed", type=int, default=42)
    eval_workflow_parser.add_argument("--out", required=True)
    eval_workflow_parser.set_defaults(func=_cmd_evaluate_workflow)

    report_parser = subparsers.add_parser("make-report", help="Generate a Markdown experiment report.")
    report_parser.add_argument("--generic", required=True)
    report_parser.add_argument("--workflow-search", required=True)
    report_parser.add_argument("--workflow-test", required=True)
    report_parser.add_argument("--out", required=True)
    report_parser.add_argument("--seed-skills", default=None)
    report_parser.add_argument("--seed-comparison", default=None)
    report_parser.add_argument("--stem-trace", default=None)
    report_parser.add_argument("--workflow-comparison", default=None)
    report_parser.add_argument("--title", default="StemDS Experiment Summary")
    report_parser.set_defaults(func=_cmd_make_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, LLMClientError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


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
    tasks = _select_tasks(tasks, limit=args.limit, seed=args.seed)
    agent = _create_agent(args)
    sandbox = PythonSandbox(timeout_sec=args.timeout)
    results = [
        _evaluate_task(task, agent, sandbox)
        for task in _progress_tasks(tasks, desc=f"run-baseline:{args.agent}")
    ]
    payload = {
        "results": [result.to_dict() for result in results],
        "metrics": aggregate_metrics(results),
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    return 0


def _cmd_inspect_dataset(args: argparse.Namespace) -> int:
    adapter = _create_dataset_adapter(args)
    tasks = adapter.load_tasks()
    example = tasks[0].to_dict() if tasks else None
    discovery = adapter.discovery
    payload = {
        "adapter": adapter.name,
        "questions_path": str(discovery.questions_path),
        "labels_path": str(discovery.labels_path),
        "csv_root": str(discovery.csv_root),
        "csv_count": discovery.csv_count,
        "converted_task_count": len(tasks),
        "example_task": example,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_convert_dataset(args: argparse.Namespace) -> int:
    adapter = _create_dataset_adapter(args)
    tasks = adapter.load_tasks()
    output_path = Path(args.out)
    adapter.write_jsonl(tasks, output_path)
    print(f"Converted {len(tasks)} tasks to {output_path}")
    return 0


def _cmd_split_data(args: argparse.Namespace) -> int:
    if args.train_frac <= 0 or args.val_frac < 0 or args.train_frac + args.val_frac >= 1:
        raise ValueError("--train-frac and --val-frac must be positive fractions summing to less than 1")
    tasks = load_tasks_jsonl(args.data)
    shuffled = list(tasks)
    random.Random(args.seed).shuffle(shuffled)
    train_end = int(len(shuffled) * args.train_frac)
    val_end = train_end + int(len(shuffled) * args.val_frac)
    splits = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(args.data).stem.removesuffix("_tasks")
    for split_name, split_tasks in splits.items():
        save_tasks_jsonl(split_tasks, out_dir / f"{prefix}_{split_name}.jsonl")
    print(
        json.dumps(
            {name: len(split_tasks) for name, split_tasks in splits.items()},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_sample_data(args: argparse.Namespace) -> int:
    tasks = load_tasks_jsonl(args.data)
    for task in tasks[: args.n]:
        print(json.dumps(task.to_dict(), sort_keys=True))
    return 0


def _cmd_init_seed_skills(args: argparse.Namespace) -> int:
    library = SkillLibrary(create_seed_prompt_skills())
    library.save_to_dir(args.out)
    print(f"Wrote {len(library.skills)} seed skills to {args.out}")
    return 0


def _cmd_analyze_run(args: argparse.Namespace) -> int:
    report = analyze_run(Path(args.run))
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_out:
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_failure_analysis_markdown(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.to_dict().items() if k != "failures"}, indent=2, sort_keys=True))
    return 0


def _cmd_compare_runs(args: argparse.Namespace) -> int:
    comparison = compare_runs(Path(args.a), Path(args.b))
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_out:
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_comparison_markdown(comparison), encoding="utf-8")
    print(json.dumps(comparison["deltas"], indent=2, sort_keys=True))
    return 0


def _cmd_develop(args: argparse.Namespace) -> int:
    developer = StemDeveloper(
        llm_client=OpenAIClient(),
        model=args.model,
        train_data=Path(args.train),
        val_data=Path(args.val),
        out_dir=Path(args.out_dir),
        max_candidates=args.max_candidates,
        val_limit=args.val_limit,
        seed=args.seed,
        min_delta=args.min_delta,
    )
    result = developer.develop()
    print(json.dumps(result.summary_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_evaluate_developed(args: argparse.Namespace) -> int:
    tasks = load_tasks_jsonl(args.data)
    tasks = _select_tasks(tasks, limit=args.limit, seed=args.seed)
    skills_path = Path(args.skills)
    library = SkillLibrary.load_from_dir(skills_path) if skills_path.exists() else SkillLibrary()
    if library.skills:
        agent = SkillAugmentedAnalysisAgent(
            model=args.model,
            llm_client=OpenAIClient(),
            skill_library=library,
            seed=args.seed,
        )
        mode = "skill_openai"
    else:
        agent = OpenAIGenericAnalysisAgent(model=args.model, llm_client=OpenAIClient(), seed=args.seed)
        mode = "openai_fallback_no_accepted_skills"
        print("No accepted skills found; falling back to generic OpenAI baseline.", file=sys.stderr)
    sandbox = PythonSandbox()
    results = [
        _evaluate_task(task, agent, sandbox)
        for task in _progress_tasks(tasks, desc="evaluate-developed")
    ]
    payload = {
        "mode": mode,
        "skills_path": str(skills_path),
        "results": [result.to_dict() for result in results],
        "metrics": aggregate_metrics(results),
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    return 0


def _cmd_search_workflows(args: argparse.Namespace) -> int:
    searcher = WorkflowSearcher(
        llm_client=OpenAIClient(),
        model=args.model,
        val_data=Path(args.val),
        out_dir=Path(args.out_dir),
        val_limit=args.val_limit,
        seed=args.seed,
        min_delta=args.min_delta,
    )
    outcome = searcher.search()
    print(json.dumps(outcome.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_evaluate_workflow(args: argparse.Namespace) -> int:
    workflow = WorkflowSpec.load_json(Path(args.workflow))
    tasks = load_tasks_jsonl(args.data)
    tasks = _select_tasks(tasks, limit=args.limit, seed=args.seed)
    sandbox = PythonSandbox()
    agent = WorkflowAnalysisAgent(
        workflow=workflow,
        model=args.model,
        llm_client=OpenAIClient(),
        seed=args.seed,
        sandbox=sandbox,
    )
    results = [
        _evaluate_task(task, agent, sandbox)
        for task in _progress_tasks(tasks, desc=f"evaluate-workflow:{workflow.workflow_id}")
    ]
    payload = {
        "workflow": workflow.to_dict(),
        "results": [result.to_dict() for result in results],
        "metrics": aggregate_metrics(results),
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    return 0


def _cmd_make_report(args: argparse.Namespace) -> int:
    warnings = write_experiment_report(
        generic_path=args.generic,
        seed_skills_path=args.seed_skills,
        seed_comparison_path=args.seed_comparison,
        stem_trace_path=args.stem_trace,
        workflow_search_path=args.workflow_search,
        workflow_test_path=args.workflow_test,
        workflow_comparison_path=args.workflow_comparison,
        output_path=args.out,
        title=args.title,
    )
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"Wrote report to {args.out}")
    return 0


def _create_agent(args: argparse.Namespace) -> DummyBaselineAgent | OpenAIGenericAnalysisAgent | SkillAugmentedAnalysisAgent:
    if args.agent == "dummy":
        return DummyBaselineAgent()
    client = OpenAIClient()
    if args.agent == "skill_openai":
        if not args.skills:
            raise ValueError("--skills is required when using --agent skill_openai")
        return SkillAugmentedAnalysisAgent(
            model=args.model,
            llm_client=client,
            skill_library=SkillLibrary.load_from_dir(args.skills),
            seed=args.seed,
        )
    return OpenAIGenericAnalysisAgent(model=args.model, llm_client=client, seed=args.seed)


def _create_dataset_adapter(args: argparse.Namespace) -> DABenchAdapter:
    if args.adapter == "dabench":
        return DABenchAdapter(root_dir=Path(args.root))
    raise ValueError(f"Unsupported adapter: {args.adapter}")


def _evaluate_task(
    task: DataAnalysisTask,
    agent: BaseDataAnalysisAgent,
    sandbox: PythonSandbox,
) -> TaskEvalResult:
    agent_output = agent.solve(task)
    if agent_output.metadata.get("answer_contract") == "ANSWER":
        sandbox_result = sandbox.run_generated_analysis(agent_output.code, task.dataset_path)
    else:
        sandbox_result = sandbox.run(agent_output.code, task.dataset_path)
    predicted_answer = sandbox_result.answer if sandbox_result.answer is not None else sandbox_result.extracted_answer
    invalid_code = sandbox_result.status != "success" or predicted_answer is None
    if sandbox_result.status == "pass":
        invalid_code = predicted_answer is None
    correct = compare_answers(task.answer, predicted_answer, task.answer_type, task.tolerance, metadata=task.metadata)
    subquestion_total, subquestion_correct = dabench_pair_counts(task.answer, predicted_answer, task.tolerance)
    return TaskEvalResult(
        task_id=task.task_id,
        expected_answer=task.answer,
        predicted_answer=predicted_answer,
        correct=correct,
        invalid_code=invalid_code,
        sandbox_status=sandbox_result.status,
        llm_calls=agent_output.llm_calls,
        duration_sec=sandbox_result.duration_sec,
        tags=task.tags,
        question=task.question,
        error_message=sandbox_result.stderr.strip() or None,
        stdout=sandbox_result.stdout,
        stderr=sandbox_result.stderr,
        generated_code=agent_output.code,
        metadata={
            **task.metadata,
            "agent_output": agent_output.metadata,
            "selected_skill_ids": agent_output.metadata.get("selected_skill_ids", []),
            "subquestion_total": subquestion_total,
            "subquestion_correct": subquestion_correct,
        },
    )


def _progress_tasks(tasks: Iterable[DataAnalysisTask], desc: str) -> Iterator[DataAnalysisTask]:
    return iter(tqdm(tasks, desc=desc, unit="task", file=sys.stderr, disable=not sys.stderr.isatty()))


def _select_tasks(tasks: list[DataAnalysisTask], limit: int | None, seed: int | None) -> list[DataAnalysisTask]:
    if limit is None:
        return tasks
    if limit < 1:
        raise ValueError("--limit must be a positive integer")
    if limit >= len(tasks):
        return tasks
    indexed_tasks = list(enumerate(tasks))
    random.Random(seed).shuffle(indexed_tasks)
    selected = sorted(indexed_tasks[:limit], key=lambda item: item[0])
    return [task for _index, task in selected]


if __name__ == "__main__":
    raise SystemExit(main())
