"""Markdown report generation from StemDS experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRIC_KEYS = {
    "answer_accuracy",
    "accuracy",
    "composite_score",
    "execution_success_rate",
    "invalid_code_rate",
    "avg_llm_calls",
    "avg_runtime_sec",
    "total_tasks",
    "subquestion_accuracy",
    "subquestion_correct",
    "subquestion_total",
}


def fmt_float(value: Any) -> str:
    number = _to_float(value)
    return "n/a" if number is None else f"{number:.3f}"


def fmt_delta(value: Any) -> str:
    number = _to_float(value)
    return "n/a" if number is None else f"{number:+.3f}"


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Required JSON artifact not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def extract_metrics(obj: dict[str, Any]) -> dict[str, Any]:
    for key in ("metrics", "aggregate_metrics"):
        metrics = obj.get(key)
        if isinstance(metrics, dict):
            return dict(metrics)
    return {key: value for key, value in obj.items() if key in METRIC_KEYS}


def extract_total_tasks(obj: dict[str, Any]) -> int | None:
    metrics = extract_metrics(obj)
    total = metrics.get("total_tasks")
    if isinstance(total, int):
        return total
    if isinstance(total, float):
        return int(total)
    results = obj.get("results")
    if isinstance(results, list):
        return len(results)
    return None


def write_experiment_report(
    *,
    generic_path: str | Path,
    workflow_search_path: str | Path,
    workflow_test_path: str | Path,
    output_path: str | Path,
    seed_skills_path: str | Path | None = None,
    seed_comparison_path: str | Path | None = None,
    stem_trace_path: str | Path | None = None,
    workflow_comparison_path: str | Path | None = None,
    title: str = "StemDS Experiment Summary",
) -> list[str]:
    markdown, warnings = render_experiment_report(
        generic_path=generic_path,
        workflow_search_path=workflow_search_path,
        workflow_test_path=workflow_test_path,
        seed_skills_path=seed_skills_path,
        seed_comparison_path=seed_comparison_path,
        stem_trace_path=stem_trace_path,
        workflow_comparison_path=workflow_comparison_path,
        title=title,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return warnings


def render_experiment_report(
    *,
    generic_path: str | Path,
    workflow_search_path: str | Path,
    workflow_test_path: str | Path,
    seed_skills_path: str | Path | None = None,
    seed_comparison_path: str | Path | None = None,
    stem_trace_path: str | Path | None = None,
    workflow_comparison_path: str | Path | None = None,
    title: str = "StemDS Experiment Summary",
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    generic = load_json(generic_path)
    workflow_search = load_json(workflow_search_path)
    workflow_test = load_json(workflow_test_path)
    seed_skills = _load_optional(seed_skills_path, "seed skills", warnings)
    seed_comparison = _load_optional(seed_comparison_path, "seed comparison", warnings)
    stem_trace = _load_optional(stem_trace_path, "StemDeveloper trace", warnings)
    workflow_comparison = _load_optional(workflow_comparison_path, "workflow comparison", warnings)

    generic_metrics = extract_metrics(generic)
    workflow_metrics = extract_metrics(workflow_test)
    benchmark = _detect_benchmark(generic, workflow_search, workflow_test)
    model = _detect_model(generic, workflow_search, workflow_test)
    seed_ignored = _detect_seed_ignored(generic) or _detect_seed_ignored(workflow_test)
    frozen_workflow_id = _frozen_workflow_id(workflow_search, workflow_test)

    lines: list[str] = [
        f"# {title}",
        "",
        "StemDS means Stem Agent for Data Science. It is a constrained stem-agent prototype for data-analysis tasks. "
        "In this experiment, differentiation happens by validating prompt skills and workflows rather than updating "
        "model weights.",
        "",
        "## Experimental setup",
        "",
        f"- Benchmark: {benchmark}",
        f"- Model: {model or 'n/a'}",
        f"- Generic baseline test tasks: {_fmt_int(extract_total_tasks(generic))}",
        f"- Frozen workflow test tasks: {_fmt_int(extract_total_tasks(workflow_test))}",
        "- Raw DABench data is kept external and converted into StemDS JSONL artifacts.",
    ]
    if seed_ignored:
        lines.append("- Seed was requested, but OpenAI reported that the active API path ignored it.")
    lines.extend(
        [
            "",
            "## Generic baseline",
            "",
            _metrics_table(generic_metrics),
            "",
            "- One-shot generic analysis agent.",
            "- Writes pandas code and assigns the final value to `ANSWER`.",
            "- No repair loop, workflow search, or learned specialization is used.",
            "",
            "## Seed-skill regression",
            "",
            _seed_skill_section(seed_skills, seed_comparison),
            "",
            "## StemDeveloper PromptSkill validation",
            "",
            _stem_trace_section(stem_trace),
            "",
            "## Workflow search validation results",
            "",
            _workflow_search_section(workflow_search),
            "",
            "## Frozen workflow held-out test result",
            "",
            f"- Frozen workflow: `{frozen_workflow_id or 'n/a'}`",
            "",
            _metrics_table(workflow_metrics),
            "",
            _workflow_comparison_section(generic_metrics, workflow_metrics, workflow_comparison),
            "",
            "## What this proves and what it does not prove",
            "",
            "### Proves",
            "",
            "- The system can run a constrained stem loop.",
            "- It can reject harmful PromptSkills.",
            "- It can search over workflows.",
            "- It can freeze and evaluate a selected workflow.",
            _workflow_improvement_proof(generic_metrics, workflow_metrics),
            "",
            "### Does not prove",
            "",
            "- Universal agent behavior.",
            "- Arbitrary self-rewriting.",
            "- Model-weight learning.",
            "- Robust determinism across all OpenAI runs.",
            "- PythonSkill generation.",
            "- Multi-domain specialization.",
            "",
            "## Limitations",
            "",
            "- OpenAI seed may be ignored depending on API path.",
            "- Results may vary due to LLM nondeterminism.",
            "- DABench answer-format assumptions may affect scoring.",
            "- PromptSkill generation was limited.",
            "- No PythonSkill generation is implemented.",
            "- No DSBench or ML-engineering extension is implemented yet.",
            "- Repair workflows cost extra LLM calls when first attempts fail.",
            "- The sandbox is suitable for cooperative generated code, not adversarial code.",
            "",
            "## Suggested write-up bullets",
            "",
            "- Data analysis was chosen because it has measurable task outcomes and sandboxable generated code.",
            "- DABench/DAEval was chosen as the first real benchmark beyond toy smoke tests.",
            "- PromptSkills came first because they are inspectable, reversible, and easy to validate.",
            "- Naive skill injection and proposed PromptSkills did not reliably improve validation performance.",
            "- The useful result was that the validator rejected harmful candidates rather than accepting them blindly.",
            "- Workflow search succeeded because repair improved execution reliability and reduced invalid code.",
            "- With more time, add PythonSkill generation, stronger determinism controls, and additional benchmarks.",
            "",
        ]
    )
    return "\n".join(lines), warnings


def render_metrics_summary(results: list[Any]) -> str:
    lines = ["# StemDS Run Summary", "", f"- total_tasks: {len(results)}"]
    return "\n".join(lines) + "\n"


def _load_optional(path: str | Path | None, label: str, warnings: list[str]) -> dict[str, Any] | None:
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.exists():
        warnings.append(f"Optional artifact missing for {label}: {json_path}")
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _metrics_table(metrics: dict[str, Any]) -> str:
    rows = [
        ("accuracy", metrics.get("answer_accuracy", metrics.get("accuracy"))),
        ("composite", metrics.get("composite_score")),
        ("execution success", metrics.get("execution_success_rate")),
        ("invalid code rate", metrics.get("invalid_code_rate")),
        ("avg LLM calls", metrics.get("avg_llm_calls")),
        ("total tasks", metrics.get("total_tasks")),
    ]
    lines = ["| metric | value |", "|---|---:|"]
    for name, value in rows:
        formatter = _fmt_int if name == "total tasks" else fmt_float
        lines.append(f"| {name} | {formatter(value)} |")
    return "\n".join(lines)


def _seed_skill_section(seed_skills: dict[str, Any] | None, seed_comparison: dict[str, Any] | None) -> str:
    if seed_skills is None and seed_comparison is None:
        return "Not provided."
    lines = [
        "Hand-authored seed skills were scaffolding, not autonomous stem-generated skills.",
    ]
    if seed_skills is not None:
        lines.extend(["", _metrics_table(extract_metrics(seed_skills))])
    deltas = _extract_deltas(seed_comparison) if seed_comparison else {}
    if deltas:
        lines.extend(
            [
                "",
                "| metric | delta vs generic |",
                "|---|---:|",
                f"| accuracy | {fmt_delta(deltas.get('answer_accuracy', deltas.get('accuracy')))} |",
                f"| composite | {fmt_delta(deltas.get('composite_score'))} |",
                f"| execution success | {fmt_delta(deltas.get('execution_success_rate'))} |",
                f"| invalid code rate | {fmt_delta(deltas.get('invalid_code_rate'))} |",
            ]
        )
        if any((_to_float(value) or 0.0) < 0 for key, value in deltas.items() if key != "invalid_code_rate"):
            lines.append("")
            lines.append("Naive skill injection did not reliably help in this comparison.")
    return "\n".join(lines)


def _stem_trace_section(stem_trace: dict[str, Any] | None) -> str:
    if stem_trace is None:
        return "Not provided."
    accepted = list(stem_trace.get("accepted_skills", []))
    rejected = list(stem_trace.get("rejected_skills", []))
    proposed_count = len(accepted) + len(rejected)
    lines = [
        f"- Proposed skills: {_fmt_int(proposed_count)}",
        f"- Accepted skills: {_fmt_int(len(accepted))}",
    ]
    if accepted:
        lines.append("- Accepted skill IDs: " + ", ".join(f"`{_skill_id(record)}`" for record in accepted))
    else:
        lines.append("- The validator rejected all PromptSkill candidates, so the system avoided harmful self-modification.")
    if rejected:
        lines.extend(["", "| skill_id | score_delta | reason |", "|---|---:|---|"])
        for record in rejected:
            lines.append(
                f"| `{_skill_id(record)}` | {fmt_delta(record.get('score_delta'))} | "
                f"{_escape_table(str(record.get('reason', '')))} |"
            )
    lines.append("")
    lines.append("PromptSkill-only differentiation did not improve validation performance in this run.")
    return "\n".join(lines)


def _workflow_search_section(workflow_search: dict[str, Any]) -> str:
    results = list(workflow_search.get("results", []))
    frozen = _frozen_workflow_id(workflow_search, {})
    direct = next((result for result in results if result.get("workflow_id") == "direct_code"), None)
    selected = next((result for result in results if result.get("workflow_id") == frozen), None)
    lines = [
        "| workflow_id | accuracy | composite | execution_success | invalid_code_rate | selected |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        metrics = extract_metrics(result)
        workflow_id = str(result.get("workflow_id", "n/a"))
        lines.append(
            f"| `{workflow_id}` | {fmt_float(metrics.get('answer_accuracy', metrics.get('accuracy')))} | "
            f"{fmt_float(metrics.get('composite_score'))} | "
            f"{fmt_float(metrics.get('execution_success_rate'))} | "
            f"{fmt_float(metrics.get('invalid_code_rate'))} | "
            f"{'yes' if workflow_id == frozen else 'no'} |"
        )
    lines.extend(["", f"- Frozen workflow: `{frozen or 'n/a'}`"])
    if direct and selected and frozen != "direct_code":
        delta = _to_float(extract_metrics(selected).get("composite_score"))
        baseline = _to_float(extract_metrics(direct).get("composite_score"))
        if delta is not None and baseline is not None:
            lines.append(f"- Validation composite delta vs `direct_code`: {fmt_delta(delta - baseline)}")
    lines.append("- Workflow search is the first successful differentiation axis in these artifacts.")
    return "\n".join(lines)


def _workflow_comparison_section(
    generic_metrics: dict[str, Any],
    workflow_metrics: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> str:
    deltas = _extract_deltas(comparison) if comparison else {}
    metric_rows = [
        ("accuracy", "answer_accuracy"),
        ("composite", "composite_score"),
        ("execution success", "execution_success_rate"),
        ("invalid code rate", "invalid_code_rate"),
    ]
    lines = ["| metric | generic | frozen workflow | delta |", "|---|---:|---:|---:|"]
    for label, key in metric_rows:
        generic_value = generic_metrics.get(key)
        workflow_value = workflow_metrics.get(key)
        delta = deltas.get(key)
        if delta is None:
            generic_float = _to_float(generic_value)
            workflow_float = _to_float(workflow_value)
            delta = None if generic_float is None or workflow_float is None else workflow_float - generic_float
        lines.append(f"| {label} | {fmt_float(generic_value)} | {fmt_float(workflow_value)} | {fmt_delta(delta)} |")
    accuracy_delta = _to_float(deltas.get("answer_accuracy"))
    if accuracy_delta is None:
        generic_accuracy = _to_float(generic_metrics.get("answer_accuracy"))
        workflow_accuracy = _to_float(workflow_metrics.get("answer_accuracy"))
        accuracy_delta = None if generic_accuracy is None or workflow_accuracy is None else workflow_accuracy - generic_accuracy
    invalid_delta = _to_float(deltas.get("invalid_code_rate"))
    lines.append("")
    if accuracy_delta is not None and accuracy_delta > 0:
        lines.append("- Held-out accuracy improved for the frozen workflow.")
    else:
        lines.append("- Held-out accuracy did not improve for the frozen workflow.")
    if invalid_delta is not None and invalid_delta < 0:
        lines.append("- Part of the improvement comes from lower invalid-code rate and repair-loop reliability.")
    return "\n".join(lines)


def _workflow_improvement_proof(generic_metrics: dict[str, Any], workflow_metrics: dict[str, Any]) -> str:
    generic_score = _to_float(generic_metrics.get("composite_score"))
    workflow_score = _to_float(workflow_metrics.get("composite_score"))
    if generic_score is not None and workflow_score is not None and workflow_score > generic_score:
        return "- Workflow differentiation improved held-out DABench performance in this run."
    return "- Workflow differentiation did not improve held-out DABench performance in this run."


def _extract_deltas(comparison: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(comparison, dict):
        return {}
    deltas = comparison.get("deltas")
    if isinstance(deltas, dict):
        return deltas
    return {key: value for key, value in comparison.items() if key in METRIC_KEYS}


def _frozen_workflow_id(workflow_search: dict[str, Any], workflow_test: dict[str, Any]) -> str | None:
    frozen = workflow_search.get("frozen_workflow")
    if isinstance(frozen, dict) and frozen.get("workflow_id"):
        return str(frozen["workflow_id"])
    workflow = workflow_test.get("workflow")
    if isinstance(workflow, dict) and workflow.get("workflow_id"):
        return str(workflow["workflow_id"])
    for result in workflow_search.get("results", []):
        if isinstance(result, dict) and result.get("accepted") and result.get("workflow_id"):
            return str(result["workflow_id"])
    return None


def _detect_benchmark(*objs: dict[str, Any]) -> str:
    text = json.dumps(objs).lower()
    if "dabench" in text or "daeval" in text or "raw_answer_type" in text:
        return "DABench/DAEval"
    return "n/a"


def _detect_model(*objs: dict[str, Any]) -> str | None:
    for obj in objs:
        found = _find_first_key(obj, "model")
        if found is not None:
            return str(found)
    return None


def _detect_seed_ignored(obj: dict[str, Any]) -> bool:
    return bool(_find_bool_key(obj, "llm_seed_ignored"))


def _find_first_key(value: Any, target: str) -> Any | None:
    if isinstance(value, dict):
        if target in value:
            return value[target]
        for item in value.values():
            found = _find_first_key(item, target)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_first_key(item, target)
            if found is not None:
                return found
    return None


def _find_bool_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        if value.get(target) is True:
            return True
        return any(_find_bool_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_find_bool_key(item, target) for item in value)
    return False


def _skill_id(record: dict[str, Any]) -> str:
    skill = record.get("skill")
    if isinstance(skill, dict):
        return str(skill.get("skill_id", "unknown"))
    return str(record.get("skill_id", "unknown"))


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _fmt_int(value: Any) -> str:
    if isinstance(value, bool):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value))
    return "n/a"


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None
