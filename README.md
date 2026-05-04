# StemDS

StemDS is a small command-line proof of concept for a constrained stem agent. The first target domain is data analysis: given a CSV and a natural-language question, an agent writes Python/pandas code, executes it in a subprocess sandbox, and returns a measured answer.

The current skeleton provides the runnable foundation for constrained stem differentiation: task schemas, toy data, DABench/DAEval adapter infrastructure, sandbox execution, answer metrics, baseline agents, CLI commands, seed skill metadata, PromptSkill validation, workflow search, and tests.

## Setup

```bash
uv sync --python 3.12 --extra dev
```

For OpenAI-backed commands, put `OPENAI_API_KEY=...` in the repo-root `.env` file or export it in your shell. `.env` is ignored by git. Tests and the dummy baseline do not require a key.

## Run Tests

```bash
uv run pytest
```

## Validate Data

```bash
uv run python -m stemds.cli validate-data --data data/toy_analysis_train.jsonl
```

`toy_analysis` is only a smoke test for the CLI, sandbox, and metrics. The first meaningful external benchmark target is InfiAgent-DABench/DAEval, which uses real data-analysis questions, constraints, labels, and CSV files.

## Run Dummy Baseline

```bash
uv run python -m stemds.cli run-baseline --data data/toy_analysis_test.jsonl --agent dummy --out runs/baseline_dummy.json
```

The dummy baseline is an offline smoke test for the CLI, sandbox, and metrics. It is hardcoded for toy tasks and should not be used as the real before/after comparison.

## Run OpenAI Baseline

```bash
uv run python -m stemds.cli run-baseline --data data/toy_analysis_test.jsonl --agent openai --model gpt-4.1-mini --out runs/baseline_openai.json
```

The OpenAI baseline requires `OPENAI_API_KEY`. Unit tests and the dummy baseline do not.

## DABench / DAEval

Clone the benchmark repository as ignored external raw data:

```bash
git clone https://github.com/InfiAgent/InfiAgent.git external/InfiAgent
```

Inspect the detected DAEval layout:

```bash
uv run python -m stemds.cli inspect-dataset --adapter dabench --root external/InfiAgent
```

Convert the public DAEval records to StemDS JSONL:

```bash
uv run python -m stemds.cli convert-dataset --adapter dabench --root external/InfiAgent --out data/dabench/dabench_tasks.jsonl
```

Create deterministic train/val/test splits:

```bash
uv run python -m stemds.cli split-data --data data/dabench/dabench_tasks.jsonl --out-dir data/dabench --train-frac 0.7 --val-frac 0.15 --seed 42
```

Run a small OpenAI baseline smoke test on the converted test split:

```bash
uv run python -m stemds.cli run-baseline --data data/dabench/dabench_test.jsonl --agent openai --model gpt-4.1-mini --limit 10 --out runs/dabench_openai_baseline_10.json
```

Analyze failures from a run:

```bash
uv run python -m stemds.cli analyze-run --run runs/dabench_openai_baseline_10.json --out runs/dabench_openai_baseline_10_analysis.json --markdown-out runs/dabench_openai_baseline_10_analysis.md
```

## Seed Skills

Seed skills are hand-authored prompt scaffolding, not autonomous stem-generated skills. They give the next baseline reusable instructions for common DABench failure modes while keeping the rest of StemDS dataset-agnostic.

Initialize the seed skill library:

```bash
uv run python -m stemds.cli init-seed-skills --out skills/seed
```

Run the skill-augmented OpenAI baseline:

```bash
uv run python -m stemds.cli run-baseline --data data/dabench/dabench_test.jsonl --agent skill_openai --skills skills/seed --model gpt-4.1-mini --limit 10 --out runs/dabench_skill_seed_10.json
```

Compare the generic and skill-augmented runs:

```bash
uv run python -m stemds.cli compare-runs --a runs/dabench_openai_baseline_10.json --b runs/dabench_skill_seed_10.json --out runs/dabench_openai_vs_skill_seed_10.json --markdown-out runs/dabench_openai_vs_skill_seed_10.md
```

The next milestone is `StemDeveloper`: use failure analysis to propose candidate `PromptSkill`s, validate them on a held-out split, and accept only skills that improve measured performance.

## Stem Development Loop v0

Seed skills are hand-authored and can hurt performance on real DABench slices. `StemDeveloper` is the first constrained self-differentiation loop: it proposes candidate `PromptSkill`s from failure analysis, validates each candidate by ablation on a validation subset, and accepts only skills that improve measured validation composite score without materially increasing invalid code rate.

Run development:

```bash
uv run python -m stemds.cli develop --train data/dabench/dabench_train.jsonl --val data/dabench/dabench_val.jsonl --model gpt-4.1-mini --out-dir runs/stem/dev_001 --max-candidates 3 --val-limit 38 --seed 42 --min-delta 0.05
```

Evaluate the accepted skill library:

```bash
uv run python -m stemds.cli evaluate-developed --data data/dabench/dabench_test.jsonl --skills runs/stem/dev_001/accepted_skills --model gpt-4.1-mini --limit 20 --seed 42 --out runs/stem/dev_001/test_eval.json
```

This version generates only prompt skills. It does not generate Python skills, edit source code, or rewrite the StemDS repository.

## Reproducing The Headline DABench Experiment

Run the generic baseline on the fixed 40-task test split:

```bash
uv run python -m stemds.cli run-baseline \
  --data data/dabench/dabench_test.jsonl \
  --agent openai \
  --model gpt-4.1-mini \
  --limit 40 \
  --seed 42 \
  --out runs/stem/dev_002/test_generic.json
```

Develop prompt skills from train failures and validate them on the full DABench validation split:

```bash
uv run python -m stemds.cli develop \
  --train data/dabench/dabench_train.jsonl \
  --val data/dabench/dabench_val.jsonl \
  --model gpt-4.1-mini \
  --out-dir runs/stem/dev_002 \
  --max-candidates 3 \
  --val-limit 38 \
  --seed 42 \
  --min-delta 0.05
```

Evaluate the accepted skill library on the same fixed 40-task test split:

```bash
uv run python -m stemds.cli evaluate-developed \
  --data data/dabench/dabench_test.jsonl \
  --skills runs/stem/dev_002/accepted_skills \
  --model gpt-4.1-mini \
  --limit 40 \
  --seed 42 \
  --out runs/stem/dev_002/test_developed.json
```

Compare before/after results:

```bash
uv run python -m stemds.cli compare-runs \
  --a runs/stem/dev_002/test_generic.json \
  --b runs/stem/dev_002/test_developed.json \
  --out runs/stem/dev_002/generic_vs_developed.json \
  --markdown-out runs/stem/dev_002/generic_vs_developed.md
```

## Workflow Search v0

PromptSkill-only differentiation rejected all candidates in `dev_003`, which is the intended safeguard when proposed skills regress validation performance. Workflow search is the next differentiation axis: StemDS searches over small prompt/control-flow architectures, validates them on DABench validation tasks, freezes the best workflow only if it beats `direct_code`, and evaluates that frozen architecture on held-out test tasks.

Search workflow candidates on validation:

```bash
uv run python -m stemds.cli search-workflows \
  --val data/dabench/dabench_val.jsonl \
  --model gpt-4.1-mini \
  --out-dir runs/stem/dev_004 \
  --val-limit 38 \
  --seed 42 \
  --min-delta 0.03
```

Evaluate the frozen workflow on the same fixed 40-task test split:

```bash
uv run python -m stemds.cli evaluate-workflow \
  --data data/dabench/dabench_test.jsonl \
  --workflow runs/stem/dev_004/frozen_workflow.json \
  --model gpt-4.1-mini \
  --limit 40 \
  --seed 42 \
  --out runs/stem/dev_004/test_frozen_workflow.json
```

Compare generic baseline to the frozen workflow:

```bash
uv run python -m stemds.cli compare-runs \
  --a runs/stem/dev_003/test_generic.json \
  --b runs/stem/dev_004/test_frozen_workflow.json \
  --out runs/stem/dev_004/generic_vs_frozen_workflow.json \
  --markdown-out runs/stem/dev_004/generic_vs_frozen_workflow.md
```

Workflow search v0 still does not generate Python skills or modify repository code. The frozen workflow is the current specialized architecture for this benchmark slice.

## Generate Experiment Report

Generate a concise Markdown summary from existing run artifacts without making new OpenAI calls:

```bash
uv run python -m stemds.cli make-report \
  --generic runs/stem/dev_004/test_generic_rerun.json \
  --seed-skills runs/dabench_skill_seed_10.json \
  --seed-comparison runs/dabench_openai_vs_skill_seed_10.json \
  --stem-trace runs/stem/dev_003/development_trace.json \
  --workflow-search runs/stem/dev_004/workflow_search_results.json \
  --workflow-test runs/stem/dev_004/test_frozen_workflow.json \
  --workflow-comparison runs/stem/dev_004/generic_rerun_vs_frozen_workflow.json \
  --out reports/stemds_experiment_summary.md
```

The report is intended for write-up drafting. It summarizes setup, generic baseline behavior, seed-skill regression, PromptSkill validation, workflow-search validation, frozen-workflow test performance, limitations, and suggested write-up bullets.
