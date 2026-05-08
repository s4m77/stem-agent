# StemDS

![Python](https://img.shields.io/badge/python-3.12-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-purple)
![pytest](https://img.shields.io/badge/tests-pytest-green)
![pandas](https://img.shields.io/badge/pandas-data%20analysis-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML%20extension-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4.1--mini-black)

[Read the full challenge report](report.md): methodology, experiments, negative results, and limitations.

StemDS is a small command-line proof of concept for a constrained stem agent. The first target domain is data analysis: given a CSV and a natural-language question, an agent writes Python/pandas code, executes it in a subprocess sandbox, and returns a measured answer.

The current skeleton provides the runnable foundation for constrained stem differentiation: task schemas, toy data, DABench/DAEval adapter infrastructure, sandbox execution, answer metrics, baseline agents, CLI commands, seed skill metadata, PromptSkill validation, AI-assisted developer-curated workflow search, generated workflow-graph search, and tests.

The strongest held-out DABench result comes from selecting the human-authored `code_then_repair` workflow. Generated workflow-graph search is included as a useful negative result: generated candidates were validated and rejected when they underperformed.

## Setup

```bash
uv sync --python 3.12 --extra dev
```

For OpenAI-backed commands, put `OPENAI_API_KEY=...` in the repo-root `.env` file or export it in your shell. Tests and the dummy baseline do not require a key.

## Docker Setup

The local uv workflow remains the primary development path. Docker is available for reviewers who want an isolated environment for tests, data validation, report generation, and optional OpenAI-backed smoke runs.

Create a local `.env` file if you plan to run OpenAI-backed commands:

```bash
echo "OPENAI_API_KEY=..." > .env
```

Build the image:

```bash
docker compose build
```

Run tests:

```bash
docker compose run --rm stemds uv run pytest
```

Validate the converted DABench test split:

```bash
docker compose run --rm stemds uv run python -m stemds.cli validate-data --data data/dabench/dabench_test.jsonl
```

Generate the experiment report from existing artifacts:

```bash
docker compose run --rm stemds uv run python -m stemds.cli make-report \
  --generic runs/stem/dev_004/test_generic_rerun.json \
  --seed-skills runs/dabench_skill_seed_10.json \
  --seed-comparison runs/dabench_openai_vs_skill_seed_10.json \
  --stem-trace runs/stem/dev_003/development_trace.json \
  --workflow-search runs/stem/dev_004/workflow_search_results.json \
  --workflow-test runs/stem/dev_004/test_frozen_workflow.json \
  --workflow-comparison runs/stem/dev_004/generic_rerun_vs_frozen_workflow.json \
  --out reports/stemds_experiment_summary.md
```

Run a small optional OpenAI baseline smoke test:

```bash
docker compose run --rm stemds uv run python -m stemds.cli run-baseline \
  --data data/dabench/dabench_test.jsonl \
  --agent openai \
  --model gpt-4.1-mini \
  --limit 5 \
  --seed 42 \
  --out runs/docker_openai_smoke.json
```

The compose service mounts `./data`, `./external`, `./runs`, `./reports`, and `./skills` into `/app` so benchmark data and outputs persist on the host. Docker does not download or clone InfiAgent automatically. To convert raw DABench inside Docker, clone `external/InfiAgent` on the host first so it is mounted into `/app/external`. Converted JSONL files under `data/dabench` can be used directly as long as their referenced CSV paths are available; if tasks reference `external/InfiAgent` CSV files, mount `external/`.

Run arbitrary StemDS CLI commands with this pattern:

```bash
docker compose run --rm stemds uv run python -m stemds.cli <command>
```

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

## AI-Assisted Developer-Curated Workflow Search v0

PromptSkill-only differentiation rejected all candidates in `dev_003`, which is the intended safeguard when proposed skills regress validation performance. AI-assisted developer-curated workflow search is the next differentiation axis: StemDS searches over a small menu of prompt/control-flow architectures that I designed with AI coding assistance and then reviewed/implemented as externally supplied candidates. StemDS validates them on DABench validation tasks, freezes the best workflow only if it beats `direct_code`, and evaluates that frozen architecture on held-out test tasks. This is useful, but it is not autonomous architecture invention.

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
  --a runs/stem/dev_004/test_generic_rerun.json \
  --b runs/stem/dev_004/test_frozen_workflow.json \
  --out runs/stem/dev_004/generic_rerun_vs_frozen_workflow.json \
  --markdown-out runs/stem/dev_004/generic_rerun_vs_frozen_workflow.md
```

Workflow search v0 still does not generate Python skills, generate arbitrary tools, or modify repository code. The frozen `code_then_repair` workflow is the strongest DABench result in this repository, but it is selection from an AI-assisted developer-curated menu rather than fully generative self-assembly. I used AI assistance while designing/implementing the menu, so this result should be read as validated selection over externally supplied workflows, not autonomous workflow invention.

## Generative Workflow Search

The predefined workflow search above is useful, but it is still selection-heavy: the workflow menu was externally supplied, with AI assistance used during development and human review/implementation deciding what entered the menu. Generative workflow search lets the stem loop propose workflow graphs from training failures while staying inside a safe DSL. The human defines primitives such as `schema_summary`, `llm_plan`, `llm_code`, `python_execute`, bounded `llm_repair`, and `stop`; the LLM proposes JSON graphs; StemDS validates graph structure and budget before any evaluation.

This tests a more generative path because the system can propose a candidate control-flow graph, not only select a prompt template. The contrast is deliberate: the successful `code_then_repair` result came from the externally supplied workflow menu, while this section tests whether the system can generate workflow graphs itself. Validation still acts as the immune system: malformed graphs, unknown nodes, unbounded cycles, and non-improving workflows are rejected. Existing human-authored workflow search remains available and is not replaced.

Generate candidate workflows, validate them, and freeze the best one only if it beats `direct_code` on validation:

```bash
uv run python -m stemds.cli generate-workflows \
  --train data/dabench/dabench_train.jsonl \
  --val data/dabench/dabench_val.jsonl \
  --model gpt-4.1-mini \
  --out-dir runs/stem/gen_001 \
  --max-candidates 3 \
  --val-limit 38 \
  --seed 42 \
  --min-delta 0.03
```

Evaluate the frozen generated workflow on the fixed 40-task DABench test slice:

```bash
uv run python -m stemds.cli evaluate-generated-workflow \
  --data data/dabench/dabench_test.jsonl \
  --workflow runs/stem/gen_001/frozen_generated_workflow.json \
  --model gpt-4.1-mini \
  --limit 40 \
  --seed 42 \
  --out runs/stem/gen_001/test_generated_workflow.json
```

Compare generated workflow performance to the generic baseline:

```bash
uv run python -m stemds.cli compare-runs \
  --a runs/stem/dev_004/test_generic_rerun.json \
  --b runs/stem/gen_001/test_generated_workflow.json \
  --out runs/stem/gen_001/generic_vs_generated_workflow.json \
  --markdown-out runs/stem/gen_001/generic_vs_generated_workflow.md
```

The current `gen_001` result is negative: 3 workflow graphs were proposed, 2 were structurally valid, neither beat `direct_code` on validation, and the generated-workflow path froze `direct_code`. Its held-out result was accuracy `0.300` and composite `0.261`, below both the generic rerun and the human-authored `code_then_repair` workflow. See `reports/generative_workflow_search_summary.md`.

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

The report is intended for write-up drafting. It summarizes setup, generic baseline behavior, seed-skill regression, PromptSkill validation, human-authored workflow-search validation, frozen-workflow test performance, limitations, and suggested write-up bullets. The generated workflow negative result is summarized separately in `reports/generative_workflow_search_summary.md`.

## Mini ML-Engineering Extension

This is a stretch demo, not the main DABench result. It uses only built-in `sklearn.datasets` data and tests whether StemDS can support a second data-science specialization with a different task schema, output contract, sandbox, and metric. It does not integrate MLAgentBench, MLE-bench, DSBench, or PythonSkill generation.

Create the built-in ML task file:

```bash
uv run python -m stemds.cli create-ml-tasks --out data/ml/sklearn_tasks.jsonl
```

Run the offline dummy baseline:

```bash
uv run python -m stemds.cli run-ml-baseline --data data/ml/sklearn_tasks.jsonl --agent dummy --out runs/ml/ml_dummy.json
```

Run an optional OpenAI ML baseline:

```bash
uv run python -m stemds.cli run-ml-baseline \
  --data data/ml/sklearn_tasks.jsonl \
  --agent openai \
  --model gpt-4.1-mini \
  --limit 5 \
  --seed 42 \
  --out runs/ml/ml_generic.json
```

Search ML workflows:

```bash
uv run python -m stemds.cli search-ml-workflows \
  --data data/ml/sklearn_tasks.jsonl \
  --model gpt-4.1-mini \
  --limit 5 \
  --seed 42 \
  --min-delta 0.03 \
  --out-dir runs/ml/dev_001
```

Evaluate the frozen ML workflow:

```bash
uv run python -m stemds.cli evaluate-ml-workflow \
  --data data/ml/sklearn_tasks.jsonl \
  --workflow runs/ml/dev_001/frozen_ml_workflow.json \
  --model gpt-4.1-mini \
  --limit 5 \
  --seed 42 \
  --out runs/ml/dev_001/ml_test_frozen_workflow.json
```

The ML extension does not replace the DABench headline result; it is a smoke demo that the same framework can host a scoped ML-engineering task family.

## DSBench Exploratory Extension

DABench remains the main validated result. DSBench is included only as an exploratory future-work path for broader data-science tasks with mixed analysis/modeling layouts. Raw DSBench data should live under `external/DSBench` and should not be committed.

Clone DSBench externally if needed:

```bash
git clone https://github.com/liqiangjing/dsbench.git external/DSBench
```

Inspect the local DSBench structure:

```bash
uv run python -m stemds.cli inspect-dsbench --root external/DSBench --out reports/dsbench_inspection.md
```

Try converting only an unambiguous simple CSV/question/answer subset if such metadata exists:

```bash
uv run python -m stemds.cli convert-dsbench-subset --root external/DSBench --out data/dsbench/dsbench_subset.jsonl --limit 10
```

The current DSBench adapter is deliberately conservative. It converts only explicit simple tabular records with `dataset_path`, `question`, `answer`, and `answer_type` fields. Native DSBench analysis assets may include Excel workbooks, images, notebooks, archived question files, and data-modeling competition metadata, so full integration requires dedicated extraction and scoring rules.
