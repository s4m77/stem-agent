# StemDS

StemDS is a small command-line proof of concept for a constrained stem agent. The first target domain is toy data analysis: given a CSV and a natural-language question, an agent writes Python/pandas code, executes it in a subprocess sandbox, and returns a measured answer.

The current skeleton does not implement stem differentiation yet. It provides the runnable foundation: task schemas, toy data, sandbox execution, answer metrics, baseline agents, CLI commands, seed skill metadata, workflow placeholders, and tests.

## Setup

```bash
uv sync --python 3.12 --extra dev
cp .env.example .env
export OPENAI_API_KEY=...
```

## Run Tests

```bash
uv run pytest
```

## Validate Data

```bash
uv run python -m stemds.cli validate-data --data data/toy_analysis_train.jsonl
```

## Run Dummy Baseline

```bash
uv run python -m stemds.cli run-baseline --data data/toy_analysis_test.jsonl --agent dummy --out runs/baseline_dummy.json
```

## Run OpenAI Baseline

```bash
uv run python -m stemds.cli run-baseline --data data/toy_analysis_test.jsonl --agent openai --model gpt-4.1-mini --out runs/baseline_openai.json
```

The OpenAI baseline requires `OPENAI_API_KEY`. Unit tests and the dummy baseline do not.
