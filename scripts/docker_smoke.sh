#!/usr/bin/env bash
set -euo pipefail

uv run pytest

if [[ -f data/dabench/dabench_test.jsonl ]]; then
  uv run python -m stemds.cli validate-data --data data/dabench/dabench_test.jsonl
else
  echo "DABench converted test split not found at data/dabench/dabench_test.jsonl; skipping validation."
fi
