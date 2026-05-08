"""Sandbox for generated ML-engineering code."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from stemds.ml.datasets import load_sklearn_dataset
from stemds.ml.tasks import MLEngineeringTask
from stemds.sandbox import PythonSandbox


@dataclass(slots=True)
class MLSandboxResult:
    status: str
    score: float | None
    metric: str | None
    model: str | None
    stdout: str
    stderr: str
    duration_sec: float
    llm_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class MLEngineeringSandbox:
    def __init__(self, timeout_sec: float = 30.0, split_seed: int = 42) -> None:
        self.timeout_sec = timeout_sec
        self.split_seed = split_seed
        self._safety = PythonSandbox(timeout_sec=timeout_sec)

    def run(self, code: str, task: MLEngineeringTask) -> MLSandboxResult:
        unsafe_reason = self._safety._unsafe_reason(code)
        if unsafe_reason is not None:
            return MLSandboxResult(
                status="runtime_error",
                score=None,
                metric=None,
                model=None,
                stdout="",
                stderr=f"Rejected unsafe code containing: {unsafe_reason}",
                duration_sec=0.0,
            )

        dataframe, target_column = load_sklearn_dataset(task.dataset_name)
        stratify = dataframe[target_column] if task.problem_type == "classification" else None
        train_df, test_df = train_test_split(
            dataframe,
            test_size=0.25,
            random_state=self.split_seed,
            stratify=stratify,
        )

        with tempfile.TemporaryDirectory(prefix="stemds_ml_") as temp_dir:
            work_dir = Path(temp_dir)
            train_path = work_dir / "train.csv"
            test_path = work_dir / "test.csv"
            solution_path = work_dir / "solution.py"
            runner_path = work_dir / "_stemds_ml_runner.py"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            solution_path.write_text(code, encoding="utf-8")
            runner_path.write_text(
                _runner_source(
                    train_path=train_path,
                    test_path=test_path,
                    target_column=target_column,
                    metric=task.metric,
                    problem_type=task.problem_type,
                ),
                encoding="utf-8",
            )

            start = time.monotonic()
            try:
                completed = subprocess.run(
                    [sys.executable, str(runner_path.name)],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return MLSandboxResult(
                    status="timeout",
                    score=None,
                    metric=None,
                    model=None,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    duration_sec=time.monotonic() - start,
                )

            duration = time.monotonic() - start
            if completed.returncode == 65:
                return MLSandboxResult("syntax_error", None, None, None, completed.stdout, completed.stderr, duration)
            if completed.returncode == 67:
                return MLSandboxResult("invalid_result", None, None, None, completed.stdout, completed.stderr, duration)
            if completed.returncode != 0:
                return MLSandboxResult("runtime_error", None, None, None, completed.stdout, completed.stderr, duration)
            payload = _extract_result_payload(completed.stdout)
            if payload is None:
                return MLSandboxResult(
                    "invalid_result",
                    None,
                    None,
                    None,
                    completed.stdout,
                    "Generated code did not emit RESULT",
                    duration,
                )
            score = _coerce_float(payload.get("score"))
            metric = payload.get("metric")
            model = payload.get("model")
            if score is None or not isinstance(metric, str) or not isinstance(model, str):
                return MLSandboxResult(
                    "invalid_result",
                    None,
                    str(metric) if metric is not None else None,
                    str(model) if model is not None else None,
                    completed.stdout,
                    "RESULT must contain float score, string metric, and string model",
                    duration,
                    metadata={"raw_result": payload},
                )
            return MLSandboxResult(
                status="success",
                score=score,
                metric=metric,
                model=model,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_sec=duration,
                metadata={"raw_result": payload},
            )


def _runner_source(train_path: Path, test_path: Path, target_column: str, metric: str, problem_type: str) -> str:
    return f'''from __future__ import annotations

import json
import pathlib
import sys
import traceback

TRAIN_CSV_PATH = {str(train_path)!r}
TEST_CSV_PATH = {str(test_path)!r}
TARGET_COLUMN = {target_column!r}
METRIC = {metric!r}
PROBLEM_TYPE = {problem_type!r}


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {{str(key): _to_jsonable(item) for key, item in value.items()}}
    return str(value)


namespace = {{
    "TRAIN_CSV_PATH": TRAIN_CSV_PATH,
    "TEST_CSV_PATH": TEST_CSV_PATH,
    "TARGET_COLUMN": TARGET_COLUMN,
    "METRIC": METRIC,
    "PROBLEM_TYPE": PROBLEM_TYPE,
}}

try:
    source = pathlib.Path("solution.py").read_text(encoding="utf-8")
    compiled = compile(source, "solution.py", "exec")
    exec(compiled, namespace)
except SyntaxError:
    traceback.print_exc()
    sys.exit(65)
except BaseException:
    traceback.print_exc()
    sys.exit(66)

if "RESULT" not in namespace:
    print("Generated code did not assign RESULT", file=sys.stderr)
    sys.exit(67)

print("__STEMDS_ML_RESULT_JSON__:" + json.dumps(_to_jsonable(namespace["RESULT"])))
'''


def _extract_result_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("__STEMDS_ML_RESULT_JSON__:"):
            raw = stripped.split("__STEMDS_ML_RESULT_JSON__:", 1)[1]
            loaded = json.loads(raw)
            return loaded if isinstance(loaded, dict) else None
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None
