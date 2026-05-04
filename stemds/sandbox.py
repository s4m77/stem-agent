"""Subprocess execution sandbox for generated Python code."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from stemds.config import DEFAULT_SANDBOX_TIMEOUT_SEC

SANDBOX_STATUSES = {"success", "syntax_error", "runtime_error", "timeout", "unsafe_code"}


@dataclass(slots=True)
class SandboxResult:
    status: str
    stdout: str
    stderr: str
    duration_sec: float
    extracted_answer: str | None
    answer: Any | None = None


class PythonSandbox:
    unsafe_snippets = (
        "import os",
        "from os",
        "import subprocess",
        "subprocess",
        "import socket",
        "from socket",
        "open(",
        "eval(",
        "exec(",
        "__import__",
        "requests",
        "shutil.rmtree",
    )
    unsafe_patterns = (re.compile(r"\bpip\b", flags=re.IGNORECASE),)

    def __init__(self, timeout_sec: float = DEFAULT_SANDBOX_TIMEOUT_SEC) -> None:
        self.timeout_sec = timeout_sec

    def run(self, code: str, dataset_path: str | Path) -> SandboxResult:
        unsafe_reason = self._unsafe_reason(code)
        if unsafe_reason is not None:
            return SandboxResult(
                status="unsafe_code",
                stdout="",
                stderr=f"Rejected unsafe code containing: {unsafe_reason}",
                duration_sec=0.0,
                extracted_answer=None,
            )

        source_dataset = self._resolve_dataset_path(dataset_path)
        if not source_dataset.exists():
            return SandboxResult(
                status="runtime_error",
                stdout="",
                stderr=f"Dataset not found: {source_dataset}",
                duration_sec=0.0,
                extracted_answer=None,
            )

        with tempfile.TemporaryDirectory(prefix="stemds_") as temp_dir:
            work_dir = Path(temp_dir)
            shutil.copy2(source_dataset, work_dir / source_dataset.name)
            solution_path = work_dir / "solution.py"
            solution_path.write_text(code, encoding="utf-8")

            start = time.monotonic()
            try:
                completed = subprocess.run(
                    [sys.executable, str(solution_path.name)],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                duration = time.monotonic() - start
                return SandboxResult(
                    status="timeout",
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    duration_sec=duration,
                    extracted_answer=None,
                )

            duration = time.monotonic() - start
            status = self._status_from_completed_process(completed)
            return SandboxResult(
                status=status,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_sec=duration,
                extracted_answer=self._extract_answer(completed.stdout) if status == "success" else None,
            )

    def run_generated_analysis(self, code: str, dataset_path: str | Path) -> SandboxResult:
        unsafe_reason = self._unsafe_reason(code)
        if unsafe_reason is not None:
            return SandboxResult(
                status="unsafe_code",
                stdout="",
                stderr=f"Rejected unsafe code containing: {unsafe_reason}",
                duration_sec=0.0,
                extracted_answer=None,
                answer=None,
            )

        source_dataset = self._resolve_dataset_path(dataset_path)
        if not source_dataset.exists():
            return SandboxResult(
                status="runtime_error",
                stdout="",
                stderr=f"Dataset not found: {source_dataset}",
                duration_sec=0.0,
                extracted_answer=None,
                answer=None,
            )

        with tempfile.TemporaryDirectory(prefix="stemds_") as temp_dir:
            work_dir = Path(temp_dir)
            temp_dataset = work_dir / source_dataset.name
            shutil.copy2(source_dataset, temp_dataset)
            solution_path = work_dir / "solution.py"
            wrapper_path = work_dir / "_stemds_runner.py"
            solution_path.write_text(code, encoding="utf-8")
            wrapper_path.write_text(_analysis_runner_source(temp_dataset), encoding="utf-8")

            start = time.monotonic()
            try:
                completed = subprocess.run(
                    [sys.executable, str(wrapper_path.name)],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                duration = time.monotonic() - start
                return SandboxResult(
                    status="timeout",
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    duration_sec=duration,
                    extracted_answer=None,
                    answer=None,
                )

            duration = time.monotonic() - start
            status = self._analysis_status_from_returncode(completed.returncode)
            answer = self._extract_json_answer(completed.stdout) if status == "pass" else None
            return SandboxResult(
                status=status,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_sec=duration,
                extracted_answer=str(answer) if answer is not None else None,
                answer=answer,
            )

    def _unsafe_reason(self, code: str) -> str | None:
        lowered = code.lower()
        for snippet in self.unsafe_snippets:
            if snippet in lowered:
                return snippet
        for pattern in self.unsafe_patterns:
            if pattern.search(code):
                return pattern.pattern
        return None

    def _resolve_dataset_path(self, dataset_path: str | Path) -> Path:
        path = Path(dataset_path).expanduser()
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()

    def _status_from_completed_process(self, completed: subprocess.CompletedProcess[str]) -> str:
        if completed.returncode == 0:
            return "success"
        if "SyntaxError" in completed.stderr:
            return "syntax_error"
        return "runtime_error"

    def _analysis_status_from_returncode(self, returncode: int) -> str:
        if returncode == 0:
            return "pass"
        if returncode == 65:
            return "syntax_error"
        if returncode == 124:
            return "timeout"
        return "runtime_error"

    def _extract_answer(self, stdout: str) -> str | None:
        for line in reversed(stdout.splitlines()):
            stripped = line.strip()
            if stripped.startswith("FINAL_ANSWER:"):
                return stripped.split("FINAL_ANSWER:", 1)[1].strip()
        return None

    def _extract_json_answer(self, stdout: str) -> Any | None:
        for line in reversed(stdout.splitlines()):
            stripped = line.strip()
            if stripped.startswith("__STEMDS_ANSWER_JSON__:"):
                raw_answer = stripped.split("__STEMDS_ANSWER_JSON__:", 1)[1]
                return json.loads(raw_answer)
        return None


def _analysis_runner_source(csv_path: Path) -> str:
    csv_path_text = str(csv_path)
    return f'''from __future__ import annotations

import json
import pathlib
import sys
import traceback

CSV_PATH = {csv_path_text!r}


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


namespace = {{"CSV_PATH": CSV_PATH}}
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

if "ANSWER" not in namespace:
    print("Generated code did not assign ANSWER", file=sys.stderr)
    sys.exit(66)

print("__STEMDS_ANSWER_JSON__:" + json.dumps(_to_jsonable(namespace["ANSWER"])))
'''
