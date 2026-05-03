"""Subprocess execution sandbox for generated Python code."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from stemds.config import DEFAULT_SANDBOX_TIMEOUT_SEC

SANDBOX_STATUSES = {"success", "syntax_error", "runtime_error", "timeout", "unsafe_code"}


@dataclass(slots=True)
class SandboxResult:
    status: str
    stdout: str
    stderr: str
    duration_sec: float
    extracted_answer: str | None


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
        "pip",
        "requests",
        "shutil.rmtree",
    )

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

    def _unsafe_reason(self, code: str) -> str | None:
        lowered = code.lower()
        for snippet in self.unsafe_snippets:
            if snippet in lowered:
                return snippet
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

    def _extract_answer(self, stdout: str) -> str | None:
        for line in reversed(stdout.splitlines()):
            stripped = line.strip()
            if stripped.startswith("FINAL_ANSWER:"):
                return stripped.split("FINAL_ANSWER:", 1)[1].strip()
        return None

