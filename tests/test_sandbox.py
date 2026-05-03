from __future__ import annotations

from stemds.sandbox import PythonSandbox


def test_sandbox_extracts_final_answer(tmp_path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("value\n1\n", encoding="utf-8")

    result = PythonSandbox(timeout_sec=2).run('print("FINAL_ANSWER: 42")\n', dataset)

    assert result.status == "success"
    assert result.extracted_answer == "42"


def test_sandbox_rejects_unsafe_code(tmp_path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("value\n1\n", encoding="utf-8")

    result = PythonSandbox(timeout_sec=2).run("import os\nprint('FINAL_ANSWER: 1')\n", dataset)

    assert result.status == "unsafe_code"

