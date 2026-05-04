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


def test_sandbox_allows_sklearn_pipeline_import(tmp_path) -> None:
    sandbox = PythonSandbox(timeout_sec=2)

    assert sandbox._unsafe_reason("from sklearn.pipeline import Pipeline\nANSWER = 1\n") is None


def test_sandbox_rejects_pip_word(tmp_path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("value\n1\n", encoding="utf-8")

    result = PythonSandbox(timeout_sec=2).run_generated_analysis("# pip install x\nANSWER = 1\n", dataset)

    assert result.status == "unsafe_code"


def test_sandbox_captures_answer_variable_from_generated_pandas_code(tmp_path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("group,value\nA,1\nA,2\nB,10\n", encoding="utf-8")
    code = """
import pandas as pd
df = pd.read_csv(CSV_PATH)
ANSWER = df.groupby("group")["value"].sum().idxmax()
"""

    result = PythonSandbox(timeout_sec=5).run_generated_analysis(code, dataset)

    assert result.status == "pass"
    assert result.answer == "B"
