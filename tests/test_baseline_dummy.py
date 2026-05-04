from __future__ import annotations

from stemds.agents.baseline import DummyBaselineAgent, OpenAIGenericAnalysisAgent, extract_python_code
from stemds.llm import MockLLMClient
from stemds.metrics import compare_answers
from stemds.sandbox import PythonSandbox
from stemds.tasks import DataAnalysisTask


def test_dummy_baseline_solves_known_task() -> None:
    task = DataAnalysisTask(
        task_id="sales_004",
        dataset_path="data/toy_csvs/sales.csv",
        question="Which region has the lowest total revenue?",
        answer="South",
        answer_type="string",
        tags=["groupby"],
    )

    output = DummyBaselineAgent().solve(task)
    result = PythonSandbox(timeout_sec=15).run(output.code, task.dataset_path)

    assert result.status == "success"
    assert compare_answers(task.answer, result.extracted_answer, task.answer_type)


def test_extract_python_code_from_fenced_response() -> None:
    response = "```python\nprint('FINAL_ANSWER: West')\n```"

    assert extract_python_code(response) == "print('FINAL_ANSWER: West')\n"


def test_extract_python_code_from_unfenced_response() -> None:
    response = "import pandas as pd\nANSWER = 'West'"

    assert extract_python_code(response) == "import pandas as pd\nANSWER = 'West'\n"


def test_openai_generic_analysis_agent_uses_mock_llm() -> None:
    task = DataAnalysisTask(
        task_id="sales_004",
        dataset_path="data/toy_csvs/sales.csv",
        question="Which region has the lowest total revenue?",
        answer="South",
        answer_type="string",
        tags=["groupby"],
    )
    llm = MockLLMClient(
        "```python\n"
        "import pandas as pd\n"
        "df = pd.read_csv(CSV_PATH)\n"
        "ANSWER = df.groupby('region')['revenue'].sum().idxmin()\n"
        "```"
    )

    output = OpenAIGenericAnalysisAgent(model="mock-model", llm_client=llm).solve(task)
    result = PythonSandbox(timeout_sec=15).run_generated_analysis(output.code, task.dataset_path)

    assert output.llm_calls == 1
    assert "CSV_PATH" in llm.prompts[0]
    assert output.metadata["answer_contract"] == "ANSWER"
    assert result.status == "pass"
    assert compare_answers(task.answer, result.answer, task.answer_type)
