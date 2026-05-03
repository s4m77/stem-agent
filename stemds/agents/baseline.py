"""Baseline agents for toy data-analysis tasks."""

from __future__ import annotations

import re
from pathlib import Path

from stemds.agents.base import AgentOutput
from stemds.llm import LLMClient
from stemds.tasks import DataAnalysisTask


class DummyBaselineAgent:
    """Offline baseline with deterministic pandas snippets for known toy tasks."""

    def solve(self, task: DataAnalysisTask) -> AgentOutput:
        csv_name = Path(task.dataset_path).name
        code = self._known_solution(task.task_id, csv_name)
        if code is None:
            code = self._schema_inspection_solution(csv_name)
        return AgentOutput(
            code=code,
            raw_response=None,
            llm_calls=0,
            metadata={"agent": "dummy", "task_id": task.task_id},
        )

    def _known_solution(self, task_id: str, csv_name: str) -> str | None:
        solutions = {
            "sales_001": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df.groupby("region")["revenue"].sum().idxmax()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "employees_001": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df.loc[df["department"] == "Engineering", "salary"].mean()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "customers_001": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = int((df["country"] == "Germany").sum())
print(f"FINAL_ANSWER: {{answer}}")
""",
            "orders_001": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
months = pd.to_datetime(df["order_date"]).dt.month_name()
answer = months.value_counts().idxmax()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "products_001": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df.groupby("category")["price"].mean().idxmax()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "orders_002": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = (df["status"].eq("cancelled").mean()) * 100
print(f"FINAL_ANSWER: {{answer}}")
""",
            "sales_002": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = (df["revenue"] * (1 - df["discount"])).sum()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "employees_002": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df.loc[df["salary"].idxmax(), "name"]
print(f"FINAL_ANSWER: {{answer}}")
""",
            "orders_003": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df.loc[df["status"] == "completed", "total"].sum()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "customers_002": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df["segment"].value_counts().idxmax()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "sales_003": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df.groupby("employee_id")["revenue"].sum().idxmax()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "employees_003": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df["age"].median()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "orders_004": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = int((df["status"] == "pending").sum())
print(f"FINAL_ANSWER: {{answer}}")
""",
            "products_002": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df.loc[df["category"] == "Hardware", "price"].mean()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "customers_003": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df["country"].value_counts().idxmax()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "sales_004": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df.groupby("region")["revenue"].sum().idxmin()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "employees_004": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df["salary"].mean()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "orders_005": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df["customer_id"].value_counts().idxmax()
print(f"FINAL_ANSWER: {{answer}}")
""",
            "products_003": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = int((df["category"] == "Software").sum())
print(f"FINAL_ANSWER: {{answer}}")
""",
            "orders_006": f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = df.loc[df["status"] == "cancelled", "total"].sum()
print(f"FINAL_ANSWER: {{answer}}")
""",
        }
        code = solutions.get(task_id)
        return code.strip() + "\n" if code is not None else None

    def _schema_inspection_solution(self, csv_name: str) -> str:
        return f"""
import pandas as pd
df = pd.read_csv("{csv_name}")
answer = ", ".join(map(str, df.columns))
print(f"FINAL_ANSWER: {{answer}}")
""".strip() + "\n"


class OpenAIBaselineAgent:
    def __init__(self, model: str, llm_client: LLMClient) -> None:
        self.model = model
        self.llm_client = llm_client

    def solve(self, task: DataAnalysisTask) -> AgentOutput:
        prompt = self._build_prompt(task)
        raw_response = self.llm_client.generate_text(prompt, model=self.model, temperature=0.2)
        code = extract_python_code(raw_response)
        return AgentOutput(
            code=code,
            raw_response=raw_response,
            llm_calls=1,
            metadata={"agent": "openai", "model": self.model},
        )

    def _build_prompt(self, task: DataAnalysisTask) -> str:
        csv_name = Path(task.dataset_path).name
        return f"""You are solving a data-analysis task with Python and pandas.

CSV file available in the current working directory: {csv_name}
Question: {task.question}

Write only executable Python code.
Requirements:
- read the CSV file named above
- compute the answer directly from the data
- print the final answer on the last line prefixed exactly with FINAL_ANSWER:
- do not include explanations or markdown
"""


def extract_python_code(text: str) -> str:
    python_match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if python_match:
        return python_match.group(1).strip() + "\n"
    generic_match = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip() + "\n"
    return text.strip() + "\n"

