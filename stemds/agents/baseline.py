"""Baseline agents for toy data-analysis tasks."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from stemds.agents.base import AgentOutput
from stemds.llm import BaseLLMClient
from stemds.skills.base import PromptSkill
from stemds.skills.library import SkillLibrary
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


class OpenAIGenericAnalysisAgent:
    def __init__(self, model: str, llm_client: BaseLLMClient, seed: int | None = 42) -> None:
        self.model = model
        self.llm_client = llm_client
        self.seed = seed

    def solve(self, task: DataAnalysisTask) -> AgentOutput:
        profile = inspect_csv(task.dataset_path)
        prompt = self._build_prompt(task, profile)
        raw_response = self.llm_client.generate_text(prompt, model=self.model, temperature=0.0, seed=self.seed)
        code = extract_python_code(raw_response)
        return AgentOutput(
            code=code,
            raw_response=raw_response,
            llm_calls=1,
            metadata={
                "agent": "openai_generic",
                "answer_contract": "ANSWER",
                "model": self.model,
                "prompt": prompt,
                "csv_shape": profile["shape"],
                "csv_columns": profile["columns"],
                "seed": self.seed,
                "llm_api_path": getattr(self.llm_client, "last_api_path", None),
                "llm_seed_ignored": getattr(self.llm_client, "last_seed_ignored", False),
            },
        )

    def _build_prompt(self, task: DataAnalysisTask, profile: dict[str, object]) -> str:
        return f"""You are a careful data analyst writing Python pandas code.

The dataset is available at the variable CSV_PATH.
Question: {task.question}
Expected answer type: {task.answer_type}

Dataset shape: {profile["shape"]}
Columns:
{profile["columns_text"]}

Sample rows as CSV:
{profile["sample_csv"]}

Requirements:
- write Python code only
- use pandas
- read the dataset with pd.read_csv(CSV_PATH)
- assign the final answer to a variable named ANSWER
- do not print prose
- do not make plots
- do not read external files
- do not use network
- keep code simple
- if the answer is numeric, assign an int or float
- if the answer is categorical or string, assign a string
"""


class OpenAIBaselineAgent(OpenAIGenericAnalysisAgent):
    """Backward-compatible name for the generic OpenAI baseline."""


class SkillAugmentedAnalysisAgent(OpenAIGenericAnalysisAgent):
    def __init__(
        self,
        model: str,
        llm_client: BaseLLMClient,
        skill_library: SkillLibrary,
        k: int = 5,
        seed: int | None = 42,
    ) -> None:
        super().__init__(model=model, llm_client=llm_client, seed=seed)
        self.skill_library = skill_library
        self.k = k

    def solve(self, task: DataAnalysisTask) -> AgentOutput:
        profile = inspect_csv(task.dataset_path)
        selected_skills = self.skill_library.retrieve(task.tags, k=self.k)
        prompt = self._build_skill_prompt(task, profile, selected_skills)
        raw_response = self.llm_client.generate_text(prompt, model=self.model, temperature=0.0, seed=self.seed)
        code = extract_python_code(raw_response)
        return AgentOutput(
            code=code,
            raw_response=raw_response,
            llm_calls=1,
            metadata={
                "agent": "skill_openai",
                "answer_contract": "ANSWER",
                "model": self.model,
                "prompt": prompt,
                "csv_shape": profile["shape"],
                "csv_columns": profile["columns"],
                "selected_skill_ids": [skill.skill_id for skill in selected_skills],
                "seed": self.seed,
                "llm_api_path": getattr(self.llm_client, "last_api_path", None),
                "llm_seed_ignored": getattr(self.llm_client, "last_seed_ignored", False),
            },
        )

    def _build_skill_prompt(
        self,
        task: DataAnalysisTask,
        profile: dict[str, object],
        selected_skills: list[PromptSkill],
    ) -> str:
        base_prompt = self._build_prompt(task, profile)
        skills_text = "\n".join(
            f"- {skill.skill_id}: {skill.prompt_instructions}" for skill in selected_skills
        )
        if not skills_text:
            skills_text = "- No retrieved skills. Solve directly and carefully."
        return base_prompt.replace(
            "\nRequirements:",
            f"\nRelevant analysis skills to apply:\n{skills_text}\n\nRequirements:",
        )


def inspect_csv(dataset_path: str | Path, sample_rows: int = 5) -> dict[str, object]:
    df = pd.read_csv(dataset_path)
    dtypes = {column: str(dtype) for column, dtype in df.dtypes.items()}
    columns_text = "\n".join(f"- {column}: {dtype}" for column, dtype in dtypes.items())
    sample_csv = df.head(sample_rows).to_csv(index=False).strip()
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "dtypes": dtypes,
        "columns_text": columns_text,
        "sample_csv": sample_csv,
    }


def extract_python_code(text: str) -> str:
    python_match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if python_match:
        return python_match.group(1).strip() + "\n"
    generic_match = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip() + "\n"
    return text.strip() + "\n"
