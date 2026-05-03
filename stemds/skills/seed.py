"""Metadata-only seed prompt skills."""

from __future__ import annotations

from stemds.skills.base import PromptSkill

SEED_PROMPT_SKILLS = [
    PromptSkill(
        name="schema_inspection",
        description="Inspect columns, dtypes, and sample rows before choosing code.",
        tags=["schema", "inspection"],
        prompt="Inspect the CSV schema before writing analysis code.",
    ),
    PromptSkill(
        name="groupby_aggregation",
        description="Use pandas groupby operations for grouped sums, means, counts, and rankings.",
        tags=["groupby", "aggregation", "ranking"],
        prompt="For grouped questions, aggregate by the requested key and sort or select idxmax/idxmin.",
    ),
    PromptSkill(
        name="date_parsing",
        description="Parse date columns with pandas.to_datetime before extracting months or periods.",
        tags=["date", "month", "time"],
        prompt="When a question references dates, parse the date column before deriving date parts.",
    ),
    PromptSkill(
        name="numeric_answer_verification",
        description="Check numeric units, percentages, and rounding before printing the final answer.",
        tags=["number", "percentage", "verification"],
        prompt="Verify numeric answers and keep percentages in the same units as the question.",
    ),
    PromptSkill(
        name="missing_value_awareness",
        description="Consider missing values before aggregation, filtering, and averaging.",
        tags=["missing_values", "cleaning"],
        prompt="Check whether relevant columns contain missing values and handle them deliberately.",
    ),
]

