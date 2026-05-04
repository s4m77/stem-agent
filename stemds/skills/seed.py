"""Hand-authored seed prompt skills for data-analysis agents."""

from __future__ import annotations

from stemds.skills.base import PromptSkill


def create_seed_prompt_skills() -> list[PromptSkill]:
    return [
        PromptSkill(
            skill_id="ensure_answer_variable",
            name="Ensure ANSWER Variable",
            description="Always assign the final result to ANSWER instead of only printing.",
            applies_to_tags=[],
            applies_to_failure_categories=["execution_error", "missing_answer_variable"],
            prompt_instructions="Always assign the final result to ANSWER. Never only print the answer.",
            priority=95,
        ),
        PromptSkill(
            skill_id="robust_csv_loading",
            name="Robust CSV Loading",
            description="Use the provided CSV_PATH variable exactly and avoid hard-coded paths.",
            applies_to_tags=[],
            applies_to_failure_categories=["execution_error"],
            prompt_instructions="Use CSV_PATH exactly with pandas, usually pd.read_csv(CSV_PATH). Do not hard-code paths or read external files.",
            priority=90,
        ),
        PromptSkill(
            skill_id="numeric_answer_format",
            name="Numeric Answer Format",
            description="Return plain numeric values without units or formatted strings.",
            applies_to_tags=[],
            applies_to_failure_categories=["numeric_tolerance_issue", "answer_format_mismatch"],
            prompt_instructions="For numeric answers, assign a plain int or float to ANSWER. Do not include units, labels, commas, or explanatory text.",
            priority=80,
        ),
        PromptSkill(
            skill_id="multi_answer_format",
            name="Multi Answer Format",
            description="Use deterministic structured outputs for multi-value questions.",
            applies_to_tags=[],
            applies_to_failure_categories=["unsupported_multi_answer", "answer_format_mismatch"],
            prompt_instructions="If multiple values are requested, return a deterministic dict or list in the requested order, with concise scalar values.",
            priority=70,
        ),
        PromptSkill(
            skill_id="summary_statistics_skill",
            name="Summary Statistics",
            description="Compute requested summary statistics explicitly and handle missing values.",
            applies_to_tags=["summary_statistics"],
            applies_to_failure_categories=[],
            prompt_instructions="Compute requested statistics explicitly. Check whether relevant columns have missing values and handle them deliberately.",
            priority=50,
        ),
        PromptSkill(
            skill_id="correlation_analysis_skill",
            name="Correlation Analysis",
            description="Use pandas or numpy correlation methods on numeric columns.",
            applies_to_tags=["correlation_analysis"],
            applies_to_failure_categories=[],
            prompt_instructions="For correlation tasks, ensure columns are numeric, drop only required missing values, and return the requested coefficient or label.",
            priority=50,
        ),
        PromptSkill(
            skill_id="distribution_analysis_skill",
            name="Distribution Analysis",
            description="Be careful with bins, frequencies, proportions, and sorting.",
            applies_to_tags=["distribution_analysis"],
            applies_to_failure_categories=[],
            prompt_instructions="For distribution tasks, identify bins or categories exactly, compute frequencies/proportions carefully, and sort only when requested.",
            priority=50,
        ),
        PromptSkill(
            skill_id="outlier_detection_skill",
            name="Outlier Detection",
            description="Apply the rule in the question before counting or filtering outliers.",
            applies_to_tags=["outlier_detection"],
            applies_to_failure_categories=[],
            prompt_instructions="For outlier tasks, identify the stated rule, compute thresholds explicitly, then filter or count according to that rule.",
            priority=50,
        ),
        PromptSkill(
            skill_id="feature_engineering_skill",
            name="Feature Engineering",
            description="Create derived columns explicitly and verify intermediate values.",
            applies_to_tags=["feature_engineering"],
            applies_to_failure_categories=[],
            prompt_instructions="For feature engineering tasks, create derived columns explicitly, use clear column names, and check intermediate values before computing ANSWER.",
            priority=50,
        ),
        PromptSkill(
            skill_id="machine_learning_skill",
            name="Machine Learning",
            description="Use sklearn only when needed and make simple ML calculations deterministic.",
            applies_to_tags=["machine_learning"],
            applies_to_failure_categories=[],
            prompt_instructions="For simple ML tasks, use sklearn only when needed, set random_state for deterministic behavior, and assign the requested metric or value to ANSWER.",
            priority=50,
        ),
    ]


SEED_PROMPT_SKILLS = create_seed_prompt_skills()
