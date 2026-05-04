from __future__ import annotations

from stemds.agents.baseline import SkillAugmentedAnalysisAgent
from stemds.llm import MockLLMClient
from stemds.skills.base import PromptSkill
from stemds.skills.library import SkillLibrary
from stemds.skills.seed import create_seed_prompt_skills
from stemds.tasks import DataAnalysisTask


def test_seed_skill_creation() -> None:
    skills = create_seed_prompt_skills()

    assert len(skills) == 10
    assert {skill.skill_id for skill in skills} >= {"ensure_answer_variable", "summary_statistics_skill"}


def test_skill_library_save_load_retrieve(tmp_path) -> None:
    library = SkillLibrary(create_seed_prompt_skills())
    library.save_to_dir(tmp_path)

    loaded = SkillLibrary.load_from_dir(tmp_path)
    retrieved = loaded.retrieve(["summary_statistics"], k=3)

    assert len(loaded.skills) == 10
    assert retrieved[0].skill_id == "summary_statistics_skill"


def test_skill_library_retrieves_global_skill_for_arbitrary_tags() -> None:
    global_skill = PromptSkill(
        skill_id="global",
        name="Global",
        description="Global skill",
        applies_to_tags=[],
        applies_to_failure_categories=[],
        prompt_instructions="Always be careful.",
        priority=5,
    )

    retrieved = SkillLibrary([global_skill]).retrieve(["unknown_tag"], k=5)

    assert [skill.skill_id for skill in retrieved] == ["global"]


def test_skill_library_retrieval_ranking_is_deterministic() -> None:
    skills = [
        PromptSkill("b", "B", "", ["summary_statistics"], [], "b", priority=10),
        PromptSkill("a", "A", "", ["summary_statistics"], [], "a", priority=10),
        PromptSkill("global", "Global", "", [], [], "global", priority=50),
    ]

    retrieved = SkillLibrary(skills).retrieve(["summary_statistics"], k=3)

    assert [skill.skill_id for skill in retrieved] == ["a", "b", "global"]


def test_prompt_skill_from_dict_coerces_tags_to_strings() -> None:
    skill = PromptSkill.from_dict(
        {
            "skill_id": "coerce",
            "name": "Coerce",
            "description": "Coerce tags",
            "applies_to_tags": [123],
            "applies_to_failure_categories": [456],
            "prompt_instructions": "Be robust.",
        }
    )

    assert skill.applies_to_tags == ["123"]
    assert skill.applies_to_failure_categories == ["456"]


def test_skill_augmented_agent_prompt_includes_relevant_skills() -> None:
    llm = MockLLMClient(
        "```python\n"
        "import pandas as pd\n"
        "df = pd.read_csv(CSV_PATH)\n"
        "ANSWER = df['revenue'].mean()\n"
        "```"
    )
    task = DataAnalysisTask(
        task_id="skill_test",
        dataset_path="data/toy_csvs/sales.csv",
        question="What is the mean revenue?",
        answer=875,
        answer_type="number",
        tags=["summary_statistics"],
    )

    output = SkillAugmentedAnalysisAgent(
        model="mock-model",
        llm_client=llm,
        skill_library=SkillLibrary(create_seed_prompt_skills()),
    ).solve(task)

    assert "Relevant analysis skills to apply" in llm.prompts[0]
    assert "summary_statistics_skill" in llm.prompts[0]
    assert output.metadata["selected_skill_ids"] == ["summary_statistics_skill"]
