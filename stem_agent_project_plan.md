# Stem Agent Project Plan: From Toy Specialization to Data/ML Agents

## 1. Project framing

The goal is to build a **stem agent**: a constrained meta-agent that starts from a minimal generic setup and differentiates into a more specialized agent for a task family.

In this project, “differentiation” does not mean fine-tuning model weights or letting an agent rewrite arbitrary source code. It means the agent can:

1. attempt tasks with a minimal baseline workflow;
2. observe failures through executable feedback;
3. propose reusable skills;
4. validate those skills against a metric;
5. keep accepted skills in a skill library;
6. search over small workflow structures;
7. freeze the best specialist workflow;
8. evaluate before/after performance on held-out tasks.

The strongest long-term vision is an IDE-native **Stem Agent Factory**: a coding assistant such as Codex, Claude Code, or a JetBrains-style IDE agent receives a task, classifies the task family, spawns a stem agent, and lets it specialize into a debugging, data-analysis, ML-engineering, QA, refactoring, or migration specialist.

For the current deliverable, the project should remain smaller and measurable:

> Build a command-line proof of concept where the same stem framework can specialize into at least one data-science-oriented agent, with a stretch goal of showing a second specialization.

---

## 2. Recommended staged plan

### Stage 0 — Define the stem framework

Build the shared infrastructure:

- task schemas;
- dataset loading;
- Python execution sandbox;
- LLM wrapper;
- baseline agent;
- skill library;
- workflow representation;
- evaluator;
- trace logging;
- report generation.

This stage should be domain-agnostic. The framework should not assume QA, data analysis, or ML engineering specifically.

### Stage 1 — Toy benchmark to validate the mechanism

Purpose: prove the development loop works quickly.

Best initial domain: **toy data analysis**.

Why toy data analysis instead of QA?

- It is still easy to evaluate automatically.
- It is more aligned with the later data/ML direction.
- It avoids the project looking like it started as a QA-only agent.
- It gives a natural route toward JetBrains DataSpell/PyCharm relevance.

Example task:

```json
{
  "task_id": "sales_001",
  "domain": "data_analysis",
  "dataset_path": "data/toy/sales.csv",
  "question": "Which region had the highest total revenue?",
  "answer": "West",
  "answer_type": "string",
  "tolerance": null,
  "tags": ["groupby", "aggregation", "sales"]
}
```

The baseline agent receives the CSV schema and question, writes pandas code, executes it, and returns an answer.

The stem-developed agent can learn skills such as:

- schema inspection;
- groupby aggregation;
- date parsing;
- missing-value handling;
- numeric tolerance checking;
- answer sanity checking.

### Stage 2 — Main benchmark: realistic data analysis

Purpose: show the project is not just a toy.

Suggested target: a subset of **DSBench**.

DSBench is designed for evaluating data-science agents on realistic data-analysis and data-modeling tasks sourced from ModelOff/Eloquence and Kaggle. Its project page describes 466 data-analysis tasks and 74 data-modeling tasks, making it highly relevant for a data-science stem-agent project.

Use a small subset first, for example:

- 20 data-analysis tasks;
- deterministic evaluation where possible;
- tasks with clear answers and manageable files;
- no heavy multimodal/image tasks in the first pass.

### Stage 3 — Stretch specialization: ML engineering

Purpose: demonstrate that the same stem framework can become a different specialist.

The ML-engineering domain receives:

- CSV/dataset;
- target column;
- problem type or inferred problem type;
- metric;
- train/test split or evaluation script.

Example task:

```json
{
  "task_id": "churn_001",
  "domain": "ml_engineering",
  "dataset_path": "data/ml/churn.csv",
  "target_column": "churn",
  "problem_type": "classification",
  "metric": "f1",
  "min_score": 0.72,
  "tags": ["classification", "categorical", "missing_values"]
}
```

The ML-engineering specialist may learn skills such as:

- target type detection;
- train/validation splitting;
- categorical encoding;
- missing-value imputation;
- metric selection;
- sklearn baseline training;
- leakage checks;
- error analysis.

Good benchmarks to study later:

- **MLAgentBench**, which evaluates language agents on end-to-end machine-learning experimentation tasks.
- **MLE-bench**, which evaluates ML-engineering agents on 75 curated Kaggle competitions. This is highly relevant but likely too heavy for the first implementation.

### Optional Stage — QA smoke test

QA is still useful as a tiny smoke-test domain because executable feedback is very clean:

- generated tests pass/fail;
- bug detection is measurable;
- false positives are measurable if reference implementations exist.

However, if the final story is data/ML/IDE specialization, QA should be optional rather than the core domain.

---

## 3. Candidate datasets for Step 1

Step 1 should use datasets that are simple, controllable, and fast. The point is to debug the stem loop, not to win a benchmark immediately.

### Option A — Custom toy data-analysis benchmark — recommended

Create 20–40 small CSV tasks yourself.

Each task includes:

- dataset path;
- natural-language question;
- ground-truth answer;
- answer type;
- tolerance if numeric;
- tags.

Example task types:

1. highest/lowest aggregate value;
2. groupby mean/sum/count;
3. filtering rows by condition;
4. date parsing and monthly aggregation;
5. missing-value handling;
6. correlation calculation;
7. outlier detection;
8. joining two small tables;
9. percentage/rate calculation;
10. simple ranking/top-k.

Pros:

- fastest to start;
- fully controlled;
- exact answers known;
- easy to create train/validation/test splits;
- ideal for debugging the sandbox, metrics, and skill validation.

Cons:

- less externally credible;
- must be followed by a more realistic benchmark if time allows.

### Option B — DSBench subset — main benchmark after the toy stage

DSBench is a benchmark for realistic data-science agents, with data-analysis and data-modeling tasks collected from sources such as Eloquence/ModelOff and Kaggle. The project page reports 466 data-analysis tasks and 74 data-modeling tasks.

Pros:

- highly aligned with data-analysis and ML-engineering agents;
- more realistic than handcrafted toy tasks;
- good benchmark credibility.

Cons:

- setup may be heavier;
- some tasks may include images/tables or ambiguous instructions;
- full benchmark is unnecessary for the MVP.

Recommended use:

- start with 10–30 manageable data-analysis tasks;
- avoid multimodal tasks at first;
- integrate as a dataset adapter after the custom JSONL pipeline works.

### Option C — Small sklearn-style ML tasks — good stretch

Use small tabular datasets from sklearn or synthetic CSVs:

- Iris classification;
- Wine classification;
- Breast cancer classification;
- Diabetes regression;
- synthetic churn;
- synthetic house prices.

Pros:

- easy to load;
- deterministic enough;
- good for testing an ML-engineering specialist;
- lets you evaluate against dummy baselines.

Cons:

- toy-like;
- may not reflect realistic ML engineering.

Recommended use:

- use as the first ML-engineering domain before attempting MLAgentBench or MLE-bench.

### Option D — MLAgentBench — later benchmark

MLAgentBench is a suite of end-to-end machine-learning experimentation tasks where an agent receives a dataset and task description and autonomously develops or improves an ML model.

Pros:

- directly aligned with the “ML engineer agent” idea;
- more realistic than toy sklearn tasks;
- evaluates end-to-end experimentation.

Cons:

- heavier than toy data analysis;
- more moving parts;
- may be difficult to integrate quickly.

Recommended use:

- stretch goal or future work;
- study its task format early, but do not depend on it for the first working prototype.

### Option E — MLE-bench — future work / inspiration

MLE-bench evaluates ML-engineering agents using 75 curated Kaggle competitions with standardized scoring.

Pros:

- very credible;
- strong match for ML-engineering agents;
- product-relevant if targeting serious autonomous ML workflows.

Cons:

- too heavy for an initial implementation;
- Kaggle-style tasks involve datasets, scoring scripts, runtime budgets, and significant engineering overhead.

Recommended use:

- cite as future direction;
- do not make it the MVP dependency.

### Option F — QuixBugs — optional QA baseline

QuixBugs contains 40 Python and Java programs, each with a one-line defect, corrected versions, and tests where available.

Pros:

- small;
- Python available;
- good for testing QA/code-repair loops;
- objective pass/fail feedback.

Cons:

- less aligned with data/ML direction;
- algorithmic bugs rather than data-science workflows.

Recommended use:

- optional smoke test if you want to demonstrate that the stem framework can support a non-data domain.

---

## 4. Recommended project scope

### Minimum credible project

Build a stem framework and evaluate it on toy data-analysis tasks.

Deliverables:

- runnable CLI;
- custom toy dataset;
- baseline data-analysis agent;
- stem-developed data-analysis agent;
- accepted/rejected skills;
- frozen workflow;
- before/after metrics;
- markdown report.

### Stronger project

Add a DSBench subset.

Deliverables:

- DSBench adapter;
- subset evaluation;
- comparison between toy and DSBench performance;
- analysis of where the stem mechanism generalizes or fails.

### Ambitious project

Add an ML-engineering specialization.

Deliverables:

- toy ML benchmark;
- ML specialist workflow;
- ML skill library;
- comparison of learned data-analysis vs ML-engineering skills;
- discussion of IDE-native Stem Agent Factory.

---

## 5. Metrics

### Data-analysis metrics

- `answer_accuracy`: exact or tolerance-based match against ground truth.
- `execution_success_rate`: generated code runs without error.
- `invalid_code_rate`: syntax errors, runtime errors, unsafe code, or no parseable answer.
- `avg_llm_calls`: average LLM calls per task.
- `avg_runtime_sec`: average execution time per task.
- `composite_score`:

```text
answer_accuracy
- 0.25 * invalid_code_rate
- 0.02 * avg_llm_calls
```

### ML-engineering metrics

- `valid_pipeline_rate`: pipeline trains and evaluates successfully.
- `score_vs_dummy`: improvement over a dummy baseline.
- `score_vs_simple_baseline`: improvement over a fixed sklearn baseline.
- `avg_llm_calls`.
- `avg_runtime_sec`.
- `composite_score`:

```text
normalized_model_score
+ 0.2 * improvement_over_dummy
- 0.25 * invalid_pipeline_rate
- 0.02 * avg_llm_calls
```

---

## 6. Key artifacts to save

Every development run should save:

```text
runs/<run_id>/
  config.yaml
  baseline_results.json
  developed_results.json
  accepted_skills/
  rejected_skills/
  frozen_workflow.yaml
  traces.jsonl
  summary.md
```

This is important for the write-up. The evaluators care about the path your thinking took, not only the final code.

---

## 7. Initial Codex prompt to initialize the code structure

Paste the following into Codex.

```text
You are implementing the initial skeleton for a project called StemDS.

Project goal:
Build a constrained “stem agent” framework that can differentiate into specialized data-science agents. The first target domain is toy data analysis: given a CSV and a natural-language question, the agent writes Python/pandas code, executes it, and returns an answer. Later, the same framework should support DSBench-style data-analysis tasks and a stretch ML-engineering domain.

Important:
Do not implement the full stem-development loop yet. First create a clean, runnable foundation with package structure, task schemas, sandbox execution, metrics, baseline agents, CLI, tests, and README.

Tech stack:
- Python 3.11+
- pandas
- pytest
- openai Python SDK, but unit tests must not require an API key
- python-dotenv optional
- plain Python first; no LangGraph, CrewAI, AutoGen, or heavy frameworks
- use dataclasses unless Pydantic becomes clearly useful

Repository/package name:
stemds

Create this structure:

stemds/
  __init__.py
  cli.py
  config.py
  llm.py
  tasks.py
  sandbox.py
  metrics.py
  agents/
    __init__.py
    base.py
    baseline.py
  skills/
    __init__.py
    base.py
    library.py
    seed.py
  workflows/
    __init__.py
    base.py
    executor.py
    candidates.py
  stem/
    __init__.py
    developer.py
    evaluator.py
    compiler.py
  reporting/
    __init__.py
    report.py

data/
  toy_analysis_train.jsonl
  toy_analysis_val.jsonl
  toy_analysis_test.jsonl
  toy_csvs/
    sales.csv
    employees.csv
    customers.csv
    orders.csv
    products.csv

tests/
  test_tasks.py
  test_sandbox.py
  test_metrics.py
  test_baseline_dummy.py

Other files:
  README.md
  pyproject.toml
  .env.example
  .gitignore

Core data model:
Implement a DataAnalysisTask dataclass with:
- task_id: str
- domain: str = "data_analysis"
- dataset_path: str
- question: str
- answer: str | float | int | bool
- answer_type: str  # one of: string, number, boolean
- tolerance: float | None
- tags: list[str]
- notes: str | None = None

JSONL format:
Each line is one DataAnalysisTask as JSON.

Create a toy dataset:
- about 10 train tasks
- about 5 validation tasks
- about 5 test tasks

Task examples:
- Which region has the highest total revenue?
- What is the average salary in Engineering?
- How many customers are from Germany?
- Which month has the highest number of orders?
- What product category has the largest average price?
- What percentage of orders were cancelled?
- Which employee has the highest sales total?
- What is the correlation between age and salary?
- Which customer placed the most orders?
- What is the total revenue after applying discounts?

Sandbox:
Implement sandbox.py with a PythonSandbox class.

It should:
1. Create a temporary working directory.
2. Copy the relevant CSV file into that directory or expose an absolute dataset path safely.
3. Write generated code to solution.py.
4. Run the code in a subprocess with timeout.
5. Capture stdout/stderr.
6. Return a SandboxResult dataclass containing:
   - status: one of "success", "syntax_error", "runtime_error", "timeout", "unsafe_code"
   - stdout: str
   - stderr: str
   - duration_sec: float
   - extracted_answer: str | None

Generated code contract:
The generated code must print the final answer on the last line prefixed with:
FINAL_ANSWER:

Example:
print(f"FINAL_ANSWER: {answer}")

Safety constraints:
For the first version, implement simple static checks before execution. Reject code containing:
- import os
- import subprocess
- import socket
- open(
- eval(
- exec(
- __import__
- pip
- requests
- shutil.rmtree

Only allow normal pandas/numpy/sklearn/matplotlib imports for now. Matplotlib is optional and should not be needed for tests.

Metrics:
Implement metrics.py with:
- TaskEvalResult dataclass:
  - task_id
  - expected_answer
  - predicted_answer
  - correct: bool
  - invalid_code: bool
  - sandbox_status
  - llm_calls: int
  - duration_sec: float
- aggregate_metrics(results) returning:
  - total_tasks
  - answer_accuracy
  - execution_success_rate
  - invalid_code_rate
  - avg_llm_calls
  - avg_runtime_sec
  - composite_score

Composite score:
answer_accuracy - 0.25 * invalid_code_rate - 0.02 * avg_llm_calls

Answer checking:
- string: case-insensitive normalized exact match
- boolean: accept true/false variants
- number: parse float and compare using tolerance if provided, otherwise exact float comparison with small default tolerance 1e-6

Agents:
In agents/base.py:
Create BaseDataAnalysisAgent abstract class/protocol:
- solve(task: DataAnalysisTask) -> AgentOutput

AgentOutput dataclass:
- code: str
- raw_response: str | None
- llm_calls: int
- metadata: dict

In agents/baseline.py:
Implement:
1. DummyBaselineAgent
   - returns hardcoded pandas code for a few known task_ids
   - returns a simple schema-inspection code otherwise
   - used for offline tests

2. OpenAIBaselineAgent
   - accepts model name and LLM client
   - prompts the model to generate Python/pandas code only
   - instructs the model to print FINAL_ANSWER on the last line
   - extracts fenced Python code if present
   - returns AgentOutput
   - tests must not require real OpenAI calls

LLM wrapper:
In llm.py:
Implement LLMClient abstraction and OpenAIResponsesClient.
Read OPENAI_API_KEY from environment.
Use the OpenAI Python SDK.
Keep wrapper minimal:
- generate_text(prompt: str, model: str, temperature: float = 0.2) -> str

CLI:
Use argparse.

Commands:
1. validate-data
   python -m stemds.cli validate-data --data data/toy_analysis_train.jsonl

2. run-baseline with dummy agent
   python -m stemds.cli run-baseline --data data/toy_analysis_test.jsonl --agent dummy --out runs/baseline_dummy.json

3. run-baseline with OpenAI
   python -m stemds.cli run-baseline --data data/toy_analysis_test.jsonl --agent openai --model gpt-4.1-mini --out runs/baseline_openai.json

For each task:
- load task
- agent generates code
- sandbox executes code
- extract predicted answer
- compare predicted answer with ground truth
- save per-task results and aggregate metrics to JSON

Skills/workflows/stem placeholders:
Create placeholder modules/classes only. Do not implement the full loop yet.

Add TODO comments:
- SkillLibrary will store accepted PromptSkills/PythonSkills.
- StemDeveloper will analyze baseline failures and propose candidate skills.
- WorkflowSearcher will evaluate candidate workflow configs.
- Compiler will freeze the best workflow.

Seed skills:
In skills/seed.py, define metadata-only seed prompt skills:
- schema_inspection
- groupby_aggregation
- date_parsing
- numeric_answer_verification
- missing_value_awareness

Workflow placeholders:
In workflows/candidates.py, define candidate workflow names:
- direct_code
- schema_then_code
- plan_then_code
- skill_retrieval_then_code
- code_then_verify

README:
Include:
- project goal
- setup:
  python -m venv .venv
  source .venv/bin/activate
  pip install -e ".[dev]"
  cp .env.example .env
  export OPENAI_API_KEY=...
- run tests:
  pytest
- validate data
- run dummy baseline
- run OpenAI baseline
- explain current limitation: first skeleton does not yet implement stem differentiation.

pyproject.toml:
Add dependencies:
- pandas
- numpy
- openai
- python-dotenv
- pytest under dev extras

Quality requirements:
- type hints
- readable code
- tests pass without OPENAI_API_KEY
- no heavy frameworks
- generated files should be small and inspectable
- use subprocess timeouts for execution

After implementation:
Run:
pytest
python -m stemds.cli validate-data --data data/toy_analysis_train.jsonl
python -m stemds.cli run-baseline --data data/toy_analysis_test.jsonl --agent dummy --out runs/baseline_dummy.json

Show me:
- files created
- commands run
- test results
- any known limitations
```

---

## 8. Suggested next Codex prompts after the skeleton

### Prompt 2 — Add OpenAI baseline robustness

```text
Improve the OpenAIBaselineAgent.

Goals:
- Add prompt templates for data-analysis code generation.
- Include CSV schema preview in the prompt: column names, dtypes, first 5 rows.
- Add robust code extraction from fenced code blocks.
- Add retry once if generated code does not print FINAL_ANSWER.
- Save traces for each task: prompt, raw response, code, sandbox result, predicted answer.
- Keep tests offline by mocking the LLM client.
```

### Prompt 3 — Add skill library MVP

```text
Implement the skill library MVP for StemDS.

Goals:
- Implement Skill, PromptSkill, PythonSkill metadata classes.
- Implement SkillLibrary save/load/search by tags.
- Store skills as YAML or JSON files.
- Add seed PromptSkills for schema inspection, groupby aggregation, date parsing, missing values, and numeric verification.
- Implement SkillRetrievalAgent that retrieves relevant skills by task tags and includes them in the code-generation prompt.
- Add CLI command run-skill-agent.
- Compare baseline vs skill agent metrics.
```

### Prompt 4 — Add stem development loop MVP

```text
Implement a simple StemDeveloper.

Goals:
- Run baseline on train tasks.
- Analyze failures using the LLM.
- Propose new PromptSkills only.
- Validate each proposed skill on validation tasks.
- Accept a skill only if composite_score improves over the current skill library.
- Save accepted and rejected skills with evidence.
- Save development traces.
- Add CLI command:
  python -m stemds.cli develop --train data/toy_analysis_train.jsonl --val data/toy_analysis_val.jsonl --out runs/dev_001
```

### Prompt 5 — Add workflow search and freezing

```text
Add workflow search and freezing.

Goals:
- Implement finite candidate workflow configs:
  direct_code
  schema_then_code
  plan_then_code
  skill_retrieval_then_code
  code_then_verify
- Evaluate each workflow on validation tasks.
- Save the best workflow to frozen_workflow.yaml.
- Add CLI command:
  python -m stemds.cli evaluate-frozen --workflow runs/dev_001/frozen_workflow.yaml --data data/toy_analysis_test.jsonl
- Add compare command that prints baseline vs developed metrics.
```

---

## 9. Sources to inspect

- DSBench: realistic data-analysis and data-modeling tasks for data-science agents.
- MLAgentBench: end-to-end ML experimentation benchmark for language agents.
- MLE-bench: OpenAI benchmark for ML-engineering agents using 75 Kaggle competitions.
- QuixBugs: small Python/Java program-repair benchmark with 40 one-line defects.
