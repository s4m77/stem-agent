# StemDS Experiment Summary

StemDS means Stem Agent for Data Science. It is a constrained stem-agent prototype for data-analysis tasks. In this experiment, differentiation happens by validating prompt skills and workflows rather than updating model weights.

## Experimental setup

- Benchmark: DABench/DAEval
- Model: gpt-4.1-mini
- Generic baseline test tasks: 40
- Frozen workflow test tasks: 40
- Raw DABench data is kept external and converted into StemDS JSONL artifacts.
- Seed was requested, but the OpenAI client reported that the requested seed was ignored.

## Generic baseline

| metric | value |
|---|---:|
| accuracy | 0.375 |
| composite | 0.336 |
| execution success | 0.925 |
| invalid code rate | 0.075 |
| avg LLM calls | 1.000 |
| total tasks | 40 |

- One-shot generic analysis agent.
- Writes pandas code and assigns the final value to `ANSWER`.
- No repair loop, workflow search, or learned specialization is used.

## Seed-skill regression

Hand-authored seed skills were scaffolding, not autonomous stem-generated skills.

| metric | value |
|---|---:|
| accuracy | 0.200 |
| composite | 0.055 |
| execution success | 0.500 |
| invalid code rate | 0.500 |
| avg LLM calls | 1.000 |
| total tasks | 10 |

| metric | delta vs generic |
|---|---:|
| accuracy | -0.100 |
| composite | -0.100 |
| execution success | +0.000 |
| invalid code rate | +0.000 |

Naive skill injection did not reliably help in this comparison.

## StemDeveloper PromptSkill validation

- Proposed skills: 3
- Accepted skills: 0
- The validator rejected all PromptSkill candidates, so harmful prompt-level changes were not accepted.

| skill_id | score_delta | reason |
|---|---:|---|
| `ps001` | -0.039 | Rejected: composite delta -0.039 did not exceed min_delta 0.050. |
| `ps002` | -0.026 | Rejected: composite delta -0.026 did not exceed min_delta 0.050. |
| `ps003` | -0.020 | Rejected: composite delta -0.020 did not exceed min_delta 0.050. |

PromptSkill-only differentiation did not improve validation performance in this run.

## Human-authored workflow search validation results

| workflow_id | accuracy | composite | execution_success | invalid_code_rate | selected |
|---|---:|---:|---:|---:|---|
| `direct_code` | 0.526 | 0.487 | 0.921 | 0.079 | no |
| `schema_then_code` | 0.579 | 0.539 | 0.921 | 0.079 | no |
| `plan_then_code` | 0.579 | 0.546 | 0.947 | 0.053 | no |
| `strict_answer_contract` | 0.368 | 0.335 | 0.947 | 0.053 | no |
| `code_then_repair` | 0.605 | 0.584 | 1.000 | 0.000 | yes |
| `plan_code_repair` | 0.579 | 0.558 | 1.000 | 0.000 | no |

- Frozen workflow: `code_then_repair`
- Validation composite delta vs `direct_code`: +0.097
- This is the strongest positive DABench result, but it is selection from a human-authored workflow menu.

## Frozen workflow held-out test result

- Frozen workflow: `code_then_repair`

| metric | value |
|---|---:|
| accuracy | 0.450 |
| composite | 0.429 |
| execution success | 1.000 |
| invalid code rate | 0.000 |
| avg LLM calls | 1.050 |
| total tasks | 40 |

| metric | generic | frozen workflow | delta |
|---|---:|---:|---:|
| accuracy | 0.375 | 0.450 | +0.075 |
| composite | 0.336 | 0.429 | +0.093 |
| execution success | 0.925 | 1.000 | +0.075 |
| invalid code rate | 0.075 | 0.000 | -0.075 |

- Held-out accuracy improved for the frozen workflow.
- Part of the improvement comes from lower invalid-code rate and repair-loop reliability.
- Comparison artifact: `runs/stem/dev_004/generic_rerun_vs_frozen_workflow.json`.

## Generated workflow search negative result

The generated workflow search attempted a more generative path using a constrained workflow-graph DSL. The model proposed 3 graphs; 2 were structurally valid, but neither beat the `direct_code` validation baseline.

| workflow_id | accuracy | composite | execution_success | invalid_code_rate | selected |
|---|---:|---:|---:|---:|---|
| `direct_code` | 0.395 | 0.362 | 0.947 | 0.053 | fallback |
| generated candidates | 0.368 | 0.315 | 0.947 | 0.053 | no |

- Frozen generated-workflow path: `direct_code`
- Held-out generated path: accuracy 0.300, composite 0.261, execution success 0.925, invalid code rate 0.075.
- Delta vs `runs/stem/dev_004/test_generic_rerun.json`: accuracy -0.075, composite -0.075.
- Delta vs human-authored `code_then_repair`: accuracy -0.150, composite -0.168.
- Full summary: `reports/generative_workflow_search_summary.md`.

## What this proves and what it does not prove

### Proves

- The system can run a constrained stem loop.
- It can reject harmful PromptSkills.
- It can search over a human-authored workflow menu.
- It can freeze and evaluate a selected workflow.
- Human-authored workflow selection improved held-out DABench performance in this run.
- It can attempt generated workflow-graph search and reject worse generated candidates.

### Does not prove

- Universal agent behavior.
- Arbitrary self-rewriting.
- Model-weight learning.
- Robust determinism across all OpenAI runs.
- PythonSkill generation.
- Benchmark-quality multi-domain specialization.

## Limitations

- OpenAI seed may be ignored depending on API path.
- Results may vary due to LLM nondeterminism.
- DABench answer-format assumptions may affect scoring.
- PromptSkill generation was limited.
- No PythonSkill generation is implemented.
- Only a small ML-engineering smoke extension is implemented; it is not benchmark-quality.
- Generated workflow search did not beat the `direct_code` validation baseline.
- Repair workflows cost extra LLM calls when first attempts fail.
- The sandbox is suitable for cooperative generated code, not adversarial code.

## Suggested write-up bullets

- Data analysis was chosen because it has measurable task outcomes and sandboxable generated code.
- DABench/DAEval was chosen as the first real benchmark beyond toy smoke tests.
- PromptSkills came first because they are inspectable, reversible, and easy to validate.
- Naive skill injection and proposed PromptSkills did not reliably improve validation performance.
- The useful result was that the validator rejected harmful candidates rather than accepting them blindly.
- Human-authored workflow search succeeded because repair improved execution reliability and reduced invalid code.
- Generated workflow search is a useful negative result: generation existed mechanically, but validation rejected worse generated architectures.
- With more time, add PythonSkill generation, stronger determinism controls, and additional benchmarks.

## Mini ML-Engineering Extension

A small sklearn-based ML-engineering smoke extension exists, but it is secondary to the DABench result and not benchmark-quality because workflow search and evaluation used the same limited task slice.
