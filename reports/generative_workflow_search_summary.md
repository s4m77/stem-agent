# Generative Workflow Search Summary

This extension lets StemDS propose workflow graphs from failure analysis instead of only selecting from a human-authored menu.
The generated graphs are constrained by a safe DSL and accepted only if validation performance improves.

## Proposal Summary

- Generated proposals: `3`
- Structurally valid proposals: `2`
- Frozen workflow: `direct_code` baseline fallback
- Differentiated: `False`
- Baseline composite: `0.361579`
- Selected composite: `0.361579`
- Min delta: `0.030000`

## Generated Candidates

- `invalid`: invalid; Malformed workflow spec: 'id'; errors: 'id'
- `plan_code_execute_normalize_stop`: valid; Workflow graph parsed and validated.; errors: none
- `plan_code_execute_answercheck_stop`: valid; Workflow graph parsed and validated.; errors: none

## Validation Results

| workflow_id | accuracy | composite | execution_success | invalid_code_rate | outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| `direct_code` | 0.395 | 0.362 | 0.947 | 0.053 | baseline fallback |
| `plan_code_execute_normalize_stop` | 0.368 | 0.315 | 0.947 | 0.053 | rejected |
| `plan_code_execute_answercheck_stop` | 0.368 | 0.315 | 0.947 | 0.053 | rejected |

## Comparison With Human-Authored Workflow Search

Human-authored workflow search remains the current validated DABench headline path. This generative layer tested whether the stem loop could create safe workflow architectures, not just choose from a predefined grid. In this run, no generated workflow improved validation performance.

## Held-Out Evaluation

The frozen generated workflow was `direct_code` because no generated candidate beat validation baseline by `min_delta`.

Held-out 40-task DABench test metrics:

- accuracy: `0.300`
- composite: `0.261`
- execution_success: `0.925`
- invalid_code_rate: `0.075`
- subquestion_accuracy: `0.729`

Comparison against `runs/stem/dev_004/test_generic_rerun.json`:

- accuracy delta: `-0.075`
- composite delta: `-0.075`
- execution_success delta: `0.000`
- invalid_code_rate delta: `0.000`

Comparison against the human-authored frozen `code_then_repair` workflow:

- accuracy delta: `-0.150`
- composite delta: `-0.168`
- execution_success delta: `-0.075`
- invalid_code_rate delta: `+0.075`

This result is a negative but useful generative-workflow result: the DSL proposal layer worked, validation rejected non-improving generated candidates, and the held-out generated path did not replace the stronger human-authored `code_then_repair` workflow.

## Limitations

- v0 linearizes validated graphs rather than implementing a fully general graph engine.
- `answer_normalize` and `llm_answer_check` are reserved no-op nodes in this pass.
- Generated workflows are still constrained to fixed primitives and bounded repair loops.
- Results may vary because OpenAI API seeding is not fully deterministic.
