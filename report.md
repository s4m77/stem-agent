# StemDS: Validated Stem-Agent Development for Data Analysis

## 1. Problem framing

The challenge asks for a “stem agent”: a minimal system that reads signals from its environment and becomes specialized for a class of tasks. I interpret that as **constrained self-differentiation**, not model-weight training or arbitrary self-modifying code. A useful stem agent should be able to try a baseline approach, observe failures, propose or select changes, validate those changes, and only keep changes that improve measured performance.

StemDS, short for **Stem Agent for Data Science**, applies this idea to CSV-based data analysis. I chose data analysis because it gives executable feedback: the agent can generate Python/pandas code, run it in a sandbox, and compare the resulting answer against ground truth. That makes the development loop measurable rather than subjective.

The main benchmark is DABench/DAEval from InfiAgent-DABench. Raw benchmark data stays outside the repository under `external/`; an adapter converts it into StemDS JSONL tasks. The project is not a universal agent. It is a constrained, inspectable prototype for one task family.

## 2. Agent definition and system design

I distinguish between three levels. A model maps input context to output text. A workflow is a fixed sequence of model and tool calls. An agent observes state, chooses actions, uses tools, receives feedback, updates state, and stops under a condition.

StemDS has an inner solver and an outer development loop. The inner solver takes a question and CSV, prompts an OpenAI model for pandas code, executes that code, and scores the captured `ANSWER`. The outer loop observes failures and validation results, then proposes or selects candidate PromptSkills, AI-assisted developer-curated workflows, and generated workflow graphs. The workflow menu was not autonomously invented by StemDS: I used AI coding assistance while designing it, then selected, reviewed, and implemented the candidate workflows myself. A candidate is accepted only if it improves validation performance.

The architecture contains:

- a DABench adapter that converts DAEval records into `DataAnalysisTask` JSONL;
- a generic OpenAI baseline that writes pandas code assigning `ANSWER`;
- a subprocess sandbox with timeout and answer capture;
- metrics for accuracy, subquestion accuracy, execution success, invalid-code rate, LLM calls, and composite score;
- failure analysis by category and task tag;
- a PromptSkill library and `StemDeveloper` validator;
- AI-assisted developer-curated workflow search;
- generated workflow-graph search using a constrained DSL.

The development loop is:

```text
task family + benchmark
→ generic baseline
→ failure analysis
→ candidate skills/workflows
→ validation
→ accepted/frozen specialist
→ held-out test
```

Safety is deliberate. StemDS does not mutate the repository, generate arbitrary tools, or execute unbounded orchestration code. Generated analysis code runs in a controlled subprocess sandbox. PythonSkill generation is left for future work because it would require stronger isolation.

## 3. Benchmark and evaluation

DABench/DAEval was selected because it contains closed-form data-analysis questions over CSV files. I converted 257 tasks into StemDS format and used a deterministic split: 179 train tasks, 38 validation tasks, and 40 test tasks. Train is used for failure analysis and candidate proposal. Validation is used for accepting or rejecting candidates. Test is reserved for final held-out comparison.

The main metrics are:

- **Accuracy**: exact or tolerance-based answer correctness.
- **Subquestion accuracy**: pair-level correctness for DABench multi-answer labels such as `@name[value]`.
- **Execution success**: whether generated code completed successfully.
- **Invalid code rate**: whether code failed, timed out, or did not produce an answer.
- **Average LLM calls**: a cost proxy.
- **Composite score**: answer accuracy minus penalties for invalid code and LLM calls.

Composite score is useful because reliability matters, but it is a design choice. I therefore report raw accuracy separately from composite.

## 4. Experiments

### 4.1 Generic baseline

The generic baseline is a one-shot code-generation workflow. It inspects the CSV, asks the model for pandas code, and requires the final value to be assigned to `ANSWER`. It has no repair loop, no accepted skills, and no workflow differentiation.

On the fixed 40-task DABench test slice, the generic rerun produced:

| Metric | Value |
|---|---:|
| Accuracy | 0.375 |
| Composite | 0.33625 |
| Execution success | 0.925 |
| Invalid code rate | 0.075 |
| Subquestion accuracy | 0.6875 |
| Avg LLM calls | 1.000 |

### 4.2 Seed PromptSkills and PromptSkill validation

I first tried hand-authored seed PromptSkills, such as reminders to assign `ANSWER`, use the provided CSV path, and be careful with summary statistics. On an early 10-task smoke slice, these regressed: generic smoke accuracy was 0.300 with composite 0.155, while the seed-skill run scored 0.200 accuracy and 0.055 composite. This was not a final result, but it showed that “more instructions” is not automatically better.

After fixing the evaluator and split hygiene, `StemDeveloper` proposed three PromptSkills from train failures and validated each on the held-out validation split. It accepted zero:

| Skill | Validation composite delta | Decision |
|---|---:|---|
| `ps001` | -0.039 | rejected |
| `ps002` | -0.026 | rejected |
| `ps003` | -0.020 | rejected |

This is a useful negative result. The validator prevented harmful prompt-level changes from being accepted. PromptSkill-only adaptation did not improve validation performance, so the system correctly froze no PromptSkill changes.

### 4.3 AI-assisted developer-curated workflow search

The next axis was workflow search over an AI-assisted, developer-curated menu. I used AI coding assistance while brainstorming and implementing these prompt/control-flow architectures, but the important point is that the menu was still externally supplied: StemDS did not autonomously invent these workflows. StemDS evaluated them on the 38-task validation split:

| Workflow | Accuracy | Composite | Execution success | Invalid code rate | Selected |
|---|---:|---:|---:|---:|---|
| `direct_code` | 0.5263 | 0.4866 | 0.9211 | 0.0789 | no |
| `schema_then_code` | 0.5789 | 0.5392 | 0.9211 | 0.0789 | no |
| `plan_then_code` | 0.5789 | 0.5458 | 0.9474 | 0.0526 | no |
| `strict_answer_contract` | 0.3684 | 0.3353 | 0.9474 | 0.0526 | no |
| `code_then_repair` | 0.6053 | 0.5837 | 1.0000 | 0.0000 | yes |
| `plan_code_repair` | 0.5789 | 0.5579 | 1.0000 | 0.0000 | no |

StemDS selected `code_then_repair`. It beat `direct_code` by +0.0971 validation composite and was frozen as the specialist workflow. This is the strongest positive result, but it is selection-based rather than fully generative: the system selected from an AI-assisted, developer-curated workflow menu rather than generating the successful architecture itself.

### 4.4 Held-out DABench result

The final comparison uses the same fixed 40-task DABench test slice.

| Metric | Generic baseline | Frozen `code_then_repair` | Delta |
|---|---:|---:|---:|
| Accuracy | 0.375 | 0.450 | +0.075 |
| Composite | 0.33625 | 0.429 | +0.09275 |
| Execution success | 0.925 | 1.000 | +0.075 |
| Invalid code rate | 0.075 | 0.000 | -0.075 |
| Subquestion accuracy | 0.6875 | 0.750 | +0.0625 |

In raw counts, the generic baseline solved 15/40 tasks and the frozen `code_then_repair` workflow solved 18/40. The comparison artifact reports 4 improved tasks, 1 regressed task, and 35 unchanged tasks. Much of the gain comes from reliability: `code_then_repair` executes generated code, detects failure or missing `ANSWER`, and sends the error back to the model once for repair.

## 5. Generative workflow search: a negative result

A fair critique of the previous result is that it is closer to grid search than biological differentiation: I wrote the workflows, and StemDS selected among them. To address this, I added a constrained workflow-graph DSL. The model could compose safe primitives such as `schema_summary`, `llm_plan`, `llm_code`, `python_execute`, `llm_repair`, and `stop`, but it could not write arbitrary orchestration code or create unbounded loops. Generated graphs were structurally validated before execution and empirically validated before acceptance.

The model generated 3 workflow proposals. Two were structurally valid. The generated-workflow validation baseline was `direct_code` with accuracy 0.395, composite 0.362, and invalid-code rate 0.053. Both valid generated candidates scored accuracy 0.368, composite 0.315, and invalid-code rate 0.053. No generated workflow beat `direct_code`, so this path froze the baseline rather than accepting a generated graph.

The held-out generated-workflow path scored accuracy 0.300, composite 0.261, execution success 0.925, and invalid-code rate 0.075. Compared with the generic rerun, this was -0.075 accuracy and -0.075 composite. Compared with the AI-assisted developer-curated `code_then_repair` workflow, it was -0.150 accuracy and -0.168 composite.

This is a useful negative result. Generative self-assembly existed mechanically, but the generated architectures were worse. The validator rejected them, so the “immune system” worked. The next frontier is improving generation quality, not weakening validation.

## 6. What surprised me / what failed

The first surprise was that seed skills regressed. I expected basic reminders about answer format and CSV loading to help. Instead, naive skill injection reduced smoke performance. This pushed the design toward validation-first adaptation rather than accumulating prompt tips.

The second important result was that all generated PromptSkills were rejected. This looked like failure at first, but it is exactly the safeguard the challenge suggests: if something goes wrong mid-transformation, the system should pull back.

There were also evaluation bugs. DABench labels can contain multiple named answers such as `@mean[value]` and `@std[value]`; raw string comparison is wrong because order should not matter. I added name-based pair comparison with numeric tolerance. Missing dependencies such as `scipy` and `scikit-learn` also initially made generated code fail for environmental reasons rather than agent reasons.

Exact reproducibility remains imperfect. A seed is threaded through the CLI and OpenAI client, but the OpenAI client reported that the requested seed was ignored. I reran the generic baseline on the same test slice before comparing it to the frozen workflow. The frozen workflow still won, but the margin is a measured run, not a deterministic theorem.

The key lesson is that generation and inhibition are different capabilities. Inhibition worked: bad PromptSkills and worse generated workflows were rejected. Generation quality remains the hard part.

## 7. Limitations and future work

StemDS does not yet generate PythonSkills, create arbitrary tools, update model weights, or prove robust multi-domain specialization. A further limitation is that the successful workflow menu was not produced autonomously by the stem loop: I used AI coding assistance during design and implementation, then treated the resulting workflows as externally supplied candidates. The sandbox is suitable for cooperative generated code, not adversarial code. The main validated result is DABench-only. Composite score is a design choice and should not replace raw accuracy. The repair loop also increases cost: average LLM calls rose from 1.00 to 1.05 on the held-out test slice.

As a stretch demo, I added a small `ml_engineering` domain using built-in scikit-learn datasets. The generic ML baseline failed on the first five tasks because the model often overwrote provided path variables such as `TRAIN_CSV_PATH` with literal strings. A repair-oriented workflow achieved valid-run rate 1.0 and composite 0.537 on the same limited slice, with all five tasks beating the dummy baseline and four of five exceeding the minimum score. I treat this as a smoke signal that the framework can host another small task family, not as a benchmark-quality result.

I also inspected DSBench as a broader future benchmark. The assets are heterogeneous—Excel files, images, notebooks, text question files, and zipped archives—so full integration would require richer extraction and task-specific schemas.

The most important next step is PythonSkill generation under stronger isolation, for example Docker or firejail, with validation before any generated skill is accepted. A second step is multi-cycle development: alternate between failure analysis, skill proposal, workflow generation, validation, and test-time audit. More robust evaluation would also require multiple OpenAI reruns and confidence intervals.

## 8. Conclusion

StemDS does not solve fully autonomous stem-agent differentiation. It demonstrates a validated development loop: prompt-level adaptations were generated and rejected, AI-assisted developer-curated workflow selection improved held-out DABench performance, and generated workflow graphs were attempted but rejected by validation.

The main lesson is that a stem agent needs both generation and inhibition. In this prototype, inhibition and validation were reliable. The strongest held-out result came from selecting an AI-assisted developer-curated workflow: `code_then_repair` improved DABench test accuracy from 15/40 to 18/40 on the fixed 40-task slice. The generative layer was implemented, but its candidates were worse. That failure is part of the result: safe self-assembly is possible to attempt, but useful self-assembly remains the hard problem.
