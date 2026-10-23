# First review: testing the integration process

We reviewed all **63 benchmark candidates**, then tested the integration process on **three worked examples**. We started with 59 configurations; the goal was to establish a reliable process for completing the integrations.

**Current checkpoint: review the three examples before applying the process across the inventory.** One meets its declared acceptance criteria; two need follow-up.

## Stages and current position

| Stage | Goal | Where we are |
|---|---|---|
| 1. Define the benchmark | Identify the dataset, supported tasks, dependencies, and scoring procedure. | Sources and scope reviewed for all 63; some access and protocol questions remain. |
| 2. Test the integration | Confirm setup works and the scorer handles correct, incorrect, and malformed outputs. | Tested direct answers, scientific coding, and a tool-using agent. |
| 3. Run a fixed sample | Run Nemotron with saved prompts, responses, tool activity, and outcomes. | Used 50 ScienceQA questions, 30 SciCode problems, and one CompChem task. |
| 4. Verify the results | Rescore saved outputs and inspect examples to check that execution and scoring behaved as intended. | Reproduced scores for all three examples; identified remaining issues below. |
| 5. Publish and accept | Document what ran, its results, and its limitations. | Three reports published. ScienceQA’s named 50-question sample is accepted. |

## What we did and found

- **[ScienceQA](benchmarks/scienceqa.md):** Reviewed 50 retained Nemotron responses, tested a fresh local setup, and reproduced every score. Accuracy was **47/50**. This text-only sample is accepted as an engineering integration; only five questions are biology. Linux deployment still needs verification.
- **[SciCode](benchmarks/scicode.md):** Reran the same 30 scientific coding problems with a larger output allowance. **Five complete problems passed**, and rescoring the saved programs reproduced every result. One generation still exhausted its allowance. A subsequent larger-allowance run scored 29 problems; a recording-system bug stopped one.
- **[CompChem](benchmarks/compchem.md):** Ran one molecular-descriptor task through Gym–Harbor. The agent generated the required files, passed the creator’s tests, and passed again when we restored and rescored its saved workspace. The planned 20-task sample remains to be run.

All three paths ran and reproduced scores. Scientific validity still needs separate assessment.

## Recommendation and decision

**Continue across the 63 candidates, starting with three parallel assignments:** fix and verify SciCode’s recovery, expand CompChem to 20 tasks, and verify ScienceQA on a Linux worker. Reuse existing machines initially. As each path passes review, continue with the next ready benchmark.

The decision now is whether this process is sufficient to proceed with that next wave. Execution remains stopped pending review.

[Full inventory](inventory.md) · [Completion plan, parallel assignments, and costs](PLAN.md)
