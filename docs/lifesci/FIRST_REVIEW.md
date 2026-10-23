# First stage gate: review the integration process

The goal of this stage was to assess readiness across **63 candidates** and test the process on **three representative benchmarks** before expanding. We reviewed all 63 and completed the three worked examples.

## Overall expansion stages

| Stage | Goal |
|---|---|
| **1. Qualify the process — current review gate** | **Review the inventory, test three representative workflows, and agree on the process for expansion.** |
| 2. Complete benchmark setup | Confirm each benchmark’s scope and dependencies, finish its Gym integration, and test its scorer. |
| 3. Run validation samples | Run fixed samples with Nemotron and retain every task’s outputs and outcome. |
| 4. Independently verify results | Reproduce scores from saved outputs, inspect trajectories, and resolve execution problems. |
| 5. Publish the completed inventory | Give each of the 63 candidates a verified result or a documented blocker. |

Stages 2–5 will overlap across benchmarks as ready work advances.

## What we did and found

- **[ScienceQA](benchmarks/scienceqa.md):** Reviewed 50 retained Nemotron responses, tested a fresh local setup, and reproduced every score: **47/50 correct**. The declared text-only sample meets its engineering acceptance criteria. Only five questions are biology; Linux deployment remains to be checked.
- **[SciCode](benchmarks/scicode.md):** Ran 30 scientific coding problems with a larger output allowance. **Five complete problems passed**, and rescoring the saved programs reproduced every result. One generation still exhausted its allowance. A subsequent larger-allowance run scored 29 problems; a recording-system bug stopped one.
- **[CompChem](benchmarks/compchem.md):** Ran one molecular-descriptor task through Gym–Harbor. The agent produced the required files and passed the creator’s tests. Restoring and rescoring its saved workspace reproduced the passing result.

All three execution paths ran and reproduced scores. Scientific validity still needs separate assessment.

## What remains to pass this gate?

**The technical work for this first review is finished. The remaining step is your decision on whether to use this process across the inventory.**

SciCode’s recording bug, CompChem’s larger 20-task sample, and ScienceQA’s Linux check are follow-up work for the next stages. They prevent claiming those broader results are finished, but the first gate required worked examples and identified limitations rather than full acceptance of all three benchmarks.

## Recommendation

Proceed to stage 2 with three parallel assignments: fix and verify SciCode’s recovery, prepare CompChem’s 20-task run, and check ScienceQA on Linux. Then continue through the remaining candidates, running and independently reviewing each ready integration. Reuse existing machines initially.

Execution remains stopped pending your review.

[Full inventory](inventory.md) · [Completion plan, parallel assignments, and costs](PLAN.md)
