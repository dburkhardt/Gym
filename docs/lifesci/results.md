# Selected execution and scoring results

These summaries describe completed records and their limitations. Source code is in this repository; raw trajectories, provider receipts, and large execution artifacts are retained separately. The uniform 63-benchmark acceptance review is specified in [PLAN.md](PLAN.md).

## SciCode

The reviewed sample contains 30 problems and 116 model-generated functions. Four whole problems and 47 functions passed their tests. One creator-provided function is excluded from the generated-function denominator. All 30 problem records were reproduced by executing the same saved programs in the same scoring environment, without model calls.

The comparison checked reward, exclusion flag, failure kind, failure explanation, per-function results, measurement-validity metadata, passed-function count, total-function count, whole-problem accuracy, and function-level accuracy. All matched. These fields are implementation consistency checks, not ten independent scientific validations.

Review of all 116 captured responses found 19 with a 16,384-token limit, exactly 16,384 reasoning tokens, and no visible answer. The generation setting therefore needs a controlled follow-up before using the result as a broad measure of coding capability. Use the same complete 30-problem sample, verify the endpoint's supported limits, and choose one adequate allowance for the revised condition. Preserve the original results. Replaying only the failed generations and replacing their scores is not an unbiased comparison.

[Upstream](https://github.com/scicode-bench/SciCode) · [Adapter](../../benchmarks/scicode/README.md) · [Saved-generation reconstruction](../../benchmarks/scicode/saved_regrade.py)

## BLADE

For the AMTL example, a saved analysis and 15 ordered judge responses reproduced all ten reported scoring metrics without new model calls. The comparison covered hit rates and coverage of models, conceptual variables, and transformation representations. Repeated identical prompts had different recorded responses, so the replay had to preserve each occurrence in sequence. This establishes scoring reproducibility; it does not independently validate the judge's assessments.

[Upstream](https://github.com/behavioral-data/BLADE) · [Adapter](../../benchmarks/blade/README.md)

## BioAgent and NatureBench

A BioAgent single-cell scorer could award full credit for any one matching cluster-label pair. A successful execution therefore did not establish correct labeling of the remaining cells. NatureBench required immutable copies of the exact input files consumed by each evaluation because later agent actions could modify those files. A controlled test preserved three snapshots containing six files. These examples motivate separate execution, replay, and scientific-validity assessments.

[BioAgent configuration](../../benchmarks/bioagentbench/config.yaml) · [NatureBench configuration](../../benchmarks/naturebench/config.yaml)
