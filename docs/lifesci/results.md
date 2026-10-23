# Selected execution and scoring results

These summaries describe completed records and their limitations. Source code is in this repository; raw trajectories, provider receipts, and large execution artifacts are retained separately. The uniform 63-benchmark acceptance review is specified in [PLAN.md](PLAN.md).

## ScienceQA

The accepted text-only 50-task engineering sample scored 47/50. Every saved score was reproduced without new model calls. The sample contains five biology questions; 23 source images are intentionally omitted under the creator’s text-completion recipe. A fresh macOS setup and deterministic CLI controls qualify that local path; Linux deployment remains separate. [Full report and sample](benchmarks/scienceqa.md).

## SciCode

The complete 65,536-token condition passed 5/30 problems and 56/116 generated functions. A fresh model-free scoring pass reproduced all 30 problem records across ten fields. One generation still exhausted its allowance. At 131,072 tokens, 29 problems received scores and one retained a capture-storage failure after four successful model calls. These conditions remain separate. [Full report, source, exact sample, and failure diagnosis](benchmarks/scicode.md).

## BLADE

For the AMTL example, a saved analysis and 15 ordered judge responses reproduced all ten reported scoring metrics without new model calls. The comparison covered hit rates and coverage of models, conceptual variables, and transformation representations. Repeated identical prompts had different recorded responses, so the replay had to preserve each occurrence in sequence. This establishes scoring reproducibility; it does not independently validate the judge's assessments.

[Upstream](https://github.com/behavioral-data/BLADE) · [Adapter](../../benchmarks/blade/README.md)

## BioAgent and NatureBench

A BioAgent single-cell scorer could award full credit for any one matching cluster-label pair. A successful execution therefore did not establish correct labeling of the remaining cells. NatureBench required immutable copies of the exact input files consumed by each evaluation because later agent actions could modify those files. A controlled test preserved three snapshots containing six files. These examples motivate separate execution, replay, and scientific-validity assessments.

[BioAgent configuration](../../benchmarks/bioagentbench/config.yaml) · [NatureBench configuration](../../benchmarks/naturebench/config.yaml)
