# Nemotron LifeSci Eval Expansion

[First review](docs/lifesci/FIRST_REVIEW.md) · [Execution plan and costs](docs/lifesci/PLAN.md) · [All 63 benchmarks](docs/lifesci/inventory.md) · [Selected results](docs/lifesci/results.md) · [Running an evaluation](docs/campaign-quickstart.md)

This repository retains the NeMo Gym fork structure for development and eventual upstream contributions.

## Introduction

I wanted to test whether coding agents could rapidly expand the breadth of scientific benchmarks available in [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym). The aim was to produce reusable integrations that others could run, inspect, and extend, while retaining the model trajectories for later failure analysis.

I used Codex to work through 63 benchmark candidates from a 302-entry scientific evaluation inventory. We created or extended Gym configurations for 59 entries and collected execution and scoring evidence across a range of scientific tasks. The work focused on running the evaluations, preserving their trajectories, and reproducing their scores. We did not systematically assess the scientific validity of the benchmarks or their answers.

## Methods

### 1. Selecting the benchmarks

From the 302-entry inventory, we identified **63 entries containing an LLM or agent evaluation, a documented public scored release, and creator-reported model results**. We then checked whether the data, scoring software, and runtime dependencies could be obtained and used. Blockers included an inaccessible ScienceAgentBench archive, missing CSMeD evaluation documents, restricted VibeProteinBench source data, and unavailable likelihood-scoring support for WMDP. Some agent workflows also required specific GPU hardware or repairs to their execution environment.

The work produced 59 configured Gym entries, three partial implementations, and one unresolved source-access case. Execution and scoring checks established different levels of readiness within those 59. The [integrated inventory](docs/lifesci/inventory.md) records the result and remaining work for every entry. The [original selection logic](docs/lifesci/inventory.md) records how the 63 were identified.

A **benchmark** consists of a dataset and its evaluation procedure. A **task** is one question or problem within it.

### 2. Characteristics of the configured benchmarks

Of the **59 configured entries, 36 generate one response per task**, covering multiple-choice questions, numerical answers, molecular structures, and structured text. **Twenty-one use multi-step workflows** for activities such as writing scientific code, running computational chemistry tools, analyzing data, or searching for evidence. **Two use likelihood scoring**, which compares the probabilities the model assigns to alternative texts. Thus, 38 entries avoid an interactive agent environment; some still require specialized scoring software.

The multi-step integrations include **11 using Gym–Harbor** and **10 using other benchmark-specific agent implementations**. These counts describe the implemented evaluation paths; a benchmark can support additional workflows beyond the subset evaluated here. [Configuration audit](docs/lifesci/inventory.md).

### 3. Building the adapters

An **adapter** is the code and configuration that connects a benchmark to Gym. It loads the task data, constructs the model instructions, provides any required tools and execution environment, and passes the output to the benchmark's scorer. It also records the task identity, configuration, trajectory, and result.

The initial source review identified **three existing Gym adapters and three upstream Harbor adapters** that could be reused or extended. Other entries required new integration components or additional source preparation. For the 36 direct-response evaluations, the adapter could use Gym's single-response agent and a benchmark-specific scorer. The 21 multi-step workflows required an agent loop or a sequence of model calls, with the tools and software needed by that benchmark. [Initial source review](docs/lifesci/inventory.md).

### 4. Running and verifying the evaluations

We exercised adapters on fixed samples, generally expanding from five tasks to 20–50. Nemotron 3.5 Super VL filled the benchmark's LLM roles, including solver and judge. We retained prompts, responses, tool interactions, generated files, and scoring outputs for later analysis.

We then scored saved outputs again. For a coding task, this means executing the Python program the model already wrote and repeating its tests. For an LLM-assisted scorer, we can replay the recorded judge responses to verify the surrounding scoring code. The reviewed records contain these checks for **41 benchmark entries**, with coverage ranging from small subsets to 50-task samples.

### 5. Defining a verified integration

For a benchmark or explicitly named subset to enter the verified release, it must have:

1. **A reproducible setup:** pinned source and data versions, documented dependencies, and a tested Gym command.
2. **A complete validation sample:** fixed task IDs, retained outputs, and a result or explained disposition for every task. Execution failures must be resolved; generation limits must be checked for suitability.
3. **Reproducible scoring:** the saved outputs reproduce the reported scores, with numerical tolerances documented where needed.
4. **An inspectable result:** task counts, scores, trajectories, supported scope, and known limitations are linked from its report.

These criteria establish an engineering result. Scientific validity remains a separate review.

## Results

### Current implementation coverage

The October 23 review inspected the source and implemented scope of all 63 candidates. **One named integration has completed the uniform engineering checks: ScienceQA’s text-only 50-task sample.** Its sample includes five biology questions, so it is a general-science process example. SciCode has a complete 30-problem scoring replay, with remaining generation and infrastructure qualifications. CompChem provides the Harbor workflow example. The [first review](docs/lifesci/FIRST_REVIEW.md) separates these outcomes and the next steps.

| Outcome | Benchmark entries | Evidence |
|---|---:|---|
| Selected for integration | 63 | Public scored release and creator results identified in the inventory review |
| Gym configuration present | 59 | Tracked configuration and implementation paths inspected |
| Partial implementation | 3 | ScienceAgentBench, CSMeD, and AgentIdeaBench |
| Source access unresolved | 1 | VibeProteinBench |
| Historical saved-output scoring checks recorded | 41 | Earlier checks of varying scope, linked in the integrated inventory |
| Source and implemented scope reviewed | 63 | Fresh source inspection and appropriate model-free checks |
| Accepted named engineering integration | 1 | ScienceQA text-only protocol; 50 distinct validation tasks |

The 41 scoring-check records overlap the configured entries and were collected retrospectively. Final release acceptance requires all four checks above for the declared sample. The [execution plan](docs/lifesci/PLAN.md) applies the same acceptance process to all 63 candidates. Each will receive an accepted result with a reconciled task count or a specific blocker with supporting evidence.

### SciCode: from a scientific problem to an inspectable result

[SciCode](https://github.com/scicode-bench/SciCode) asks a model to write Python code for scientific problems. A problem is broken into smaller programming steps: the model writes a function for one step, then receives its earlier code as context when writing the next. The benchmark executes the functions against supplied tests. A whole problem passes only when all of its scored steps pass.

We evaluated the same fixed 30 problems under 16,384- and 65,536-token output allowances. They required 116 generated functions; one additional function was supplied by the benchmark. The first condition passed 4 whole problems and 47 functions. The larger-budget condition passed 5 whole problems and 56 functions. The number of generations that exhausted their allowance without visible code fell from 19 to one.

To verify the second result, we reconstructed the accumulated Python program for each function from its recorded model response and reran the released tests in the same pinned environment. All 30 problem records matched across ten compared fields, including function-level results, whole-problem results, and failure classifications. This scoring pass made no model calls.

A third condition used the provider’s advertised 131,072-token ceiling. It produced scores for 29 problems: 4 complete solutions and 51 of 110 tested functions passed. One problem stopped after four successful model responses when a pending capture marker disappeared during a filesystem read. Its two remaining generations were never sent. We retained that failure and the four responses; the 29-problem subset is not presented as a complete 30-problem result.

These are descriptive runs, not a causal estimate of the effect of more reasoning tokens. Concurrency also changed from one problem slot in the original run to three in the later runs. The [SciCode report](docs/lifesci/benchmarks/scicode.md) links the upstream dataset, adapter, exact sample, separate conditions, scoring replay, inspected examples, and remaining qualifications.

### BLADE: reproducing an LLM-assisted scorer

[BLADE](https://github.com/behavioral-data/BLADE) evaluates agents performing open-ended data analysis. Its scorer examines aspects of the analysis rather than relying only on a single final answer. This makes it a useful test of whether an integration can preserve a more complex evaluation process.

For the AMTL example, we replayed a saved analysis and **15 saved judge responses** through the scorer. All ten reported metrics matched the original values without new model calls. An initial replay implementation incorrectly assumed that identical prompts could share one cached response. In fact, some prompts had occurred several times and received different answers. The corrected replay preserves each response in its original sequence. This tests reproducibility of the scoring implementation; it does not establish that the judge's original assessments were correct. See the [benchmark paper](https://aclanthology.org/2024.findings-emnlp.815/), [our integration](https://github.com/dburkhardt/Gym/tree/dburkhardt/life-sci-evals/benchmarks/blade) and [replay results](docs/lifesci/results.md).

### What execution checks revealed about interpretation

Some workflows ran successfully but exposed limits in what their scores meant. In a BioAgent single-cell analysis example, the scorer could award full credit when **any one predicted cluster label matched a reference label**. That does not establish that the remaining cell labels or the analysis were correct. We retained the raw result while flagging its interpretation. The [BioAgent entry](docs/lifesci/inventory.md) links the implementation and evidence.

NatureBench exposed a different issue: an agent can modify a file after an evaluator has read it. Saving only the final file is then insufficient to reproduce an earlier evaluation. We implemented and tested preservation of the exact input files supplied to each evaluation. The controlled test retained three snapshots containing six files despite subsequent changes. This established file-preservation behavior, not a valid model score. See the [NatureBench integration](https://github.com/dburkhardt/Gym/tree/dburkhardt/life-sci-evals/benchmarks/naturebench) and [snapshot review](docs/lifesci/results.md).

## Interpretation and limitations

The work demonstrates that coding agents can contribute integrations across a broad set of scientific benchmarks and preserve enough execution evidence to inspect and reproduce selected results. It does not yet quantify how much faster this is than manual integration: we did not run a human baseline or consistently measure engineering time per benchmark.

The resulting collection is best treated as an engineering and analysis resource. Sample sizes, task types, and metrics differ, and some implementations use explicitly documented variants of creator settings. Scores should not be averaged across benchmarks or compared directly with published leaderboards without matching those conditions. Systematic expert assessment of scientific validity, representative customer-workflow coverage, and model comparisons remain separate studies.

The [first review](docs/lifesci/FIRST_REVIEW.md) proposes the next execution wave. Further work should qualify recovery from the observed capture failure, carry each ready integration through the same checks, and use retained trajectories to investigate weak grading rules. Expert review remains necessary before using these results to prioritize scientific training data.
