# SciCode: scientific code generation

**Scoring reproduced for a fixed 30-problem sample**, reviewed October 23, 2026. Nemotron 3.5 Super VL passed 56 of 116 generated functions, and all functions within 5 of 30 complete problems, with a 65,536-token output allowance. A fresh scoring pass reproduced every result without new model calls. A subsequent 131,072-token condition finished 29 problems and exposed a capture-storage race on the remaining problem; that condition is not accepted as a complete run.

## Source, sample, and protocol

[SciCode](https://github.com/scicode-bench/SciCode) evaluates scientific Python code through problems divided into functions. Each prompt includes scientific background and the code generated for earlier functions. The verifier executes the accumulated code against the released tests and numerical reference data. This is a scientific coding workflow with a custom sequential harness; the model does not operate an interactive terminal or receive test feedback between functions.

The [public dataset](https://huggingface.co/datasets/SciCode1/SciCode) was pinned to revision `4510f6a6aa27c43fad7b43da2c59602a86e88480`. We retained the same previously frozen 30 problems for all three conditions. These contain 117 declared functions; problem 62's first function is supplied by the benchmark, leaving 116 generated and scored functions. The sample spans scientific computing, including physics, numerical methods, and ecological modeling. It is not a life-sciences-only sample or a random estimate of the full benchmark's accuracy. The [sample manifest](scicode-sample.json) lists every problem ID, per-problem outcome, source hash, and protocol setting.

[Gym configuration](../../../benchmarks/scicode/config.yaml) · [Setup instructions](../../../benchmarks/scicode/README.md) · [Agent](../../../responses_api_agents/scicode_agent/app.py) · [Verifier](../../../resources_servers/scicode/app.py) · [Saved-output reconstruction](../../../benchmarks/scicode/saved_regrade.py)

All model requests use `nvidia/nvidia/nemotron-3.5-super-vl`, temperature 0, no tools, one provider attempt, and a 600-second provider timeout. The scientific code runs in the same pinned Docker image, with two verifier processes, one CPU and 4 GiB per container, and a 30-second timeout per function. The canonical numerical-reference file is checked by SHA256. The executed maintained source was revision [`7e645a6af`](https://github.com/dburkhardt/Gym/tree/7e645a6afac7b51f096b1e25e8492116c71ffdbd). Its [agent](https://github.com/dburkhardt/Gym/blob/7e645a6afac7b51f096b1e25e8492116c71ffdbd/responses_api_agents/scicode_agent/app.py) and [verifier](https://github.com/dburkhardt/Gym/blob/7e645a6afac7b51f096b1e25e8492116c71ffdbd/resources_servers/scicode/app.py) identify the executed versions; relative links above show the current implementation. Exact scientific files are also bound in the manifest. The provider does not expose a model-weight hash.

## Results and scoring verification

| Observation | 16,384-token condition | 65,536-token condition | 131,072-token condition |
|---|---:|---:|---:|
| Problems with a final scientific score | 30/30 | 30/30 | 29/30 |
| Whole problems passing every scored function | 4/30 | 5/30 | 4/29 completed problems |
| Functions passing tests | 47/116 | 56/116 | 51/110 scored functions |
| Retained model requests, all HTTP 200 | 116 | 116 | 114, including four on the failed problem |
| Generations exhausting their output allowance | 19 | 1 | 0 among the 114 retained calls |
| Input tokens | 196,040 | 221,467 | 211,708 |
| Output tokens, including reasoning | 593,646 | 673,904 | 631,082 |
| Captured request span | 1,697.089 seconds | 616.430 seconds | 618.695 seconds |
| Concurrent problem slots | 1 | 3 | 3 |
| Fresh saved-output scoring | No new claim here | 30 problems; all 300 compared fields matched | Not executed |

The 131,072-token counts include the four successful generations from problem 67, although that problem has no scientific score. Its two remaining generations were not dispatched. Keep that missing problem visible when interpreting the completed subset. Request span excludes preparation and cleanup. Concurrency changed between the first and later conditions, so these timings do not isolate the effect of output allowance. Single runs, even at temperature 0, also do not establish that score differences were caused by the allowance change.

For the 65,536-token condition, the independent preparer reconstructed each function's accumulated Python program from its exact captured response and preceding functions. It checked every prompt and generation setting, joined results by problem ID, and rejected missing, duplicated, reordered, or changed evidence. The fresh scoring pass executed those same programs against the same tests, reference data, image, and timeout. Ten fields matched for every problem: reward, mask, failure kind, failure reason, individual function results, measurement-validity findings, passing-function count, total-function count, problem accuracy, and function accuracy. All 116 execution receipts confirmed container cleanup; no model call was made during this scoring pass.

Independent terminal checks confirmed that the 65,536-token model run, its scoring replay, and the 131,072-token model run had stopped: their owner processes and four native service scopes were absent, no SciCode verifier container remained, and every retained execution receipt confirmed cleanup. The final 131,072-token run contains 110 scientific execution receipts. The run supervisor's successful exit alone does not imply that every benchmark task succeeded.

## Inspected examples

We inspected complete trajectories for six scored problems and the failed capture sequence. The public IDs below identify the corresponding entries in the [released dataset](https://huggingface.co/datasets/SciCode1/SciCode).

| Public problem ID | Example from the 131,072-token condition | Observed outcome |
|---|---|---|
| 14 | Brownian motion in an optical tweezer | First function passed; second reached the unchanged 30-second scientific timeout. |
| 53 | Stochastic Lotka–Volterra simulation | Three functions passed; the final function failed a numerical assertion. |
| 57 | Harmonic-oscillator numerical solution | Two of five functions passed. The fifth now returned visible code, but failed its numerical test. |
| 59 | Variational quantum eigensolver | All five functions passed. |
| 61 | X-ray coordinate conversion | Four of five functions passed; the first failed a numerical assertion. |
| 56 | Temporal ecological niches | All four functions passed. |
| 67 | Layered-electron-gas Dyson equation | Four responses were captured; a storage-guard exception stopped execution before the fifth of six generations. No scientific score was produced. |

Problem 57's fifth generation used its entire 65,536-token allowance for reasoning and returned no code in the earlier condition. At 131,072 tokens it completed with visible code and failed the released test. This distinguishes an absent answer from a completed answer that did not pass. The original outcomes remain separate.

## Infrastructure finding and remaining qualification

The problem-67 failure occurred in `CaptureStore.bind_logical_call`, while the next call scanned a pending capture marker. The preceding successful call had already been durably appended to its capture file; background cleanup removed its marker between directory enumeration and reading. The file name in the traceback exactly matches the fourth successful captured call. This caused a `FileNotFoundError` before the fifth provider dispatch. All four responses and the original failure are preserved.

A model-free regression reproduces the race. A proposed narrow repair accepts a disappeared marker only when its exact durable completed capture exists with matching lineage. Controls refuse missing evidence, changed requests, and incomplete captures, and replay a settled identical call rather than sending it again. This proposal has not been applied to the live source or used to continue problem 67. No persisted native checkpoint exists for that failed run, and the maintained restart path does not authorize restarting a generic agent failure with successful earlier calls.

The next review can admit a partial scoring replay of the 29 completed problems, explicitly retain problem 67 as unresolved, and decide whether to develop and qualify a continuation that reuses its four exact responses. No continuation, replacement generation, or partial replay has been run. Reproducible scoring establishes that the integration can execute and rescore these saved programs; scientific reference validity, broader sample coverage, and agreement with published model results remain separate checks.
