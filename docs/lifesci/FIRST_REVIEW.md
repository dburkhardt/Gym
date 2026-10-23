# First review: benchmark integration process

October 23, 2026. **Execution stops at this review. Further benchmark work requires Daniel’s instruction to continue.** The first goal covered readiness across all 63 candidates and representative tests of the acceptance process. The findings were ready about two hours after execution began, before the four-hour target, so there is no reason to spend the remaining time starting more evaluations.

## What reached each stage

| Outcome | Entries or tasks | What the count means |
|---|---:|---|
| Selected candidates | 63 | Entries containing an LLM or agent evaluation, a documented public scored release, and creator-reported results |
| Existing configurations | 59 | 36 direct-response, 21 multi-step, and two likelihood paths; configuration presence does not establish a completed run |
| Source and implemented scope reviewed | 63 | Fresh inspection of creator sources, existing code, dependency access, and available evidence |
| Historical scoring-check records | 41 entries | Earlier checks with varying sample sizes and coverage; not 41 uniformly accepted integrations |
| Accepted named engineering integration | 1 | ScienceQA text-only creator recipe, with 50 distinct validation tasks |
| Additional representative paths exercised | 2 | SciCode custom sequential coding and CompChem’s Harbor tool-using workflow; their remaining qualifications are explicit below |

The [inventory](inventory.md) and [machine-readable acceptance records](inventory.json) contain all 63 outcomes and next actions. Several integrations cover a named subset rather than a creator’s entire suite. For example, SMolInstruct currently covers two formula-conversion tasks, S²Bench covers AtomNum, and BIABench covers one wound-healing task. These subsets must remain visible in both names and coverage claims.

ScienceQA and SciCode are general-science examples with some life-sciences content. They qualify execution paths, rather than establish that the validation sample represents life-sciences customer workflows.

## Three worked examples

| Path | Completed work | Result and remaining qualification |
|---|---|---|
| [ScienceQA: direct response](benchmarks/scienceqa.md) | Inspected the retained exact-model calls, rebuilt a clean local environment, exercised the native CLI with valid and malformed fixture responses, rescored all 50 actual outputs, and independently reviewed five trajectories | **47/50**, with all scores reproduced. Accepted for the named text-only 50-task engineering sample. Only five questions are biology. A supervised Linux deployment remains a separate qualification. |
| [SciCode: custom scientific coding](benchmarks/scicode.md) | Ran the complete original 30 problems with 65,536 output tokens and rescored every generated program against the pinned tests. Then ran the same sample with a 131,072-token allowance | At 65,536 tokens, **5/30 whole problems and 56/116 functions passed**, and all 300 compared result fields matched on replay. One generation still returned no code. The larger condition scored 29 problems and retained one capture-system failure after four successful responses. It is not a complete accepted condition. |
| [CompChem: Harbor tool workflow](benchmarks/compchem.md) | Prepared the 43 pinned creator tasks, qualified native empty-agent/oracle controls, then ran one RDKit descriptor task using maintained Gym–Harbor components and replayed its saved workspace | The fresh task and its model-free replay both scored **1.0**. Four model exchanges were retained. The full benchmark has not completed its fixed 20-task acceptance sample. |

No solver output was selectively replaced with a better attempt. Historical and fresh samples remain separate. The only accepted task total is **50 distinct ScienceQA questions**; the repeated SciCode conditions are the same 30 problems and must not be added together as new tasks.

## What the process exposed

**Output allowance can prevent an answer from appearing.** In SciCode, 19 of 116 original generations exhausted a 16,384-token allowance entirely in reasoning. With 65,536 tokens, one did; at 131,072 tokens, none of the 114 retained calls did. That final run was incomplete because of an infrastructure failure. The conditions also used different concurrency settings, so these are descriptive observations, not a controlled causal estimate of accuracy or latency gains.

**Successful provider calls do not guarantee a completed task.** SciCode problem 67 retained four HTTP 200 responses, then failed while its capture guard read a pending marker that background cleanup had just removed. Two later calls were never dispatched. A model-free control reproduces the race and exercises a proposed narrow repair. The repair remains unapplied, and the original failure and responses are preserved. Continuing that problem requires a qualified recovery that does not resend its four successful calls.

**Clean setup needs to exercise the actual execution system.** A CompChem preflight initially missed that Ray requires a whole number for node CPU capacity, although individual task reservations can be fractional. The first attempt failed before any inference. A separate corrected attempt used two node CPUs while preserving the original task/container limits, and completed. Harbor’s existing resource patch also had to be enabled for both the solver and its separate verifier. These setup failures belong in the evidence; they cannot be hidden by a green import check.

**A reproduced score still needs interpretation.** ScienceQA’s released text recipe omits images and has a randomized answer-parser fallback. Other reviewed creators use permissive answer matching or require human scoring. We preserve creator behavior and report those limits. Engineering checks are not systematic scientific validation.

## Remaining work across the 63

These are mutually exclusive primary queue assignments. An entry can have additional difficulties beyond its assigned category.

| Primary next step | Count | Entries or scope |
|---|---:|---|
| Accepted named engineering sample | 1 | ScienceQA |
| Continue integration and measurement checks | 52 | Remaining ready or repairable paths, including SciCode and CompChem |
| Resolve creator access | 2 | ScienceAgentBench and VibeProteinBench |
| Obtain original source export and solver protocol | 1 | CSMeD |
| Qualify a likelihood backend | 2 | BrainBench and WMDP |
| Qualify provider image requests | 1 | ChemX |
| Obtain required scientific hardware | 1 | NatureBench |
| Decide supported scope or human scoring | 2 | MicroVQA++ and GeneHop |
| Complete a new implementation | 1 | AgentIdeaBench |

Access failures describe the checked route and environment, rather than prove that a resource does not exist. Repairable code remains implementation work. The public population counts in the inventory use different scientific units and must not be summed as completed evaluation tasks.

## Timing, resources, and costs

The two new SciCode conditions each captured requests across approximately **10.3 minutes**, with three concurrent problem slots. These spans exclude preparation and cleanup. The fresh one-task CompChem run, workspace replay, and supervised cleanup finished in approximately **80 seconds**. A clean ScienceQA dependency setup and CLI fixture ran locally; its 50 real model responses were retained from earlier execution. These observations do not establish the runtime of all scientific workflows.

No new machine was provisioned for this review. Live work reused an existing 8-vCPU, 32-GiB worker. Its retrieved catalogue quote was **$0.268/hour**, or about $1.34 for five hours of compute; retained storage is separate. Credit availability and account-wide billing were checked through Brev and retained privately. Account-wide spending includes other machines and delayed charges, so it cannot be attributed to these particular runs.

The full-project planning range remains **$700–$2,200** for compute and storage, with approximately **$5,100** as an extended scientific-repair scenario. These are capacity scenarios in [the completion plan](PLAN.md#brev-compute-and-storage-estimate), not a measured forecast derived from one RDKit task. Resolve shared setup and recovery defects before expanding the fleet. Sixty prepared evaluation jobs can run independently of the three available implementation agents; 60 idle machines would not accelerate source inspection or repairs.

## Recommended next goal

First qualify the capture-guard repair and a recovery that reuses SciCode’s four retained responses. In parallel, qualify supervised Linux execution for ScienceQA and expand CompChem to its fixed 20-task sample with the same native controls and saved-workspace replay. Give each implementation a separate reviewer.

Then keep a rolling queue of independent integrations. Prioritize ProteinLMBench, Mol-Instructions biotext QA, SMolInstruct formula conversion, SupraBench TBS base, LPM-24, and MolPuzzle after reviewing each named scope and generation settings. Run shared setup work once, reuse the verified environment, and continue preparing other benchmarks while admitted jobs execute remotely. Preserve all 63 as the project scope; external constraints receive explicit unresolved outcomes.

Before continuing, Daniel should review whether the worked examples provide sufficient engineering evidence, whether the clearly named general-science and partial-suite scopes are acceptable, and whether to proceed with this next wave. Expert scientific validation and customer-workflow prioritization remain separate workstreams.

**Review gate:** no new evaluations, worker provisioning, or remaining integration work may start while waiting. The goal is awaiting Daniel’s first review and must not be marked complete. Mark it blocked once the runtime’s required three consecutive blocked goal turns have occurred; keep the gate closed until then.
