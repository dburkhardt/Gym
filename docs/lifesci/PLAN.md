# Completing the Nemotron LifeSci Eval Expansion

## Objective and starting point

Implement, run, verify, and report every accessible benchmark among the [63 selected candidates](inventory.md). Each accepted integration must have a reproducible setup, an explicitly defined validation sample, retained trajectories, reproducible scores, and an inspectable report. Record a concrete external blocker for any benchmark that cannot be completed. A blocked entry remains unfinished; it is never counted as a successful integration.

The starting point is 59 entries with configurations, three partial implementations, and one source-access problem. Configuration presence does not establish that a workflow runs. The 41 documented scoring checks were collected retrospectively and vary in coverage. Apply the process below to all 63, reusing existing evidence only when it meets the same requirements. There is no separate 12-benchmark release target.

Codex performs implementation and review. **Nemotron 3.5 Super VL must fill every LLM role inside the benchmark**, including solver, planner, helper, and judge. Pin the actual endpoint model identifier and record any provider revision available. Do not silently substitute another model. Inference is assumed free for this project.

## 1. Establish the benchmark contract

For each entry, read the creator's dataset, code, and evaluation instructions. Record the release and source revisions, downloadable task count, supported task types, required tools, scorer, metric, and creator-reported model results. Compare the implemented path with the creator's procedure. Document adaptations such as a different judge model or missing subtask explicitly.

The unit of acceptance is a named benchmark release or a clearly named subset. For example, an AtomNum-only S²Bench integration cannot establish coverage of all nine subtasks. Audit scope before running more samples. Likelihood benchmarks require an endpoint that supplies the probabilities their procedure needs; a text-answer approximation does not satisfy that requirement.

Check data licenses and redistribution conditions. Keep restricted data, credentials, proprietary model outputs, and internal documents out of this public repository. Link to public sources and preserve private execution artifacts in an approved location.

**Output:** one contract per benchmark, linked from `inventory.json`, with any missing dependencies or unsupported scope identified.

## 2. Build and test the integration

An adapter loads the tasks, prepares prompts, supplies tools and execution state, invokes the scorer, and records the results in Gym. Reuse the creator's implementation and the existing Gym–Harbor path where appropriate. Pin dependencies and test setup from a clean worker so a developer's existing environment cannot hide missing packages or data.

The current 59 configurations comprise 36 single-response evaluations, 21 multi-step workflows, and two likelihood evaluations. Eleven multi-step workflows use Harbor; ten use other benchmark-specific implementations. Organize work by shared execution requirements so a common repair benefits several benchmarks.

Before generating paid-compute workloads, test the scorer with a known valid answer, an incorrect answer, and an empty or malformed output. For agents, also test missing artifacts and a failed tool call. Establish that execution errors cannot silently become ordinary incorrect answers or disappear from the denominator. These controls check implementation behavior; they do not establish the scientific validity of a reference answer.

**Output:** a documented setup command, a Gym run command, pinned dependencies, scorer controls, and an accessible runtime.

## 3. Run a small pilot, then a fixed validation sample

Start with up to five public tasks per implementation. Inspect actual prompts, model responses, tool calls, and scoring inputs. Confirm that the model receives the intended data, the tools execute, the judge uses Nemotron, and the output allowance permits a usable answer. Repair shared failures before scaling the worker pool.

Then freeze the task IDs and settings before running the validation sample. Planning defaults are **50 tasks for a direct-response or likelihood benchmark and 20 for a multi-step workflow**. Use the entire public set when smaller, and preserve an existing adequately sized sample when its protocol is suitable. Record the sampling method, seed, and coverage of subtasks. An unusually expensive native workflow may need a smaller declared sample; explain the reason and report its actual size.

Across the current configuration types, those defaults imply approximately **2,320 validation tasks**: 1,800 direct-response tasks, 420 workflow tasks, and 100 likelihood tasks. This is a planning count, contingent on availability and scope, not a count of completed tasks or the full released datasets. The four remaining candidates add work after their requirements are resolved. A workflow task can require many solver, helper, and judge calls.

Persist every task's terminal outcome, including model failure, infrastructure failure, generation truncation, and scoring failure. Retry infrastructure failures under a documented policy while retaining the original attempt. Do not replace unsuccessful model attempts selectively. A changed prompt, budget, or environment defines a new condition and requires a consistently evaluated sample.

For SciCode, rerun the complete original 30-problem sample under one larger supported output allowance. Nineteen original generations exhausted their allowance without producing visible code. Preserve the original condition, compare both conditions, and do not combine the best outputs from each.

**Output:** a complete fixed-sample run, with every selected task accounted for and generation limits reviewed.

## 4. Verify scoring and inspect the result

Score all saved outputs again without new solver generations. For code, execute the saved programs against the pinned tests. For LLM judges, replay the recorded judge responses in occurrence order, then separately inspect a sample of the judgments. Recorded-response replay verifies the scoring implementation; it does not establish that the original judgment was correct.

Compare task-level scores, exclusions, failure classifications, and aggregate metrics. Require exact equality for deterministic fields, and justify numerical tolerances for nondeterministic scientific software. Capture immutable copies of files at the moment a scorer reads them. Preserve repeated judge interactions individually, even when their prompts match.

Review at least five trajectories per benchmark, including a passing and failing case when available, and every distinct technical failure category. Inspect representative subtasks. Confirm that a successful score corresponds to the benchmark's declared success condition. Flag questionable creator scoring rules, such as credit for a single matching label in a larger labeling task, rather than silently rewriting the benchmark.

Resolve implementation errors and unusable output settings before acceptance. Genuine incorrect answers remain valid measurements. If a creator scorer has a substantive validity concern, report it clearly and withhold any stronger scientific conclusion. This review is engineering quality assurance, not systematic expert validation of the underlying science.

**Output:** task-level replay comparison, reviewed examples, reconciled counts, and explicit limitations.

## 5. Publish one outcome per benchmark

Use [inventory.json](inventory.json) as the acceptance record and generate its [readable table](inventory.md). Keep runtime scheduling state in the existing runner; avoid creating another competing run registry. Each benchmark report must link its upstream release, adapter, configuration, exact sample, score, replay result, examples, artifact location, and known limitations.

An accepted result requires all four criteria in the root README. Report successful integrations and their total distinct validation tasks separately from configured entries and blocked work. Never add overlapping historical samples to inflate the task total. Report external blockers with the failed command or access check, missing resource, attempted remedy, and next action. Repairable implementation work stays on the queue.

The final report should explain how many of the 63 reached each stage and why others did not. Keep the root README as the research summary, this file as the execution plan, and per-benchmark reports as the evidence. Preserve original Gym documentation and the fork relationship for later upstream contributions.

## Parallel execution and elapsed time

Start with existing qualified machines or two to four workers to validate the shared path. Expand to approximately six to twelve small CPU workers, two to six high-memory workers, and up to two GPU workers when admitted jobs justify them. Build reusable environments and cache public dependencies; isolate benchmark execution so one job cannot modify another's files. Give each integration a single owner to avoid conflicting edits, and use bounded retries and checkpointed jobs.

There are currently **three available implementation-subagent slots plus the coordinating agent in this chat**. Sixty remote evaluation jobs do not require 60 Codex agents: a scheduler can dispatch already prepared jobs independently. Sixty machines become useful only when enough validated work is queued and capacity, quotas, and endpoint throughput support them. Starting all of them during dependency repair would mostly pay for waiting and repeated downloads.

The initial endpoint planning assumption is 500 requests per five minutes per IP. Verify actual provider limits, including any key-wide or organization-wide limits, before provisioning for throughput. Distinct worker IPs do not establish unlimited aggregate capacity. Respect all applicable limits, use backoff, and measure tokens and latency as well as request counts.

Budget the first 4–8 hours for the uniform contract and dependency review and another 4–12 hours for shared repairs and pilots; these activities can overlap. Some existing integrations can finish much sooner. A practical target is useful accepted results within 12–24 hours and the accessible set in **two to five days**. Complex scientific environments may take **five to seven days or longer**. Missing public data or required hardware prevents a reliable completion-date promise.

These estimates should be replaced with observed pilot timings. As a sensitivity check, 420 workflow tasks taking 15 minutes to four hours each require 105–1,680 execution-slot hours. Sixteen usable concurrent slots imply roughly 7–105 hours of execution alone. Four concurrent tasks on each high-memory machine is an assumption to test, not a guaranteed capacity. Direct-response tasks are likely to be much faster, but long reasoning, judge calls, and retries still matter.

## Brev compute and storage estimate

The [machine-readable estimate](cost-estimate.json) records live Brev catalogue quotes retrieved on October 23, 2026. An 8-vCPU, 32-GiB GCP CPU machine was $0.268/hour; a 32-vCPU, 128-GiB machine was $1.071/hour. A stoppable single-A100 80-GB option was $1.98/hour. These are catalogue prices, not reservations or proof of capacity. The A100 quote did not specify storage billing; confirm it before launch.

| Scenario | Small CPU workers | High-memory workers | GPU workers | Compute | Storage reserve | Total including 50% contingency |
|---|---|---|---|---:|---:|---:|
| Focused completion | 6 × 48 hours | 4 × 48 hours | 2 × 24 hours | $397 | $67 | **$696** |
| Broader repair allowance | 12 × 96 hours | 6 × 96 hours | 2 × 72 hours | $1,242 | $231 | **$2,210** |
| Long scientific tail | 12 × 168 hours | 8 × 168 hours | 4 × 120 hours | $2,974 | $431 | **$5,108** |

Each row also includes a small coordinator for 72, 120, or 168 hours. Storage reserves assume 6.5 TB for 48 hours, 9 TB for 120 hours, or 12 TB for 168 hours, respectively, at $0.15622/GB-month using 730 hours/month. Confirm which disks are included in instance pricing before billing estimates are finalized. GPU hours support native scientific tools; Nemotron inference remains remote and free under the project's assumption.

A different available GPU can change the estimate. A quoted GCP A100 40-GB option was $4.408/hour; using it for the same GPU hours adds approximately $117, $350, or $1,165 before contingency. It is an alternative only when its memory and hardware satisfy the benchmark. Do not substitute hardware when doing so changes the evaluation contract.

For comparison, **60 small CPU workers for one day cost about $385 in compute, or $462 including 250 GB of storage per worker**. That excludes heavy workers, GPUs, and the coordinator. Parallelism can shorten elapsed time; it cannot remove dependency work or make unsupported tasks executable.

Use **$700–$2,200 as the initial planning range**, with approximately $5,000 as a plausible extended-repair scenario. These figures exclude prior spending, unrelated running machines, taxes, data egress, Codex usage, and any charge for inference. They estimate validation samples, not full-dataset evaluation of every task against multiple models.

Before launching, verify available Brev credits and the account's current burn rate. Reconcile accrued costs and projected next-day spending daily, and notify Daniel before credit exhaustion is likely. Do not purchase credits automatically. Stop task-owned idle workers after exporting and verifying their artifacts, within the approved lifecycle scope. Retained disks can continue billing after a VM stops: an additional 2 TB at this storage rate is about $312/month. Preserve artifacts and confirm retention needs before deleting storage. Do not alter unrelated machines.

## Deliverables and completion review

Deliver a prominent root README, 63 explicit inventory outcomes, reproducible commands for accepted integrations, linked per-benchmark reports, private retained trajectories where required, and a reconciled compute/storage record. Include accepted benchmark and task counts, technical blockers, scoring concerns, and any differences from creator protocols. The implementation review should also identify which integrations are ready for an upstream Gym pull request.

Start execution only after the plan is activated. No new instances or model runs were launched while preparing this document.
