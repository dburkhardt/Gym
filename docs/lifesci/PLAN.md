# Completing the Nemotron LifeSci Eval Expansion

## Overall expansion stages

| Stage | Goal |
|---|---|
| **1. Qualify the process — current review gate** | **Review readiness across all 63 candidates, test three representative workflows, and obtain Daniel’s first review.** |
| 2. Complete benchmark setup | Confirm scope and dependencies, finish each integration, and test its scorer. |
| 3. Run validation samples | Run fixed Nemotron samples and retain every task’s outputs and outcome. |
| 4. Independently verify results | Reproduce scores, inspect trajectories, and resolve execution problems. |
| 5. Publish the completed inventory | Record a verified result or documented blocker for every candidate. |

The first goal stops at stage 1’s review gate. After approval, stages 2–5 overlap across benchmarks: ready entries advance while others are still being prepared. The detailed benchmark tasks below support those stages.

## First /goal: establish the process and stop for Daniel’s review

**Scope this first goal to the review checkpoint below.** The overall project remains all 63 candidates. This goal establishes their readiness, exercises the common acceptance process on representative integrations, and produces a concrete proposal for the remaining work. The later sections describe the full process; they do not authorize continuing past this checkpoint without Daniel’s review.

Start the elapsed-time clock when execution begins and persist the start time, checkpoint deadline, assigned work, and current job IDs. **Target the first review at four hours, and stop admitting new work no later than five hours.** Notify Daniel earlier if the review packet is ready, or if a material access, credit, or scientific-protocol decision prevents progress. Do not wait a day: the most useful early review is whether the integration and verification process is producing trustworthy evidence before applying it broadly.

The review packet should contain:

- A status for all 63 candidates: source and scope reviewed, dependency readiness, current stage, and the next action. Label unfinished inspections explicitly rather than treating them as blockers.
- Three worked examples covering a direct-response evaluation, a Harbor tool-using workflow, and a custom multi-step workflow. Prefer existing integrations with retained evidence so review time goes toward verification. Also check likelihood endpoint support early. These examples qualify execution paths; they do not replace the 63-candidate scope.
- For each example, its actual prompt, response or tool trajectory, scorer controls, replay comparison, sample size, and remaining limitations. Aim for accepted fixed samples where feasible; five-task pilots with explicit missing checks are sufficient for the first process review, but do not count as accepted releases automatically.
- Measured setup and task durations, concurrent-job capacity, model-call counts, available credits, accrued compute/storage estimates, and a revised estimate for completing the remaining entries. Identify unavailable billing data honestly.
- A recommendation for the next execution wave, changes to the process, and the specific decisions Daniel needs to make. Save this as `docs/lifesci/FIRST_REVIEW.md`, with links to public-safe evidence and approved private artifact locations.

**The five-hour checkpoint is a time limit, not a promise that every review target will be achieved.** Deliver the packet with the evidence available then and explain any shortfall. Do not extend the initial goal to finish an attractive benchmark. Stop new admissions, stop replacement jobs, and checkpoint running tasks using the runtime’s supported mechanism. Configure bounded job timeouts before launch. Record anything that cannot safely stop immediately, its expected drain time, and its cost; only artifact preservation and safe shutdown may continue after the checkpoint.

**Mandatory review gate:** notify Daniel in this chat, link the packet, and state that further execution requires his review. Do not launch additional evaluations, provision workers, or advance the remaining integration work while waiting. Do not mark the goal complete merely because the checkpoint has been reached.

**Goal status:** mark the goal `blocked` with the reason “Awaiting Daniel’s first review” once the goal runtime permits that status. The current runtime requires the same blocking condition to persist for at least three consecutive goal turns, including the turn that first encounters it, with no meaningful work possible without user input. Until that threshold is met, keep the review gate closed and report the pending review; do not create artificial turns or perform more benchmark work to fill the wait. If the runtime’s rule changes, follow its current requirements. Daniel’s instruction to continue opens the gate for the next explicitly scoped goal.

## Overall objective and starting point

Implement, run, verify, and report every accessible benchmark among the [63 selected candidates](inventory.md). Each accepted integration must have a reproducible setup, an explicitly defined validation sample, retained trajectories, reproducible scores, and an inspectable report. Record a concrete external blocker for any benchmark that cannot be completed. A blocked entry remains unfinished; it is never counted as a successful integration.

The starting point is 59 entries with configurations, three partial implementations, and one source-access problem. Configuration presence does not establish that a workflow runs. The 41 documented scoring checks were collected retrospectively and vary in coverage. Apply the process below to all 63, reusing existing evidence only when it meets the same requirements. There is no separate 12-benchmark release target.

Codex performs implementation and review. **Nemotron 3.5 Super VL must fill every LLM role inside the benchmark**, including solver, planner, helper, and judge. Pin the actual endpoint model identifier and record any provider revision available. Do not silently substitute another model. Inference is assumed free for this project.

## Work required for each benchmark

The following five tasks are the per-benchmark checklist. They support the overall expansion stages above; the first goal exercises them on representative examples before broader execution.

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

The final report should explain how many of the 63 reached each stage and why others did not. Keep the root README as the research summary, this file as the execution plan, and per-benchmark reports as the evidence. Preserve the fork relationship and upstream implementation for later Gym contributions.

## Parallel subagent ownership at every stage

Use every available subagent slot for independent work, and replenish slots as assignments finish. At startup, inspect actual concurrency limits; this chat currently supports **three subagents plus the coordinator**. Reuse agents or finish assignments before spawning replacements. Do not assume nested agents increase the limit. The coordinator owns dispatch, shared infrastructure changes, integration of commits, inventory updates, cost accounting, and the review checkpoint.

Use a rolling queue: a benchmark can advance as soon as its prerequisites pass. Do not wait for all 63 contracts to be complete before implementing or piloting a ready entry. Each benchmark has one implementation owner, and a different agent performs its acceptance review. Partition writes by benchmark directory or isolated worktree. Agents return structured results to the coordinator; they must not concurrently edit the root README or shared inventory. Changes to shared adapters, model clients, and scheduler code have one assigned owner and must be integrated before dependent jobs run.

| Stage | Parallel subagent work | Coordinator and dependency gate |
|---|---|---|
| 1. Contracts and dependencies | Initially divide the 63 entries into three approximately equal batches, balanced by complexity. Independently read sources, inspect existing evidence, test dependency access, and identify execution requirements. Return results incrementally. | Merge observations, prioritize shared blockers, and admit ready entries immediately. Require the source, scope, and scoring contract before a live pilot. |
| 2. Adapters and controls | Assign independent ready benchmarks to all available agents. One agent can fix a shared issue while the others work on unaffected paths. Each owner tests clean setup and correct/incorrect/malformed scorer controls. | Serialize shared-file changes, review interfaces, and reject controls that conceal errors. Shared infrastructure work must not block unrelated benchmark paths. |
| 3. Pilots and validation | Agents inspect pilots for their assigned paths and prepare immutable sample manifests. Dispatch admitted jobs to independent remote workers; agents continue preparing other benchmarks while jobs execute. | Own worker capacity, rate limits, job leases, timeouts, and credits. Admit a full sample only after its pilot passes. Stop admitting work at the first-review deadline. |
| 4. Replay and review | As soon as a sample finishes, assign replay and trajectory inspection to an agent other than its implementer. Replays on remote workers can run alongside generation jobs on other benchmarks. Return defects to the owner. | Accept only complete evidence; track implementation and independent-review states separately. Reserve review capacity so implementation cannot indefinitely outrun verification. |
| 5. Reports | Benchmark owners prepare reports in separate files while reviewers confirm counts, examples, and limitations. | Integrate the single inventory and README, reconcile task counts and spending, and deliver the first-review packet. Public publication requires a source/privacy check. |

For the first review, begin with three discovery assignments and nominate one candidate per representative execution path immediately. As incremental source reviews return, move a slot onto pilots and another onto independent review; keep remaining discovery moving. At approximately three hours, reserve one slot for independent verification and the review packet. At four hours, assess readiness and notify Daniel if the packet is ready. At five hours, enforce the stop regardless of unfinished batch work. A waiting model job should not occupy an agent slot that could do useful inspection or implementation.

The first goal may run fixed samples for ready entries before the checkpoint, but must not launch long, uncheckpointable work that is expected to extend materially beyond it. Defer such jobs to the reviewed next wave. Maximize useful parallel work within the review boundary rather than expanding the initial goal’s duration.

## Remote execution and elapsed time

Start with existing qualified machines or two to four workers to validate the shared path. Expand to approximately six to twelve small CPU workers, two to six high-memory workers, and up to two GPU workers when admitted jobs justify them. Build reusable environments and cache public dependencies; isolate benchmark execution so one job cannot modify another's files. Give each integration a single owner to avoid conflicting edits, and use bounded retries and checkpointed jobs.

There are currently **three available implementation-subagent slots plus the coordinating agent in this chat**. Sixty remote evaluation jobs do not require 60 Codex agents: a scheduler can dispatch already prepared jobs independently. Sixty machines become useful only when enough validated work is queued and capacity, quotas, and endpoint throughput support them. Starting all of them during dependency repair would mostly pay for waiting and repeated downloads.

The initial endpoint planning assumption is 500 requests per five minutes per IP. Verify actual provider limits, including any key-wide or organization-wide limits, before provisioning for throughput. Distinct worker IPs do not establish unlimited aggregate capacity. Respect all applicable limits, use backoff, and measure tokens and latency as well as request counts.

The first goal timeboxes contract inspection, shared repairs, and representative pilots to the four-to-five-hour review window above. Across the full project, allow roughly 4–8 hours for contract and dependency review and another 4–12 hours for shared repairs and pilots; these activities overlap and unfinished work is proposed at the checkpoint. Some existing integrations can finish much sooner. A practical target is useful accepted results within 12–24 hours and the accessible set in **two to five days**. Complex scientific environments may take **five to seven days or longer**. Missing public data or required hardware prevents a reliable completion-date promise.

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

## Full-project deliverables after the first review

Deliver a prominent root README, 63 explicit inventory outcomes, reproducible commands for accepted integrations, linked per-benchmark reports, private retained trajectories where required, and a reconciled compute/storage record. Include accepted benchmark and task counts, technical blockers, scoring concerns, and any differences from creator protocols. The implementation review should also identify which integrations are ready for an upstream Gym pull request.

Start execution only after the plan is activated through /goal. The first activation ends at the mandatory review gate; later work requires Daniel’s instruction to continue. No new instances or model runs were launched while preparing this document.
