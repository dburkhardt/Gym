# Run a campaign benchmark

Use the installed `gym` CLI from the selected fork and its qualified environment. The commands below are templates: replace `/absolute/...` with your reviewed configuration and artifact paths. See the [operator reference](campaign-operator.md) for resource admission, service ownership and recovery details.

## Choose and prepare

```sh
gym list benchmarks --json
gym eval prepare --config /absolute/run.yaml
gym env prefetch --config /absolute/run.yaml
```

Read the chosen benchmark's README and select its pinned data, frozen sample and compatible agent configuration. Preparation and prefetch can download or install dependencies. Record the source revision and asset hashes. Keep references inaccessible to the solver.

Select model and harness through the configuration, preserving the benchmark's scoring protocol and budgets. For this campaign, explicitly bind **every LLM role** to `nvidia/nvidia/nemotron-3.5-super-vl`, including judges and helpers. A solver override does not change a separate judge. Supply credentials through the approved environment mechanism.

## Plan, then run

```sh
gym eval run --config /absolute/run.yaml --output /absolute/attempt/results.jsonl --plan-only
```

Check `results_dispatch_plan.json` and `results_planned_inputs.jsonl` for exact task IDs, ordering, repetitions and resolved model admission settings. Planning does not execute the benchmark, but it can invoke preparation and its dependency setup. Use the approved prepared input/cache settings if no new setup is intended. It also does not produce execution's materialized-input artifact.

The run configuration must set the selected dataset split (for these benchmark datasets, `split: benchmark`, or pass `--split benchmark`). If it sets `dispatch_budget_s`, it must also set a finite `num_samples_in_parallel` (or pass the reviewed `--concurrency` value) or `max_resident_rollout_tasks`; otherwise native validation refuses the run. Keep the same settings in plan, execution and resume.

Before dispatch, verify actual component imports, assets, memory, disk, ports and the complete task-plus-cleanup allowance. Enable the reviewed dispatch guard, captures and owned supervisor configuration described in the operator reference. Check the current Brev account credit balance and projected burn, including retained storage and other organization usage. Preserve enough headroom to avoid exhausting account credits and report when a top-up is needed. Earlier $500/$1,000 campaign estimates are historical, not current spending limits. Do not purchase credits or introduce a new paid service without authorization.

```sh
gym eval run --config /absolute/run.yaml --output /absolute/attempt/results.jsonl
gym eval health-check /absolute/attempt --rollouts-file results.jsonl --json
```

Retain results, failure sidecars, captures, trajectories, materialized inputs and cleanup receipts. Report attempted, validly scored, unresolved and unstarted counts separately. A supervisor that cleaned up successfully may still contain a failed child run; inspect its child return code. An infrastructure failure is not an incorrect model answer. Inspect `results_failures.jsonl` alongside `results.jsonl`: a handled HTTP200 terminal judge failure can retain completed solver artifacts while leaving scientific and creator scores null. It contributes no reward, including no automatic zero. Genuine solver-budget zeros must come from the original scorer. See [terminal judge failures](campaign-operator.md#terminal-judge-failures-are-unscored).

## Resume or regrade

For an unchanged bound run, ordinary resume skips completed rows and occurrences marked `_ng_failure_terminal: true`:

```sh
gym eval run --config /absolute/run.yaml --output /absolute/attempt/results.jsonl --resume
```

An unknown accepted request must remain unresolved until reconciled. A definitive rejection may qualify for an explicitly selected new attempt using [guarded batch restart](../nemo_gym/batch-restart.md); changing the output directory is not reconciliation. Stateful agents require their own qualified recovery boundary.

To apply a reviewed scorer to saved outputs, use a new output and the **actual execution materialized-input file**:

```sh
gym eval reverify --config /absolute/regrade.yaml \
  --inputs /absolute/attempt/actual-materialized-inputs.jsonl \
  --rollouts /absolute/attempt/results.jsonl \
  --output /absolute/regrade/results.jsonl
```

Generic reverification requires a saved `response`. PromptBio's terminal judge sidecar deliberately lacks one; it needs a separately qualified benchmark-specific saved-artifact grading path, not a generic `--judge-failed-only` retry. Reverification of supported response records preserves solver outputs and records new scorer lineage. An LLM judge may still make inference calls; only a deterministic scorer or qualified saved-judge replay is zero-inference. Inspect evidence through the [rollout tools](../fern/versions/latest/pages/observability/rollout-evidence.mdx) and the benchmark's own report.

The six command families were checked against the current CLI help. These configuration templates are not a claim that an arbitrary benchmark is runtime-qualified.
