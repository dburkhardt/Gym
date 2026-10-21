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

Check `results_dispatch_plan.json` and `results_planned_inputs.jsonl` for exact task IDs, ordering, repetitions and resolved model admission settings. Planning does not execute the benchmark. It also does not produce execution's materialized-input artifact.

Before dispatch, verify actual component imports, assets, memory, disk, ports and the complete task-plus-cleanup allowance. Enable the reviewed dispatch guard, captures and owned supervisor configuration described in the operator reference. Keep this campaign within its remaining $500 total compute/storage cap.

```sh
gym eval run --config /absolute/run.yaml --output /absolute/attempt/results.jsonl
gym eval health-check /absolute/attempt --rollouts-file results.jsonl --json
```

Retain results, failure sidecars, captures, trajectories, materialized inputs and cleanup receipts. Report attempted, validly scored, unresolved and unstarted counts separately. A supervisor that cleaned up successfully may still contain a failed child run; inspect its child return code. An infrastructure failure is not an incorrect model answer.

## Resume or regrade

For an unchanged bound run, ordinary resume preserves completed rows:

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

Reverification preserves solver outputs and records new scorer lineage. An LLM judge may still make inference calls; only a deterministic scorer or qualified saved-judge replay is zero-inference. Inspect evidence through the [rollout tools](../fern/versions/latest/pages/observability/rollout-evidence.mdx) and the benchmark's own report.

The six command families were checked against the current CLI help. These configuration templates are not a claim that an arbitrary benchmark is runtime-qualified.
