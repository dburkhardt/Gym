# Scientific evaluation campaign: operator entry point

Use the maintained Gym configuration for each benchmark. Keep its scientific inputs and scorer separate from the model, harness and execution settings. The campaign covers the frozen 63 candidates; a discoverable configuration or a passing offline plan is not a completed integration.

## Working sequence

1. **Discover and prepare.** `gym list benchmarks scienceboard --json` exercises native discovery. Follow the selected benchmark's README to stage its pinned assets and frozen task IDs. For example, [ScienceBoard](../scienceboard/README.md) requires creator source, while [BioMaze](../biomaze/README.md) takes a checksum-bound sample. Do not replace frozen selections with new samples.
2. **Plan without inference.** `gym eval plan --config /absolute/path/run.yaml --out /absolute/path/plan.json` writes task identities and input digests. This command path was exercised for BioMaze's remaining 15 tasks and ScienceBoard's 20. Inspect the plan and source receipts before admission. Planning can prepare missing data; it does not establish runtime resources or scorer validity.
3. **Bind the run.** Record source revision, actual component imports, asset hashes, model/harness settings and scoring protocol. Assign a fresh attempt output directory, capture directory and disjoint service ports. Use `nvidia/nvidia/nemotron-3.5-super-vl` for every live solver, judge and helper role in this campaign. Judges have separate configuration; changing a solver must not silently change the judge.
4. **Execute and inspect.** Use the selected integration's documented `gym eval run` command after its resource and scientific checks. Retain requests, responses, trajectories, verifier inputs, failures and native manifests. Inspect attempted, scored and unresolved counts separately; a budget-exhausted model answer remains an attempted model outcome, while an infrastructure failure is not an incorrect scientific answer.
5. **Resume or regrade deliberately.** A completed-output `--resume` must preserve identities and captures. Unknown provider outcomes require reconciliation before another request. An explicitly selected new attempt is governed by the [restart contract](../../nemo_gym/batch-restart.md), whose supported components are narrower than ordinary execution. Use `gym eval reverify` with a separate output and scorer record to grade saved responses; an LLM judge may make new judge calls even when there are no new solver calls.

Preserve original attempts when repairing a protocol. Compare systems only on matched tasks and scoring conditions. The shared controls establish specific recovery boundaries, not universal restoration of arbitrary agent or scientific-process state. Keep runtime-blocked candidates visible and respect the campaign's total compute/storage budget.

## Model and harness capability declarations

Native evaluation checks declared dataset requirements before starting services or collecting rollouts. MicroVQA requires `vision`; BrainBench requires `conditional_loglikelihood`. Supply declarations for the actual selected component instance in your run configuration, for example:

```yaml
component_capabilities:
  policy_model:
    vision: true
```

This is an operator assertion about the selected backend, not an automatic capability probe. Set `false` for known unsupported capabilities; omit unknown ones. Both unsupported and undeclared required capabilities fail closed. Changing the model does not establish that an old declaration remains true. In this campaign, only the authorized Nemotron model may be called; a successful configuration check does not authorize another model or spending.

Benchmark datasets may declare `required_model_capabilities` and `required_harness_capabilities`. The gate resolves the dataset's actual agent and its primary `model_server` reference. It rejects missing or incorrectly typed model bindings. Additional judge/helper roles can declare `required_model_roles`, explicitly naming a reachable component, its model-reference field and required capabilities; see the [role contract](../../fern/versions/latest/pages/evaluation/benchmarks.mdx). The validator follows those exact bindings and does not discover undeclared roles or arbitrary custom-agent fields. A taskset requiring validation without an explicit resolved agent fails closed. Untagged datasets receive no capability guarantee. These collection checks do not certify resource-only regrading configurations or actual provider capabilities.

Tests exercise the maintained MicroVQA and BrainBench configurations with unsupported, unknown and supported declarations without dispatching inference. BrainBench still requires a genuinely available likelihood transport; declaring support cannot create that implementation.

### Concurrent native services

Native `gym eval run` and `gym env start` acquire cooperative ownership of their resolved head and component ports before starting services. Concurrent Gym supervisors under the same local user cannot claim overlapping ports, including `localhost`, loopback and wildcard aliases. Conflicts fail before model dispatch; Gym does not silently change configured ports. Configure separate head ports and non-overlapping component ranges for simultaneous runs.

Ownership remains held until owned services are confirmed stopped. Normal shutdown and failed startup release it; supervisor process death releases kernel locks. Lock files are deliberately retained so two processes cannot lock different inodes for the same port. Durable service receipts still govern recovery of orphaned child processes; a port lock is not a session checkpoint.

This is conservative per-user coordination on one machine, not a socket reservation or a cross-host allocator. Disjoint interfaces with the same port are also rejected. Other users and non-Gym processes do not participate and can still win an OS bind race; native startup must fail rather than sending work to an unowned service. It does not coordinate arbitrary task-container or tool ports.
