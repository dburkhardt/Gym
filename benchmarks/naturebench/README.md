# NatureBench native Gym integration

`config.yaml` registers the preserved NatureBench Harbor task selection through
Gym. It uses the shared guarded `Terminus2NemoGym`, the creator's timed-agent and
Docker extension, and native separate-verifier execution. No bespoke scheduler
or external launcher is required. Live acceptance is still pending.

## Prepare and plan

Set `NATUREBENCH_FROZEN_SAMPLE` to the preserved `pilot-5.jsonl` and run:

```sh
gym eval plan --config benchmarks/naturebench/config.yaml --out naturebench-plan.json
```

Preparation preserves every original row field and adds a stable `task_id` alias
from `instance_id`; `.source.json` binds original and staged checksums. This
planning operation does not prove task archives, images or GPUs are available.
Set `NATUREBENCH_SAMPLE_SHA256` to require a particular source checksum.

To materialize the exact frozen-five task packages, additionally set
`NATUREBENCH_ARCHIVES_DIR` to the directory containing their `.tar.gz` files.
`frozen_sources.json` pins release archive checksums at dataset revision
`5527b0b8c9fa1fea207678238bf40140a5dca5a2`. Extraction rejects links, traversal,
unsupported entries and duplicate names. Each task is published into a fresh
directory after checksum validation; existing task state is never overwritten.
Receipts preserve original and prepared file hashes. Only the judge provider
routing is patched; scientific evaluators, prompts, data and task configuration
are preserved. Remove the archives environment variable for subsequent planning
of already-materialized tasks, rather than overwriting their directories.

The frozen five require four low-tier GPUs (creator RTX 3090/4090 contract) and
one high-tier GPU. Prior CPU calibration trials are outside this selection and
must not be substituted or counted as frozen-five results. Explicit CPU/RAM
resource contracts and immutable image identities still require admission.

## Protocol and guarded roles

Creator extension source is pinned at
`0f80a67d47a56fcd08ab3f1f586159d264616e2f` with its MIT license retained.
The effective solve budget remains **14,400 seconds**, excluding evaluation
service time. The task's **43,200-second** agent timeout remains a distinct wall
clock safety limit. The maintained wrapper additionally settles child tasks on
cancellation and distinguishes timer transport failure from budget exhaustion.

Solver settings preserve the campaign's no-summarization, interleaved-thinking,
49,152 input / 16,384 output context declarations, 16,384 output request budget,
and 1,200-second model transport timeout. The shared agent supplies durable
logical-call identities. Arbitrary full shell/session restart is not claimed.

The separate `NatureBenchVerifier` delegates execution/scoring to Harbor's
native verifier. Its dynamically allocated, bearer-authenticated local bridge
forwards the creator Responses request to the rollout-qualified Gym model route,
with a durable fresh scorer lineage and configuration digest covering task and
all verifier file bytes. Different judge payloads within one verification fail
closed. Existing verifier directories cannot silently restart; regrade uses a
new native scorer record. Set `NATUREBENCH_BRIDGE_HOST` to a verified host address
reachable from the separate verifier container (typically its Docker bridge).
The default loopback is suitable for local controls, not container acceptance.

Every benchmark LLM role must use `nvidia/nvidia/nemotron-3.5-super-vl`. The judge
replacement is an explicit protocol variant: creator prompt, verdict parser and
scientific evaluator remain intact. Native `reward_from_verdict` can retain a
numeric improvement when validity is unknown. Preserve that raw reward but
report unresolved validity separately; it is not a validated scientific result.

## Verification boundary

Focused controls cover source preservation and duplicate rejection, immutable
archive staging, timer-error cancellation settlement, judge authentication,
rollout/call binding, changed-request rejection and endpoint cleanup using a
local fake provider. The actual shared timed-agent constructor and native Gym
five-task plan have been exercised. No real task-container/judge acceptance or
GPU execution has been performed for this maintained integration.

`agent.py` and `transport.py` are retained historical campaign implementations;
the native config does not use them or their fixed-port proxy. Before live use,
validate creator task assets and native multi-service lifecycle on an admitted
GPU worker, verify pinned images and CPU/RAM, and test capture/scorer cleanup.

### Opt-in evaluation input snapshots

`materialize_archive(..., snapshot_evaluation_inputs=True,
snapshot_capture_timeout_seconds=60)` enables a source-bound
molecular-task operational variant. The default remains creator-original. Each
concurrent evaluation gets its own captured output directory and SHA256 manifest;
`/state/evaluation_inputs` is added to the existing Harbor artifact export list.
Detected changes during copying, links, insufficient disk, and exhausted capture
budgets refuse scoring and retain structured operational failures. Capture has a separately declared cooperative operational allowance that must
fit inside the existing collection allowance alongside the full unchanged
3,600-second scientific timeout, with positive overhead. It is recorded in the
materialization receipt and must be included in outer-lease admission;
blocking filesystem calls still depend on the enclosing supervisor's cleanup
bound. No scientific values, prompts, predictive models, or scorer formulas change.
The original molecular scorer only reads output files; its intermediate work stays
outside the read-only snapshot. Rebuild and qualify the sidecar from this prepared
context: a previously built image does not acquire this variant automatically.

Snapshots preserve the bytes actually scored, including after timeouts/errors.
They detect ordinary concurrent mutation; they do not claim atomic multi-file
publication against adversarial writers. Local fixture controls cover controller
records and the task export declaration, not a deployed Harbor export or live
scientific acceptance. Existing runs and historical scores are never rewritten.
