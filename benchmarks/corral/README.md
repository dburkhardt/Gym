# Corral native integration (in progress)

Preparation pins creator source f5a6963a9ccf6e3c0058c776fe4111b639add648 and dataset release6f7e6f080e9ec029b8c7847a563246b7e6c1ca18. It preserves the existing retrosynthesis level3 frozen5 (make1,2,5,9,10), with prior3/4 expansion recorded separately. Full native task fields remain verifier metadata, never substituted into a flat answer task. Spectra and wetlab are separate unaccepted environment routes, not silently covered by retrosynthesis.

`native_phase.py` provides single-task creator execution and separate saved-state evaluation using ShardedCommitStore, including workspace snapshots. It does not launch a scheduler, construct a model client, or establish container/DB isolation. Solver integration must inject the creator ReAct agent with native50 interactions,16384 output tokens, one attempt, exact Nemotron model, and guarded native streaming transport. Creator streaming reconstructs reasoning content as answer content when final content is absent; preserve that qualification.

Next gates: shared managed-container launch and DB-only connectivity; source/private-reference boundary; guarded streaming and durable per-call identity; native saved-state scorer controls; exact image/database/price snapshot identity. Persistent PostgreSQL/RDKit, frozen buyables and three supplemental reactions must be available. The maintained config produces a five-task native Gym plan; actual container/source and database boundary acceptance remains pending, with no live inference authorized.

Canonical scorer false positives are preserved: historical wrong-root/template/precursor controls can score1. The optional diagnostic scorer is a separate protocol, not a replacement. Task1's released target/template mismatch remains qualified. Scientific acceptance must retain raw canonical metrics and these limitations; runtime integration alone does not clear them.

The saved-state phase supports the creator's explicit `benchmark_run_id` layout as well as standalone execution stores. A copied historical `make_1_lvl3` checkpoint was regraded by the actual creator `evaluate_task` with the original commit and returned the original canonical score 0, without a solver or model calls; the source SQLite checksum remained unchanged. This host-native control proves state API compatibility, not the pending managed-container boundary. Persistence cleanup settles before repeated cancellation propagates.

`config.yaml` uses `CORRAL_SOURCE` for pinned preparation and `CORRAL_DATABASE_SNAPSHOT` for the exact frozen production dump. The custom Harbor agent runs the creator loop; its separate verifier reuses the exported state without another solver. Every scientific phase uses a fresh read-only database clone and a network-disabled runtime. The solver can reach only its authenticated, scoped Unix-socket model bridge. The creator runtime retains its previously resolved campaign lock (SHA `1aaabcffa6c7cc73b5b8021c64c3c7db6babc4e9299ed87763400bd2f452d403`), separately recorded from the upstream source lock. Local image digests must already be present; this registration does not download or rebuild them.

The adapter preserves 50 native interactions, temperature 1, 16,384 output tokens and zero transport retries. Its 3,600-second operational phase bound aborts with an infrastructure error, never a scientific zero; this is an explicit bounded acceptance setting, not a claim that the creator defines a one-hour task budget. Full interrupted-agent continuation remains unsupported and fails closed on an existing output directory.

The real isolated database control passed on the existing build worker: a separate container impersonating PostgreSQL's OS UID could not authenticate as postgres; task-role writes remained denied after disabling session read-only defaults; role escalation failed; native RDKit queries and all 427,683 reactions (including three supplements) remained available. Both owned containers were removed. Full native runtime image/source and end-to-end Harbor acceptance remain separate gates.

Creator `iteration_limit` is a model outcome, not infrastructure invalidity. The adapter checks the persisted runtime metadata, records `model_budget_exhausted` with attempted=1 and completion_failure=1, and leaves the unavailable creator score null. A surrender also has no creator scorer measurement. Provider/tool/runtime failures remain distinct unclassified failures with their original error type. Report attempted-task completion rates separately from scores available on submitted artifacts; never omit exhaustion from the attempted denominator.

### Restricted decision-boundary continuation

`continuation.py` adds an opt-in controller primitive for the pinned ReAct loop.
It stores the **actual provider history** using the creator's
`AgentSession.set_agent_state`/SQLite commits. Canonical session history is not
interchangeable: on the preserved `make_3_lvl3` tool-completed boundary, the
creator's normal reconstruction produces 31 messages instead of the actual next
request's 30, including different tool-observation roles. Missing exact history
is therefore refused.

Admission checks the native state, completed guard ordinals, authoritative Gym
capture store (including later unknown attempts), source and installed dependency
bytes, and the original absolute deadline. The inherited creator loop receives
only its remaining original 50-call allowance. Pending actions, uncertain calls,
changed state/history/runtime, expired deadlines and a different rollout/attempt
endpoint are refused. A controller must hold exclusive ownership while paused;
these helpers are not an automatic restart policy or a shared CLI checkpoint
implementation. Full database/image/resource placement still needs separate
qualification. Existing terminal historical executions must not be resumed.

The offline selected-tool control uses the unchanged frozen task prompt and tool
catalog, real pinned RDKit functional-group functions, actual creator
ReAct/AgentSession/SQLite, and Gym's guarded capture middleware with an in-process
deterministic provider. It kills the process after one completed tool and a
published pre-dispatch boundary, then restores in a fresh interpreter. The two
requests (including generation settings and logical-call headers) and two tool
results must match uninterrupted execution. Its narrow tool-execution adapter
rejects every tool outside the inspected pure-RDKit closure. This does **not**
qualify PostgreSQL/RXNMapper, the canonical scorer, arbitrary shell state, or the
shared CLI recovery path. No scientific score is produced.

From the repository root, with the source and private dependency locations used
by the campaign:

```bash
export PYSTOW_HOME=/tmp/corral-pystow
export LITELLM_LOCAL_MODEL_COST_MAP=True
export PYTHONPATH=.:../../outputs/broad-eval-campaign/foundation/scientific-recovery-proposal/corral-deps:../../outputs/broad-eval-campaign/harbor/agent-expansion/sources/29/lamalab-org-corral-f5a6963/src:../naturebench-runtime/lib/python3.13/site-packages
../bioagent-runtime/bin/python -m benchmarks.corral.qualify_tool_continuation \
  --source ../../outputs/broad-eval-campaign/harbor/agent-expansion/sources/29/lamalab-org-corral-f5a6963 \
  --database ../../outputs/broad-eval-campaign/corral/transport-v3-preparation/storage-placement/actual-make3/worker-artifacts/native/make_3_lvl3-disjoint-v3/task-make_3_lvl3/k-1/commits.sqlite3 \
  --output /absolute/path/to/a-new-control-directory
```

`qualify_continuation.py` is a separate historical-history diagnostic. It uses a
virtual historical clock and seeds a new branch in a copy from saved evidence;
it explicitly does not establish fresh capture admission or resume the original
terminal execution. The fresh selected-tool control above uses a current,
unchanged absolute deadline and newly recorded local fixture captures.

Native state export occurs inside the scientific container before it closes, so
creator-owned `2770` directories and `0640` files remain readable during packing.
The shared verified bundle is downloaded into a fresh caller-owned tree; the
original bind tree is retained without a host ownership change. Both export and
solver-to-verifier transport explicitly opt into directory-setgid preservation.
The manifest declares this policy, readers must opt in independently, and all
other special bits are refused. Archived UID/GID values are never restored.
Files retain their bytes and modes; directories retain setgid but inherit the
receiver's local group, not the creator's numeric group. Unsupported filesystems
that strip requested permission bits fail before publication. This transport
repair does not establish a successful full native scientific run.
