# MOOSE-Chem native pipeline

Pinned creator f8cda98a3bed30b0f0d85fba625e343d688461ab; SHA256 checks cover source and the official51-question workbook/150-paper corpus. The wrapper invokes creator Screening, HypothesisGenerationEA and Evaluate, preserving source prompts, parsing, ranking and reference match scorer. Configuration follows creator main.sh: strict background/survey, four retrieval rounds, three mutations and refinements, up to three inspiration search steps. Ground-truth inspirations are disabled for generation. Evaluation enables reference annotation because a benchmark metric is required.

All model roles use Nemotron3.5SuperVL via the native Gym model service; this is an explicit model/judge substitution variant. Creator8192-token generation cap and temperatures are retained. Every raw chat request/response is saved without authentication. Reference hypothesis/keypoints enter only the native evaluator. Match scoring is only applied to hypotheses with a creator-matched inspiration, so a missing matched score is not a numeric zero. A modern model may have seen these2024 papers; this run cannot reproduce the paper's unseen-literature premise.

Three no-inference controls cover source defaults, reference recall1/0, and first retrieval prompt exclusion of hypothesis/keypoints. Full native pipeline runtime and judge calibration remain in progress. The native Gym agent configuration is `responses_api_agents/moosechem_native/configs/moosechem.yaml`. It preserves full creator evaluation outputs and uses only the explicitly named last-round inspiration recall as Gym transport reward; this is not overall hypothesis quality. Importing an already completed native execution requires an explicit flag and never repeats inference. Three additional contract tests verify trace model allowlisting, native metric preservation and refusing implicit reruns.

Future runs use an explicit no-transport-retry variant: the first OpenAI API error writes its request/error receipt and aborts the native pipeline. A dedicated TransportAbort escapes the creator’s nested Exception retry loops; KeyboardInterrupt and SystemExit remain untouched. Returned-response format retries retain creator behavior. Historical runs and receipts are unchanged. The regression simulates the creator retry loop and verifies one dispatch plus durable failure evidence.

Future runs atomically claim their output directory with `run-intent.json`, persist each request intent before dispatch, and fsync terminal traces. Receipt failures abort outside native retry loops. Interruptions remain explicitly unknown outcomes requiring reconciliation; neither a trace row nor a run-intent file alone proves provider completion.

## Maintained native path for new attempts

Use `gym eval plan --config benchmarks/moosechem/config.yaml --out plan.json`, then the normal native Gym evaluation workflow with the guarded policy model configured. Preparation requires `MOOSE_FROZEN_TASKS` and its exact `MOOSE_FROZEN_SHA256`; it atomically stages every selected row without resampling. Set `MOOSE_PYTHON`, `MOOSE_SOURCE` and a fresh `MOOSE_OUTPUT`. The frozen five IDs remain18,26,38,0,33; task26 is explicitly blocked because its historical outcome is unresolved. This example plan is not authorization to rerun completed tasks.

New execution uses the shared task-scoped authenticated Unix socket model bridge and shared managed subprocess lifecycle. It requires `provider_dispatch_guard`, exact policy model identity and a native rollout capture ID. There is no normal dependency on49180,49394 or a slot-clearance file. Requests retain creator8192-token limits by default;16384 is an explicit variant. Request intents and raw responses are durably recorded before creator parsing, transport failures poison the attempt and escape creator retry loops, and cancellation settles the child process group before removing the socket. Existing output directories are never implicitly restarted. Saved execution import remains explicit.

The creator Python runs as trusted local code through LocalProvider: this is process lifecycle management, not container isolation or a hard CPU/RAM quota. Execution identity includes the orchestrator hash, timeout and interpreter hash; the child records and checks its actual native/guarded imports and interpreter before dispatch. Pin and qualify scientific dependencies separately. The trusted creator process can read reference assets; reference exclusion from generation is enforced by audited creator prompts, not an OS boundary. No live native rollout has yet qualified this migrated path; offline controls and a five-row native plan establish wiring, transport boundaries and cancellation behavior only. The full creator evaluation remains available; Gym's named inspiration-recall reward is not a validated measure of hypothesis quality.

The following proxy/clearance notes describe preserved historical attempts only.

## Historical remaining tasks0/33: bespoke admission variant

`remaining_two.py` selects only0 and33 from frozen `[18,26,38,0,33]`. Completed38 and unresolved26 are not restarted. Full native150-corpus screening, mutation/generation and evaluation remain unchanged. All roles use exact Super and16,384 output tokens. No transport retries.

Stage `admission.py` on original-host loopback49394 only after a fresh sole-producer slot. `MOOSE_ADMISSION_URL=http://127.0.0.1:49394/v1` routes to that proxy, which forwards solely to existing49180. Authorization is memory-only. Every supplied output-budget field must equal16,384; source text messages/model/label are checked. UTF8 request bytes plus4,096 framing margin plus output budget must fit100,000. Oversized requests reject before dispatch. Shared rolling usage, global drain and429 cooldown apply per request. Transport/non200/wrong model/missing usage/usage overrun stop future dispatch persistently. Unknown requests on proxy restart require reconciliation.

Queue clearance under `remaining2-admitted-v1/slot-clearance.json` must include approved:true, producer`moosechem-remaining2`, fresh issued_at, frozen sample_sha256 and implementation_sha256 for native.py/admission.py/remaining_two.py. Queue/native tasks use exclusive durable intents, preserved commands/logs/terminal receipts and no automatic restart. Source/scorer and previous results remain unchanged. This is an explicit transport/budget variant. Nothing is deployed or inferred merely by staging these files.

Follow-up launch gates pin the actual `/moosechem-source` creator modules and both prompt/corpus assets using source-sha256.json, and compare the exact existing remote Python interpreter SHA/version plus installed distribution versions to runtime-fingerprint.json. The queue must execute from the deployed overnight-gym-protein module tree; its child CWD/PYTHONPATH select that same tree. Clearance hashes now also include child_entry.py and both pin manifests. Proxy health reports its executing file SHA, which must match clearance.

Each child starts a separate process group and writes/fsyncs its own PID, PGID, Linux start ticks, boot ID and command before any native code runs. It waits up to60seconds for a matching parent release. Parent persists that identity before release. If parent dies before release, the child times out without inference; if after release, coordinator can discover the live native process by PID+PGID+startticks+bootID (not PID alone), with native command retained. No automatic relaunch follows parent interruption.

### Offline full-loop qualification

`python -m benchmarks.moosechem.qualify_replay --repo <pinned-source> --saved <complete-trace.jsonl> --output <new-directory> --task-id 24 --budget 16384` executes the unchanged creator pipeline with an offline-only client. Every stage and effective request must match the saved sequence, including the declared historical token-budget variant. Recorded completed budget-exhausted responses are returned unchanged so creator parsing behavior is preserved. Ambiguous/error records, divergence and sequence exhaustion fail closed; no provider client is constructed, and Python socket audit events are denied. This process-level test boundary is not an OS security sandbox.

Production `ReplayPrefix` recovery restrictions remain unchanged. Offline replay does not authorize new inference or reconcile ambiguous historical requests. Successful replay establishes runtime/parser fidelity, not independent scientific judgment quality. Creator `evaluate.py` uses `list(set(...))` for used-inspiration provenance; that list's ordering can differ across processes even when requests, scores and other outputs match.

New managed executions store creator files under `attempts/task<ID>/<hash-of-task-and-native-episode>/`; the readable capture key and capture-store identity are recorded in `execution-identity.jsonl`. The exact output directory returned by execution is used for scoring. Historical `import_existing` keeps the original `task<ID>` layout; legacy task-only artifacts block new execution until explicit reconciliation. No artifacts are moved or overwritten.

An independent native repeat is distinct from a retry. A new attempt of the same logical rollout requires reconciled captures, the original capture-store identity, successful managed-process termination, and creator output files. Pending/unknown provider outcomes and partial creator executions remain blocked. Bridge lineage is the persisted native capture key, not a random retry ID. These admission controls do not implement continuation, authorize inference, or certify hypothesis quality.

### Offline settled-tail qualification

`python -m benchmarks.moosechem.qualify_settled_tail --repo CREATOR --saved TRACE
--gateway LEDGER --evidence PROOF --output NEW_DIR --task-id ID --budget 16384`
replays a successful saved prefix and stops before its exact rejected HTTP503
request. The proof binds `saved_sha256`, `gateway_sha256`, the full maintained
`creator_sha256` map, and `tail_gateway_request_id`. Every saved request must have
one unique gateway start and terminal response within its recorded time interval;
completed prefix responses must match IDs and have nonempty stopped generations.
Network calls are denied by a Python audit hook. The terminal receipt reports
replayed calls, exact rejected request and zero new model calls, never a scientific
score. This is an offline compatibility check, not authorization or an implementation
of live continuation. Existing full saved-loop qualification remains unchanged.

A reviewed continuation can be configured through the native agent's optional
`settled_recovery` object (`task_id`, `saved`, `evidence`, `gateway`). This does not
clear `blocked_task_ids`: the operator must first review settlement and select a
new attempt. The saved trace, gateway ledger, proof, creator hashes, implementation
hashes and unchanged historical token budget are composed with the execution
configuration into the shared bridge digest. The child verifies that same digest
and all four actual maintained imports before entering the creator loop.
Successful saved requests replay locally; only the exactly matching rejected tail
can enter `GuardedTransport`, as new ordinal0 in a new lineage. Subsequent calls
use normal guarded capture and failure latching. An exclusive continuation claim
and the normal run-intent prevent output reuse. Replay counts are recorded apart
from new calls; historical records are never replaced. Qualification does not
itself authorize this mode or prove a complete scientific result.
