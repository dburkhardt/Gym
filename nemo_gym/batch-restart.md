# Explicit infrastructure restart for bound native batches

This opt-in starts a **new scientific attempt**, not a continuation or an
exactly-once replay. Earlier calls, partial computation, failures and costs remain
recorded. It never retries a scored zero, generic timeout or successful result to
select a better answer.

Default guarded resume still skips completed rows and refuses occupied capture
identities. The explicit path currently supports only inspected built-in
`simple_agent`, `openai_model`, `mcqa` and `example_single_tool_call` components,
including the native `legacy_agent` relay bound to that simple agent.
The exact maintained SciCode graph can also qualify when its native component
interpreters, prompt/source files, H5 checksum and locally inspected immutable
Docker image all validate. Missing environments or images fail closed; no setup,
pull or provisioning occurs. The source binding includes non-Python requirements
and build files, and component package versions and actual import origins are
checked; external editable packages are unsupported. This permits an explicit
new attempt, not restoration of a prior scientific sandbox. SciCode's existing
scientific validity findings remain unchanged.

Harbor and other external creator runtimes, skills and external prompt configuration remain
unsupported; their dependency/runtime qualification is not inferred from Python
source alone. No historical run is retroactively made eligible.

The standard guarded CLI creates an existing-schema batch manifest beside the
output when none is explicitly supplied, including for one benchmark. Its
component-recipe digest binds resolved server settings; the separate full execution
binding also covers global scientific settings. An explicit manifest remains
authoritative. Incompatible existing manifests are never overwritten.

From the first guarded run, the native `batch_manifest_fpath` gains an
`execution_binding` in its existing `batch_status.json`. It binds the manifest,
exact materialized input bytes, resolved server settings, collector configuration,
other global options, capture-store IDs/paths, Python/package versions and current
Python source under the selected implementations and shared Gym library. Actual
entrypoints must stay within their implementation roots. Binding is deliberately
conservative: library changes may require a separate run rather than reuse.

Server host/port/worker/proxy placement fields and credentials are excluded by the
existing configuration digest. Collector resume/receipt, concurrency, dispatch
order/window, memory retention and health-display fields are excluded; scientific
sampling/routing settings remain bound. Declared input bytes bind prompts and
per-task sampling. No change to a model, harness, scorer or input is permitted.

An operator selects exact prior failed attempts using a JSON receipt:

```json
{"execution_binding_sha256":"<canonical digest of status execution_binding>",
 "failures":[{"capture_key":"0-0", "failure_sha256":"<canonical digest of exact native failure row>"}]}
```

Use `nemo_gym.batch_restart.digest` to compute these hashes from saved JSON.
The receipt is an explicit decision, not a new identity registry or scheduler.
Run the same supported `gym eval` collection command and configuration with
`resume_from_cache=True` and `guarded_restart_receipt_fpath=/absolute/receipt.json`.
The option belongs to rollout-collection configuration. Keep the original output,
manifest and durable capture store. Only receipt-selected logical tasks are
scheduled in this explicit mode; other unfinished tasks remain unfinished.
Native failure counts supply monotonically higher attempt indices.

Eligible rows need a bound native failure record, no reward, and a positive
infrastructure `failure_kind`: provider unavailable/quota/OOM, session lost or
expired. An untyped outer agent failure can additionally qualify only when every
physical request in its exact bound capture is conclusively rejected HTTP 429,
with no successful or ambiguous sibling request. The raw failure row is unchanged.
HTTP 503 is intentionally not such proof: the current transport records its
acceptance as unknown, and recovery refuses to resend. A generic agent error is insufficient. All prior provider sends must be
reconciled, with no incomplete/pending captures; changing attempt suffix or capture
storage cannot evade uncertainty. Same-attempt reuse is always refused. Scientific
budget exhaustion remains in its original denominator and cannot be selected.

Invalid receipt or binding fails before status publication and dispatch. Explicit
restart does not repair/truncate old JSONL tails. Fresh invocation against an
existing bound batch is rejected before rewriting materialized inputs or clearing
results. Existing outputs/captures are never deleted by this admission path.

### Stateless creator-scoring closure

The maintained `campaign_creator_scoring` resource can qualify explicit failed-attempt restart for its **GenomeQA and MolJSON** branches. These branches extract functions from pinned creator source and score materialized references using the standard library. This is an inspected execution closure, not blanket permission for every creator scorer.

Qualification requires exactly the native OpenAI model → one-step, tool-free simple agent → creator resource graph and its legacy environment relay. It binds the creator Git revision and exact scorer bytes, the maintained adapter package, actual component imports and interpreter/package identities, resolved configuration, immutable materialized inputs, and original capture stores. Missing references, unknown settings, additional execution components, editable external dependencies, and source changes refuse restart. The scorer subprocess uses its resource service's `sys.executable`; it already verifies the imported maintained scoring module before returning a score.

All existing failed-attempt rules still apply: an explicit scoped receipt, reconciled definitive provider rejection, a higher attempt identity, and preserved completed results are required. Ambiguous 503/timeout outcomes remain blocked. This does not restore a partially executed scorer, tools, an arbitrary shell, or an old unbound campaign run. Other creator branches remain unqualified until their scientific imports and assets have an equally explicit closure. Package versions and actual loaded maintained modules are attested; this is not a hermetic OS or transitive wheel-content reproducibility claim.

The local control is `tests/e2e/coordinator/test_cli_crash.py::test_creator_resource_restart`, enabled by `NEMO_GYM_COORDINATOR_E2E=1` and `NEMO_GYM_CREATOR_RESTART_SOURCE=/absolute/path/to/pinned/genomeqa`. It uses benign fixture prompts, the actual creator resource/scorer and native CLI, and a deterministic loopback provider. It tests 200/429/200 followed by only the rejected task's new attempt, completed resume with no new calls, and ambiguous-503 refusal. It is transport/lifecycle acceptance, not a new scientific benchmark result.
