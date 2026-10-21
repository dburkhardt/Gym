# Explicit infrastructure restart for bound native batches

This opt-in starts a **new scientific attempt**, not a continuation or an
exactly-once replay. Earlier calls, partial computation, failures and costs remain
recorded. It never retries a scored zero, generic timeout or successful result to
select a better answer.

Default guarded resume still skips completed rows and refuses occupied capture
identities. The explicit path currently supports only inspected built-in
`simple_agent`, `openai_model`, `mcqa` and `example_single_tool_call` components.
Harbor/external creator runtimes, skills and external prompt configuration remain
unsupported; their dependency/runtime qualification is not inferred from Python
source alone. No historical run is retroactively made eligible.

From the first guarded run, an existing native `batch_manifest_fpath` gains an
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
expired. A generic agent error is insufficient. All prior provider sends must be
reconciled, with no incomplete/pending captures; changing attempt suffix or capture
storage cannot evade uncertainty. Same-attempt reuse is always refused. Scientific
budget exhaustion remains in its original denominator and cannot be selected.

Invalid receipt or binding fails before status publication and dispatch. Explicit
restart does not repair/truncate old JSONL tails. Fresh invocation against an
existing bound batch is rejected before rewriting materialized inputs or clearing
results. Existing outputs/captures are never deleted by this admission path.
