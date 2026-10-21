# Creator scoring lifecycle

The resource server delegates each score to the pinned creator adapter in a fresh Python process on the verifier host. It reuses Gym's LocalProvider with at most two active scoring processes per server event loop. This is trusted host execution, not container isolation or a CPU/memory quota.

`scoring_timeout_s` is a positive finite wall-clock budget (default 600 seconds), starting when the subprocess runs, not while waiting for a slot. Native scientific timeouts are unchanged: for example, ChemIQ retains its creator two-second SIGALRM handling on the worker's main thread. The outer timeout is an infrastructure failure, never a fabricated scientific zero. Creator exceptions, absent automatic scores, malformed outputs and imported-source mismatches likewise fail verification. A genuine zero returned by the creator remains zero; existing incomplete-model-response accounting is preserved.

Timeout and request cancellation kill and reap the owned process group before removing its temporary request/result directory. Spawn is settled before cleanup even if cancellation arrives during process creation. Repeated cancellation does not abandon cleanup, and cleanup failures retain the original scoring error. Deliberately detached child sessions and abrupt SIGKILL of the resource server are not covered; use an external service/container supervisor for that boundary.

The worker reports its imported deterministic adapter path and SHA-256, and the parent checks both against its selected source before accepting results. This is not a lock of every transitive dependency. Scientific result details are JSON-serialized, as they already are at the HTTP boundary.

MolecularIQ's core dependency is installed by native environment preparation from the exact MIT-licensed commit a1b89635371c3cd942e44ebeec63ec3665e7743d declared in requirements.txt. It must not be replaced by an unrelated installed package. Qualification from a freshly created component environment confirmed VCS provenance and replayed saved HiSciBench, Sino-US, MolecularIQ and FGBench answers through the maintained scorer subprocess; all four matched their preserved rewards. This is dependency/runtime qualification, not new inference or exhaustive scientific validation.

### ChemIQ dependency preflight

For ChemIQ, `/seed_session` runs the same pinned creator parser/verifier
initialization used by scoring, in a supervised subprocess of the resource
server's actual `sys.executable`. Select the qualified scientific environment
for this component; importing this server's lightweight wrapper is insufficient.
The check loads the selected UUID from the creator questions and imports NumPy,
requests and RDKit. It does not score an answer or contact OPSIN or a model.
Initialization has a 60-second operational cap (or the smaller configured scoring
cap), without changing the scoring or provider timeout.

A failed check returns HTTP 503 with `dependency_preflight_failed`, task identity
and `dispatch_status=not_sent`; the simple agent stops before its solver call.
Retain this infrastructure failure in the native failure sidecar, not as reward
zero. Successful initialization is necessary, not proof of OPSIN availability,
scientific reference correctness or runtime capacity. Other creator branches
retain their existing behavior and report `not_covered` in the seed response's
`x-nemo-gym-scorer-preflight` header; they are not certified by this check.

Focused tests accept `CHEMIQ_SOURCE` for the pinned creator checkout,
`CHEMIQ_MISSING_RDKIT_PYTHON` for an isolated interpreter with NumPy/requests but
no RDKit, and `CHEMIQ_QUALIFIED_PYTHON` for the scientific interpreter. Missing
optional test environments are reported as skips, not successful qualification.
