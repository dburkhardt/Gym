# Creator scoring lifecycle

The resource server delegates each score to the pinned creator adapter in a fresh Python process on the verifier host. It reuses Gym's LocalProvider with at most two active scoring processes per server event loop. This is trusted host execution, not container isolation or a CPU/memory quota.

`scoring_timeout_s` is a positive finite wall-clock budget (default 600 seconds), starting when the subprocess runs, not while waiting for a slot. Native scientific timeouts are unchanged: for example, ChemIQ retains its creator two-second SIGALRM handling on the worker's main thread. The outer timeout is an infrastructure failure, never a fabricated scientific zero. Creator exceptions, absent automatic scores, malformed outputs and imported-source mismatches likewise fail verification. A genuine zero returned by the creator remains zero; existing incomplete-model-response accounting is preserved.

Timeout and request cancellation kill and reap the owned process group before removing its temporary request/result directory. Spawn is settled before cleanup even if cancellation arrives during process creation. Repeated cancellation does not abandon cleanup, and cleanup failures retain the original scoring error. Deliberately detached child sessions and abrupt SIGKILL of the resource server are not covered; use an external service/container supervisor for that boundary.

The worker reports its imported deterministic adapter path and SHA-256, and the parent checks both against its selected source before accepting results. This is not a lock of every transitive dependency. Scientific result details are JSON-serialized, as they already are at the HTTP boundary.
