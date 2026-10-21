# Description

OpenAI-compatible model server using Gym's shared HTTP client.

## HTTP retries

HTTP 404 and 408 are retryable by default, alongside the existing transient
server and rate-limit errors. This applies to every dataset and to policy and
judge clients. A permanent 404 still fails after the bounded attempt budget.
Retries resend the same request with the existing fixed 0.5-second delay
between attempts and preserve the terminal error body.

`max_http_attempts` defaults to three total attempts. Set it per model server:

```yaml
judge_model:
  responses_api_models:
    openai_model:
      max_http_attempts: 5
```

A value of one disables HTTP retries. Connection-error retries are controlled
separately. Internal Gym clients retain their existing unbounded extension for
rate-limit statuses; 404 and 408 never trigger that extension.


# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0

### Literal text completions (opt-in)

Set `raw_completions_enabled: true` only for a provider that implements
`/v1/completions`. Gym forwards one literal string prompt, without converting it
into chat messages. The initial contract supports one non-streaming completion,
finite sampling parameters, a positive `max_tokens`, and string stop sequences.
Unsupported fields (including `echo`, `logprobs`, token-array prompts, `suffix`,
`best_of`, and multiple choices) fail validation. Enabled configurations validate
`extra_body` at startup; incompatible existing chat configurations can leave this
capability disabled. Provider response model identity must match `openai_model`.

The route uses the existing rollout URL, logical-call headers, capture ledger,
and uncertain-dispatch guard. Missing usage remains unknown. Raw completion
training-token capture is unsupported and fails before dispatch. This interface
provides generated text, not conditional likelihood or tokenizer alignment.
A successful local fixture does not establish that a remote provider supports
this endpoint or that its completion protocol matches a benchmark's original model.

### Optional host-local cooldown coordination

`upstream_rate_limit_cooldown_seconds` retains its per-process default behavior.
To share an observed HTTP 429 delay across native worker processes or separate
model servers on **one host**, also configure:

```yaml
upstream_max_num_tries: 1
upstream_rate_limit_cooldown_seconds: 30
upstream_rate_limit_shared_directory: /private/local/path/provider-admission
upstream_rate_limit_shared_namespace: campaign-provider-scope
```

The directory must be a trusted host-local filesystem, not a network mount.
Use the same directory and nonsecret operator-chosen namespace only for workers
that intentionally share one provider admission scope. Namespace reuse with a
different endpoint or model fails closed. Credentials are not inspected or stored;
the operator must keep separate credential/tenant quotas in separate namespaces.
Endpoint URLs containing credentials, query strings or fragments are refused in
this optional mode. No credentials, raw URLs or request bodies enter state files.

The shared state records only hashed namespace/scope/boot identities, a monotonic
deadline and diagnostic wall-clock update time. A kernel boot identifier makes
old monotonic deadlines expire after reboot; ordinary process restarts retain
the deadline. Wall-clock changes do not shorten an active wait. Linux and macOS
are supported; unavailable boot identity, corrupt state or an unavailable lock
fail closed. Atomic publication is fsynced and takes the maximum existing deadline.

Locks are held only during short state operations. Acquisition is bounded to one
second, including startup; no lock or upstream concurrency slot is held while
sleeping through cooldown. Cancellation settles the bounded state operation before
returning. Already in-flight requests are not recalled, and a concurrent request
may pass the last check just before a peer publishes a 429; this is coordination
of observed cooldown, not a globally serialized admission semaphore.

This does not infer RPM/TPM, coordinate across hosts, pool concurrency slots, alter
benchmark budgets, or establish provider capacity. Waiting still consumes the
benchmark wall-clock budget. Existing explicit-rejection retries and uncertain
no-resend semantics remain unchanged; keep configured physical retry limits
appropriate to the provider guard. Qualification evidence uses separate local
processes and a loopback fake provider, not a production rate-limit claim.

### Optional cumulative queue allowance

`upstream_admission_timeout_seconds` bounds cumulative queued time for one logical
request: pre-request jitter, shared/local cooldown, semaphore acquisition and
retry backoff. The default `null` preserves existing behavior. A peer extending
shared cooldown does not reset this allowance, nor does a rejected retry.

This is separate from `upstream_request_timeout_seconds`, which still grants
each provider attempt its configured duration. For example, a 60-second
admission allowance plus a 600-second provider timeout can occupy up to 660
seconds per role (excluding local settlement overhead). A solver followed by
a judge therefore needs a drain covering both roles; do not silently count queue
time against the scientific provider budget.

Expiry raises `UpstreamAdmissionTimeoutError` before starting the next attempt.
It is an infrastructure failure, not a model score. Captures distinguish expiry
with zero physical sends from expiry after earlier rejected attempts; the latter
remain preserved. This option neither enables retries nor makes uncertain calls
safe to resend. Configure the existing rejected-only policy separately when
authorized, and retain the overall collector/supervisor deadline.
