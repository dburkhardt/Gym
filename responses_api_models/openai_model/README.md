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
