# WMDP Bio/Chem: conditional-likelihood integration boundary

This experimental adapter stages the exact public Bio/Chem test sets and reproduces the creator's zero-shot lm-eval 0.4.2 requests. It supports native Gym discovery and planning, **not model execution**. `wmdp_conditional` refuses dispatch because no qualified conditional-likelihood backend is available. A successful plan is not an evaluation result.

The source manifest pins the [creator repository](https://github.com/centerforaisafety/wmdp), [lm-eval v0.4.2 task definition](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.2/lm_eval/tasks/wmdp), actual request/scoring methods, and [public dataset revision](https://huggingface.co/datasets/cais/wmdp/tree/7125571f22f032c56415e7980f48d877dd830ff8). Staging includes all 1,273 Bio and 408 Chem questions. Stable IDs are subset plus original row index; these are a newly materialized full test set, not a claimed historical frozen sample. Cyber is outside this inventory entry.

## Protocol

The context is the creator's subject description, stripped question, four choices, and `Answer:`. The four continuations are exactly ` A`, ` B`, ` C`, ` D`. There are no demonstrations, chat wrappers, added instructions or generated-answer substitutes. Accuracy uses the highest **summed continuation log likelihood**, first-index tie breaking, and the original zero-based answer. Do not length-normalize or invert this score. This is a hazardous-knowledge safety measure: higher accuracy is not an adoption/training-priority recommendation, even though the creator metric declares higher-is-better for accuracy.

`protocol.requests` builds the shared `nemo_gym.conditional_likelihood` contract. That contract binds raw causal conditioning, exact model/tokenizer revisions, token boundaries, truncation and positions. Its stricter boundary validation rejects tokenizer merges that make upstream's unchecked split ambiguous; it does not silently approximate. `protocol.score` reconstructs the canonical ordered question/model requests and only accepts four observations whose plans match those exact request digests. Its backend-qualified argument is a caller assertion for a separately admitted backend, **not evidence or an endpoint capability switch**. No production caller supplies it here.

The missing backend must return per-continuation token log probabilities on the exact raw context, with matching tokenizer/source identity and input/position receipts. Gym's current generated-completion route, chat prompt NLL, and a returned answer letter do not satisfy this requirement. Backend qualification and guarded durable transport must be implemented and independently checked before enabling `run`; no fallback exists.

## Local planning and controls

Place the two exact parquet files named `wmdp-bio.parquet` and `wmdp-chem.parquet` under `WMDP_DATA_ROOT`. SHA-256 validation precedes atomic publication:

```sh
WMDP_DATA_ROOT=/path/to/pinned/data gym eval plan --benchmark wmdp_bio_chem --out /path/to/plan.json
```

The prepared Responses-shaped input is a Gym planning envelope only; it must never be sent as chat. The answer stays in host-side verifier metadata. Public data and generated plan files are not committed in this adapter.

Offline tests execute the **actual pinned** lm-eval `construct_requests` and `process_results` methods with harmless fixtures, render the actual Jinja task template, and compare request strings, leading spaces, ties and accuracy. They also validate all 1,681 source rows and fail-closed native dispatch. Set `WMDP_UPSTREAM_ROOT` to the SHA-checked source tree and `WMDP_DATA_ROOT` to run those source-dependent controls. No model output, runtime measurement or scientific result is claimed.
