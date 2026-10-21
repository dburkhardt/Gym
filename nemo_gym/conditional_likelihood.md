# Raw conditional-likelihood contract (offline only)

`conditional_likelihood.py` defines serializable request, token plan and observation
objects. It installs no model endpoint, does not declare any currently served model
capable, and does not implement a WMDP adapter. Whole-chat prompt NLL and generated
multiple-choice answers do not satisfy this contract.

The reference is [lm-eval v0.4.2 TemplateLM](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.2/lm_eval/api/model.py)
and its [causal HFLM scorer](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.2/lm_eval/models/huggingface.py).
We preserve trailing-context-whitespace relocation, empty-context prefix token,
joined-string tokenization and continuation-only summed log probabilities. The
optional explicitly named left-context truncation follows the upstream max-length
plus-one causal slice. Continuation truncation is forbidden. Default truncation
rejects rather than silently dropping context.

There is one deliberate stricter boundary: upstream splits joined tokens by the
length of separately encoded context without checking equal prefixes. A tokenizer
merge across that boundary can yield an empty or misaligned continuation. This
contract rejects unequal prefixes. It does not pretend to score such a case with
canonical parity. Tokenization fixtures are benign toy controls, not qualification
of a served model's tokenizer or forward pass.

Every request binds the raw strings, model revision, tokenizer artifact and
configuration SHA256, prefix token policy, protocol and truncation through the
existing provider-dispatch configuration digest. The plan carries exact model
input IDs and continuation logit positions. The observation must match those IDs,
positions and identities, with one finite nonpositive log probability per target
and a consistent finite sum. Serialized request/plan/observation can accompany
existing capture records; this module adds no journal or retry path.

A future native modelserver adapter must first qualify the actual served model,
exact tokenizer and special-token settings against trusted forward-pass controls,
then carry these records through existing durable capture/dispatch identities.
`validate_likelihood` defaults to unavailable; explicitly passing capability true
is a caller admission decision, not proof of support. No production caller or
configuration currently enables it. Backend admission and modelserver transport
remain separate unfinished work.
