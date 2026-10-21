# GeneHop native retrieval pilot

Uses the pinned GeneGPT `main_turbo.py` header and URL-calling protocol against the GeneHop release, not GeneTuring and not closed-book QA. Source revision: `0a2a327a507be49e660597769cc53431abd6613a`. All six documentation/demonstration components are included. Public NCBI replies are captured and may differ from historical snapshots.

Preserves10 model turns,36,000-character rolling prompt truncation,20,000-character tool truncation, and native answer parsing. Protocol changes: Nemotron3.5SuperVL replaces the creator model;16,384 generation tokens per turn; allowlisted HTTPS NCBI tools; bounded HTTP timeouts and8MiB capture ceiling. Full model requests/responses and tool bytes are saved without credentials. No uncertain inference is automatically retried.

Only Disease gene location has an upstream automatic score. Other task types retain generated responses with `raw_creator_score=null` and masked Gym reward placeholders. No overall accuracy should be computed from those placeholders. Responses wrap the final native answer; provider identity/usage is in per-turn receipts.

Set `GENEHOP_SOURCE`, `GENEHOP_OUTPUT`, and `GENEHOP_SAMPLE` on the worker. Configure the shared policy model as described below. The legacy gateway path alone uses an inherited `GENEHOP_API_KEY`; never serialize it into Gym configuration. The migrated shared path remains pending live acceptance.

## Shared transport migration

The maintained benchmark config selects `policy_model` through Gym. Enable `observability_enabled` and `provider_dispatch_guard`, configure durable model-call capture, and supply credentials only to the model server. Each native chat turn uses the original payload and a rollout-qualified logical call identity. The direct gateway remains only for legacy unguarded runs; guarded runs reject that fallback. Existing task artifact directories block an automatic restart, so arbitrary interrupted-loop resume is not yet implemented. This bridge has contract tests but still requires live acceptance.

For the automatically graded Disease gene location category, budget-limited output is scored with the creator parser and remains in the denominator. Other categories remain manually graded; infrastructure failures remain masked. Historical attempts are unchanged.

Header reuse now requires matching pinned creator source, header content and captured NCBI tool-byte provenance. Tool redirects are checked before following them. Each model turn writes a durable intent before dispatch. Cancellation prevents later turns and waits for the current bounded HTTP/thread operation to settle before releasing the per-agent lock; it does not claim to restore an interrupted loop. The shared chat request retains the legacy 600-second bound.

## Native CLI deterministic control

`responses_api_agents.genehop_native.qualify_cli` runs the supported CLI against a local deterministic provider using the preserved five terminal answers and provenance-checked cached creator header. It checks request/result task and rollout indices, five captured physical calls, manual-only masking, and completed-output resume without additional calls. This does not exercise fresh NCBI retrieval, measure model quality, or establish interrupted-loop recovery. Native receipts now map to invocation/turn telemetry using actual response IDs, prompts and timestamps; tool bodies are hash-checked and represented with native truncation/RID semantics. The preserved fixture correctly yields three healthy and two hollow-answer unhealthy rows, with none unobserved and no capture or token mismatches. Earlier unobserved controls remain preserved.

Native artifact directories bind the exact task UUID and attempt-qualified rollout identity in a SHA-256 directory name, with `execution-identity.json` retaining their readable values. Repeats and explicit new attempts therefore do not collide; dispatching the same episode twice still refuses an existing directory. This does not authorize retry after an unresolved provider call. Guarded execution requires matching body-derived identity for artifacts and provider transport; context-only identity is refused before directory creation. Legacy calls without native identity retain the old task directory convention. Existing artifacts are never relocated or overwritten.
