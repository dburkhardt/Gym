# GeneHop native retrieval pilot

Uses the pinned GeneGPT `main_turbo.py` header and URL-calling protocol against the GeneHop release, not GeneTuring and not closed-book QA. Source revision: `0a2a327a507be49e660597769cc53431abd6613a`. All six documentation/demonstration components are included. Public NCBI replies are captured and may differ from historical snapshots.

Preserves10 model turns,36,000-character rolling prompt truncation,20,000-character tool truncation, and native answer parsing. Protocol changes: Nemotron3.5SuperVL replaces the creator model;16,384 generation tokens per turn; allowlisted HTTPS NCBI tools; bounded HTTP timeouts and8MiB capture ceiling. Full model requests/responses and tool bytes are saved without credentials. No uncertain inference is automatically retried.

Only Disease gene location has an upstream automatic score. Other task types retain generated responses with `raw_creator_score=null` and masked Gym reward placeholders. No overall accuracy should be computed from those placeholders. Responses wrap the final native answer; provider identity/usage is in per-turn receipts.

Set `GENEHOP_SOURCE`, `GENEHOP_OUTPUT`, `GENEHOP_SAMPLE`, and an inherited `GENEHOP_API_KEY` on the worker. The API key must never be supplied as a serialized Gym config value. Uses the existing local rate-limited gateway. Initial implementation remains pending independent review and live pilot.
