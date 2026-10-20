# NatureBench runtime preparation — in progress

This integration uses the creator's Harbor packages and separate evaluator and
validity-judge containers. It is not yet an executed benchmark integration.

The downloaded schema 1.4 tasks validate with Harbor 0.24.0. The creator's
timed-agent extension excludes evaluation time from its four-hour solve budget;
the task's twelve-hour agent timeout is a wall-clock safety limit, not the solve
budget. Preserve these distinct clocks.

`prepare_runtime.patch_judge` transforms a **copied** released `judge_readme.py`
into the explicit Nemotron judge variant. It preserves the scientific evaluator,
judge prompt, and verdict parser. It routes the exact allowed model to Responses,
requires an explicitly configured endpoint through `CAMPAIGN_JUDGE_BASE_URL`,
and rejects mismatched-model or incomplete responses. Other models and providers
are rejected. The original creator dispatch otherwise treats every non-GPT model
as an Anthropic model.

Do not modify the immutable downloaded tasks, expose reference labels to the
solver, or treat schema validation as evidence of scientific execution. Pin the
source and image digests, preserve the full solver trajectory and judge receipts,
and label the Nemotron judge replacement as a protocol variant.

The creator's `reward_from_verdict` retains a numeric improvement when
`is_valid` is `None`; only an explicit invalid verdict changes it to -1.
Keep that native reward, but report unresolved judging separately and never
count it as a validated result. Local controls cover correct clustering labels,
equivalent cluster-ID renaming, wrong predictions, empty arrays, invalid
shapes, noninteger labels, and negative labels.

The campaign transport binds loopback and the worker's private Docker bridge
on port 18885 and forwards to the existing worker tunnel on 18883. It accepts
only the exact Nemotron model and nonstream requests, adds a fixed NatureBench
accounting header, checks the returned model, and never retries uncertain calls.
Both chat and Responses live routing controls returned the required model;
this does not establish task execution or scoring correctness.
