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
