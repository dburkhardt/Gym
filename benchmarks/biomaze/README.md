# BioMaze

BioMaze has two distinct protocols: TrueFalse uses deterministic answer scoring; Openended uses the creator's conclusion-judging rubric. Keep their samples, attempts and metrics separate. Both use the maintained Gym preparation and evaluation interfaces. References stay in `verifier_metadata` and are not sent to the solver.

## Prepare a frozen sample

Supply an audited, preselected JSONL file through `CAMPAIGN_SAMPLE_JSONL` and its SHA-256 through `CAMPAIGN_SAMPLE_SHA256`. Run `python benchmarks/biomaze/prepare.py` from the repository root. The shared staging helper preserves prompts and references, exposes existing stable task IDs, rejects duplicate identities and writes a provenance receipt. It does not sample new questions. The default destination is `resources_servers/biomaze/data/benchmark.jsonl`; use an isolated checkout when preparing a different sample during an active run.

The campaign's original open-ended selections contain 50 tasks, with the 20-task selection as an exact prefix. The five native acceptance tasks match the first five prompts and references exactly. The next expansion is the remaining 15 of that open-ended selection, not tasks from the separate historical TrueFalse run. Keep previous attempts intact.

## Configure and plan

TrueFalse uses `benchmarks/biomaze/config.yaml`. Openended uses `benchmarks/biomaze/configs/openended.yaml` and requires `BIOMAZE_JUDGE_PROMPT` to point to the preserved creator template. Record the template checksum with the experiment. For an already prepared isolated sample, override the `biomaze.resources_servers.biomaze.datasets` entry's `jsonl_fpath` in a run configuration and set `use_cached_prepared_benchmarks: true`.

Run `gym eval plan --config /absolute/path/run.yaml --out /absolute/path/plan.json` before execution. This offline path was exercised on the original remaining 15 open-ended tasks. The planner sorts task identities canonically; the staged dataset retains original sample order. Planning does not establish execution admission or scientific acceptance.

## Solver and judge protocol

The separate judge server is `biomaze_judge`; the solver is `policy_model`. For this campaign, both use `nvidia/nvidia/nemotron-3.5-super-vl`. The native five-task acceptance used solver temperature 0.1 and maximum 16,384 output tokens, and judge temperature 0 with maximum 4,096 output tokens. These are campaign settings, not a claim that they reproduce the creator's model configuration. The Nemotron conclusion-judge variant replaces the creator's model-specific input-length gate with provider context enforcement. Preserve this qualification in comparisons; do not silently mix the separately recorded 8,192-token heterogeneous attempt with the 16,384-token individual attempt.

## Evidence and limitations

The native five-task probe produced five solver outputs and five judge results. Fresh saved-output regrading made five judge calls and no solver calls; one judgment changed, moving the score from 2/5 to 3/5. This small, judge-sensitive probe is integration evidence, not a reliable training-priority or leaderboard estimate. Join results by task identity rather than completion order.

Captured replay after simulated result-append loss made no new physical model calls. That experiment was not an arbitrary coordinator-kill test. Separate shared lifecycle and restart controls have their own evidence and must not be attributed to this BioMaze run. Expansion to 20 and 50 through the maintained path remains pending.

## Judge format validity

The unchanged creator parser maps malformed judgments to a Boolean, including a permissive positive fallback when `Judge:` is absent. The `explicit-creator-verdict-v1` qualification requires exactly one `Judge:` marker followed by `Correct` or `Incorrect`. Malformed judgments retain the creator raw reward and complete judge response but are masked as `judge_unparseable`; report these attempted items separately from scientific accuracy. Empty or incomplete judge responses remain judge failures. A valid `Incorrect` judgment of an empty solver answer remains a scored model failure. This qualification does not retroactively change saved results or explain judge-to-judge disagreement.
