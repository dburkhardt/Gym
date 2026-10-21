# Campaign operator quickstart

Run from the selected fork checkout with its installed `gym` CLI and qualified Python environment. This is an operating guide, not authorization to spend, dispatch inference, or replace the campaign's frozen 63-candidate scope. Registration, a successful plan, and scientific acceptance are separate gates.

## Discover and prepare

These read-only commands are runnable without benchmark assets:

```sh
gym list benchmarks --help
gym list benchmarks --json
gym eval run --help
```

Choose the maintained benchmark configuration from discovery and read its component README. Obtain its exact pinned creator source, frozen sample, reference data, tokenizer, and scientific images as required. Verify hashes, licenses/access, and actual component imports; a controller import alone does not qualify a worker. Keep references outside the solver's accessible filesystem. For example, [FGBench](../benchmarks/fgbench/README.md) explicitly documents a zero-shot variant and prohibits pooled scalar accuracy.

All commands below are **templates**. Replace each `/absolute/...` path with a reviewed configuration or artifact; they are not ready-to-run campaign launch commands. `run.yaml` must select the exact frozen cohort, benchmark resource, compatible harness, model roles, capture paths, and original scientific budgets. Use a fresh attempt directory.

```sh
gym eval prepare --config /absolute/run.yaml
gym env prefetch --config /absolute/run.yaml
```

Preparation may download data; prefetch may install dependencies. Perform these only within approved setup scope. Shared sample receipts check source, output, and normalization identity; retain old artifacts rather than overwriting history to bypass a stale-cache refusal. External prepared inputs still require operator source/identity verification.

## Bind roles and plan

For this campaign, every LLM role—including solver, judge, helper, and tool-side models—must explicitly use `nvidia/nvidia/nemotron-3.5-super-vl` at the authorized endpoint. A solver CLI model override alone does not bind judges. Preserve each benchmark's temperature, token/context limits, interaction count, scientific timeouts, scorer, and harness protocol; there is no universal temperature override. Supply credentials through the approved environment mechanism, never literal YAML, arguments, or reports.

Start from the benchmark's maintained configuration and compatible [model](../fern/versions/latest/pages/model-server) and [agent](../fern/versions/latest/pages/agent-server) settings. Require declared capabilities where relevant; do not override unsupported pairings to obtain a passing plan. Enable observability, `provider_dispatch_guard: true`, and durable captures in the reviewed runtime configuration. Record source commit, actual import hashes, input hashes, and resolved non-secret settings.

```sh
gym eval plan --config /absolute/run.yaml --validate-compatibility --out /absolute/attempt/plan.json
gym eval run --config /absolute/run.yaml --output /absolute/attempt/results.jsonl --plan-only
```

Inspect the emitted `*_dispatch_plan.json` and `*_planned_inputs.jsonl`: exact task IDs, draw IDs, input hashes, role settings, and expected counts must match the frozen selection. `--limit` selects source tasks before repetition; omitting `--num-repeats` preserves dataset defaults. Planning starts no model services but can invoke preparation. Do not combine `--plan-only` with `--no-serve`.

For managed serving, configure the input dataset in the run configuration and set `split: benchmark` when its dataset type is `benchmark` (or pass `--split benchmark`). The CLI rejects `--input` in that mode; it is for externally served `--no-serve` runs. Check the planned task IDs before dispatch rather than assuming a command copied from another mode selects the same sample.

## Run and inspect

Before removing `--plan-only`, verify runtime/image availability, actual memory and disk, owned ports, endpoint readiness, full task-plus-cleanup budget, and all prior dispatch identities. Use the existing owned supervisor with an explicit absolute bound; collector admission/drain windows must leave each admitted task its original scientific budget. On qualified Linux hosts, optional `NEMO_GYM_SERVICE_OWNERSHIP_MODE=systemd` scopes apply to new service receipts only.

```sh
gym eval run --config /absolute/run.yaml --output /absolute/attempt/results.jsonl
```

Keep native results, failure sidecars, materialized inputs, batch manifest/status, capture ledger, trajectories, and cleanup receipts together. Use the existing [rollout evidence tools](../fern/versions/latest/pages/observability/rollout-evidence.mdx) and benchmark-specific reports. Distinguish raw score from measurement validity, model failure from infrastructure failure, and attempted/scored/unresolved/undispatched counts. Never convert missing or unknown outcomes to zero; retain genuine model-budget zeros. Health and cleanup are separate from scientific scoring.

A run with no scored rows can exit nonzero after recording a definitive provider rejection. Inspect its failure sidecar, capture and batch binding before choosing recovery. A nonzero exit alone neither authorizes a fresh dispatch nor proves that no request was sent.

```sh
gym eval health-check /absolute/attempt --rollouts-file results.jsonl --json
```

A dead coordinator or absent listener does not prove provider settlement or complete cleanup. Reconcile exact owned service identities; never kill processes by name or port. For quota handling, use the reviewed [shared overlays](../configs/eval/README.md): one-attempt peer settlement is distinct from explicitly versioned rejected-only retries. Cooldown does not extend scientific deadlines or establish provider capacity.

## Resume or regrade deliberately

For the same bound run, ordinary resume skips completed rows and admits only safe remaining work:

```sh
gym eval run --config /absolute/run.yaml --output /absolute/attempt/results.jsonl --resume
```

Retain original source/configuration, inputs, outputs, captures, and manifests. Unknown accepted calls must remain refused; a new suffix or capture directory does not reconcile them. Stateful continuation requires a specifically qualified checkpoint path. An explicit new attempt after definitive rejection is a separate opt-in with exact receipt-selected failures and supported dependency closure: follow [guarded batch restart](../nemo_gym/batch-restart.md), not a generic retry command. Do not resume merely because a process disappeared.

Regrade saved responses with a separately reviewed scorer configuration and a new output:

```sh
gym eval reverify --config /absolute/regrade.yaml \
  --inputs /absolute/attempt/materialized-inputs.jsonl \
  --rollouts /absolute/attempt/results.jsonl \
  --output /absolute/regrade/results.jsonl
```

Use the **actual** materialized-input filename from the run, not an assumed reconstruction. Reverification preserves saved generation requests when present while taking task/reference fields from supplied inputs; inspect provenance differences. It does not regenerate solver answers, but an LLM judge can make new paid calls. A zero-inference regrade requires a deterministic/resource-only configuration or a qualified saved-judge replay. Keep historical raw results unchanged and compare by exact task/draw identity, not row order.

Command syntax was checked against the current CLI's `--help` for all seven command families above. Configuration-dependent templates were not executed as inference or claimed as runtime qualification.
