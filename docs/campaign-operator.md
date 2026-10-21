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

`gym eval plan` builds a benchmark plan: it requires exactly one `type: benchmark` dataset with its preparation script and uses `--out`. For an execution preflight, use the exact `gym eval run` invocation with `--plan-only` and a supported train, validation, or benchmark split. It uses `--output` and exercises the same dataset selection as execution without starting services. End-to-end execution and planning reject `example`; example files require separately started services and `eval run --no-serve --input`, which does not support `--plan-only`. A newly authored synthetic control can use `type: validation` with explicit license metadata; do not relabel scientific datasets to bypass their preparation. These two planning commands are not interchangeable.

Inspect the emitted `*_dispatch_plan.json` and `*_planned_inputs.jsonl`: exact task IDs, draw IDs, input hashes, role settings, and expected counts must match the frozen selection. `--limit` selects source tasks before repetition; omitting `--num-repeats` preserves dataset defaults. Planning starts no model services but can invoke preparation. Do not combine `--plan-only` with `--no-serve`.

For managed serving, configure the input dataset in the run configuration and set `split: benchmark` when its dataset type is `benchmark` (or pass `--split benchmark`). The CLI rejects `--input` in that mode; it is for externally served `--no-serve` runs. Check the planned task IDs before dispatch rather than assuming a command copied from another mode selects the same sample.

## Run and inspect

Use the same controller interpreter, `uv_venv_dir`, and launcher `PATH` for preflight and execution. Managed startup prints a `Launcher preflight` record for each component before starting Ray or services. Check the selected interpreter and `reuse_existing_venv` value: a reused environment needs executable `bin/python` and readable `bin/activate`; otherwise setup requires `uv` on the inherited PATH. An import in the controller environment does not prove that component environments are usable. This read-only check neither installs dependencies nor validates their scientific compatibility. See the [launcher implementation](../nemo_gym/cli/setup_command.py).

When using a staged checkout with an environment installed from another revision, finish staging and hash verification first, then set `NEMO_GYM_EXTRA_ROOTS` to the existing absolute checkout before starting a fresh Python interpreter. Creating that directory after interpreter startup can leave import discovery bound to the earlier environment. Gym gives these component roots explicit precedence; `PYTHONPATH` alone is not a source-identity guarantee after component path augmentation. Use the same setting in preflight and child execution, then inspect actual imported module paths and hashes against the selected commit. A matching controller module does not prove the agent or resource loaded from that checkout. See [component search order](../nemo_gym/__init__.py).

Before removing `--plan-only`, verify runtime/image availability, actual memory and disk, owned ports, endpoint readiness, full task-plus-cleanup budget, and all prior dispatch identities. Use the existing owned supervisor with an explicit absolute bound; collector admission/drain windows must leave each admitted task its original scientific budget. On qualified Linux hosts, optional `NEMO_GYM_SERVICE_OWNERSHIP_MODE=systemd` scopes apply to new service receipts only.

For an outer supervisor bound that includes service startup, opt in with `python -m nemo_gym.agent_utils.process_supervisor --timeout SECONDS --collector-cleanup-reserve CLEANUP_SECONDS --receipt receipt.json -- gym eval run ...`. The collector must also receive an explicit positive `drain_margin_s` covering the full original task budget. The supervisor exports a same-kernel, boot-bound monotonic deadline; the collector uses the earlier of that deadline minus cleanup reserve and its relative dispatch deadline. It rechecks after queue and checkpoint waits, leaving late rows unstarted. This does not shorten an admitted task’s timeout or extend the supervisor bound. Reserve service shutdown separately from task drain and include the supervisor’s own cleanup in the outer controller envelope. Do not copy this environment contract to another host or reuse it after reboot. Without the opt-in flag, existing relative-budget behavior is unchanged.

```sh
gym eval run --config /absolute/run.yaml --output /absolute/attempt/results.jsonl
```

Keep native results, failure sidecars, materialized inputs, batch manifest/status, capture ledger, trajectories, and cleanup receipts together. Use the existing [rollout evidence tools](../fern/versions/latest/pages/observability/rollout-evidence.mdx) and benchmark-specific reports. Distinguish raw score from measurement validity, model failure from infrastructure failure, and attempted/scored/unresolved/undispatched counts. Never convert missing or unknown outcomes to zero; retain genuine model-budget zeros. Health and cleanup are separate from scientific scoring.

A run with no scored rows can exit nonzero after recording a definitive provider rejection. Inspect its failure sidecar, capture and batch binding before choosing recovery. A nonzero exit alone neither authorizes a fresh dispatch nor proves that no request was sent.

```sh
gym eval health-check /absolute/attempt --rollouts-file results.jsonl --json
```

A dead coordinator or absent listener does not prove provider settlement or complete cleanup. Reconcile exact owned service identities; never kill processes by name or port. For quota handling, use the reviewed [shared overlays](../configs/eval/README.md): one-attempt peer settlement is distinct from explicitly versioned rejected-only retries. Cooldown does not extend scientific deadlines or establish provider capacity. New guarded captures retain allowlisted retry and quota response headers under `provider_attempts[].quota_response_headers` when exposed by the transport. These are diagnostic evidence, not automatic retry instructions; older captures may omit them.

## Resolve admission budgets from the actual workflow

Host free memory is not Ray task capacity. When Gym starts a new local cluster, optional `ray_local_object_store_memory_bytes` passes a positive integer byte count to Ray's public `object_store_memory` setting. Without it, Ray retains its default sizing. The setting is ignored when connecting to an existing cluster, including child services inheriting the original configuration; it never resizes that cluster or overrides Ray's advertised task memory. Budget the object store, scientific task, services and cleanup together, then inspect actual per-node resources before launch. A smaller object store is an explicit backend configuration choice, not a smaller benchmark memory budget or proof that the task is schedulable.

For supported Harbor jobs, set `harbor_admission_finalization_allowance_s` and an explicit `drain_margin_s`. The [phase-budget check](../responses_api_agents/harbor_agent/README.md#optional-declared-phase-budget-admission) resolves the downloaded task and installed agent before execution. Inspect `gym-phase-budget.json`: environment startup, agent setup, agent execution and verification all count, including a second environment startup for separate verification. Do not size the drain from the solver timeout alone. The declared phases plus operational allowance must fit the drain; service shutdown and supervisor cleanup need their own space. This is a check on declared budgets, not a guarantee that artifact transfer or other operational I/O will finish within the allowance.

For each model role, `upstream_admission_timeout_seconds` optionally bounds cumulative queue waiting separately from `upstream_request_timeout_seconds`. See the [model-server queue policy](../responses_api_models/openai_model/README.md#optional-cumulative-queue-allowance). A solver followed by a judge needs allowance for both roles. Queue expiry is an infrastructure disposition, and a successful solver response followed by a rejected judge request is still unscored. Preserve that solver output; generic reverification can call the judge again and is not a zero-inference replay.

The shared [peer-safe profile](../configs/eval/rate_limit_peer_safe.yaml) configures `policy_model`; it does not automatically configure a separate judge or helper. Check each role's resolved configuration. An expired cooldown means only that an observed wait has elapsed, not that provider capacity is available. With `max_attempts: 1`, a rejected call stays failed, but its cooldown still delays later distinct calls. A longer provider `Retry-After` can exhaust the cumulative queue allowance before the next send; retain that as a zero-send infrastructure failure. A 504 or interrupted response remains uncertain and is not a safe rejected retry. Host-local cooldown state does not establish shared quota scope across workers.


Read a supervisor receipt's child `return_code`, not only the supervisor command's exit status. Successful cleanup can coexist with a failed child evaluation. Confirm scored, failed, unstarted and unresolved counts from the native records before calling the attempt successful.

## Harbor cancellation and retry ownership

For guarded Harbor calls, the agent submits once; the model server owns the configured physical retry policy and shared cooldown. A one-attempt model policy must not be multiplied by an outer agent retry. Preserve rejected calls as unscored infrastructure outcomes and unknown calls as unresolved; neither becomes an automatic new attempt.

Harbor defaults to a 60-second worker-finalizer settlement window and a 70-second service shutdown grace. The latter must exceed the former and also covers surviving descendants of a logging launcher. Other services keep their existing one-second default. These cleanup windows do not extend scientific solver or verifier timeouts. Keep them inside the overall execution budget.

Inspect the job's `gym-cancellation-request.json` and `gym-cancellation-worker.json`, retained artifacts, and exact owned container/service state. A settled Ray result or worker receipt alone does not prove Docker cleanup. Missing receipts, artifact errors, or remaining containers require reconciliation before releasing resources. See the [Harbor integration guide](../responses_api_agents/harbor_agent/README.md) for the supported boundaries. Local regression tests cover this policy; each new execution environment still needs its native cleanup check.

For readonly nested artifact trees, the [opt-in bundled Docker export](../responses_api_agents/harbor_agent/README.md) preserves source permissions and validates the captured tree before publication. Select only the exact required paths through the existing environment configuration. Before a long run, check the complete tree against its 1GiB and 100,000-member limits, including expected outputs; links and special files are unsupported. Do not omit required inputs or merge a prior partial capture to make export pass. Retain both Harbor's artifact manifest and the sibling bundle receipt. A capture failure leaves the scientific result unqualified even when the solver finished.

For visual tasks, verify the complete generated-file → image observation → model request path. A vision-capable model behind a text-only terminal harness does not establish visual-task support. Keep an image-capable harness variant explicit and preserve exact image bytes in model-call evidence.

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

Use the **actual** materialized-input filename from the run, not an assumed reconstruction. Reverification preserves saved generation requests when present while taking task/reference fields from supplied inputs; inspect provenance differences. Every new reverification records a versioned scorer-lineage binding beside its output, including hashes of the supplied inputs and saved rollouts, resolved scientific configuration, and locally resolved component/shared Python sources. Resume and append check that binding before skipping cached rows or changing result files; old outputs without this binding require a distinct output path. Placement and reporting options such as ports, output paths, concurrency and aggregation do not change this binding. The binding is conservative local cache protection, not proof of external asset, dependency-package, remote-service or runtime-import identity. Changes to those external dependencies still require separate qualification and a new output. It does not regenerate solver answers, but an LLM judge can make new paid calls. A zero-inference regrade requires a deterministic/resource-only configuration or a qualified saved-judge replay. Keep historical raw results unchanged and compare by exact task/draw identity, not row order.

Command syntax was checked against the current CLI's `--help` for all seven command families above. Configuration-dependent templates were not executed as inference or claimed as runtime qualification.
