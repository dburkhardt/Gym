# SciCode Resources Server

Verifies multi-step scientific code generation on the
[SciCode](https://huggingface.co/datasets/SciCode1/SciCode) benchmark. The agent submits the code it
generated for each sub-step; this server executes each sub-step's accumulated code against
ground-truth targets in `test_data.h5` and returns the reward.

- Task type: multi-step code generation (one LLM call per sub-step) + local code execution
- Domain: `coding`
- Tasks: 65 problems / 288 evaluated sub-steps (`test` only, matching the current AA Intelligence Index setup)
- Reward: binary per problem — `1.0` iff every sub-step passes its tests

> **Run commands and `env.yaml` setup**: see [`benchmarks/scicode/README.md`](../../benchmarks/scicode/README.md),
> including how to stage the required `test_data.h5`.

## Server Composition

SciCode uses a **custom multi-step agent** — `simple_agent` will not work:

- `responses_api_agents/scicode_agent` (generates code sub-step by sub-step, accumulating prior code)
- `responses_api_models/*` (typically `policy_model`)
- `resources_servers/scicode` (this server)

The full wiring lives in `benchmarks/scicode/config.yaml`, which chains all three plus the dataset.
The agent + its `example` dataset are defined in this server's own `configs/scicode.yaml`.

## Managed scientific execution

The agent submits accumulated solutions to `/verify`. Each scored sub-step is executed in a fresh shared `AsyncSandbox` Docker sandbox, with no network, a read-only root, an unprivileged user, explicit CPU/memory limits, and only read-only program/test-data mounts. Generated code does not inherit resource-server credentials or its Python environment. Containers and descendants are removed after success, failure, timeout, or cancellation; durable lifecycle receipts remain under `execution_artifact_dir`.

Build `sandbox/Dockerfile` and configure `sandbox_image` with its immutable image ID. The base image and scientific packages are pinned: Python 3.12 and SciPy 1.13.1 preserve `scipy.integrate.simps`. Host Gym still uses its required Python version. `sandbox/requirements.txt` is separate from the resource-server requirements. The default concurrency is 2, with 1 CPU/4096 MiB per active sub-step; execution retains the native 30-second budget. Infrastructure failures raise instead of becoming incorrect answers.

The HDF5 loading/comparison helper bytes and sanitized creator assertions remain unchanged. `test_data.h5` must match the configured SHA256 (default canonical creator digest); a missing or mismatched file fails before scoring. Positive/negative scoring controls and owned lifecycle tests must pass before live use. The old synchronous `run_substep` function remains only as a legacy isolated unit-test helper; the server never calls it.

**Scoring notes** (mirroring nemo-skills):
- Sub-steps absent from `solutions` (prefilled reference steps) are excluded from the denominator.
- Out-of-context sentinels (`_ran_out_of_context_`) are counted as failed sub-steps.
- `verify()` returns `reward` + `step_results`, `num_steps_passed`, `num_steps_total`,
  `problem_accuracy`. The headline `subtask_accuracy` (total passed / total over all rollouts) is
  computed on the **agent** (`/aggregate_metrics` runs there), not this server.

## Test data (required, staged manually)

`test_data.h5` (~1 GB) holds the numeric ground-truth targets. It is **not** auto-downloaded — stage
it from the official SciCode Google Drive and point `test_data_fpath` at it (see
[`benchmarks/scicode/README.md`](../../benchmarks/scicode/README.md)). If it's missing, `verify()`
fails fast with a clear error rather than scoring everything as wrong.

## Dataset Format

Two JSONL files coexist with different shapes:

- **`benchmarks/scicode/data/scicode_benchmark.jsonl`** (65-row test split, gitignored; produced by
  `prepare.py`). Flat-field: each row has `problem_id`, `sub_steps`, `required_dependencies`, `uuid`.
- **`resources_servers/scicode/data/example.jsonl`** (5-row fixture, committed).

Unlike most benchmarks, SciCode's prompts are **not** materialized at rollout time — the agent builds
each sub-step's prompt itself from `sub_steps`. So rows carry an empty `responses_create_params.input`,
the dataset `prompt_config` is `null`, and you do **not** pass `+prompt_config` when collecting
rollouts.

## Running servers

```bash
gym env start --benchmark scicode --model-type openai_model
```

## Smoke test (5 example problems)

Requires `test_data.h5` staged. Use the benchmark config (so `test_data_fpath` is set) and the
benchmark agent:

```bash
gym eval run --no-serve \
    --agent scicode_benchmark_agent \
    --input resources_servers/scicode/data/example.jsonl \
    --output results/scicode_smoke.jsonl \
    --num-repeats 1 \
    --temperature 0.0
```

## Tests

```bash
gym env test +entrypoint=resources_servers/scicode
```

Covers test sanitization, the program builder, subprocess pass/fail/timeout, and the `verify()`
paths (no solutions, missing/unconfigured test data, all-pass, all-fail, prefilled-excluded,
out-of-context).

## Licensing

- **Code**: Apache 2.0
- **Data**: SciCode dataset (`SciCode1/SciCode`) + `test_data.h5` — Apache 2.0
- **Dependencies**:
  - nemo_gym: Apache 2.0
