# PromptBio-Bench (experimental)

This integration evaluates file-producing bioinformatics tasks with a minimal shell agent in Docker and the **unmodified upstream PromptBio evaluator**. It is a new harness configuration, not a reproduction of the published agents. Scores from a pilot subset are not full-benchmark estimates.

## Measurement contract

The public [dataset](https://huggingface.co/datasets/promptbio-ai/promptbio-bench-data) is pinned to `078420b69e25c41c99a79135724b6ae6a6c82b69` (244 tasks, Apache-2.0). The [evaluator](https://github.com/PromptBio/promptbio-bench) is pinned to `e2a1947894eac530900247f00c0b726363b540ed` (Apache-2.0). Preparation preserves task definitions and IDs, performs no answer-driven filtering, and downloads metadata only. Execution downloads the selected task assets.

The solver sees `task.json`, the declared files under `data/`, and an empty `outputs/` directory. References, `eval.json`, grading scripts and API keys are never mounted into the container. The host performs model calls; the sandbox has no network. The solver can run bash, read input files, and write requested outputs. A fixed tools image must supply the task's dependencies; missing software is an environment limitation and must be reported, not described as a general model failure. Docker resource defaults are 2 CPUs/4 GB, one task at a time. Production callers should pass an immutable Docker image ID or digest.

Default agent limits: 100 turns, 16,384 output tokens per call, temperature zero, one attempt, and the task's original wall-time limit. No optional domain skills are injected. The maintained adapter makes one physical request per completion; ambiguous errors are retained without automatic resend. Capacity waiting belongs to the shared admission layer before dispatch. Completed receipts resume only when task/model/image/settings match; interrupted attempts require inspection rather than automatic repetition. All assistant/tool messages and available token usage are saved outside the repository.

Grading uses the upstream Match → Detect → Recommend → Compare pipeline. `comparison_models` runs the requested models on the same task and passes all their output directories to **one** evaluator invocation, sharing reference detection and comparison recommendations. Reward is the upstream `avg_similarity` in [0,1], higher is better. Upstream averages files with success/invalid status; skipped/errored files are retained in reports and must be disclosed. Missing or nonfinite overall scores raise grading errors rather than becoming zero. Judge decisions are nondeterministic; `determinism: unknown`.

The historical standalone pilot default judge was `gpt-5.4`; it is not authorized for this campaign. The maintained benchmark config now matches `campaign.yaml`: every solver, helper and judge must use `nvidia/nvidia/nemotron-3.5-super-vl` through the campaign gateway, enforced by the configuration validator. Historical paid-pilot calls passed through a host-only budget proxy: durable reservations before each request, conservative uncached pricing ($2.50/M input, $15/M output), maximum 16,384 output tokens, no image or >272k-input requests, and a $100 ceiling. Unknown usage/outcomes stop further requests. These bounds are a pilot restriction; image-based grading is **unsupported** in this pilot. Use one process/ledger per paid allocation. Do not launch independent workers against the same ledger. Inspect the ledger after any unknown outcome before retrying.

## Prepare and run

From the Gym root:

```bash
gym eval prepare --benchmark promptbio_bench
gym env validate promptbio_bench
docker build -t promptbio-solver -f responses_api_agents/promptbio_bench/Dockerfile .
git clone https://github.com/PromptBio/promptbio-bench /path/to/scorer
git -C /path/to/scorer checkout e2a1947894eac530900247f00c0b726363b540ed
# Install upstream environment.yml into a separate environment; keep its Python path.
```

Use the paired pilot CLI, supplying endpoint/key paths at runtime:

```bash
python -m responses_api_agents.promptbio_bench.run_pilot \
  --task a-1-1 --model MODEL_ONE --model MODEL_TWO \
  --base-url ENDPOINT --key-file /private/solver.key \
  --cache /data/promptbio-cache --output /data/promptbio-results \
  --image IMMUTABLE_IMAGE_ID \
  --scorer /path/to/scorer --scorer-python /path/to/evaluator/python \
  --judge-key-file /private/judge.key --budget-ledger /data/promptbio-budget.json
```

Omit scorer arguments for solver-only smoke runs; their output is explicitly ungraded. First smoke task `a-1-1` is a no-download reverse-complement/FASTA task, chosen before viewing any model outputs to test the complete artifact path cheaply. It cannot establish performance on harder analysis tasks.

The Gym config exposes the same paths via `PROMPTBIO_*` variables. For paired runs set `comparison_models` to the second solver; `/run` returns the configured primary model's score and retains both outputs. Exceptions from execution/grading remain failed requests, not misleading reward-zero records. The independent paired CLI and Gym adapter use the same solver and scorer functions.

## Validation and limits

```bash
pytest responses_api_agents/promptbio_bench/tests
```

Tests check reference exclusion, path traversal, exact score preservation, malformed/missing grading, and durable budget refusal. The external engine has `resources_server: null`; fixture-based `gym env test NAME` and `gym env publish NAME` are unsupported. Use the component tests and real paired rollouts instead. Synthetic tests are not benchmark results. Full scientific-task support and published-result reproduction have not been established; leave the manifest experimental.

## Broad campaign: Nemotron-only variant

Use `python -m responses_api_agents.promptbio_bench.campaign`, **not** the historical paid-judge CLI above. Required arguments are `--selection` (JSON with seeded `task_ids`), `--key-file`, `--cache`, `--output`, `--scorer`, `--scorer-python`, `--image`, and optional `--limit 5`. This entrypoint fixes solver and every grading role to `nvidia/nvidia/nemotron-3.5-super-vl` through `http://127.0.0.1:49180/v1`; it has no paid judge route. Full solver and judge request/response receipts are saved, and wrong returned identities fail closed.

The upstream `--model` argument does not propagate to every nested format-detection call. `campaign_scorer.py` therefore explicitly overrides LangChain model initialization for **all** roles, logs the original requested model, and forces the same local proxy with a 16,384-token ceiling. Creator matching, detection, recommendation, and comparison code remains pinned/unmodified. This is a Nemotron judge and minimal-shell-agent variant, not published-agent reproduction. Completed structured tool-call judge responses are accepted; empty/incomplete judge responses fail rather than becoming valid scores.

Historical CRISPR task `a-13-1` has a demonstrated scorer mismatch (identical required guide pairs can receive different scores). LC-MS `a-8-1` has an unspecified signed-ppm convention. Preserve both regressions and raw scores; neither is repaired by model substitution. The fresh uniformly seeded sample may expose unsupported dependencies or large downloads, which must be inspected before runtime. Local contract tests do not establish live evaluator compatibility.

## Frozen campaign preparation

Set `PROMPTBIO_FROZEN_SELECTION` to the existing campaign `selection.json` and `PROMPTBIO_TASK_SNAPSHOT` to the cached pinned Hugging Face snapshot. Preparation verifies the selection hash and reconstructs all244 original definition rows to verify the recorded source checksum before publishing the selected20 in their original order. Only stable top-level task IDs are added; prompts and verifier metadata remain unchanged. Shared atomic staging protects repeated publication. This does not qualify scientific inputs, the execution runtime or scoring. Native CLI preparation still installs the declared Hugging Face dependency before invoking this offline path.
