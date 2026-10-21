# PromptBio-Bench (experimental)

This integration evaluates file-producing bioinformatics tasks with a minimal shell agent in Docker and the pinned upstream PromptBio evaluator with an explicit file-mapping safety check. It is a new harness configuration, not a reproduction of the published agents. Scores from a pilot subset are not full-benchmark estimates.

## Measurement contract

The public [dataset](https://huggingface.co/datasets/promptbio-ai/promptbio-bench-data) is pinned to `078420b69e25c41c99a79135724b6ae6a6c82b69` (244 tasks, Apache-2.0). The [evaluator](https://github.com/PromptBio/promptbio-bench) is pinned to `e2a1947894eac530900247f00c0b726363b540ed` (Apache-2.0). Preparation preserves task definitions and IDs, performs no answer-driven filtering, and downloads metadata only. Execution downloads the selected task assets.

The solver sees `task.json`, the declared files under `data/`, and an empty `outputs/` directory. References, `eval.json`, grading scripts and API keys are never mounted into the container. The host performs model calls; the sandbox has no network. The solver can run bash, read input files, and write requested outputs. A fixed tools image must supply the task's dependencies; missing software is an environment limitation and must be reported, not described as a general model failure. Docker resource defaults are 2 CPUs/4 GB, one task at a time. Production callers should pass an immutable Docker image ID or digest.

Default agent limits: 100 turns, 16,384 output tokens per call, temperature zero, one attempt, and the task's original wall-time limit. No optional domain skills are injected. The maintained adapter makes one physical request per completion; ambiguous errors are retained without automatic resend. Capacity waiting belongs to the shared admission layer before dispatch. The native managed path allocates a fresh durable attempt UUID and never reconstructs an old shell automatically. Full shell-state recovery is not implemented; interrupted attempts require inspection rather than automatic repetition. Historical standalone receipt reuse is not native resume. All assistant/tool messages and available token usage are saved outside the repository.

Grading preserves the pinned evaluator: shared Detect and Recommend work followed by Match/Compare for each output directory. `comparison_models` runs the requested models on the same task and passes all their output directories to **one** evaluator invocation, sharing reference detection and comparison recommendations. Reward is the upstream `avg_similarity` in [0,1], higher is better. Upstream averages files with success/invalid status; skipped/errored files are retained in reports and must be disclosed. Missing or nonfinite overall scores raise grading errors rather than becoming zero. Legitimate task/turn/output-token exhaustion retains its model-budget disposition and still grades the saved outputs; infrastructure failures do not proceed to scoring. Judge decisions are nondeterministic; `determinism: unknown`.

The historical standalone pilot default judge was `gpt-5.4`; it is not authorized for this campaign. The maintained benchmark config now matches `campaign.yaml`: every solver, helper and judge must use `nvidia/nvidia/nemotron-3.5-super-vl` through a rollout-qualified native Gym model server. Legacy gateway fields are retained for historical configuration compatibility. Historical paid-pilot calls passed through a host-only budget proxy: durable reservations before each request, conservative uncached pricing ($2.50/M input, $15/M output), maximum 16,384 output tokens, no image or >272k-input requests, and a $100 ceiling. Unknown usage/outcomes stop further requests. These bounds are a pilot restriction; image-based grading is **unsupported** in this pilot. Use one process/ledger per paid allocation. Do not launch independent workers against the same ledger. Inspect the ledger after any unknown outcome before retrying.

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

The supported campaign path is `gym eval` with `--benchmark promptbio_bench`, a guarded native model server, and an immutable solver image plus a **separately qualified evaluator image**. Set `PROMPTBIO_SCORER_IMAGE` to its digest and `PROMPTBIO_SCORER_PYTHON` to the Python executable inside that image. The scorer image must contain the pinned evaluator dependencies, Git, and the shared guard's HTTP client dependencies. No qualified scorer image is currently recorded, so fresh native inference remains blocked until real environment and saved-output controls pass.

Solver and judge use shared `AsyncSandbox` Docker ownership, resource limits, cancellation settlement and cleanup receipts. Both have network disabled. The judge alone receives the pinned creator checkout, materialized task references and readonly solver outputs; the solver never receives these. A task-scoped Unix socket forwards exact-model requests through the native Gym capture/dispatch guard. Provider credentials are never mounted. Durable per-role request/response records and source hashes survive failures; ambiguous outcomes are not resent.

`comparison_models` still passes all output directories to one creator invocation, preserving shared reference detection and recommendations. The campaign authorizes only the exact Nemotron model for every role. The old `run_pilot`, `campaign`, proxy and raw Docker functions remain historical artifacts and do not establish current shared-runtime acceptance. Do not use them as a substitute for this native admission gate.

## Validation and limits

```bash
pytest responses_api_agents/promptbio_bench/tests
```

Tests check reference exclusion, path traversal, exact score preservation, malformed/missing grading, and durable budget refusal. The external engine has `resources_server: null`; fixture-based `gym env test NAME` and `gym env publish NAME` are unsupported. Use the component tests and real paired rollouts instead. Synthetic tests are not benchmark results. Full scientific-task support and published-result reproduction have not been established; leave the manifest experimental.

## Broad campaign: Nemotron-only variant

The native guarded scorer overrides every LangChain model initializer, including nested format-detection helpers, to `nvidia/nvidia/nemotron-3.5-super-vl`, 16,384 output tokens and zero SDK retries. It preserves the creator's comparison/parser logic and rejects incomplete judge responses without returning a fabricated zero. The creator invocation still shares detection/recommendations across paired outputs. This is a Nemotron-judge and minimal-shell-agent variant, not published-agent reproduction.

Historical CRISPR task `a-13-1` has a demonstrated scorer mismatch (identical required guide pairs can receive different scores). LC-MS `a-8-1` has an unspecified signed-ppm convention. Preserve both regressions and raw scores; neither is repaired by model substitution. The fresh uniformly seeded sample may expose unsupported dependencies or large downloads, which must be inspected before runtime. Local contract tests do not establish live evaluator compatibility.

## Frozen campaign preparation

Set `PROMPTBIO_FROZEN_SELECTION` to the existing campaign `selection.json` and `PROMPTBIO_TASK_SNAPSHOT` to the cached pinned Hugging Face snapshot. Preparation verifies the selection hash and reconstructs all244 original definition rows to verify the recorded source checksum before publishing the selected20 in their original order. Only stable top-level task IDs are added; prompts and verifier metadata remain unchanged. Shared atomic staging protects repeated publication. This does not qualify scientific inputs, the execution runtime or scoring. Native CLI preparation still installs the declared Hugging Face dependency before invoking this offline path.

## Native scorer safety variant

The pinned matcher accepts arbitrary model-selected candidate paths. The maintained wrapper validates that each returned reference is in the supplied reference list, rejects duplicate reference mappings, and permits only null or supplied candidate paths after resolving paths. An attempted reference-as-candidate self-comparison is a verifier failure, with the raw mapping retained. This is an explicit scoring-safety variant; ordinary valid mappings and creator metrics are unchanged, but unsafe historical outputs are not silently reclassified. A model-free normal/malicious/null/duplicate control covers the boundary. Full paired image controls remain pending.

## Reproducible evaluator-image qualification

`Scorer.Dockerfile` and `scorer-requirements.txt` reproduce the historical Python judge package versions recovered from the existing worker. Build from the verified historical solver digest `sha256:7020e2509833989127e3404131d4897e33ff0b922635f98ca194a6dc88f6fcb1` (Python 3.12), then record the resulting image ID and package manifests. This is a version-pinned definition, not a claim of bitwise rebuild reproducibility: system packages are captured after installation. It does not add every optional R, Julia or single-cell dependency listed by the creator. Missing handlers stay explicit runtime blockers; the fixed20 is not resampled.

The `qualify_runtime` module executes the actual creator pipeline inside the managed judge container against the original frozen `b-4-3` reference copied as an oracle output, paired with missing output. A loopback-only deterministic provider returns preserved structured judge fixtures through the real shared UDS bridge/guard. Expected scores are `[1, 0]`; the shared strategy pass must occur once. These are infrastructure controls, not benchmark results or native Gym model-capture acceptance. No provider key is accepted by this command.

```bash
docker build --pull=false --build-arg BASE_IMAGE=sha256:7020e2509833989127e3404131d4897e33ff0b922635f98ca194a6dc88f6fcb1 -f responses_api_agents/promptbio_bench/Scorer.Dockerfile -t promptbio-scorer-qualified .
python -m responses_api_agents.promptbio_bench.qualify_runtime --task /pinned/tasks/b-4-3 --task-sha256 VERIFIED_TASK_SHA --scorer /pinned/creator --image sha256:BUILT_IMAGE_SHA --output /new/control-directory
```

Before either command, validate worker identity, available memory/disk, total spending and any existing lease deadline. No image build or runtime control has yet been accepted for this new isolated scorer. The historical host judge environment is not an approved fallback.
