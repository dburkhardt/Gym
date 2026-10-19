# GenoTEX: three-stage end-to-end GEO pilot

This **experimental subset integration** evaluates dataset filtering/selection, preprocessing,
and gene-trait statistical analysis. It is not the published GenoAgent harness, not the full
GenoTEX benchmark, and not an analysis-only task supplied with gold preprocessed data.
The solver is a generic Gym model-server shell loop; consequently results characterize the
model **and this harness**, and are not directly comparable to published GenoAgent scores.

## Provenance and selection

[Upstream GenoTEX](https://github.com/Liu-Hy/GenoTEX) is pinned to
`2ce91ce8fe326bb53a64a7cf25d57d961f75fe36`; the [dataset](https://huggingface.co/datasets/Liu-Hy/GenoTEX)
is pinned to `9d50c9020256e8c943e02b6c0ad843017cd76cf8`. Upstream code/data are CC-BY-4.0;
this adapter is Apache-2.0. No upstream code, references, or downloaded datasets are vendored.
Cite Liu, Chen, Zhang, and Wang, [GenoTEX](https://arxiv.org/abs/2406.15341).

The pinned metadata contains 1,384 trait-condition pairs. Preparation selects exactly five
problems before seeing model outputs: two unconditional, two Age/Gender-conditioned, and one
two-trait problem. Within each category, order by total required raw GEO bytes, then task ID.
Require a published regression reference and exclude traits requiring Xena/TCGA mapping.
This is deliberately a resource-bounded pilot, not a random representative sample. The selected
problems are Insomnia/None, Polycystic_Ovary_Syndrome/None, Insomnia/Age, Insomnia/Gender,
and Fibromyalgia/Insomnia. Full selection and exclusion reasons are saved outside the repo
in `selection.json`. All available GEO cohorts for the required traits are included; no
within-cohort truncation or synthetic data is used. HF LFS SHA256/Git blob SHA1 checksums are
verified for every downloaded file. Docker tags and Python dependency pins are recorded;
record the resolved image digest with each actual run.

## Task contract and isolation

`runtime.task_prompt()` defines the model-visible task and required output schemas. The
sandbox receives raw GEO files, public `tools/` helper functions, and public metadata including
known gene associations. It receives **no reference notebooks, processed matrices, regression
answers, or evaluator code**. The full task requires filtering cohorts, preprocessing clinical
and gene data, integrating samples, selecting cohorts, and statistical analysis. The selected
condition and all required trait raw inputs are available; references are mounted nowhere.

The solver gets a `shell(command)` tool in an isolated Docker container with no network,
API credentials, Docker socket, capabilities, or host filesystem mounts other than read-only
public inputs and a dedicated writable output directory. References remain host-side.
Every model starts a fresh task container and must independently produce its own preprocessing.
Output symlinks are rejected before host-side grading. Do not run on an untrusted shared Docker
daemon or mount the full upstream checkout into the solver container.

## Scoring and run conditions

The **unmodified pinned upstream metric functions** score each stage. The only scope adaptation
restricts `get_question_pairs()` to the requested pair; otherwise upstream inserts extra
unconditional pairs from metadata. All stage metrics and raw upstream omissions/undefined
values are preserved (NaN serializes as null). Missing or malformed outputs can be skipped by
upstream averages; therefore inspect `raw_upstream`, `requested_pairs`, `expected_files`, and
`missing_files` together. `complete` denotes presence of expected artifacts, not correctness
or successful parsing. No complete-benchmark score is claimed.

The scalar Gym reward is upstream **analysis F1 / 100**, with absent/undefined F1 mapped to
zero for Gym only. Dataset selection and preprocessing remain separate metrics; they are
never mixed into a synthetic composite reward. A correct gene list does not prove correct
preprocessing. Report the three stages separately and report execution failures in denominators.

Defaults: one attempt, 80 model steps, two-hour wall-clock task budget, 300-second shell-command
limit, 4 CPUs/16 GB per sandbox, concurrency one, temperature zero. Model-native reasoning is
retained by the configured Gym model server. Both compared models must use the same settings.
Tool errors are returned to the solver. Infrastructure errors are distinguished with
`mask_sample=true`; task timeouts preserve partial outputs and are scored as such. Full
trajectories, tool calls, artifacts, scorer stdout, task IDs, and elapsed time are retained
under `work_root` outside the repository. Model usage remains in each saved response.

## Preparation and execution

Run from the Gym root. Keep data and results outside the public checkout:

```bash
export GENOTEX_DATA_ROOT=/path/outside/repo/genotex
export GENOTEX_WORK_ROOT=/path/outside/repo/genotex-runs
python benchmarks/genotex/prepare.py --data-root "$GENOTEX_DATA_ROOT"
docker build -f responses_api_agents/genotex/Dockerfile -t genotex-pilot:latest .
uv venv --python 3.11 /path/outside/repo/genotex-grade-env
uv pip install --python /path/outside/repo/genotex-grade-env/bin/python \
  numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 scikit-learn==1.5.2 \
  statsmodels==0.14.4 sparse-lmm==1.4 biopython==1.85 tqdm==4.67.1
export GENOTEX_GRADER_PYTHON=/path/outside/repo/genotex-grade-env/bin/python
gym env validate genotex
# Supply the usual private Gym model-server configuration for each model.
gym eval run --benchmark genotex --model-type openai_model \
  --split benchmark --limit 1 --num-repeats 1 --output /private/results/genotex-model-smoke.jsonl
# Inspect one trajectory and grading.json before expanding. For the remaining four,
# override the configured dataset jsonl_fpath to $GENOTEX_DATA_ROOT/tasks-remaining.jsonl
# and use a NEW output/cache with --limit 4. Do not increase --limit on a resumed
# one-row cache: Gym reuses its original materialized inputs.
# For an unchanged task set/model/config, reuse output/cache and --resume.
```

Use a distinct output/cache per model and pinned configuration. Never reuse a prior model's
cache for a different model or harness setting. Credentials belong in private config only.

## Validation and limitations

`gym env validate genotex` checks the manifest and prepared inputs. As an external agent
integration (`resources_server: null`), `gym env test genotex` and `gym env publish genotex`
currently cannot use a verifier fixture. Run the dedicated tests and real rollouts instead:

```bash
GENOTEX_UPSTREAM="$GENOTEX_DATA_ROOT/upstream" /path/to/grade-env/bin/python -m pytest \
  responses_api_agents/genotex/tests/test_genotex.py --confcutdir=responses_api_agents/genotex/tests -q
.venv/bin/python -m pytest responses_api_agents/genotex/tests/test_agent.py \
  --confcutdir=responses_api_agents/genotex/tests -q
```

Tests cover deterministic selection, all three metrics on exact artifacts, wrong genes,
missing and malformed outputs, source/reference separation, and path/symlink rejection.
Tests using upstream scoring require the pinned checkout plus the grading dependencies.
Unit tests do not certify a model rollout or reproduce published results. TCGA-backed problems
and the remaining benchmark are intentionally not yet implemented by this pilot preparation.
