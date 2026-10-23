# CORE-Bench: one frozen medium capsule

This native Harbor binding covers only `capsule-9070543`, from creator revision `e32a2980e72fe6eb04ee04eb749458f570625663`. It does not represent full CORE-Bench coverage. The creator scientific task, intended code/data inputs and scorer are preserved. The campaign uses Terminus through Gym instead of the creator's AutoGPT-CORE harness.

Set `COREBENCH_PREPARED_DIR`, `COREBENCH_SOURCE_AUDIT`, and `COREBENCH_RUNTIME_ARCHIVE`, then run `python benchmarks/corebench/prepare.py`. Preparation verifies all 59 frozen source files and the recovered Python3.11.10 archive SHA, safely extracts the runtime and publishes the native dataset through Gym's shared staging helper. It adds workspace artifact capture for native source-job regrading. No dataset or runtime is downloaded by preparation.

The scoring wrapper writes a scalar `reward` and separate creator components, correcting the previous Harbor reward-schema rejection. Reference answers and scorer remain outside the solver image. The reference-based offline control is scorer qualification, not a model evaluation.

The creator image tag was resolved read-only on October20 to digest `sha256:b0f02683bde424099bb568607b13175d7b0c2603fb0b278e3f9926d9275615b5`; preparation pins that digest. The old worker no longer held that base image, so equality to the historical campaign image was not established. Apt and transitive pip packages are resolved during image build and their installed versions are recorded under `/opt/corebench-build-metadata`; record the final built image digest before inference. This is not a claim of fully locked rebuilds.

Offline tests verify immutable staging, modified-input refusal, reference exclusion, and creator-scored correct/incorrect/missing submissions. Actual native Docker correctness, incorrect-answer and isolation controls remain required before live use. Native Harbor owns sandbox lifecycle, captured workspace regrade and all solver routing. Model selection and dispatch guards come from the campaign runtime configuration.

## Frozen20 preparation

`frozen20.yaml` / `prepare_frozen20.py` retain the separate frozen20 selection (43 questions). Set `COREBENCH20_MEDIUM_ROOT` to the preserved medium trees, `COREBENCH20_REFERENCE_ROOT` to the original per-capsule prompt/evaluator folders, and `COREBENCH20_QUALIFICATIONS` to image-control receipts. Preparation validates the entire source set before publishing qualified rows; the receipt lists every blocked capsule. No image qualification means no runnable dataset, rather than silently treating source acquisition as integration.

`image_controls.stage_context` builds a checked context using the pinned recovered Python archive. Harness Python lives at `/opt/corebench-harness`; creator PATH and scientific interpreters are preserved. `SCIENTIFIC_PROBE` must match in creator and final images (use the same `/tmp` working directory), and `HARNESS_PROBE` must pass before a receipt is accepted. The recipe refuses nonempty existing `/code`, `/data`, or `/results` rather than hiding creator files. Runtime variants remain explicit. Qualification binds exact source/recipe hashes, final image ID, resource profile, tool checks, reference isolation and cleanup.

The six previously measured workflows are candidates for image controls, not already qualified agent integrations. CUDA capsules remain blocked without GPU resources; the observed OOM requires an explicit resource review. The standalone9070543 result does not qualify any of these20 images. Native source-job regrade remains the supported saved-output path.

New frozen20 image qualifications require `qualification_sha256 = image_controls.qualification_digest(row)`. This binds the exact scientific and harness probes, recovered Python archive, recipe, medium files, instruction, reference and scorer. The scientific probe now records R package versions, build versions and library paths as well as interpreter versions. Historical receipts remain evidence of their original controls; they are not silently upgraded to this binding and must be requalified before new publication through the updated preparation path.

Capsule6133093 explicitly selects `ubuntu-bionic-isolated-v1` for harness tmux installation. The historical recipe failed because its third-party CRAN repository has an expired signature. The variant uses only exact existing Ubuntu bionic archive/security source lines with private apt lists and signature validation enabled, leaving original source files and keys unchanged. Every previously installed dpkg package must retain its version; only tmux, libevent-2.1-6 and libutempter0 may be newly installed. The helper bytes are bound into the recipe digest. This is a harness installation variant, not a scientific dependency upgrade or a claim of a hermetic Ubuntu package archive.

Image context publication stages verified files in a private cache outside the Docker COPY tree, then atomically publishes the clean context under the shared staging lock convention. Generated staging locks never enter capsule or harness trees. Legitimate creator filenames ending in `.stage.lock` remain intact; exact final-image source checking is unchanged.

Scientific execution can explicitly increase a qualified capsule's memory through
`COREBENCH20_EXECUTION_RESOURCES='{"capsule-8197429":{"memory_mb":8192}}'`.
Preparation retains the original image qualification receipt and records the separate
execution allocation. For capsule 8197429 the image control used 4 GiB, while the
historical scientific workflow used 8 GiB. This option does not claim that the image
probe ran with 8 GiB; CPU, storage and scientific deadlines remain unchanged.
Only integer memory increases for already qualified capsules are supported.

A narrowly reviewed compatibility transition admits the original capsule8197429
v2 receipt under the exact later qualification implementation that only added
capsule1906954 to the bionic allowlist. It requires the original qualification
hash, default variant, exact reviewed implementation hash, and recomputed recipe,
source, reference, scorer, runtime archive and probe inputs. All image/control
checks still apply. Preparation preserves the original receipt and adds a separate
`qualification_compatibility_reviews` entry; it does not claim a new image probe.
Unknown implementations or changed inputs require review and remain refused.

A future digest schema should bind the selected generated recipe and probes plus
a versioned qualification engine, avoiding repeated mappings for unrelated
capsule allowlist additions. That redesign is not part of this exception.

`auxiliary_controls.observe_auxiliaries` provides optional, source-bound diagnostic
observations for scientific executables beyond the R/Python package inventory,
for example PhantomJS or TeX. Each selection specifies an executable name or
absolute path, `--version`/`-V`, and the exact frozen source-file hashes explaining
the selection. It reuses the existing no-network owned image probe (120-second
execution timeout, 180-second sandbox TTL), preserving the creator PATH and
recording executable hashes, version exit status and `ldd` output. Its separate
`auxiliary-observation.json` binds the selections, full source manifest, published
image, observed image identity and both probe implementation hashes. Missing tools
and failed version queries are diagnostic outcomes, not successful qualification.
No package installation, workflow execution, default probe change or automatic
upgrade of historical qualification receipts occurs. A reviewed environment
variant or readiness decision remains separate.

The source-file hashes identify the reviewed frozen instructions used to select
tools; this diagnostic does not inspect those files inside the creator image.
Actual image contents remain subject to the separate maintained source-image
qualification boundary. Image identity alone is not source-file verification.

CORE313's `ubuntu-bionic-isolated-v1` harness variant admits only its exact
published digest and the existing signed Ubuntu-source installer. CRAN sources
and all installed scientific package versions remain preserved. The opt-in
`qualify_required_tools` auxiliary control requires successful dynamic-tool
queries and exact before/after executable paths, hashes, version output and
linked-library hashes; only loader ASLR addresses are normalized. It supplements,
never replaces, ordinary image/source/reference qualification. Missing tools or
changed libraries refuse the additional control. Diagnostic-only observations
retain their previous behavior.

The CORE313 allowlist-only engine transition explicitly admits the preceding
exact engine hash for the two observed preceding-engine receipts (1683542 and 6746514), recomputing every original
recipe/source/reference/probe binding. Historical receipts are not rewritten;
older already-incompatible engines are not newly admitted. A future digest
schema should separately bind the selected generated recipe/probes and a
versioned qualification engine, avoiding recurring mappings for unrelated
allowlist additions. This change does not implement that broader redesign.

Fresh frozen20 preparation adds evaluator-only `item_diagnostics.json` to the
collected verifier logs. The observer qualifies the exact creator scorer bytes,
loaded function and AST structure, then observes the single existing evaluation:
raw/normalized predictions, reference observations, actual numeric intervals and
original comparison/counter outcomes. It does not reimplement scoring, adjust
rounding/tolerances, regrade, change rewards or mask samples. Missing/unreached
comparisons and caught exceptions remain distinct; non-finite intervals are
explicit. A preexisting trace is untouched, and an unsupported source or failed
observer produces unavailable diagnostics without changing original scoring.
The reference-bearing artifact stays outside the solver workspace. Historical
prepared tasks/results and the legacy single-capsule preparer are unchanged.
A zero-width interval can expose rounding sensitivity, but this metadata does
not by itself establish scientific invalidity or authorize a score correction.

For R images, `auxiliary_controls.qualify_creator_r_startup` supplies an additional before/after check using normal image startup. A `--vanilla` inventory may omit creator site libraries, as observed for capsule 8912293. The check compares the Rscript executable, R version, library paths, selected startup variables and installed package inventory through owned no-network probes. It supplements `qualify_images`; it does not exercise project-local startup files or establish scientific execution. Keep its receipt with the ordinary image/source qualification.

Capsule 8912293 requires the explicit `filesystem_layout_variant: 8912293-relative-data-v1` when using the workspace overlay. Its unmodified R scripts reference `../../data` and `../../results`; physical resolution of `/code` otherwise misses these paths. The variant adds workspace aliases to the existing data/results directories, is bound to the exact creator image, and changes the recipe identity. Requalify the image and verify relative-path resolution before execution. Historical receipts do not establish this variant's readiness.

For capsule 8912293, `external_input_variant: 8912293-tse-2016-current-v1` explicitly pins the currently retrieved creator-referenced election ZIP (SHA256 `0e511c46a2e22c1766c21dd5d8d5d9f50596e1316cc8243a52af74b384784ddd`). Supply it through `stage_context(..., external_input_path=...)`; missing or changed bytes fail staging. The image adds a root R startup profile that verifies the ZIP and copies it into each R process's temporary cache, using the creator package's existing offline path. The build refuses to overwrite an existing root profile. Scientific scripts remain unchanged. This is a versioned external-input variant, not proof of the original historical data bytes. The layout variant is required. Full scientific execution and startup parity must still be validated; unit tests alone do not establish readiness.

### Saved-output replay

For a completed model trial, use Harbor's native `source_jobs` regrade route.
Prepare a separate replay task with `benchmarks.corebench.saved_regrade.prepare_task`;
it changes only the verifier mode and preserves the original scorer/reference
files. Harbor restores declared `/workspace` artifacts to `/workspace`, so do not
rewrite the scorer to read `/logs/artifacts/workspace`. Keep the live task and
original result unchanged. Compare creator components and report hashes, confirm
no new model captures, and require a successful trial result and cleanup; job
process exit alone does not prove the verifier succeeded.

GPU placement may be added to an otherwise CPU-qualified image without rewriting
its qualification receipt. Pass an execution `gpus` override with separate
`placement_evidence` (CLI preparation: `COREBENCH20_PLACEMENT_EVIDENCE`). The
hash-bound owned workflow receipts must match the frozen source manifest,
immutable final image, CPU/memory/GPU resources and original agent deadline, with
successful execution, fresh stdout and exact container cleanup. The original
image qualification remains unchanged in preparation metadata. This attests
execution placement, not model performance or absence of historical outputs.
