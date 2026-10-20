# CORE-Bench: one frozen medium capsule

This native Harbor binding covers only `capsule-9070543`, from creator revision `e32a2980e72fe6eb04ee04eb749458f570625663`. It does not represent full CORE-Bench coverage. The creator scientific task, intended code/data inputs and scorer are preserved. The campaign uses Terminus through Gym instead of the creator's AutoGPT-CORE harness.

Set `COREBENCH_PREPARED_DIR`, `COREBENCH_SOURCE_AUDIT`, and `COREBENCH_RUNTIME_ARCHIVE`, then run `python benchmarks/corebench/prepare.py`. Preparation verifies all 59 frozen source files and the recovered Python3.11.10 archive SHA, safely extracts the runtime and publishes the native dataset through Gym's shared staging helper. It adds workspace artifact capture for native source-job regrading. No dataset or runtime is downloaded by preparation.

The scoring wrapper writes a scalar `reward` and separate creator components, correcting the previous Harbor reward-schema rejection. Reference answers and scorer remain outside the solver image. The reference-based offline control is scorer qualification, not a model evaluation.

The creator image tag was resolved read-only on October20 to digest `sha256:b0f02683bde424099bb568607b13175d7b0c2603fb0b278e3f9926d9275615b5`; preparation pins that digest. The old worker no longer held that base image, so equality to the historical campaign image was not established. Apt and transitive pip packages are resolved during image build and their installed versions are recorded under `/opt/corebench-build-metadata`; record the final built image digest before inference. This is not a claim of fully locked rebuilds.

Offline tests verify immutable staging, modified-input refusal, reference exclusion, and creator-scored correct/incorrect/missing submissions. Actual native Docker correctness, incorrect-answer and isolation controls remain required before live use. Native Harbor owns sandbox lifecycle, captured workspace regrade and all solver routing. Model selection and dispatch guards come from the campaign runtime configuration.
