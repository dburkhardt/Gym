# CORE460 complete-workflow control

Run the frozen capsule's unchanged `bash run.sh`, which invokes the full
`Gazer_walkthrough.R`. Preserve both retained gaze/pupil tables, preprocessing,
filtering, aggregation and plotting. This control does not install packages,
download substitute inputs or answer questions from evaluator references.

The caller supplies the frozen 125-file checksum manifest as
`/solution/source.sha256` and starts with an empty results directory. The control
retains before/after source checks, original stdout/stderr, mount/network records,
and both generated PNGs. PNG signature and dimensions establish artifact
production only; they do not establish scientific or visual-answer correctness.

The empty `report.json` deliberately contributes no model-quality score. Actual
execution requires the separately qualified immutable image and original task
resources/time budgets. A successful import probe or shell syntax check does not
qualify this full workflow. Keep the oracle solution out of native solver tasks.
