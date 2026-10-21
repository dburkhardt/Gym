# AIRS-Bench molecular research tasks

This integration preserves five research tasks: ZINC graph regression and four QM9 property targets. Each requires predictions for the complete creator test set (5,000 ZINC or10,000 QM9 rows). They are five tasks over two datasets, not50 sampled questions. Raw MAE is lower-is-better and is not normalized into a leaderboard score.

`frozen_sources.json` binds the preserved task definitions, creator scorers, full data arrays and qualified image IDs to their historical receipts. `prepare.py` requires a JSON mapping of all five task names to complete source directories in `AIRS_TASK_ROOTS_JSON`. It validates all scientific assets before publication and uses shared checksum-verified staging. Partial laptop caches containing Arrow metadata without arrays are rejected. Mac AppleDouble sidecars are excluded from scientific identity.

Native configuration and runtime overlays select immutable images and separate artifact-only scoring. Model-free native oracle, offset, missing-submission and saved-output regrade controls have passed for all five targets, including complete test rows and reference isolation. Missing submissions retain an attempted model outcome with undefined raw MAE; they are not infrastructure failures. Live solver acceptance remains pending.

The historical ZINC and Cv image IDs could not be recovered. Their source/base/data-bound reconstructions are available only through explicit `AIRS_RUNTIME_REVISION=reconstructed-v1`; default preparation retains the original images. The revision manifest binds the historical frozen-manifest SHA and original-to-qualified image mapping, and preparation writes a runtime-revision receipt. Scoring records both actual and historical image IDs. This is a versioned runtime qualification, not a claim of byte-identical recovery.

The preserved variant uses Harbor, CPU2/3GiB,3,600seconds,40turns and16,384 output tokens rather than the published MLGym/aira-dojo setup. Creator data split differences and target scaling remain intact. All future in-benchmark LLM roles use the campaign-authorized Nemotron model.

Each new saved-output regrade records the creator revision, scientific scorer asset hashes, manifest hash and regrade implementation hash. It writes to a new output directory and cannot overwrite historical scoring records. MAEs from different property targets must not be averaged into a single comparable performance score.

The native Gym configuration disables aggregation. Machine-readable scoring identities retain the target, minimize direction, exact creator label multiplier and creator-declared units separately; physical unit conversion is not independently established. No cross-target normalized reward is invented.

Native artifact capture excludes immutable `/workspace/data`, which remains available from the pinned image in both environments. This avoids copying read-only public inputs as if they were agent-produced artifacts. The verifier checks Harbor's artifact manifest before scoring: failed or incomplete transfer is an infrastructure failure, not a missing model submission. Only manifest-listed regular scorer files are uploaded into the separate verifier; extras, symlinks and special files are rejected.


Pinned Harbor 0.24.0 clears solver compose overrides for separate verifiers. The optional `responses_api_agents.harbor_agent.runtime_patches.sidecar_resources` utility therefore supplies a narrow, checksum-gated built-in sidecar resource variant (128 MiB, 0.25 CPU) for that exact Harbor revision. Apply it only to an isolated runtime copy and retain its original-file/provenance receipt. It does not forward arbitrary solver compose settings. Actual solver and verifier resource qualification is required before live use.
