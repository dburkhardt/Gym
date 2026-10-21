# AIRS-Bench molecular research tasks

This integration preserves five research tasks: ZINC graph regression and four QM9 property targets. Each requires predictions for the complete creator test set (5,000 ZINC or10,000 QM9 rows). They are five tasks over two datasets, not50 sampled questions. Raw MAE is lower-is-better and is not normalized into a leaderboard score.

`frozen_sources.json` binds the preserved task definitions, creator scorers, full data arrays and qualified image IDs to their historical receipts. `prepare.py` requires a JSON mapping of all five task names to complete source directories in `AIRS_TASK_ROOTS_JSON`. It validates all scientific assets before publication and uses shared checksum-verified staging. Partial laptop caches containing Arrow metadata without arrays are rejected. Mac AppleDouble sidecars are excluded from scientific identity.

Preparation currently preserves the old task packages. Native registration, runtime overlays, separate artifact-only verifier integration and actual execution acceptance are still pending. Do not launch the preserved mutable image tags directly. The pinned images and creator scorer controls must be requalified on the selected worker before live execution.

The preserved variant uses Harbor, CPU2/3GiB,3,600seconds,40turns and16,384 output tokens rather than the published MLGym/aira-dojo setup. Creator data split differences and target scaling remain intact. All future in-benchmark LLM roles use the campaign-authorized Nemotron model.
