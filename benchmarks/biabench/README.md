# BIABench native integration — in progress

Preparation validates 80 preserved creator files and 34 assets for the existing wound-healing task. It stages 12 public inputs and the exact frozen CPU prompt under `solver/`, references under `references/`, and creator scorer source under `verifier_source/`, then publishes task identity through Gym's checksum-verified staging. Unexpected files or symlinks in solver staging fail before dataset publication. The local archive name provides only a short source revision; the manifest pins exact bytes rather than claiming a verified full Git commit. This is one materialized task, not a replacement for other frozen campaign selections.

A maintained Harbor config and separate process-verifier hook now exist, with local native preparation/plan and deterministic creator controls. Actual container execution of this new path and runtime/image qualification remain required; this is not live acceptance. Preserve full-resolution inputs and the documented basic-tier CPU prompt variant. References, rubric and checklist must never be mounted into the solver.

The historical outcome score and incomplete process checklist remain distinct. Preserve the original 2048-token judge protocol and separately recorded 16384-token variant, unresolved provider request, and skipped floating-TIFF evidence. Do not silently repair the scorer, omit process scoring, or resend an uncertain request. Preparation is not scientific acceptance.

`regrade_outcome.py` recomputes the creator outcome metric vector from saved predictions in a separate verifier process. It refuses existing output files, validates pinned creator/task bytes and actual imported creator modules, hashes prediction files, and rejects prediction mutation. The local saved-output replay reproduced outcome 0.79 and every historical metric exactly, with 30 prediction files and 29 creator imports verified. The empty-output control returned zero. Neither result is a process-checklist score or proof of a Linux managed run.

Supply `BIABENCH_SOURCE`, `BIABENCH_TASK`, and `BIABENCH_DATA` to `prepare.py`. The native scorer requires the creator scientific dependencies, including imageio, tifffile, scikit-image and openpyxl because its evaluator package imports multiple task modules. Use an isolated verifier environment; do not install into a running solver environment. `regrade_outcome.py --help` describes the four explicit paths for saved-output replay. Only `solver/` may be mounted into the solver; mounting the preparation root would expose references.

`managed_regrade.regrade_managed` wraps outcome replay in Gym's owned Docker sandbox lifecycle. It requires a full image digest, disables networking, mounts source/task/predictions read-only, retains logs and execution status on verifier errors, and refuses reuse of an attempt directory. Its local lifecycle controls and real Linux saved-output replay pass. Image `sha256:9229b10d07ace7834eadd86997816f8f8943b756f7cc9f91faa444dce8763740` reproduced the full historical metric object (outcome 0.79) and all 30 prediction hashes with zero model calls and confirmed owned-container cleanup. Two earlier attempts retained missing-source/input failures as verifier errors; they produced no scores. This entrypoint does not perform process-checklist judging or qualify the solver.


## Maintained execution path

Set `BIABENCH_SOURCE`, `BIABENCH_TASK`, and `BIABENCH_DATA` to the preserved inputs, then use `gym eval plan --benchmark biabench --out plan.json`. `harbor_prepare.py` stages the exact CPU prompt and one native task. The solver image is the historical immutable v2 image; its 4 CPU/8 GiB/7200-second, 40-turn, temperature-zero protocol is preserved. Solver and process judge both route through Gym; process judging retains the creator's 2048-token budget, chunk size two and one sample. No ambiguous provider retry is permitted.

`BIAVerifier` requires a complete `/workspace` artifact capture and a full solver trajectory. Harbor's separate environment restores that capture before the hook reads it. The hook constructs a creator submission, then `managed_process.score_managed` runs the exact pinned creator evaluator in an owned network-disabled verifier container with a task-scoped UDS model bridge. Source/task/submission mounts are read-only; no verifier references are mounted into the solver. The managed scorer requests another 2 CPU/4 GiB container, so total simultaneous Harbor/verifier memory must be qualified before execution; local controls do not establish that capacity.

The hook returns outcome and, when numeric, process as separate metrics. Native unknown process results remain null in retained reports; aggregate calculation is disabled. Failed transport, incomplete checklist, truncated judge output or partial capture refuses an accepted score. A new process-scoring attempt needs a new output directory and lineage; arbitrary interrupted-session recovery is not implemented.

Local deterministic creator qualification processed all 49 checklist items in 25 mocked provider calls, preserving outcome 0.79 while all-unknown judgments produced process null. A second control injected a disconnect: only one physical request occurred and no process score was accepted. These are correctness controls, not Nemotron performance. Four creator-selected TIFF reference images remain unreadable by the unchanged encoder (100 skipped encodings across 25 chunks); their omission is retained explicitly and limits visual evidence. No conversion or scorer repair was applied. The historical unresolved provider attempt remains untouched.

Resource admission reserves 22 GiB through native `harbor_ray_task_memory_bytes`:
8 GiB solver, 8 GiB separate verifier and 4 GiB managed process scorer can overlap.
The additional 2 GiB is an unqualified host-services/headroom assumption, not a
hard host memory bound. Ray cannot place the trial on a node without that logical
capacity; it may queue rather than immediately fail. Actual Docker and sidecar
bounds still require runtime qualification.

The runtime must explicitly declare `component_capabilities.policy_model.vision: true`.
This is required for the process judge's reference images; merely selecting a
model name does not establish compatibility. Currently all model roles share the
same authorized vision model. Independent future judge configurations must retain
that requirement.

The measurement-review hook binds raw metric values to the retained pinned-source
result. Outcome validity here means faithful execution of that scorer, not universal
scientific correctness. The four unreadable creator-selected TIFF references make
process coverage unresolved. Raw scores remain unchanged; the existing shared hook
conservatively masks the whole scalar row when process evidence is incomplete.
Separately reviewed outcome reporting must use its explicit per-metric disposition.

A separately qualified reconstruction may set `BIABENCH_SOLVER_IMAGE` to a full
lowercase `sha256:` image ID; the historical image remains the default. The
operator must qualify exact public inputs, dependency versions, reference
isolation, and runtime behavior before using an override. Preparation records
the image and published file hashes. Changed images, sources, or prepared bytes
refuse reuse: use a fresh versioned source/preparation directory and retain the
old artifacts. An image ID alone is not scientific equivalence evidence.
