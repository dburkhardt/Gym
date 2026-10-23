# AInsteinBench native pilot adapter

This adapter retains the creator's DockerAgent loop and four tools for two pinned PySCF tasks (PR2373 and PR2396). It is a native pilot wrapper, not a completed Gym resource server or a verified benchmark baseline. `source-sha256.json` pins the creator files consumed by the wrapper.

The explicit protocol variant uses Nemotron 3.5 Super VL, a16,384-token response budget,1CPU/4GiB container limits and disabled container networking. It corrects the checkout directory and removes evaluator patch files and future Git history before exposing the workspace. Tracked file bytes and symlink targets must remain unchanged; the resulting history must contain only the base snapshot. Credential access remains on the host.

The two campaign pilots exhausted the creator's30-iteration limit without a submitted patch. They are model-budget outcomes, not independently scored scientific failures. No test-pass rate is inferred. Electron-transfer tasks remain separately held for source/configuration problems; they are not implemented by this wrapper.

Request and response receipts are retained in `model-traces.jsonl`; do not re-run an existing output directory or silently retry uncertain dispatches. Before another run, inspect the creator loop's transport-error handling as well as SDK retry configuration. The SDK uses zero retries, which alone does not prove absence of retries inside the creator loop.

## Maintained native migration

`config.yaml` and `prepare.py` register all four released IDs. Two PySCF tasks use the immutable creator images; both Einstein Toolkit IDs remain visible with explicit source/runtime blockers and fail before model dispatch. Their latest preserved correct-control build timed out, and the earlier cached-test perturbation blindspot remains unresolved.

`harbor_native.py` runs the four-tool loop asynchronously in Harbor-owned containers. It reuses the exact checksum-pinned creator prompt, output extraction and PySCF parser function bodies, without importing the creator's Docker-owning controller. A scripted trace control compares requests and output against the original creator loop. The parser's count/set constructor is represented directly; no test-selection or score formula changes. Each request has persisted lineage, ordinal and protocol digest, goes through Gym capture with retries disabled, and retains unresolved dispatch evidence. Full shell continuation is unsupported and existing attempt directories fail closed.

Before the first request, the adapter verifies the base checkout, removes evaluator patches/scripts and future Git history, then checks original tracked bytes and symlink targets are unchanged. The separate verifier restores only the submitted patch/status, runs the original immutable image's `fix-run.sh`, and retains creator test IDs, counts, raw percentage and success flag. A zero-test apply/setup outcome remains qualified; infrastructure exceptions and signal termination are not scientific zeroes. Genuine 30-iteration exhaustion, early empty output and infrastructure failure are distinct outcomes; attempted tasks remain visible even when no submitted patch can be scored.

The explicit safety caps are 600 seconds per model request, 18,600 seconds for the agent phase and 3,500 seconds for the scientific command within a 3,600-second verifier phase. The creator's constructor timeout was unused in its loop; these are bounded campaign variants, not paper settings. The prior 16,384-token limit, four tools and 10,000-character tool-output truncation remain. Native container controls are documented below; fresh solver acceptance remains pending. No new inference is authorized by registration or unit tests.

Native model-free controls now pass on both pinned PySCF images through Harbor-owned solver isolation, separate verification and source-job replay. Gold passes4/4 and9/9; a behavior-preserving no-fix patch passes2/4 and8/9. **The raw percentage is not a solved-task rate**: most tests can pass before the requested repair. Empty patches produce the creator's explicit `git apply` rejection, zero tests and raw0. All eight native scores and individual test IDs match preserved creator controls; originals remain unchanged and owned containers were removed. These used2GiB control containers and saved patches, not live model execution.

The maintained qualification layer always retains an available raw creator metric. Recognized patch rejection is a model outcome. Signal/missing-executable errors, unclassified zero-test runs, or incomplete test collection retain raw diagnostics and raise an unscored failure; they are not presented as scientific zeroes. Expected control counts4/9 and observed counts are reported separately, without replacing the creator metric. The qualifier was replayed against all eight saved controls and tested against injected import/collection/partial-run failures. Full native Gym model-server/agent acceptance and a guarded live run remain pending.

Native Harbor routing uses `ainsteinbench::<task_id>` only for `instance_id`.
The creator `task_id`, `uuid`, task directory and instruction remain unchanged.
Preparation refuses an existing legacy unnamespaced manifest: choose a fresh,
versioned `AINSTEIN_DATA` directory and point the native dataset path there.
Do not reuse a cached legacy manifest or overwrite historical attempt inputs.
A native plan alone does not exercise Harbor's `/run` routing contract; the
preparation regression validates the request, parser and dataset job selection.

The parent Harbor server passes its authoritative guard decision as
`gym_provider_dispatch_guard` to the custom agent. The worker refuses a missing,
false or non-boolean decision, and retains the exact-model and rollout-scoped
endpoint checks. It does not parse Gym/Hydra configuration in a Ray worker;
Ray's own command-line arguments must remain untouched.

For ET image preparation, `et_cache.audit_object` provides a fail-closed audit of
individual ELF relocatable objects and their original compiler `.d` records.
Supply independently verified original and candidate image filesystem manifests,
actual object/dependency bytes, and explicit forbidden source paths. It resolves
symlinks and requires each expanded compiler dependency to retain its exact source
or header hash in the candidate. Target aliases, identical target copies, missing
or changed dependencies, unsupported make syntax, archives and executables fail.
The pinned Fortran compiler's empty additional rule for the same object is accepted;
additional object-to-object rules remain unsupported rather than being ignored.

A cache receipt neither publishes an image nor establishes scientific readiness.
Bind both full manifest hashes and immutable image identity in preparation evidence;
retain source/metadata provenance and preserve cache timestamps during any separately
reviewed restoration. Do not infer static-library safety from member names, warm a
solver image by compiling the target answer, or replace original reference/negative
controls with cache checks. The ET adapter records returned command output before
rejecting abnormal termination; this preserves available diagnostics without
recovering bytes discarded upstream or changing the original 300-second deadline.

### GRHydro dependency-bound cache runtime

GRHydro uses its own isolated image `e45df9c19c046fa88e9839d4b5854a5706afd1e02d310c14b97be63b4b0be611`.
Its 120 restored individual objects and dependency files passed the maintained dependency audit;
no target object, aggregate library, executable, or target source was restored. The complete
isolated baseline filesystem is unchanged apart from those audited additions and their directories.
Original creator reference controls passed 10/10 tests; omitting the source-term call passed 6/10
(score 80, including compilation and execution credit). This demonstrates sensitivity to that
specific defect, not comprehensive scorer sensitivity or model performance. Both controls retained
2 CPUs / 4 GiB, network isolation, the original 300-second command timeout and creator 600-second
test command. Native preparation preserves the original solver/verifier ceilings and requires a
fresh versioned destination; older preparation receipts are not overwritten.
