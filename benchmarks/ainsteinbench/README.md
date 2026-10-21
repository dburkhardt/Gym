# AInsteinBench native pilot adapter

This adapter retains the creator's DockerAgent loop and four tools for two pinned PySCF tasks (PR2373 and PR2396). It is a native pilot wrapper, not a completed Gym resource server or a verified benchmark baseline. `source-sha256.json` pins the creator files consumed by the wrapper.

The explicit protocol variant uses Nemotron 3.5 Super VL, a16,384-token response budget,1CPU/4GiB container limits and disabled container networking. It corrects the checkout directory and removes evaluator patch files and future Git history before exposing the workspace. Tracked file bytes and symlink targets must remain unchanged; the resulting history must contain only the base snapshot. Credential access remains on the host.

The two campaign pilots exhausted the creator's30-iteration limit without a submitted patch. They are model-budget outcomes, not independently scored scientific failures. No test-pass rate is inferred. Electron-transfer tasks remain separately held for source/configuration problems; they are not implemented by this wrapper.

Request and response receipts are retained in `model-traces.jsonl`; do not re-run an existing output directory or silently retry uncertain dispatches. Before another run, inspect the creator loop's transport-error handling as well as SDK retry configuration. The SDK uses zero retries, which alone does not prove absence of retries inside the creator loop.

## Maintained native migration (runtime qualification pending)

`config.yaml` and `prepare.py` register all four released IDs. Two PySCF tasks use the immutable creator images; both Einstein Toolkit IDs remain visible with explicit source/runtime blockers and fail before model dispatch. Their latest preserved correct-control build timed out, and the earlier cached-test perturbation blindspot remains unresolved.

`harbor_native.py` runs the four-tool loop asynchronously in Harbor-owned containers. It reuses the exact checksum-pinned creator prompt, output extraction and PySCF parser function bodies, without importing the creator's Docker-owning controller. A scripted trace control compares requests and output against the original creator loop. The parser's count/set constructor is represented directly; no test-selection or score formula changes. Each request has persisted lineage, ordinal and protocol digest, goes through Gym capture with retries disabled, and retains unresolved dispatch evidence. Full shell continuation is unsupported and existing attempt directories fail closed.

Before the first request, the adapter verifies the base checkout, removes evaluator patches/scripts and future Git history, then checks original tracked bytes and symlink targets are unchanged. The separate verifier restores only the submitted patch/status, runs the original immutable image's `fix-run.sh`, and retains creator test IDs, counts, raw percentage and success flag. A zero-test apply/setup outcome remains qualified; infrastructure exceptions and signal termination are not scientific zeroes. Genuine 30-iteration exhaustion, early empty output and infrastructure failure are distinct outcomes; attempted tasks remain visible even when no submitted patch can be scored.

The explicit safety caps are 600 seconds per model request, 18,600 seconds for the agent phase and 3,500 seconds for the scientific command within a 3,600-second verifier phase. The creator's constructor timeout was unused in its loop; these are bounded campaign variants, not paper settings. The prior 16,384-token limit, four tools and 10,000-character tool-output truncation remain. Native container controls and fresh solver acceptance have not yet passed for this migration; no new inference is authorized by registration or unit tests.
