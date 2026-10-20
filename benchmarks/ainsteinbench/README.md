# AInsteinBench native pilot adapter

This adapter retains the creator's DockerAgent loop and four tools for two pinned PySCF tasks (PR2373 and PR2396). It is a native pilot wrapper, not a completed Gym resource server or a verified benchmark baseline. `source-sha256.json` pins the creator files consumed by the wrapper.

The explicit protocol variant uses Nemotron 3.5 Super VL, a16,384-token response budget,1CPU/4GiB container limits and disabled container networking. It corrects the checkout directory and removes evaluator patch files and future Git history before exposing the workspace. Tracked file bytes and symlink targets must remain unchanged; the resulting history must contain only the base snapshot. Credential access remains on the host.

The two campaign pilots exhausted the creator's30-iteration limit without a submitted patch. They are model-budget outcomes, not independently scored scientific failures. No test-pass rate is inferred. Electron-transfer tasks remain separately held for source/configuration problems; they are not implemented by this wrapper.

Request and response receipts are retained in `model-traces.jsonl`; do not re-run an existing output directory or silently retry uncertain dispatches. Before another run, inspect the creator loop's transport-error handling as well as SDK retry configuration. The SDK uses zero retries, which alone does not prove absence of retries inside the creator loop.
