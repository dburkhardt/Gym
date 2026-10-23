# CompChemBench: native scientific-tool workflow

**One RDKit task completed through the maintained Gym–Harbor integration**, reviewed October 23, 2026. Nemotron 3.5 Super VL generated molecular-descriptor outputs that passed the creator's programmatic verifier. Restoring its saved workspace into a separate verifier reproduced the score without calling a model. This qualifies one concrete execution and scoring path; the larger CompChemBench sample retains unresolved checks.

## Source and scope

The integration uses [CompChemBench](https://github.com/Chenghao-Wu/compchem-bench) at revision [`772d3f3e1a93b875f49ddb692e6dfc3db1d3593d`](https://github.com/Chenghao-Wu/compchem-bench/tree/772d3f3e1a93b875f49ddb692e6dfc3db1d3593d). That pinned release contains 43 computational-chemistry tasks. The current creator README describes 46 tasks, and an older baseline describes 40; their denominators refer to different releases.

| Work performed | Scope | Outcome |
|---|---|---|
| Source preparation | All 43 pinned tasks | Creator files and recorded integration overlays passed checksum validation |
| Retained Nemotron execution | Original 20-task sample | 18 usable native scores: 17 passes and one failure; two attempts remain masked |
| Retained scoring comparison | The 18 usable saved results | All 18 native regrades match; historical workspace access and scientific checks remain separate |
| Fresh deterministic controls | One RDKit task | No-operation agent: 0; creator oracle: 1; restored oracle-workspace replay: 1 |
| Fresh Nemotron pilot | One original RDKit task | Native score: 1; restored model-workspace replay: 1; no masked outcome |

The fresh pilot selects `compchem::rdkit-descriptors-batch`, the first task in the original frozen five-task prefix. It is a compact CPU workflow with real scientific software and an executable verifier. It was chosen to exercise the integration before expanding execution. The [public-safe manifest](compchem-sample.json) records its identity, settings, source revisions, scores, artifact hashes, and evidence hashes.

[Gym configuration](../../../benchmarks/compchem/config.yaml) · [Preparation and source validation](../../../benchmarks/compchem/prepare.py) · [Integration instructions](../../../benchmarks/compchem/README.md) · [Creator task and instructions](https://github.com/Chenghao-Wu/compchem-bench/tree/772d3f3e1a93b875f49ddb692e6dfc3db1d3593d/tasks/rdkit-descriptors-batch)

## What the model did

The task supplies 12 molecules as SMILES strings and asks for five RDKit descriptors: molecular weight, calculated logP, topological polar surface area, hydrogen-bond donors, and hydrogen-bond acceptors. The solver must write a CSV containing the canonical molecular representation and descriptors, plus a JSON file reporting the input and output counts.

The native trajectory contains four model responses and seven shell commands. The model inspected the input file, wrote a Python program using RDKit's descriptor functions, ran it, and displayed the two generated files. The outputs contain 12 data rows and matching counts. For example, its ethanol row reports molecular weight 46.069, calculated logP −0.0014, polar surface area 20.23, one hydrogen-bond donor, and one acceptor. The program is preserved in the recorded shell-command trajectory; both output files and the input asset are preserved in the captured workspace.

The [creator verifier](https://github.com/Chenghao-Wu/compchem-bench/blob/772d3f3e1a93b875f49ddb692e6dfc3db1d3593d/tasks/rdkit-descriptors-batch/tests/verify.py) first checks the input asset's hash and output schemas. It independently recomputes every molecule's descriptors using the image's RDKit installation, compares rows in input order, and checks reference counts and the molecular-weight sum. Floating-point tolerance is 0.0001; canonical SMILES and integer descriptors must match exactly. The original and replay verifier logs both report `PASS: rdkit-descriptors-batch`.

## Execution, scoring, and resources

Generation followed maintained Gym components: `SimpleModelServer` → `HarborAgent` → `Terminus2NemoGym` → task container → creator verifier. An isolated lifecycle controller loaded the maintained benchmark configuration and selected the single task; it did not implement another solver or scorer. Native Harbor source-job regrading then restored the captured workspace and reran the verifier with zero additional model calls.

| Check | Observed result |
|---|---|
| Model identity | `nvidia/nvidia/nemotron-3.5-super-vl` for every model role; deterministic verifier |
| Native result | Reward 1.0; unmasked; no native exception |
| Saved-workspace replay | Reward 1.0; identical descriptor CSV, count JSON, and input asset hashes |
| Model exchanges | Four logical calls, four physical provider attempts, all HTTP 200 and normal stop termination |
| Output allowance | 16,384 tokens per call; no response exhausted the allowance |
| Usage | 11,086 prompt tokens and 3,752 completion tokens across the four calls |
| Timing | 80.47 seconds for the supervised attempt, including 72.83 seconds for startup, execution, and replay |
| Original task budgets | Solver: 900 seconds; verifier: 60 seconds; main container: 1 CPU and 2 GiB |
| Sidecar budget | 0.25 CPU and 128 MiB |
| Direct resource observation | Exact limits confirmed for solver containers and the separate replay-verifier containers |
| Cleanup | Four directly observed owned containers removed normally; none remained; owned services and local Ray cluster stopped |

Requests used temperature 0, interleaved reasoning, the JSON command parser, and disabled summarization. Correlated capture and the provider dispatch guard bound calls to the declared rollout. Both transport and provider retry layers allowed one attempt, and no automatic retries occurred. The replay result retains the original agent's token metadata; those copied fields do not represent new inference usage.

The qualified image uses Python 3.11.9, RDKit 2024.09.5, and NumPy 1.26.4. Its immutable image identity and file hashes appear in the manifest. The Gym source was commit `ec3c3ab6c1e08cf3825e17efc176b0caa19ec2e6`, with Harbor 0.24.0 from commit `96ff9c6db90cc015f0fc0987d896670c9c5f1fad`.

Harbor's existing [checksum-gated sidecar runtime patch](../../../responses_api_agents/harbor_agent/runtime_patches/sidecar_resources.py) was applied to an isolated copy of the matching Harbor package. A solver-only compose overlay does not bound the separate verifier's sidecar. The canonical patch covers both. No shared runtime was modified. The original run's brief separate verifier was not directly captured by the resource observer; the separate saved-workspace verifier was observed at the declared limits.

The first startup attempt failed because Ray requires integer node CPU capacity. It ended before any native task request or provider dispatch, and its zero-dispatch failure receipt was preserved. A newly admitted attempt used a two-CPU Ray node with a 1.25-CPU per-task reservation. Scientific container and timeout budgets stayed unchanged. A process-group watchdog imposed a 1,200-second outer deadline with graceful drain; the successful attempt completed without deadline signals.

## Remaining limits

This one-task result demonstrates a model-to-tool-to-verifier workflow and reproducible scoring of its saved outputs. It does not establish accuracy across the 43 tasks, reproduce a creator leaderboard comparison, or validate the scientific appropriateness of every task and reference. The endpoint supplied a model identifier but no model-weight hash.

The historical twenty-task measurement still includes two masked attempts and unresolved provenance and scientific checks, including LAMMPS scheduling, CP2K basis/electrostatics, and quantum-chemistry position conventions. Those limitations remain attached to that measurement. The fresh RDKit result does not repair or replace them.

Exact execution commands, raw model captures, the generated-program trajectory, output workspaces, native results, replay receipts, and cleanup evidence are retained privately. The manifest binds their hashes without publishing host paths or credentials. The pinned creator repository is distributed under Apache 2.0; task assets and any future external dependencies should continue to be checked against their own terms.
