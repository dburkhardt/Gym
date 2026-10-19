# BixBench3 — experimental external integration

This adapter executes the native [Edison Scientific BixBench3](https://github.com/EdisonScientific/BixBench3) Inspect task and separately calls its deterministic host-side artifact grader. Upstream is pinned to `d0e0bbb41222335b1b8878f533a58466a3a782dc`. The upstream CC-BY-SA-4.0 source is installed separately, not vendored into Apache-2.0 Gym. Third-party datasets and licensed software retain their terms.

## Readiness and measurement

This integration is experimental and requires operator staging. Real-case execution has not yet been validated. A tiny public synthetic fixture has completed the native Inspect/Docker lifecycle with both requested Nemotron models, including tool execution, output collection, deterministic host grading, and container/network cleanup. This validates the integration mechanics only: it uses a lightweight synthetic sandbox rather than the licensed full runtime and is not a real benchmark-performance result. There is no claimed reproduction of published results.

The task is study reproduction through native Inspect tools and Docker. Preserve the upstream task prompt, 24-hour task limit, output contract, runtime image, network gateway, and read-only inputs. Both models must receive the same case and settings. No optional skills are added. Artifact grading is the reward; Inspect's output-collection score is only completeness. Process judging is not run by this adapter and must be reported separately if subsequently enabled. Logs are retained outside the repository.

The [dataset](https://huggingface.co/datasets/EdisonScientific/BixBench3) is gated. Obtain authorized access and the required Cell Ranger 6.0.2 and Cell Ranger ARC 2.0.2 runtime archives before following upstream runtime build instructions. Do not use the Edison-owned cloud project or bucket named in upstream operator notes. This adapter uses pre-staged local Docker execution and never provisions a cloud VM itself.

## Operator preparation

Install the pinned upstream package into an isolated Python environment. Stage each case in a distinct absolute `run_root` with `dataset/case.json`, its referenced prompt, `broker/policy.json`, and empty workspace/output directories. Download input manifests and verify their upstream checksums before staging. Keep grading specifications and gold files in a separate host-only directory outside every mounted path. Build the upstream sandbox and gateway images without removing licensed tools or weakening network policy.

An operator-owned settings JSON outside the repository specifies `task_id`, `model`, `upstream_checkout`, `upstream_python`, `run_root`, `data_dir`, `grading_specification`, `sandbox_image`, `gateway_image`, `broker_port`, `broker_token_file`, `solver_base_url`, `solver_key_file`, `budget_proxy_url`, `budget_proxy_key_file`, `model_context_length`, and `model_context_source`. Set the context limit from a verified model source; the adapter rejects missing provenance rather than using Inspect’s unknown-model fallback. `max_output_tokens` defaults to 16384; generation uses temperature zero and the Responses API. The paid network-adjudicator proxy must be reachable from the gateway container and enforce the campaign spending limit. Credentials are host files, not dataset fields. The solver key is never mounted in the agent sandbox.

Create ignored `benchmarks/bixbench3/local-preparation.json` with `settings_root` and a `cases` list. Each entry has a relative `settings_file` and `case_sha256` of its staged `dataset/case.json`. Use separate settings roots and run directories for the two models. Preparation writes one Gym row per case in the explicit configured order and fails on duplicate IDs or changed manifests.

```bash
python benchmarks/bixbench3/prepare.py
gym env validate bixbench3
python -m pytest responses_api_agents/bixbench3/tests -q
```

Override the Gym agent's `settings_root` and `runtime_python` with operator configuration, then use `gym eval run --benchmark bixbench3` after confirming Docker images, paid-judge budget enforcement, authorized data, storage, and task duration. The execution adapter uses the pinned Inspect Python API; its own process log and Inspect logs remain in the run directory. A failed harness subprocess raises an infrastructure error rather than returning a fabricated zero accuracy. A fully graded existing run can be read back; partial runs require inspection before resubmission.

This external engine has no Gym resources-server verifier fixture, so `gym env test bixbench3` and `gym env publish bixbench3` are not supported. Use adapter tests, the upstream fixture, and actual model rollouts. Static manifest validation alone is not execution validation.
