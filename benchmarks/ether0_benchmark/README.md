# ether0_benchmark pilot

Experimental: requires real smoke and independent review. Set `CAMPAIGN_SAMPLE_JSONL` to the audited sample20.jsonl in outputs/broad-eval-campaign/qa-sources/ether0_benchmark. References are confined to verifier_metadata. Use Nemotron 3.5 Super VL for policy_model; there is no model grader for the metrics in this adapter.

```sh
CAMPAIGN_SAMPLE_JSONL=/absolute/path/sample20.jsonl python benchmarks/ether0_benchmark/prepare.py
gym eval run --benchmark ether0_benchmark --model-type openai_model --split benchmark --limit 5 --num-repeats 1 --temperature 0.3 --max-output-tokens 16384 --output /absolute/private/results/ether0_benchmark-smoke.jsonl
```

The generation cap is a pilot setting, not a claimed creator setting. After reviewing smoke, run all20 and preserve failures. Inspect the campaign qa-integration-report.md for source pins, known scorer defects and incomplete protocol coverage.

Native asset staging: after installing the resource environment, run
`python -m resources_servers.ether0_benchmark.stage_assets` in that environment
from the repository root. The creator wheel omits two BloomFilter files;
missing files crash molbloom in native code. Staging downloads the exact pinned
creator files and checks SHA256. The scorer checks both before invoking native
code, and each verification runs in an isolated subprocess. Native failure or
missing assets is a masked infrastructure failure, never a model zero.
