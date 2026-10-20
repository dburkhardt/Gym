# fdarxbench pilot

Experimental: requires real smoke and independent review. Set `CAMPAIGN_SAMPLE_JSONL` to the audited sample20.jsonl in outputs/broad-eval-campaign/qa-sources/fdarxbench. References are confined to verifier_metadata. Use Nemotron 3.5 Super VL for policy_model; the grader also uses exact Nemotron 3.5 Super VL with the pinned creator FDA rubric (temperature 0, 4096 tokens). The rubric requires `LETTER: short reason`; A is correct, B incorrect, and C not attempted. Malformed or incomplete judge outputs remain explicit failures.

```sh
CAMPAIGN_SAMPLE_JSONL=/absolute/path/sample20.jsonl python benchmarks/fdarxbench/prepare.py
gym eval run --benchmark fdarxbench --model-type openai_model --split benchmark --limit 5 --num-repeats 1 --temperature 0 --max-output-tokens 16384 --output /absolute/private/results/fdarxbench-smoke.jsonl
```

The generation cap is a pilot setting, not a claimed creator setting. After reviewing smoke, run all20 and preserve failures. Inspect the campaign qa-integration-report.md for source pins, known scorer defects and incomplete protocol coverage.
