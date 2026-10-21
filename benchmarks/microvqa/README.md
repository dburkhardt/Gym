# microvqa pilot

Experimental: requires real smoke and independent review. Set `CAMPAIGN_SAMPLE_JSONL` to the audited sample20.jsonl in outputs/broad-eval-campaign/qa-sources/microvqa. References are confined to verifier_metadata. Use Nemotron 3.5 Super VL for policy_model; there is no model grader for the metrics in this adapter.

```sh
CAMPAIGN_SAMPLE_JSONL=/absolute/path/sample20.jsonl python benchmarks/microvqa/prepare.py
gym eval run --benchmark microvqa --model-type openai_model --split benchmark --limit 5 --num-repeats 1 --temperature 0.3 --max-output-tokens 16384 --output /absolute/private/results/microvqa-smoke.jsonl
```

The generation cap is a pilot setting, not a claimed creator setting. After reviewing smoke, run all20 and preserve failures. Inspect the campaign qa-integration-report.md for source pins, known scorer defects and incomplete protocol coverage.

Native verification uses the shared explicit-completion diagnostic helper. Provider-reported output-budget exhaustion is retained without changing creator reward or mask; provider errors and unspecified incomplete responses are distinct, and earlier failure diagnoses take precedence. Native task/rollout identity and image-bearing input messages survive reverify. The preserved45 responses are all completed; incomplete-status controls are explicitly synthetic wiring checks, not new observed model failures. Image preparation, held-task admission and the creator answer parser are unchanged.
