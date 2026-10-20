# biomaze pilot

Experimental: requires real smoke and independent review. Set `CAMPAIGN_SAMPLE_JSONL` to the audited sample20.jsonl in outputs/broad-eval-campaign/qa-sources/biomaze. References are confined to verifier_metadata. Use Nemotron 3.5 Super VL for policy_model; the TrueFalse path is deterministic. Openended uses the pinned creator conclusion rubric with the same exact Nemotron model as judge (temperature 0, 4096 tokens); configure judge_model_server and judge_prompt_file. These are separate protocols.

```sh
CAMPAIGN_SAMPLE_JSONL=/absolute/path/sample20.jsonl python benchmarks/biomaze/prepare.py
gym eval run --benchmark biomaze --model-type openai_model --split benchmark --limit 5 --num-repeats 1 --temperature 0.3 --max-output-tokens 16384 --output /absolute/private/results/biomaze-smoke.jsonl
```

The generation cap is a pilot setting, not a claimed creator setting. After reviewing smoke, run all20 and preserve failures. Inspect the campaign qa-integration-report.md for source pins, known scorer defects and incomplete protocol coverage.

## Open-ended judge configuration

Use `--config benchmarks/biomaze/configs/openended.yaml` with a frozen open-ended sample in `CAMPAIGN_SAMPLE_JSONL`, its checksum in `CAMPAIGN_SAMPLE_SHA256`, and the pinned creator prompt path in `BIOMAZE_JUDGE_PROMPT`. This config gives the judge its own model-server instance (`biomaze_judge`) while retaining the campaign-authorized Nemotron model. Solver configuration remains separate. Judge generation settings remain the adapter's temperature 0 and 4096 output tokens. The prompt file must be preserved and hashed with the experiment artifacts. This configuration alone does not establish live judge acceptance.
