# fdarxbench pilot

Experimental: requires real smoke and independent review. Set `CAMPAIGN_SAMPLE_JSONL` to the audited sample20.jsonl in outputs/broad-eval-campaign/qa-sources/fdarxbench. References are confined to verifier_metadata. Use Nemotron 3.5 Super VL for policy_model; the grader also uses exact Nemotron 3.5 Super VL with the pinned creator FDA rubric (temperature 0, 4096 tokens). The rubric requires `LETTER: short reason`; A is correct, B incorrect, and C not attempted. Malformed or incomplete judge outputs remain explicit failures.

The resource binds a separate `judge_model` component; overriding the solver does not change the judge. Save the following component as `fdarx-runtime.yaml` and pass `--config /absolute/path/fdarx-runtime.yaml`. Supply `FDARX_JUDGE_PROMPT` as the absolute path to the pinned `creator-judge-prompt.txt` from the audited FDARx source bundle. Add this component in your runtime YAML (credentials stay in environment variables):

```yaml
judge_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_base_url: ${oc.env:INFERENCE_BASE_URL}
      openai_api_key: ${oc.env:INFERENCE_API_KEY}
      openai_model: nvidia/nvidia/nemotron-3.5-super-vl
```

The runtime must also configure the solver and guarded capture. Each judge call uses the persisted native `verification_id` and a fixed rubric-call identity, binding the exact rubric bytes, verifier/parser source and judge generation settings. Missing identity fails before dispatch; changing rubric bytes changes the dispatch digest. This does not authorize automatic retry or certify the scientific rubric. Native request metadata is retained in the result.

```sh
export FDARX_JUDGE_PROMPT=/absolute/path/creator-judge-prompt.txt
CAMPAIGN_SAMPLE_JSONL=/absolute/path/sample20.jsonl python benchmarks/fdarxbench/prepare.py
gym eval run --benchmark fdarxbench --config /absolute/path/fdarx-runtime.yaml --model-type openai_model --split benchmark --limit 5 --num-repeats 1 --temperature 0 --max-output-tokens 16384 --output /absolute/private/results/fdarxbench-smoke.jsonl
```

The generation cap is a pilot setting, not a claimed creator setting. After reviewing smoke, run all20 and preserve failures. Inspect the campaign qa-integration-report.md for source pins, known scorer defects and incomplete protocol coverage.
