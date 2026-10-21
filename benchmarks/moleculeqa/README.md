# moleculeqa pilot

Experimental: requires real smoke and independent review. Set `CAMPAIGN_SAMPLE_JSONL` to the audited sample20.jsonl in outputs/broad-eval-campaign/qa-sources/moleculeqa. References are confined to verifier_metadata. Use Nemotron 3.5 Super VL for policy_model; there is no model grader for the metrics in this adapter.

```sh
CAMPAIGN_SAMPLE_JSONL=/absolute/path/sample20.jsonl python benchmarks/moleculeqa/prepare.py
gym eval run --benchmark moleculeqa --model-type openai_model --split benchmark --limit 5 --num-repeats 1 --temperature 0.3 --max-output-tokens 16384 --output /absolute/private/results/moleculeqa-smoke.jsonl
```

The generation cap is a pilot setting, not a claimed creator setting. After reviewing smoke, run all20 and preserve failures. Inspect the campaign qa-integration-report.md for source pins, known scorer defects and incomplete protocol coverage.

## Protocol boundary

This adapter is the campaign's **zero-shot, single-instance, strict-letter variant**, not the published general-model baseline. The [creator paper](https://aclanthology.org/2024.findings-emnlp.216.pdf), appendix A.7/Table 14, describes aspect-specific ten-shot prompts and labeled instance answers. Table 14 illustrates only two Source demonstrations followed by an ellipsis. The [creator repository](https://github.com/IDEA-XL/MoleculeQA) and [released dataset](https://huggingface.co/datasets/IDEA-AI4S/MoleculeQA) inspected for this integration do not supply the complete ordered demonstration sets or an evaluation/extraction script.

A canonical configuration must be separate and bind those source assets, per-aspect system prompts, instance grouping, generation settings and extraction/error policy. Do not construct substitute demonstrations from arbitrary training examples or relabel these frozen prompts/results as canonical reproduction. Accuracy of the strict-letter variant does not establish parity with the published baseline.
