# evidencebench pilot

Experimental: requires real smoke and independent review. Set `CAMPAIGN_SAMPLE_JSONL` to the audited sample20.jsonl in outputs/broad-eval-campaign/qa-sources/evidencebench. References are confined to verifier_metadata. Use Nemotron 3.5 Super VL for policy_model; there is no model grader for the metrics in this adapter.

```sh
CAMPAIGN_SAMPLE_JSONL=/absolute/path/sample20.jsonl python benchmarks/evidencebench/prepare.py
gym eval run --benchmark evidencebench --model-type openai_model --split benchmark --limit 5 --num-repeats 1 --temperature 0.3 --max-output-tokens 16384 --output /absolute/private/results/evidencebench-smoke.jsonl
```

The generation cap is a pilot setting, not a claimed creator setting. After reviewing smoke, run all20 and preserve failures. Inspect the campaign qa-integration-report.md for source pins, known scorer defects and incomplete protocol coverage.

## Explicit creator saved-answer adaptation

`python -m resources_servers.evidencebench.creator_replay --input saved-native-results.jsonl --output new-projection.json --creator-root /path/to/pinned/EvidenceBench --parser parse_gpt_output` executes the unchanged pinned creator evaluation loop in input order, with one private Python RNG seeded42 and100,000 in-place shuffles for each oversized answer. It verifies revision and actual parser/evaluator bytes, records source/input/RNG/runtime identities, and refuses an existing output. Original native rewards, masks and failure kinds are retained alongside projected scores; projection is not scientific acceptance.

This is explicitly `creator-gpt-parser-saved-answer-adaptation-v1`, not a canonical Nemotron protocol. The ordinary creator parser selects GPT/Claude/Gemini by model name, whereas the step-by-step script uses a different fallback. Neither defines Nemotron. Selecting the GPT parser here matches the existing campaign format by explicit choice. Creator regeneration of over-limit answers is absent; replay preserves the answers that actually exist. Ordered subset RNG is not equivalent to RNG consumption across the full creator dataset. The default native verifier remains the separately named exact-expectation variant.

Malformed integer lists and unknown indices in oversize output retain creator exceptions rather than being converted to scientific zeros. Fixed positive limits are supported; optimal-k mode is not silently substituted. The existing saved50 replay has identical scores under this explicit adaptation because no answer consumes oversize RNG; that does not establish equivalence for arbitrary future outputs.

The pinned GPT parser never reads its `prompt` argument; replay therefore passes an empty prompt, with an actual-source invariance control. Missing responses remain unscored/null, not empty-answer zeros. Null original rewards yield null deltas. Per-record identities, content hashes, masks and failure reasons remain in the projection, including records excluded from the scorer input stream; the reported mean is only raw creator replay, never an accepted-quality mean.
