# csmed_ft pilot

Experimental: requires real smoke and independent review. Set `CAMPAIGN_SAMPLE_JSONL` to the audited sample20.jsonl in outputs/broad-eval-campaign/qa-sources/csmed_ft. References are confined to verifier_metadata. Use Nemotron 3.5 Super VL for policy_model; there is no model grader for the metrics in this adapter.

```sh
CAMPAIGN_SAMPLE_JSONL=/absolute/path/sample20.jsonl python benchmarks/csmed_ft/prepare.py
gym eval run --benchmark csmed_ft --model-type openai_model --split benchmark --limit 5 --num-repeats 1 --temperature 0.3 --max-output-tokens 16384 --output /absolute/private/results/csmed_ft-smoke.jsonl
```

The generation cap is a pilot setting, not a claimed creator setting. After reviewing smoke, run all20 and preserve failures. Inspect the campaign qa-integration-report.md for source pins, known scorer defects and incomplete protocol coverage.

## Separately named creator parser and merge contract

`creator_protocol.py` implements `csmed-ft-creator-chunk-labels-81cd354`, independently of the existing whole-paper strict-label grader. Source parity controls extract only the two pure functions from pinned creator commit `81cd35498bdea8981e5310842c32398625a10e95` and verify both file SHA256 hashes; they never import its API-key-reading module. The parser lowercases without stripping whitespace, accepts included/include/yes and excluded/exclude/no, and marks everything else -1. The paper merge returns excluded if any chunk is excluded, otherwise included—even when every chunk is invalid. It retains the first reference label and exposes conflicting references and invalid chunk counts. Those creator quirks are preserved, not endorsed as measurement quality. This module does not change existing scores or enable a canonical generation run.

Canonical generation still needs the original review title, review criteria, paper title, abstract, main text, review/paper IDs and integer dataset row identity. The creator constructs `Systematic review:` and `Candidate paper:` text, uses `tiktoken.encoding_for_model(model_name)` and LangChain `TokenTextSplitter` with zero overlap, truncates the review to its first half-context chunk, and splits the paper at `max(context_size // 2, context_size - original_review_token_count)`. Crucially, that token count precedes review truncation. Each paper chunk receives the original eligibility question and `Answer: ` suffix; predictions must retain paper and chunk identities before merging. A whole-paper response cannot reconstruct these unobserved chunk responses.

The selected upstream example is GPT-3.5-Turbo-16k with a 16,000-token window; the script also defines 4,000/8,000-token alternatives. A Nemotron run needs an explicitly declared tokenizer/context-window adaptation, pinned splitter/tokenizer versions, and frozen chunk/prompt hashes. It cannot inherit a GPT tokenizer by accident or claim creator-model reproduction. Upstream leaves temperature/output cap at API defaults; the campaign's 0.3/16,384 settings are a separate variant. Upstream unbounded exception retry must not replace Gym's uncertainty guard. Canonical reporting additionally requires the creator's accuracy, macro precision/recall, binary/macro/micro F1 and macro F3 over merged papers, with explicit invalid-chunk and conflicting-reference dispositions. None of these generation or reporting gates is closed by parser parity alone.
