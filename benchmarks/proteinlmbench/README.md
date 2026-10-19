# ProteinLMBench

Experimental native Gym integration of [ProteinLMBench](https://huggingface.co/datasets/tsynbio/ProteinLMBench), [paper](https://arxiv.org/abs/2406.05540). This is protein-knowledge multiple choice, not an agentic analysis benchmark.

## Measurement contract

The public `ProteinLMBench.json` is pinned to `f1397963c7f727a4a2f00cdd691e6e219c36e992`, SHA256 `f2dce5c1b54e8ce73897a4af48ae8ba05507076d511b9d02d57cddf31c42ed96`. Its dataset card declares Apache-2.0. Only 944 evaluation questions are downloaded; the much larger SFT corpus is not used. All rows and option order are retained, including 2–10 options per question. Source order provides stable IDs. No filtering, shuffling, synthetic additions, or training-data conversion occurs.

The prompt and first-number scoring follow [the author's evaluator](https://github.com/tsynbio/ProteinLMDataset/blob/d8586e22ff85f6805edea0bbc23002aaccf525c4/benchmark/benchmark_your_model.py). The unusual upstream instruction to output option 7 when unsure is deliberately retained, even when only six options exist. Inputs include only the source question and options, never the explanation or gold answer. No tools, retrieval, or skills are enabled. Gold labels stay in verifier metadata and are not sent to the model.

Binary reward is equality between the first numeral in the final answer and the reference option. Unparseable/missing answers score zero; accuracy is mean reward. This parser can credit an incidental numeral; `strict_answer` and `strict_correct` expose a secondary exact-format diagnostic rather than silently changing the published rule. A number outside the available options is flagged. Separate reasoning items and `<think>`/`<thinking>` blocks are excluded before grading for modern reasoning models. This exclusion is an explicit adaptation of the older evaluator.

Use one attempt per task and temperature 0.1, matching upstream. The overnight reasoning-model comparison uses max_output_tokens 16384 rather than upstream's 20-token generation cap so native reasoning is not artificially truncated. Record this deviation; these are not exact reproductions of the paper's model scores. Endpoint timeouts/retries and actual token usage must accompany results. Never drop infrastructure failures silently from denominators.

## Prepare and validate

From the repository root, with the Gym development environment activated:

```bash
python benchmarks/proteinlmbench/prepare.py
gym env validate proteinlmbench
gym env test proteinlmbench
python -m pytest resources_servers/proteinlmbench/tests -q
gym env publish proteinlmbench
```

The verifier is stateless and deterministic; full model runs are not declared deterministic. Five source rows are included as examples. The benchmark data file is gitignored.

## Evaluate

Configure model credentials and endpoint outside the repository using Gym's model configuration. Run each model on identical row IDs, first ten source rows for smoke, then all 944:

```bash
gym eval run --benchmark proteinlmbench --model-type openai_model \
  --split benchmark --limit 10 --num-repeats 1 --temperature 0.1 \
  --max-output-tokens 16384 --output /absolute/private/results/smoke.jsonl
```

Inspect trajectories and grading before removing `--limit 10`. Save model configuration, source revision, selected IDs, resource limits, output, and aggregate accuracy outside public Git. No claim of a verified baseline is made by this experimental manifest.
