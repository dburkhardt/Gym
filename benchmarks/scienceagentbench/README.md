# ScienceAgentBench frozen preparation

This directory implements verified metadata selection and required-asset preflight. It is **not yet a runnable Gym agent/verifier configuration**, and does not establish archive authenticity, scientific correctness or runtime qualification. Missing prerequisites fail before task files are published. No acquisition, package installation or model request occurs.

`frozen50.json` separately binds creator code `c26e151ed601ba109dc4d35e057ff8e73fec469d`, dataset revision `9c6e96c9e74572e979b0930ee735041cef528cb7`, the verified parquet bytes and every selected metadata row. Preserve both zero-based source row index and creator `instance_id`; they are not interchangeable. The creator's exact first-line slicing for `dataset_path` is retained. Gold/evaluator selectors remain outside `creator_task`, but the host-side task JSON and manifest must never be mounted into a solver sandbox.

From the Gym root, with pandas and a parquet reader available, run this template with real local paths and a **fresh** destination:

```sh
python -m benchmarks.scienceagentbench.prepare \
  --source /path/to/pinned/ScienceAgentBench \
  --parquet /path/to/scienceagentbench-verified.parquet \
  --assets /path/to/extracted/benchmark \
  --destination /path/to/new/prepared-output
```

`--assets` contains the creator `datasets/`, `gold_programs/` and `eval_programs/` directories. Every concrete path enumerated in each frozen folder tree must exist, and selected gold programs/evaluation scripts must exist. Symlinks and traversal are rejected. Fresh checksums are an inventory, **not** proof that files match the creator archive. For checksum verification additionally provide `--asset-manifest /path/to/expected.json --asset-manifest-sha256 <trusted-SHA256>`, where JSON maps relative file paths to expected SHA256 and covers every selected asset. That caller-supplied binding still needs independent provenance approval. The retained evidence does not contain an authenticated checksum for the inaccessible `benchmark_verified.zip`.

The receipt deliberately remains `required_paths_present_not_runtime_or_archive_provenance_qualified`, even with supplied checksum verification. Auxiliary files imported by private evaluation scripts, runtime dependencies and archive provenance still need qualification. Future execution admission must refuse this status. Existing destinations are refused, preserving historical preparations; interrupted staging may leave owned partial output for inspection. `--use-knowledge` is explicit and defaults false, as in the creator command.

## Bridge constraints identified from the pinned creator

`run_infer.py` defaults to validation despite the verified release; the frozen preparation explicitly uses verified metadata. `agent.py` uses the model-specific LiteLLM tokenizer, a 28000-token context cutoff, 0.2 temperature/top_p0.95, optional knowledge and up to10 self-debug steps. Each debug execution has a900s bound and dynamic dependency installation. `engine/base_engine.py` routes non-gpt/o1 names to Bedrock; the OpenAI engine combines SDK retries10 and an outer backoff decorator. A Nemotron bridge must inject an explicit guarded engine and qualified tokenizer without retaining hidden retries or silently selecting a GPT tokenizer. Cost-table bookkeeping must not be mistaken for scientific behavior.

`compute_scores.py` computes CodeBERT similarity, then executes the submitted prediction with gold/evaluation files already present before importing the task evaluator. Reference isolation therefore requires separate prediction execution (no gold/eval mounts), followed by trusted scoring over resulting artifacts. This changes the creator's execution arrangement and must be recorded; merely excluding gold from the prompt is insufficient. The unavailable archive prevents qualification of evaluation-script auxiliary dependencies today.

`gpt4_visual_judge.py` requests three GPT-4o visual samples at0.2/top_p0.95/max_tokens1000 and averages parsed scores. All-Nemotron substitution must preserve three distinct samples and the exact prompt/parser and be labeled a judge-model adaptation. `calculate_metrics.py`/creator README describe selecting the best of three independent runs; a one-run control cannot claim that aggregate protocol. No native config is published until the guarded creator bridge, isolated verifier and their meaningful local-provider controls exist.

Tests in `tests/unit_tests/test_scienceagentbench_prepare.py` use real frozen metadata plus clearly synthetic prerequisite files: all50 identity checks, exact creator `format_task_dict` parity for both knowledge modes, missing enumerated file/refusal before output, source tampering, externally pinned asset checksum tampering, symlink refusal and historical output preservation. Synthetic files do not qualify any scientific dataset or scorer.
