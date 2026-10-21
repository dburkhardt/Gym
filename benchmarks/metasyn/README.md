# MetaSyn native frozen dataset

`gym eval plan --benchmark metasyn` prepares the preserved native BM25 top200 fixed-pool RAG variant. Set `METASYN_DATA` to the existing materialized directory. Default preparation combines `gym-pilot5.jsonl` and `gym-remaining15.jsonl` in original order; `METASYN_SAMPLE_COUNT=5` selects only the unchanged pilot. It does not invent a native50-row sample.

All20 prompts were checked against their actual `rag/<review_id>/prompt.txt` files, including immutable byte hashes. They contain400620–517165 characters. The separate `agent-inputs-20.jsonl` corpus-search prompts use a different schema and scientific protocol and are deliberately not used. Preserved retrieval source revision is `c8fa07d89c44093d623f9a213c6bf070f40ab960`, with native BM25 top200 and source-review exclusion from140585 corpus records.

Preparation checks frozen source and prompt hashes, preserves every original row field and adds only `task_id=metasyn::<review_id>`. The remaining15 rows originally lacked instance_id; no positional IDs or new sampling are introduced. Shared `stage_file` publishes output and a receipt linking both original files, every prompt hash and the normalized output checksum.

The existing native runner interface remains unchanged: `METASYN_PYTHON`, `METASYN_RUNNER`, `METASYN_DATA`, `METASYN_OUTPUT` and `METASYN_REFERENCES`. Planning does not invoke the runner or establish live readiness. The native wrapper still uses an external subprocess and has not passed the current shared lifecycle, isolation and durable-call guard acceptance gates. A successful dataset plan must not authorize a new run. Historical Nemotron generator/judge results and full evaluator metrics remain separate; scalar inclusion_f1 is not an invented overall benchmark score.
