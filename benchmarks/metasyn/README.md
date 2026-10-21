# MetaSyn native frozen dataset

`gym eval plan --benchmark metasyn` prepares the preserved native BM25 top200 fixed-pool RAG variant. Set `METASYN_DATA` to the existing materialized directory. Default preparation combines `gym-pilot5.jsonl` and `gym-remaining15.jsonl` in original order; `METASYN_SAMPLE_COUNT=5` selects only the unchanged pilot. It does not invent a native50-row sample.

All20 prompts were checked against their actual `rag/<review_id>/prompt.txt` files, including immutable byte hashes. They contain400620–517165 characters. The separate `agent-inputs-20.jsonl` corpus-search prompts use a different schema and scientific protocol and are deliberately not used. Preserved retrieval source revision is `c8fa07d89c44093d623f9a213c6bf070f40ab960`, with native BM25 top200 and source-review exclusion from140585 corpus records.

Preparation checks frozen source and prompt hashes, preserves every original row field and adds only `task_id=metasyn::<review_id>`. The remaining15 rows originally lacked instance_id; no positional IDs or new sampling are introduced. Shared `stage_file` publishes output and a receipt linking both original files, every prompt hash and the normalized output checksum.

The existing native runner interface remains unchanged: `METASYN_PYTHON`, `METASYN_RUNNER`, `METASYN_DATA`, `METASYN_OUTPUT` and `METASYN_REFERENCES`. Planning does not invoke the runner or establish live readiness. The native wrapper uses an external creator subprocess. The native fixture control described below qualifies guarded calls, collection, shutdown and completed-output reuse for its tested configuration; it does not qualify arbitrary interrupted-state recovery, full live execution or every isolation boundary. A successful dataset plan must not authorize a new run. Historical Nemotron generator/judge results and full evaluator metrics remain separate; scalar inclusion_f1 is not an invented overall benchmark score.

Native CLI model-free qualification now covers the complete prepared 20-row dataset with `--limit 1` selecting unchanged review 203, actual `/run` metadata routing, guarded model capture, creator/encoder execution and completed-output `--resume`. The representative produced five linked solver/judge turns and a healthy native inspection; resume added no provider calls. These deterministic responses are transport/scorer fixtures, not model results. MetaSyn still has no serializable agent-session checkpoint hooks: completed-output reuse does not prove interrupted creator/encoder recovery. Keep all 20 tasks in scope and require separate live/resource admission.

## Reconcile preserved results without inference

`python -m benchmarks.metasyn.import_saved --materialized <metasyn-materialized> --creator-source <pinned-src/metasyn> --transport <metasyn-transport> --output <new-directory>` imports the historical5+15 cohorts. It requires their original files and the exact creator source manifest. It checks frozen prompts, candidate pools, reports, evaluations, available raw75 HTTP pairs and provider receipts, then calls only the creator's deterministic selection/ID metrics. It never invokes generator, judges or encoder. Output creation is exclusive; original records are unchanged and projected records add shared measurement-review metadata. The shared report qualifies deterministic metric parity only; saved judge/encoder values remain unresolved historical observations. Cohort-local execution indices are reported separately, not conflated into a new run.

Pilot5 raw HTTP bodies are unavailable. The remaining15 generator requests bind directly to frozen prompts/reports; judge calls are checked against the cohort's raw receipts, without inventing per-review attribution from timestamps. Historical successful-response ledgers do not rule out an unanswered request before logging. The receipt retains that uncertainty and creates no modern capture or episode identities. This historical reconciliation is not fresh live acceptance or interrupted-state recovery.

## Pinned Linux CPU scientific environment

The maintained `dependencies/` directory records the qualified 71-wheel Linux x86_64/Python 3.11 closure. `torch==2.3.1+cpu` is an explicit Linux CPU adaptation of the historical `torch==2.3.1` environment, not historical binary parity. Other package versions remain pinned. Creator Python is loaded from the separately hash-checked source directory; it is not replaced with a PyPI `metasyn-benchmark` package. Keep this scientific environment separate from the native Gym controller/model/agent environments; set `METASYN_PYTHON` to its interpreter.

The following are operator templates, run from the repository root on Linux x86_64 with Python 3.11. Set `WHEELHOUSE`, `SCIENTIFIC_ENV` and `ENCODER` to absolute directories; the environment must be new. Acquisition needs public package access and disk capacity. It is not part of an inference run.

```sh
LOCKS="$PWD/benchmarks/metasyn/dependencies"
python3.11 -m pip download --only-binary=:all: --no-deps --require-hashes \
  -r "$LOCKS/linux-cp311-download.lock" --dest "$WHEELHOUSE"
# Supply the separately retained, qualified ANTLR wheel described below.
(cd "$WHEELHOUSE" && sha256sum --check "$LOCKS/wheels.sha256")
python3.11 -m venv "$SCIENTIFIC_ENV"
"$SCIENTIFIC_ENV/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" \
  --only-binary=:all: --no-deps --require-hashes -r "$LOCKS/linux-cp311.lock"
"$SCIENTIFIC_ENV/bin/python" -m pip check
(cd "$ENCODER" && sha256sum --check "$LOCKS/encoder.sha256")
```

The download lock preserves the exact official PyTorch CPU wheel URL and SHA, rather than relying on a default-index version lookup. All other downloads require the qualified wheel hashes. The checksum manifest additionally fixes filenames. No wheels or encoder weights are stored in this repository. `encoder.json` provides the 13 exact URLs, sizes and hashes for revision `4328cf26390c98c5e3c738b4460a05b95f4911f5`; use the existing shared checksum staging facility to acquire them. Preserve the snapshot contents and point the creator similarity-model configuration's `local_path` to that directory. Do not change model identity or silently download a newer revision.

ANTLR 4.9.3 has no upstream wheel. `antlr-build.json` records its sole approved source build: exact official sdist SHA, setuptools 69.5.1 and wheel 0.43.0 URLs/hashes, and the qualified wheel SHA. The isolated Python 3.11 build used `pip wheel --no-index --no-deps --no-build-isolation --wheel-dir <new-output> <verified-sdist>` after installing only those verified build tools. All 56 packaged Python files matched the sdist; version metadata was 4.9.3 and no native extension was present. This was package-manager offline mode, not a network sandbox guarantee. Obtain the qualified wheel from the retained artifact store and verify its hash before installing. If rebuilding, retain the original lock: ZIP timestamp byte identity was not established, so a different wheel digest requires a separate source-byte/metadata review and explicitly versioned lock; never simply remove `--require-hashes` or authorize other source builds.

This closure qualified actual creator/encoder imports and a native local-provider review-203 control, including five linked calls and zero additional calls on completed-output resume. It does not authorize new inference, establish historical binary parity, or prove interrupted creator recovery. Preserve the frozen full20 inputs and references separately; dependencies do not replace data/source preflight or resource admission.
