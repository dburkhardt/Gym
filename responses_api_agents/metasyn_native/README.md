# MetaSyn native RAG integration

This Gym agent runs the creator's BM25 top-200 fixed-pool RAG baseline and complete evaluator. It preserves the native report and every metric. Gym's scalar is specifically `inclusion_f1`, not an overall quality score. Generator and judges use `nvidia/nvidia/nemotron-3.5-super-vl`; this is a Super-judge variant, not a reproduction of the paper's MA-retriever or judge model.

The preserved creator source is `THUIR/MetaSyn@3ee040f31b9a476c4efe4012d909aac6e7cce8d6`; the dataset revision is `c8fa07d89c44093d623f9a213c6bf070f40ab960`. The native nongenerative similarity resource is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2@4328cf26390c98c5e3c738b4460a05b95f4911f5`. These are distinct identities. Preparation verifies the frozen five- or twenty-review sample, exact prompt bytes and source hashes; it does not use the separate corpus-search protocol.

## Prepare and inspect

Configure `METASYN_DATA` with the preserved materialized dataset, `METASYN_CREATOR_SOURCE` with the pinned `src/metasyn` directory, `METASYN_PYTHON` with the isolated scientific interpreter, `METASYN_RUNNER` with this directory's `native_runner.py`, `METASYN_REFERENCES` with the corresponding full reference JSONL, and `METASYN_OUTPUT` with a fresh output root. `METASYN_SAMPLE_COUNT` accepts 5 or 20. References remain in the trusted evaluator; the model receives only creator-generated prompts.

From the Gym checkout, the discoverable benchmark name is `metasyn`. Listing does not require the runtime environment variables or credentials. Use the supported offline inspection and plan interface:

```sh
gym list benchmarks metasyn --json
gym eval plan --benchmark metasyn --out /absolute/path/metasyn-plan.json
```

The frozen five- and twenty-task plans have been exercised. The creator interpreter needs the scientific dependencies plus the shared guard's imports, including OmegaConf 2.3.0. The local qualification captured installed versions separately; it does not establish a portable clean-worker environment.

## Execution and evidence

Execution is disabled by default. After qualification, `METASYN_INFERENCE_ENABLED=true` sets `inference_enabled: true`; execution also requires the authorized model, provider dispatch guard and native rollout identity. The maintained parent opens a task-scoped Unix socket bridge to the Gym model endpoint. Both creator SDK constructors use the shared guarded synchronous transport with SDK retries disabled. No provider credential enters the child. The legacy standalone endpoint is historical only and is not the maintained launch path.

The shared LocalProvider owns the creator process group, enforces the unchanged 3600-second execution budget and cleans up descendants. Each review retains its identity, native runtime log, dispatch evidence, report, complete metrics and execution receipt. Existing review output directories are never overwritten. Changed source, prompt, references, candidate pool or encoder files fail identity checks before inference.

`METASYN_SCIENTIFIC_THREADS` configures scientific threads (default one), which are recorded in the evaluation identity; offline encoder loading prevents implicit downloads. The actual pinned creator, encoder, managed child, SDK and Unix socket bridge passed a local deterministic control using the full 447,052-character review 203 prompt and five calls. This is integration evidence, not model-performance evidence: the endpoint supplied synthetic responses and token counts. Earlier historical live results predate this shared path and do not qualify it.

Native Gym model capture, real-provider execution, clean-worker resources, interruption controls and unattended heterogeneous acceptance remain pending. Keep live enablement closed until those gates pass. The local runtime evidence is in the campaign's `foundation/metasyn-native/RUNTIME-CONTROL.md` and `full-runtime-probe.json`.

## Repeat the local runtime controls

Run this maintained command from the checkout with the configured source and dataset paths. Use a new output directory for every attempt. `success` completes the creator evaluator; `judge-disconnect` aborts the first judge request and checks that no further physical requests occur and no reward is accepted. Both use a deterministic loopback endpoint, with no external model calls.

```sh
python -m responses_api_agents.metasyn_native.qualify_runtime \
  --mode judge-disconnect \
  --data "$METASYN_DATA" --references "$METASYN_REFERENCES" \
  --creator-source "$METASYN_CREATOR_SOURCE" --python "$METASYN_PYTHON" \
  --output /absolute/path/new-qualification
```

The maintained disconnect command was exercised with the actual creator/encoder. Its failed native evaluation is deliberately preserved as evidence; a failed judge result is never a valid scalar reward. This control does not instantiate the native Gym model server or qualify provider-ledger reconciliation.

### Repeated tasks and explicit attempts

Native execution artifacts are isolated by the exact review ID and Gym rollout/attempt identity under `output_root/attempts/`. The creator still receives an output root with its original `<review_id>` child directory. A second invocation of the same episode refuses to overwrite or resend work. Independent logical repeats use separate directories and do not claim to reconcile earlier repeats. A new physical attempt of the same logical rollout requires successful prior creator/evaluation receipts and reconciled capture records. Missing, failed, legacy task-only artifacts, a changed capture store, or an unknown provider outcome require explicit reconciliation; changing the attempt suffix does not bypass these checks. This is artifact isolation, not arbitrary MetaSyn restart recovery. Existing native completed-output resume remains the supported way to reuse completed results.

The local regression runs a real deterministic child through the maintained process lifecycle and checks isolated file effects, unchanged first-attempt artifacts, and guarded refusal. It makes no external model calls and does not requalify live scientific execution.

### Public generator isolation

New live configurations must provide `public_generator_image` as a qualified immutable `sha256:` image ID. Enabling inference without it fails config validation and the `/run` preflight. The generator receives only the exact prepared prompt and pinned `rag.py`, minimal guarded transport code, its own output directory and scoped Unix socket. It receives no creator repository, candidate/reference files, encoder cache, provider credential or host network. The Docker allocation is an explicit 1 CPU/2 GiB operational placement; qualify the selected image and actual mount boundary before live use.

After generator cleanup, the unchanged trusted evaluator consumes the digest-bound saved report and usage. It retains the original reference-derived protocol query, candidates and scoring implementation. Generator and evaluator share one absolute 3600-second operational envelope including setup; an expired envelope refuses the next phase. Generation uses the separate `<rollout>:generator` lineage and ordinal zero; evaluator calls retain the parent lineage and their own contiguous ordinals. The returned trajectory combines both phases without changing stored call records. Historical saved-output controls may still use the legacy direct evaluator path; they do not qualify fresh unisolated generation.
