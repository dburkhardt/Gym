# MetaSyn native RAG integration

This Gym agent runs the creator's BM25 top-200 fixed-pool RAG baseline and complete evaluator. It preserves the native report and every metric. Gym's scalar is specifically `inclusion_f1`, not an overall quality score. Generator and judges use `nvidia/nvidia/nemotron-3.5-super-vl`; this is a Super-judge variant, not a reproduction of the paper's MA-retriever or judge model.

The preserved creator source is `THUIR/MetaSyn@3ee040f31b9a476c4efe4012d909aac6e7cce8d6`; the dataset revision is `c8fa07d89c44093d623f9a213c6bf070f40ab960`. The native nongenerative similarity resource is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2@4328cf26390c98c5e3c738b4460a05b95f4911f5`. These are distinct identities. Preparation verifies the frozen five- or twenty-review sample, exact prompt bytes and source hashes; it does not use the separate corpus-search protocol.

## Prepare and inspect

Configure `METASYN_DATA` with the preserved materialized dataset, `METASYN_CREATOR_SOURCE` with the pinned `src/metasyn` directory, `METASYN_PYTHON` with the isolated scientific interpreter, `METASYN_RUNNER` with this directory's `native_runner.py`, `METASYN_REFERENCES` with the corresponding full reference JSONL, and `METASYN_OUTPUT` with a fresh output root. `METASYN_SAMPLE_COUNT` accepts 5 or 20. References remain in the trusted evaluator; the model receives only creator-generated prompts.

From the Gym checkout, use the supported offline preparation and plan interface:

```sh
gym eval plan --benchmark metasyn --out /absolute/path/metasyn-plan.json
```

The frozen five- and twenty-task plans have been exercised. The creator interpreter needs the scientific dependencies plus the shared guard's imports, including OmegaConf 2.3.0. The local qualification captured installed versions separately; it does not establish a portable clean-worker environment.

## Execution and evidence

Execution is disabled by default. Enabling requires `inference_enabled: true`, the authorized model, provider dispatch guard and native rollout identity. The maintained parent opens a task-scoped Unix socket bridge to the Gym model endpoint. Both creator SDK constructors use the shared guarded synchronous transport with SDK retries disabled. No provider credential enters the child. The legacy standalone endpoint is historical only and is not the maintained launch path.

The shared LocalProvider owns the creator process group, enforces the unchanged 3600-second execution budget and cleans up descendants. Each review retains its identity, native runtime log, dispatch evidence, report, complete metrics and execution receipt. Existing review output directories are never overwritten. Changed source, prompt, references, candidate pool or encoder files fail identity checks before inference.

Scientific threads default to one and are recorded in the evaluation identity; offline encoder loading prevents implicit downloads. The actual pinned creator, encoder, managed child, SDK and Unix socket bridge passed a local deterministic control using the full 447,052-character review 203 prompt and five calls. This is integration evidence, not model-performance evidence: the endpoint supplied synthetic responses and token counts. Earlier historical live results predate this shared path and do not qualify it.

Native Gym model capture, real-provider execution, clean-worker resources, interruption controls and unattended heterogeneous acceptance remain pending. Keep live enablement closed until those gates pass. The local runtime evidence is in the campaign's `foundation/metasyn-native/RUNTIME-CONTROL.md` and `full-runtime-probe.json`.
