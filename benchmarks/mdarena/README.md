# MDArena frozen dataset binding

`--benchmark mdarena` loads the existing JSON Harbor configuration. Use `--config responses_api_agents/harbor_agent/configs/harbor_agent_mdarena_xml.yaml` for XML. Their dataset names are `mdarena_json` and `mdarena_xml`. They consume the same preserved sample and differ only in their existing Terminus2 parser configuration. Registration changes no scientific task files, images, budgets or judge protocol.

Set `MDARENA_FROZEN_SAMPLE` to preserved `pilot-5.jsonl`, `pilot-20.jsonl`, or `pilot-50.jsonl`. Preparation verifies original byte checksum and ordered IDs against `frozen-samples.json`, preserves every field and adds only `task_id` as an alias for `instance_id`. Shared `stage_file` publishes data and an original-source/output checksum receipt. There is no resampling, acquisition or inference.

Keep `MDARENA_TASKS_DIR` and `MDARENA_IMAGE_ID` pointed at a separately qualified task deployment. Planning is not runtime admission: scientific images, explicit resources, isolated full-workspace verifier and guarded judge remain required. Both configurations currently use one configured verifier image; heterogeneous tasks need their actual compatible images qualified before execution. A50-row plan does not establish50 runnable tasks.

Creator source: `weitse-hsu/MDArena@d1d789d92a9935e716182e31d99393edcd4ac7d6` (MIT). Retain its canaries and exclusion from training corpora. See the [native preparation guide](../../responses_api_agents/harbor_agent/benchmarks/mdarena/README.md) for protocol and Nemotron judge variant. Prior live demonstrations and failed held attempts remain separate evidence.
