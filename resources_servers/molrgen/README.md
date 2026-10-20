# MolRGen creator-service bridge (runtime validation pending)

Source https://github.com/Fransou/MolRGen at `f033a3dfea50c3be5782a979e73eeeb842a7bae1`; data https://huggingface.co/datasets/Franso/MolRGen at `a63fd8b3b7f895f551bb1a5bc6f8fb408839bdc4`. Use the archive's `test_data/test_prompts_ood_boxed.jsonl` (1,000 records with original system/user messages and full reward metadata), not the stripped validation parquet. Deterministic seed20261020 samples5/20/50 and all receptor assets are prepared outside Git.

This bridge sends the actual completion and isolated creator metadata to the creator `/get_reward` service in singleton mode. It records the entire creator receipt and labels reported docking oracle. No SMILES-validity approximation replaces docking. Set `creator_url` to the isolated service endpoint, `docking_oracle` to its actual configured condition, and `CAMPAIGN_SAMPLE_JSONL` before preparation. The service must use boxed parsing to match these prompts.

The CPU PyScreener/Vina route is a creator-supported **oracle variant**, not equivalent to the default AutoDock-GPU scores. The current existing host has no GPU docking runtime. A real positive/invalid-molecule docking fixture and full error inspection remain required. Two tests validate Config, native HTTP /verify and mocked creator-service payload/receipt handling only; these do not validate chemistry or actual docking. Integration remains verified:false.

Per-completion reward is not the paper's diversity-aware top-k result. Repeated generation, fingerprint clustering and creator aggregate replay remain separate required steps for that metric. Preserve all molecular properties, reward scaling, docking exhaustiveness and target settings. Empty/failed scoring must remain visible; explicit creator errors are masked and recorded with zero only as serialization placeholder. Do not silently omit failed attempts from reported denominators.

The creator's published Docker image `fransou/molgendata:latest` could not be pulled (access denied/not found). A CPU build from the pinned source is being staged separately; dependency and runtime validation remain open. No model or docking runs are claimed by this commit.

## Managed CPU oracle

Set `managed_oracle` instead of `creator_url` to run a separate task-owned Docker
sandbox for each verification. Required fields are `source_dir`, `source_sha256`,
`data_dir`, `data_sha256`, and `artifact_dir`. The source digest is SHA-256 of the sorted lines
`relative_python_path sha256_of_file\n`, excluding Apple `._` files. The preserved
campaign instrumented source digest is
`8df235f5ed9187868d3dccbc8ccbffb8f66e3e8447af4a2a964488e1c54758aa`.

The image is pinned by immutable ID in `managed_oracle.py`. Each invocation uses
2 CPUs, 3 GiB RAM, 256 MiB shared memory, a dynamically published localhost port,
and Gym ownership labels. Keep evaluation concurrency at one on an 8 GiB worker.
Source and data mounts are read-only. Provision the creator dataset separately.
The required data digest hashes sorted `relative_path sha256_of_file\n` lines for
every dataset file, excluding Apple `._` files. It is checked before each container.
Preserved campaign data digest:
`f9ca645d73b3b4cf816f639a137c5fa599f871c2a33a20279721380505d5b35a`. CPU Vina1.1.2,
boxed parsing, property reward, rescaling, and exhaustiveness 8 match the preserved
native CPU service. This is not the published AutoDock-GPU protocol.

The artifact directory retains the service script, source hash, serialized sandbox
handle, native receipts, raw oracle values, logs, and docking scratch files.
Success, HTTP timeout, and cancellation all stop the exact owned sandbox, including
Ray/Vina children. A 900-second Docker TTL bounds orphan lifetime after controller
crash; cleanup errors propagate and must be reconciled before retrying. The HTTP
request is never automatically resent. A failed oracle is masked, not scored as a
scientific zero. Per-completion reward is still not the diversity-aware top-k metric.

`gym eval plan --benchmark molrgen` now stages frozen sample rows via the shared
campaign preparation interface, preserving prompts and metadata and recording the
source hash. Supply `CAMPAIGN_SAMPLE_JSONL` and optionally `CAMPAIGN_SAMPLE_SHA256`.
