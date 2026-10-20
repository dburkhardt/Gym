# MolBench screening baseline

Pinned creator source c16be3a70deef4228f19930b4495ec2748fafa38; SHA256 checks cover loader, prompt templates, baseline runner, scorer and ms-3 dataset. This integration reproduces the creator tool-free prompt/parser/scorer path, not the MolClaw tool agent. Frozen public ms-3 contains25 tasks, five calibration and20 including those five; ms-1/ms-2, editing and optimization remain outside this adapter.

Solver messages are reconstructed from the native loader and compared exactly with frozen public prompts. Ground-truth answer metadata stays in the verifier. The creator parser truncates a requested60-item ranking to3. Preserve top3_hit_rate and all native metrics, including rank computed from that truncated list. Duplicates receive repeated native hit credit; duplicate_ranking reports this without rewriting native scores. This measures reference ranking agreement, not independently validated molecular binding.

Stage the creator repository, frozen sample JSONL, and set CAMPAIGN_UPSTREAM_REPO/CAMPAIGN_SAMPLE_JSONL. Native Gym registration is config.yaml. The prepare hook copies the selected JSONL to a literal path; runtime configuration should likewise use a literal dataset path. Exact campaign model is nvidia/nvidia/nemotron-3.5-super-vl; temperature0 matches baseline. The creator baseline usesseed42; the campaign Responses transport does not yet establish seed parity. Explicit16,384 output-token budget, seed omission and single-attempt transport are campaign variations.

Tests cover exact25 prompt reconstruction, source isolation, positive/malformed/duplicate controls, full Gym configuration, service startup, actual /verify and dataset collation. No inference is implied by these controls.
