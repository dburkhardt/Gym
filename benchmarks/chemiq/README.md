# chemiq campaign integration

Run the pinned preparation in `benchmarks/scientific_agent_campaign/deterministic.py`. Set `CAMPAIGN_UPSTREAM_REPO` to the absolute pinned creator checkout and `CAMPAIGN_SAMPLE_JSONL` to the absolute seeded JSONL file. This integration sends only responses_create_params to the model; verifier_metadata remains server-side. Use Nemotron3.5SuperVL exclusively.

Start with `gym env start --benchmark chemiq --model-type openai_model --config runtime.yaml`, then `gym eval run --no-serve --benchmark chemiq --model-type openai_model --config runtime.yaml --agent chemiq_agent --input "$CAMPAIGN_SAMPLE_JSONL" --limit 5 --num-repeats 1 --output pilot5.jsonl`. This explicit input avoids E2E dataset-precedence surprises. Runtime must configure the model endpoint, exact allowlisted model, and credentials securely.

Creator code and data remain external with their original licenses; pin checks fail closed. Shared dependencies: `resources_servers/campaign_creator_scoring/requirements.txt`. For MolecularIQ install the pinned local core checkout separately. All settings/protocol qualifications and fixture evidence are in the campaign report. Config verified=false pending live rollout and independent review.

OPSIN observation preserves the pinned creator's raw score, cache and two-second
alarm. HTTP connection/timeouts, 429/5xx and malformed successful JSON are recorded
as dependency failures; an alarm outside HTTP is separately a verifier timeout.
These observed failures mask quality aggregation without removing raw scores.
HTTP 400/404 and cached values are diagnostic, not automatic infrastructure
exclusions. Ordinary correct/wrong/empty results receive no new measurement review
and remain in native quality aggregation. This does not certify scientific validity
for the separate opt-in reviewed-metric mean. Existing generation failure diagnoses
remain visible even when a concurrent verification failure masks the measurement.
Pinned-source controls require CHEMIQ_CREATOR_REPO (or the local sibling checkout),
check the verifier SHA256, and explicitly skip when the source is unavailable.
