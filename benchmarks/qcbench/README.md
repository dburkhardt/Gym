# qcbench campaign integration

Run the pinned preparation in `benchmarks/scientific_agent_campaign/deterministic.py`. Set `CAMPAIGN_UPSTREAM_REPO` to the absolute pinned creator checkout and `CAMPAIGN_SAMPLE_JSONL` to the absolute seeded JSONL file. This integration sends only responses_create_params to the model; verifier_metadata remains server-side. Use Nemotron3.5SuperVL exclusively.

Start with `gym env start --benchmark qcbench --model-type openai_model --config runtime.yaml`, then `gym eval run --no-serve --benchmark qcbench --model-type openai_model --config runtime.yaml --agent qcbench_agent --input "$CAMPAIGN_SAMPLE_JSONL" --limit 5 --num-repeats 1 --output pilot5.jsonl`. This explicit input avoids E2E dataset-precedence surprises. Runtime must configure the model endpoint, exact allowlisted model, and credentials securely.

Creator code and data remain external with their original licenses; pin checks fail closed. Shared dependencies: `resources_servers/campaign_creator_scoring/requirements.txt`. For MolecularIQ install the pinned local core checkout separately. All settings/protocol qualifications and fixture evidence are in the campaign report. Config verified=false pending live rollout and independent review.

## Measurement validity

The creator numeric scorer rejects some of its own references, including frozen task 233 (`3.51e-09`), because its decimal-place rounding ignores the exponent. Preserve the original `reward` and `creator_numeric` metric. `scoring_details.reference_self_score`, `measurement_valid`, and `measurement_invalid_reason` expose this defect separately; a normal incorrect model answer can still have `measurement_valid: true`.

Do not silently remove affected tasks or relabel their raw scores as model failures. Report attempted/completed counts, raw upstream scores, and the count of interpretable measurements separately. Historical results remain unchanged. This metadata is a source-scorer consistency check, not a comprehensive certification of the task or reference.
