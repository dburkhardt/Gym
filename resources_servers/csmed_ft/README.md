# CSMeD-FT input admission

See [benchmark setup](../../benchmarks/csmed_ft/README.md). Full benchmark acceptance remains experimental; the qualification below covers input admission.

The maintained `/seed_session` validates the frozen preparation's `verifier_metadata.input_valid` before SimpleAgent begins its model loop. Only explicit `true` is admitted. Missing full text/criteria, missing validity evidence, and malformed metadata return structured HTTP422 with task/review identity and `csmed_ft:invalid_input`; no scientific zero is invented. Raw source fields are not separately available in this request, so the server does not reconstruct or replace them from prompts.

Keep invalid frozen rows in the input. Use native `route_failures_to_sidecar: true` to retain rejected rows and continue valid tasks. The current legacy environment wraps the inner422 as outer500 `agent_run_error`; its failure-sidecar response body retains the precise reason/task ID, while native task/rollout indices join the original materialized inputs. Report those input failures separately from scientific/model performance.

Qualification used actual native CLI with one invalid and one valid frozen row and a local deterministic provider: two selected, one failed before model dispatch, one valid result, exactly one provider call, zero external inference. Both legacy and modern task_data seed contracts are tested; ordinary valid verification is unchanged.
