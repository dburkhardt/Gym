# Harbor folder targets and execution support

Folder/task-registry materialization is not executable harness support. The `gym eval run <folder-or-harbor:dataset> --agent <harness>` route requires `resources_servers/harbor/app.py` through the normal Gym component search roots. This checkout does not ship that resources implementation. The CLI now refuses early, before downloading a task registry dataset or writing run outputs, when it is absent or not a file. A plugin may supply it through the existing roots; file presence is only a prerequisite, not runtime qualification.

The maintained `responses_api_agents/harbor_agent` route instead owns an entire Harbor Job, including its native agent/environment/verifier lifecycle. Use an existing benchmark configuration selecting that adapter through `gym eval run --benchmark <registered-name>`. It is not a drop-in substitution for the folder route's chosen tool harness or `single_agent_turn` protocol.

The missing route still needs an actual resources implementation and native runtime/tool/verifier acceptance. Successful task discovery, row generation, schema validation or these unit tests must not be reported as that acceptance. No new execution framework or automatic protocol conversion is introduced by this refusal gate.
