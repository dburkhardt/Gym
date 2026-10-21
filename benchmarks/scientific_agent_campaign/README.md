
## Model and harness capability declarations

Native evaluation checks declared dataset requirements before starting services or collecting rollouts. MicroVQA requires `vision`; BrainBench requires `conditional_loglikelihood`. Supply declarations for the actual selected component instance in your run configuration, for example:

```yaml
component_capabilities:
  policy_model:
    vision: true
```

This is an operator assertion about the selected backend, not an automatic capability probe. Set `false` for known unsupported capabilities; omit unknown ones. Both unsupported and undeclared required capabilities fail closed. Changing the model does not establish that an old declaration remains true. In this campaign, only the authorized Nemotron model may be called; a successful configuration check does not authorize another model or spending.

Benchmark datasets may declare `required_model_capabilities` and `required_harness_capabilities`. The gate resolves the dataset's actual agent and its primary `model_server` reference. It rejects missing or incorrectly typed model bindings. It currently does not validate judge/secondary model roles or arbitrary custom-agent fields. A taskset requiring this validation without an explicit resolved agent fails closed. Untagged datasets receive no capability guarantee. Do not claim generic system compatibility from this gate alone.

Tests exercise the maintained MicroVQA and BrainBench configurations with unsupported, unknown and supported declarations without dispatching inference. BrainBench still requires a genuinely available likelihood transport; declaring support cannot create that implementation.

### Concurrent native services

Native `gym eval run` and `gym env start` acquire cooperative ownership of their resolved head and component ports before starting services. Concurrent Gym supervisors under the same local user cannot claim overlapping ports, including `localhost`, loopback and wildcard aliases. Conflicts fail before model dispatch; Gym does not silently change configured ports. Configure separate head ports and non-overlapping component ranges for simultaneous runs.

Ownership remains held until owned services are confirmed stopped. Normal shutdown and failed startup release it; supervisor process death releases kernel locks. Lock files are deliberately retained so two processes cannot lock different inodes for the same port. Durable service receipts still govern recovery of orphaned child processes; a port lock is not a session checkpoint.

This is conservative per-user coordination on one machine, not a socket reservation or a cross-host allocator. Disjoint interfaces with the same port are also rejected. Other users and non-Gym processes do not participate and can still win an OS bind race; native startup must fail rather than sending work to an unowned service. It does not coordinate arbitrary task-container or tool ports.
