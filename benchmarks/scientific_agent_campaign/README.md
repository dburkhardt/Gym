
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
