# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

from nemo_gym.batch_restart import ROW_BINDING_KEY, admitted_prior_attempts, bind_execution, digest
from nemo_gym.config_types import ConfigError


def fixture(tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text('{"task": "original"}\n')
    tracker = SimpleNamespace(manifest_sha256="m" * 64, execution_binding=None)
    config = {
        "model": {
            "responses_api_models": {"openai_model": {"entrypoint": "app.py", "openai_model": "m", "port": 1234}}
        },
        "agent": {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py", "temperature": 0.1}}},
    }
    binding = bind_execution(
        tracker, config, inputs, resume=False, require_prior=False, capture_dirs=[tmp_path / "captures"]
    )
    row = {
        "_ng_task_index": 1,
        "_ng_rollout_index": 0,
        "_ng_failure_class": "environment_server_failed",
        "failure_kind": "session_lost",
        ROW_BINDING_KEY: binding,
    }
    failures = tmp_path / "failures.jsonl"
    failures.write_text(json.dumps(row) + "\n")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {"execution_binding_sha256": binding, "failures": [{"capture_key": "1-0", "failure_sha256": digest(row)}]}
        )
    )
    kwargs = dict(
        binding_sha256=binding,
        failures_path=failures,
        pending_rows=[{"_ng_task_index": 1, "_ng_rollout_index": 0, "_ng_attempt_index": 1}],
        completed_keys=set(),
    )
    return tracker, config, inputs, row, receipt, kwargs


def test_exact_infrastructure_admission_preserves_files(tmp_path):
    tracker, config, inputs, row, receipt, kwargs = fixture(tmp_path)
    before = kwargs["failures_path"].read_bytes()
    assert admitted_prior_attempts(receipt, **kwargs) == {"1-0"}
    assert kwargs["failures_path"].read_bytes() == before
    config["model"]["responses_api_models"]["openai_model"]["port"] = 9999
    assert (
        bind_execution(tracker, config, inputs, resume=True, require_prior=True, capture_dirs=[tmp_path / "captures"])
        == kwargs["binding_sha256"]
    )


@pytest.mark.parametrize("change", ["model", "harness", "protocol", "task"])
def test_changed_scientific_binding_refused(tmp_path, change):
    tracker, config, inputs, *_ = fixture(tmp_path)
    if change == "model":
        config["model"]["responses_api_models"]["openai_model"]["openai_model"] = "other"
    elif change == "harness":
        config["agent"]["responses_api_agents"]["simple_agent"]["entrypoint"] = "other.py"
    elif change == "protocol":
        config["agent"]["responses_api_agents"]["simple_agent"]["temperature"] = 0.5
    else:
        inputs.write_text("changed")
    with pytest.raises(ConfigError, match="binding changed"):
        bind_execution(tracker, config, inputs, resume=True, require_prior=True, capture_dirs=[tmp_path / "captures"])


@pytest.mark.parametrize(
    "change", ["edited_failure", "completed", "budget_zero", "generic_timeout", "same_attempt", "unbound"]
)
def test_unsafe_receipts_refused(tmp_path, change):
    tracker, config, inputs, row, receipt, kwargs = fixture(tmp_path)
    if change == "completed":
        kwargs["completed_keys"] = {(1, 0)}
    elif change == "same_attempt":
        kwargs["pending_rows"][0]["_ng_attempt_index"] = 0
    elif change == "unbound":
        row.pop(ROW_BINDING_KEY)
    elif change == "budget_zero":
        row.update(reward=0, failure_kind="model_budget_exhausted")
    elif change == "generic_timeout":
        row.update(failure_kind="timeout_exceeded")
    else:
        row["failure_reason"] = "edited"
    kwargs["failures_path"].write_text(json.dumps(row) + "\n")
    if change in {"budget_zero", "generic_timeout"}:
        receipt.write_text(
            json.dumps(
                {
                    "execution_binding_sha256": kwargs["binding_sha256"],
                    "failures": [{"capture_key": "1-0", "failure_sha256": digest(row)}],
                }
            )
        )
    with pytest.raises(ConfigError):
        admitted_prior_attempts(receipt, **kwargs)


def test_unbound_historical_run_refused(tmp_path):
    tracker, config, inputs, *_ = fixture(tmp_path)
    tracker.execution_binding = None
    with pytest.raises(ConfigError, match="previously bound"):
        bind_execution(tracker, config, inputs, resume=True, require_prior=True, capture_dirs=[tmp_path / "captures"])


def test_capture_store_switch_and_missing_store_refused(tmp_path):
    tracker, config, inputs, *_ = fixture(tmp_path)
    for stores in ([tmp_path / "other-captures"], []):
        with pytest.raises(ConfigError):
            bind_execution(tracker, config, inputs, resume=True, require_prior=True, capture_dirs=stores)


def test_collector_scientific_protocol_bound(tmp_path):
    tracker, config, inputs, *_ = fixture(tmp_path)
    with pytest.raises(ConfigError, match="binding changed"):
        bind_execution(
            tracker,
            config,
            inputs,
            resume=True,
            require_prior=True,
            capture_dirs=[tmp_path / "captures"],
            collector_config={"temperature": 0.9},
        )


def test_explicit_guard_never_launders_unknown_or_same_attempt(tmp_path):
    from nemo_gym._checkpoint.control import ControlError
    from nemo_gym.base_responses_api_model import CaptureStore, assert_guarded_rollout_dispatch_safe

    store = CaptureStore(tmp_path, provider_dispatch_guard=True)
    row = {"_ng_task_index": 1, "_ng_rollout_index": 0, "_ng_attempt_index": 1}
    store.path_for("1-0").write_text(
        json.dumps({"model_call_id": "call", "provider_attempts": [{"attempt_id": "a", "outcome": "captured"}]}) + "\n"
    )
    before = store.path_for("1-0").read_bytes()
    assert_guarded_rollout_dispatch_safe([row], [tmp_path], admitted_prior_attempts=frozenset({"1-0"}))
    assert store.path_for("1-0").read_bytes() == before
    with pytest.raises(ControlError):
        assert_guarded_rollout_dispatch_safe(
            [{**row, "_ng_attempt_index": 0}], [tmp_path], admitted_prior_attempts=frozenset({"1-0"})
        )
    store.write_dispatch("1-0", "unknown", {"provider_attempts": [{"attempt_id": "b", "outcome": "unknown"}]})
    with pytest.raises(ControlError, match="unresolved"):
        assert_guarded_rollout_dispatch_safe([row], [tmp_path], admitted_prior_attempts=frozenset({"1-0"}))


def test_rejected_fresh_run_preserves_every_artifact(tmp_path):
    from nemo_gym.batch_restart import reject_bound_overwrite

    for name in ("manifest.json", "inputs.jsonl", "results.jsonl", "failures.jsonl"):
        (tmp_path / name).write_text("original-" + name)
    (tmp_path / "batch_status.json").write_text(json.dumps({"execution_binding": {"manifest_sha256": "old"}}))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ConfigError, match="cannot be overwritten"):
        reject_bound_overwrite(tmp_path / "manifest.json", resume=False)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [False, True])
async def test_native_collector_rejection_preserves_artifacts_and_dispatches_nothing(tmp_path, monkeypatch, fresh):
    from omegaconf import OmegaConf

    import nemo_gym.rollout_collection as rc
    from nemo_gym.batch_status import BatchStatusTracker
    from tests.unit_tests.test_batch_status import materialized_row, write_manifest

    rows = [materialized_row("agent", "test", 0, 0)]
    inputs = tmp_path / "source.jsonl"
    inputs.write_text(json.dumps(rows[0]) + "\n")
    output = tmp_path / "results.jsonl"
    output.write_text("")
    manifest = write_manifest(tmp_path / "batch_manifest.json", rows)
    receipt = tmp_path / "bad-receipt.json"
    receipt.write_text("{}")
    config = rc.RolloutCollectionConfig(
        input_jsonl_fpath=str(inputs),
        output_jsonl_fpath=str(output),
        batch_manifest_fpath=str(manifest),
        resume_from_cache=not fresh,
        guarded_restart_receipt_fpath=None if fresh else str(receipt),
        disable_health_check=True,
    )
    config.materialized_jsonl_fpath.write_text(json.dumps(rows[0]) + "\n")
    failures = rc.failures_path_for(output)
    failures.write_text(
        json.dumps({**rows[0], "_ng_failure_class": "environment_server_failed", "failure_kind": "session_lost"})
        + "\n"
    )
    global_config = OmegaConf.create(
        {
            "provider_dispatch_guard": True,
            "observability_enabled": True,
            "model_call_capture_dir": str(tmp_path / "capture"),
            "model": {"responses_api_models": {"openai_model": {"entrypoint": "app.py", "openai_model": "fixture"}}},
            "agent": {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py"}}},
        }
    )
    tracker = BatchStatusTracker(manifest, rows)
    bind_execution(
        tracker,
        global_config,
        config.materialized_jsonl_fpath,
        resume=False,
        require_prior=False,
        collector_config=config.model_dump(mode="json"),
        capture_dirs=[tmp_path / "capture"],
    )
    tracker.write_status([], [], force=True)
    monkeypatch.setattr(rc, "get_global_config_dict", lambda: global_config)
    monkeypatch.setattr(
        rc.RolloutCollectionHelper,
        "setup_server_client",
        lambda self: (_ for _ in ()).throw(AssertionError("No dispatch allowed")),
    )
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(
        ConfigError, match="cannot be overwritten" if fresh else "Restart receipt execution binding mismatch"
    ):
        await rc.RolloutCollectionHelper().run_from_config(config)
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.asyncio
async def test_native_collector_restart_selects_only_explicit_failure(tmp_path, monkeypatch):
    fresh = False
    from omegaconf import OmegaConf

    import nemo_gym.rollout_collection as rc
    from nemo_gym.batch_status import BatchStatusTracker
    from tests.unit_tests.test_batch_status import materialized_row, write_manifest

    rows = [materialized_row("agent", "test", i, 0) for i in range(4)]
    inputs = tmp_path / "source.jsonl"
    inputs.write_text(json.dumps(rows[0]) + "\n")
    output = tmp_path / "results.jsonl"
    output.write_text("")
    manifest = write_manifest(tmp_path / "batch_manifest.json", rows)
    receipt = tmp_path / "bad-receipt.json"
    receipt.write_text("{}")
    config = rc.RolloutCollectionConfig(
        input_jsonl_fpath=str(inputs),
        output_jsonl_fpath=str(output),
        batch_manifest_fpath=str(manifest),
        resume_from_cache=not fresh,
        guarded_restart_receipt_fpath=None if fresh else str(receipt),
        disable_health_check=True,
    )
    config.materialized_jsonl_fpath.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config.disable_aggregation = True
    output.write_text(json.dumps({**rows[3], "reward": 1.0}) + "\n")
    failures = rc.failures_path_for(output)
    failures.write_text(
        json.dumps({**rows[0], "_ng_failure_class": "environment_server_failed", "failure_kind": "session_lost"})
        + "\n"
    )
    global_config = OmegaConf.create(
        {
            "provider_dispatch_guard": True,
            "observability_enabled": True,
            "model_call_capture_dir": str(tmp_path / "capture"),
            "model": {"responses_api_models": {"openai_model": {"entrypoint": "app.py", "openai_model": "fixture"}}},
            "agent": {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py"}}},
        }
    )
    tracker = BatchStatusTracker(manifest, rows)
    bind_execution(
        tracker,
        global_config,
        config.materialized_jsonl_fpath,
        resume=False,
        require_prior=False,
        collector_config=config.model_dump(mode="json"),
        capture_dirs=[tmp_path / "capture"],
    )
    binding = digest(tracker.execution_binding)
    failed = [
        {
            **rows[i],
            "_ng_failure_class": "environment_server_failed",
            "failure_kind": "session_lost",
            ROW_BINDING_KEY: binding,
        }
        for i in (0, 1)
    ]
    failures.write_text("".join(json.dumps(row) + "\n" for row in failed))
    receipt.write_text(
        json.dumps(
            {
                "execution_binding_sha256": binding,
                "failures": [{"capture_key": "0-0", "failure_sha256": digest(failed[0])}],
            }
        )
    )
    tracker.write_status([], [], force=True)
    monkeypatch.setattr(rc, "get_global_config_dict", lambda: global_config)
    dispatched = []

    async def completed(row):
        return rc._CompletedRollout(row, {"reward": 0.0}, 1.0)

    def run_examples(self, examples, **kwargs):
        dispatched.extend(examples)
        return iter([completed(row) for row in examples])

    monkeypatch.setattr(rc.RolloutCollectionHelper, "_run_examples_with_metadata", run_examples)
    preserved = {p: p.read_bytes() for p in (inputs, config.materialized_jsonl_fpath, failures, receipt)}
    original_output = output.read_bytes()
    await rc.RolloutCollectionHelper().run_from_config(config)
    assert [(row["_ng_task_index"], row["_ng_attempt_index"]) for row in dispatched] == [(0, 1)]
    assert all(path.read_bytes() == data for path, data in preserved.items())
    assert output.read_bytes().startswith(original_output)
    results = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(row["_ng_task_index"], row.get("_ng_attempt_index", 0)) for row in results] == [(3, 0), (0, 1)]


def test_changed_entrypoint_bytes_refused(tmp_path, monkeypatch):
    root = tmp_path / "responses_api_models/openai_model"
    root.mkdir(parents=True)
    (root / "app.py").write_text("original = True\n")
    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.write_text("fixed")
    tracker = SimpleNamespace(manifest_sha256="m" * 64, execution_binding=None)
    config = {"model": {"responses_api_models": {"openai_model": {"entrypoint": "app.py"}}}}
    bind_execution(tracker, config, inputs, resume=False, require_prior=False, capture_dirs=[tmp_path / "capture"])
    (root / "app.py").write_text("changed = True\n")
    with pytest.raises(ConfigError, match="binding changed"):
        bind_execution(tracker, config, inputs, resume=True, require_prior=True, capture_dirs=[tmp_path / "capture"])


def test_external_harness_remains_unsupported(tmp_path):
    tracker, config, inputs, *_ = fixture(tmp_path)
    config["agent"] = {"responses_api_agents": {"harbor_agent": {"entrypoint": "app.py"}}}
    with pytest.raises(ConfigError, match="unsupported"):
        bind_execution(tracker, config, inputs, resume=True, require_prior=True, capture_dirs=[tmp_path / "captures"])
