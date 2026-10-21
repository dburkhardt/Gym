import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.cli.admission_evidence import resolved_admission_evidence


@pytest.fixture(autouse=True)
def offline_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def model(**overrides):
    return dict(
        host="localhost",
        port=1,
        entrypoint="app.py",
        openai_api_key="SECRET",
        openai_base_url="https://provider.test/v1",
        openai_model="fixture",
        **overrides,
    )


def test_defaults_and_redaction():
    result = resolved_admission_evidence(
        OmegaConf.create({"solver": {"responses_api_models": {"openai_model": model()}}})
    )
    row = result["roles"][0]
    assert row["admission_timeout_seconds"] is None
    assert row["cooldown_seconds"] is None
    assert row["max_attempts"] == 1
    assert "SECRET" not in json.dumps(result)
    assert "provider.test" not in json.dumps(result)


@pytest.mark.parametrize(
    "settings",
    [dict(upstream_rate_limit_shared_directory="/private/scope"), dict(upstream_retry_policy={"max_attempts": 2})],
)
def test_existing_validator_refuses_incomplete_policy_without_leaking_input(settings):
    with pytest.raises(ValueError, match="Cannot validate admission policy") as exc:
        resolved_admission_evidence(
            OmegaConf.create({"judge": {"responses_api_models": {"openai_model": model(**settings)}}})
        )
    assert "SECRET" not in str(exc.value)


def test_native_plan_resolves_nested_profiles_and_independent_roles(tmp_path):
    source = Path(__file__).resolve().parents[2]
    sample = tmp_path / "sample.jsonl"
    sample.write_text(
        json.dumps(
            {
                "task_id": "one",
                "responses_create_params": {"input": [{"role": "user", "content": "Choose 1"}]},
                "verifier_metadata": {"expected_answer": 1, "option_count": 2},
            }
        )
        + "\n"
    )
    inner = tmp_path / "inner.json"
    inner.write_text(
        json.dumps(
            {
                "policy_model": {
                    "responses_api_models": {
                        "openai_model": {
                            "upstream_max_num_tries": 1,
                            "upstream_rate_limit_cooldown_seconds": 60,
                            "upstream_rate_limit_shared_directory": "/never-created",
                            "upstream_rate_limit_shared_namespace": "private-namespace",
                            "upstream_request_timeout_seconds": 540,
                        }
                    }
                }
            }
        )
    )
    outer = tmp_path / "outer.json"
    outer.write_text(
        json.dumps(
            {
                "config_paths": [str(inner)],
                "policy_model": {"responses_api_models": {"openai_model": {"upstream_admission_timeout_seconds": 60}}},
            }
        )
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "config_paths": ["benchmarks/proteinlmbench/config.yaml", str(outer)],
                "policy_api_key": "SECRET",
                "policy_base_url": "https://provider.test/v1",
                "judge_model": {"responses_api_models": {"openai_model": model(upstream_request_timeout_seconds=600)}},
                "use_cached_prepared_benchmarks": True,
                "proteinlmbench": {
                    "resources_servers": {
                        "proteinlmbench": {
                            "datasets": [
                                {
                                    "name": "proteinlmbench",
                                    "type": "benchmark",
                                    "jsonl_fpath": str(sample),
                                    "prepare_script": "benchmarks/proteinlmbench/prepare.py",
                                    "num_repeats": 1,
                                }
                            ]
                        }
                    }
                },
            }
        )
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from nemo_gym.cli.main import main; main()",
            "eval",
            "run",
            "--config",
            str(runtime),
            "--model-type",
            "openai_model",
            "--model",
            "fixture",
            "--split",
            "benchmark",
            "--plan-only",
            "--output",
            str(tmp_path / "results.jsonl"),
        ],
        cwd=source,
        env=dict(os.environ, PYTHONPATH=str(source), NEMO_GYM_EXTRA_ROOTS=str(source)),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    receipt = json.loads((tmp_path / "results_dispatch_plan.json").read_text())
    roles = {r["role"]: r for r in receipt["model_admission_policy"]["roles"]}
    assert roles["policy_model"]["cooldown_seconds"] == 60
    assert roles["policy_model"]["admission_timeout_seconds"] == 60
    assert roles["policy_model"]["provider_timeout_seconds"] == 540
    assert roles["judge_model"]["provider_timeout_seconds"] == 600
    assert roles["judge_model"]["cooldown_seconds"] is None
    assert not any(secret in json.dumps(receipt) for secret in ("SECRET", "private-namespace", "provider.test"))


@pytest.mark.parametrize(
    "config,expected",
    [
        ({}, []),
        (
            {"other": {"responses_api_models": {"custom": {"secret": "hidden"}}}},
            [{"role": "other", "implementation": "custom", "status": "unsupported"}],
        ),
    ],
)
def test_non_openai_configs_do_not_import_implementation(monkeypatch, config, expected):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "responses_api_models.openai_model.app":
            raise AssertionError("OpenAI implementation must not be imported")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert resolved_admission_evidence(OmegaConf.create(config))["roles"] == expected


def test_retry_behavior_is_distinguished_in_evidence():
    def evidence(policy):
        return resolved_admission_evidence(
            OmegaConf.create(
                {
                    "solver": {
                        "responses_api_models": {
                            "openai_model": model(
                                upstream_max_num_tries=1,
                                upstream_rate_limit_cooldown_seconds=60,
                                upstream_retry_policy=policy,
                            )
                        }
                    }
                }
            )
        )

    base = evidence({"max_attempts": 3})
    changed = evidence(
        {
            "max_attempts": 3,
            "terminal_http_status_codes": [504, 429],
            "pre_request_jitter_seconds": [1, 2],
            "backoff_initial_seconds": 3,
            "backoff_multiplier": 4,
            "backoff_jitter_fraction": 0.5,
        }
    )
    assert base["sha256"] != changed["sha256"]
    row = changed["roles"][0]
    assert row["retryable_http_status_codes"] == [429]
    assert row["terminal_http_status_codes"] == [429, 504]
    assert row["pre_request_jitter_seconds"] == (1, 2)
    assert (row["backoff_initial_seconds"], row["backoff_multiplier"], row["backoff_jitter_fraction"]) == (3, 4, 0.5)


TEXT = "responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym:Terminus2NemoGym"
IMAGE = "responses_api_agents.harbor_agent.custom_agents.terminus_2_images_nemo_gym:Terminus2ImagesNemoGym"


def transport_config(path=TEXT, **kwargs):
    return OmegaConf.create(
        {
            "provider_dispatch_guard": True,
            "solver": {
                "responses_api_models": {
                    "openai_model": model(
                        upstream_max_num_tries=1,
                        max_http_attempts=1,
                        upstream_rate_limit_cooldown_seconds=60,
                        upstream_request_timeout_seconds=600,
                        upstream_admission_timeout_seconds=120,
                    )
                }
            },
            "task": {
                "responses_api_agents": {
                    "harbor_agent": {
                        "harbor_agent_import_path": path,
                        "model_server": {"type": "responses_api_models", "name": "solver"},
                        "harbor_agent_kwargs": kwargs,
                    }
                }
            },
        }
    )


@pytest.mark.parametrize("path", [TEXT, IMAGE])
@pytest.mark.parametrize(
    "timeout,expected,margin",
    [(None, "client_shorter_than_server_window", -600), (750, "covers_declared_queue_plus_provider_only", 30)],
)
def test_actual_harbor_constructor_timeout_and_explicit_override(path, timeout, expected, margin):
    config = transport_config(path, **({} if timeout is None else {"nemo_model_server_timeout_sec": timeout}))
    before = OmegaConf.to_container(config)
    result = resolved_admission_evidence(config)
    link = result["harbor_transport_links"][0]
    assert link["status"] == expected
    assert link["client_timeout_seconds"] == (120 if timeout is None else timeout)
    assert link["server_window_seconds"] == 720
    assert link["margin_seconds"] == margin
    assert all(len(digest) == 64 for digest in link["runtime_source_sha256"].values())
    assert OmegaConf.to_container(config) == before
    assert "SECRET" not in json.dumps(result)


def test_exact_binding_does_not_fall_back_to_policy_model():
    config = transport_config()
    config.task.responses_api_agents.harbor_agent.model_server.name = "missing"
    config.policy_model = config.solver
    link = resolved_admission_evidence(config)["harbor_transport_links"][0]
    assert link["reason"] == "unsupported_or_missing_model_policy"


def test_independent_harbor_links_and_digest():
    config = transport_config()
    config.judge = config.solver
    config.judge.responses_api_models.openai_model.upstream_request_timeout_seconds = 10
    config.review = config.task
    config.review.responses_api_agents.harbor_agent.model_server.name = "judge"
    links = {r["agent"]: r for r in resolved_admission_evidence(config)["harbor_transport_links"]}
    assert links["review"]["server_window_seconds"] == 130
    assert links["task"]["server_window_seconds"] == 720
    before = resolved_admission_evidence(config)["sha256"]
    config.task.responses_api_agents.harbor_agent.harbor_agent_kwargs.nemo_model_server_timeout_sec = 750
    assert resolved_admission_evidence(config)["sha256"] != before


@pytest.mark.parametrize(
    "change,reason",
    [
        ("unguarded", "unguarded_agent_retry_schedule"),
        ("multiple", "multiple_or_unknown_attempt_schedule"),
        ("unbounded", "unbounded_or_unknown_server_window"),
        ("invalid", "invalid_client_timeout"),
        ("llm", "custom_llm_override"),
    ],
)
def test_unknown_schedules_are_not_coverage_claims(change, reason):
    config = transport_config()
    if change == "unguarded":
        config.provider_dispatch_guard = False
    elif change == "multiple":
        config.solver.responses_api_models.openai_model.upstream_retry_policy = {"max_attempts": 3}
    elif change == "unbounded":
        config.solver.responses_api_models.openai_model.upstream_admission_timeout_seconds = None
    elif change == "invalid":
        config.task.responses_api_agents.harbor_agent.harbor_agent_kwargs.nemo_model_server_timeout_sec = True
    else:
        config.task.responses_api_agents.harbor_agent.harbor_agent_kwargs.llm = "SECRET"
    link = resolved_admission_evidence(config)["harbor_transport_links"][0]
    assert link["status"] == "unknown"
    assert link["reason"] == reason
    assert "margin_seconds" not in link


def test_unsupported_harness_never_imported_and_runtime_unavailable_redacted(monkeypatch):
    from nemo_gym.cli import admission_evidence

    def unavailable(path):
        raise RuntimeError("SECRET")

    monkeypatch.setattr(admission_evidence, "_harbor_timeout_contract", unavailable)
    assert (
        resolved_admission_evidence(transport_config("arbitrary.module:Class"))["harbor_transport_links"][0]["status"]
        == "unsupported"
    )
    result = resolved_admission_evidence(transport_config())
    assert (
        result["harbor_transport_links"][0]["reason"]
        == "harbor_runtime_contract_unavailable_or_offline_import_not_enabled"
    )
    assert "SECRET" not in json.dumps(result)


def test_schema_v2_hash_binds_roles_and_links():
    import hashlib

    result = resolved_admission_evidence(transport_config(nemo_model_server_timeout_sec=750))
    assert result["version"] == 2
    payload = {key: value for key, value in result.items() if key != "sha256"}
    assert (
        result["sha256"]
        == hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    link = result["harbor_transport_links"][0]
    assert "pre_request_jitter" in link["excluded_overhead"]
    assert "not_end_to_end_deadline" in link["scope"]


def test_cold_runtime_without_offline_flag_refuses_import(monkeypatch):
    from nemo_gym.cli import admission_evidence

    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.delitem(sys.modules, "litellm", raising=False)
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match="local cost map"):
        admission_evidence._harbor_timeout_contract(TEXT)
    assert dict(os.environ) == before


def test_fresh_process_without_flag_never_imports_optional_agents():
    script = r"""
import builtins
import os
import socket
import sys
from nemo_gym.cli.admission_evidence import _harbor_timeout_contract
assert "litellm" not in sys.modules
assert "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ
attempts = []
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith("responses_api_agents") or name.startswith("litellm"):
        attempts.append(name)
        raise AssertionError("optional import attempted")
    return original(name, *args, **kwargs)
def blocked(*args, **kwargs):
    attempts.append("network")
    raise AssertionError("network attempted")
builtins.__import__ = guarded
socket.create_connection = blocked
socket.socket.connect = blocked
before = dict(os.environ)
try:
    _harbor_timeout_contract("responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym:Terminus2NemoGym")
except RuntimeError as exc:
    assert "local cost map" in str(exc)
else:
    raise AssertionError("cold contract must be unavailable")
assert attempts == []
assert dict(os.environ) == before
"""
    env = dict(os.environ)
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
