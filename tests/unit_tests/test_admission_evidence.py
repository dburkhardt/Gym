import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.cli.admission_evidence import resolved_admission_evidence


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
