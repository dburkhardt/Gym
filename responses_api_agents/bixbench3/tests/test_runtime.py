# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from responses_api_agents.bixbench3.runtime import check_gold_isolation, read_artifact_score


def test_gold_outside_mounts(tmp_path):
    gold = tmp_path / "host_gold"
    data = tmp_path / "input"
    gold.mkdir()
    data.mkdir()
    check_gold_isolation(gold, [data])


def test_gold_inside_mount_rejected(tmp_path):
    gold = tmp_path / "gold"
    gold.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        check_gold_isolation(gold, [tmp_path])


def test_symlink_mount_rejected(tmp_path):
    gold = tmp_path / "gold"
    gold.mkdir()
    link = tmp_path / "input"
    link.symlink_to(gold, target_is_directory=True)
    with pytest.raises(ValueError, match="overlaps"):
        check_gold_isolation(gold, [link])


@pytest.mark.parametrize("score", [0, 0.5, 1])
def test_artifact_score(tmp_path, score):
    path = tmp_path / "grade.json"
    path.write_text(json.dumps({"score": score, "artifact_count": 1, "artifacts": [{"score": score}]}))
    assert read_artifact_score(path) == score


@pytest.mark.parametrize(
    "payload",
    [
        {"score": 1},  # Inspect completeness is not an artifact score.
        {"score": float("nan"), "artifact_count": 1, "artifacts": [{}]},
        {"score": True, "artifact_count": 1, "artifacts": [{}]},
        {"score": 2, "artifact_count": 1, "artifacts": [{}]},
        {"score": 1, "artifact_count": 2, "artifacts": [{}]},
        {"score": 0, "artifact_count": 0, "artifacts": []},
    ],
)
def test_invalid_score_rejected(tmp_path, payload):
    path = tmp_path / "grade.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_artifact_score(path)


def grading_fixture(tmp_path):
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "counts.csv").write_text("gene_id,sample\na,1\n")
    case = {
        "task_id": "synthetic",
        "doi": "10.0000/synthetic",
        "expected_outputs": [{"artifact_id": "counts", "path": "outputs/counts.csv"}],
    }
    spec = {
        "paper_id": "synthetic",
        "doi": "10.0000/synthetic",
        "artifacts": [
            {
                "artifact_id": "counts",
                "ground_truth": {"path": "counts.csv"},
                "expected_output": {"path": "outputs/counts.csv"},
            }
        ],
    }
    path = gold / "grading.json"
    path.write_text(json.dumps(spec))
    mount = tmp_path / "mount"
    mount.mkdir()
    return case, spec, path, mount


def test_grading_identity_and_artifact_contract(tmp_path):
    from responses_api_agents.bixbench3.runtime import validate_grading_contract

    case, spec, path, mount = grading_fixture(tmp_path)
    validate_grading_contract(case=case, specification=path, mounts=[mount])
    spec["paper_id"] = "wrong"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="paper_id"):
        validate_grading_contract(case=case, specification=path, mounts=[mount])
    spec["paper_id"] = "synthetic"
    spec["artifacts"][0]["expected_output"]["path"] = "outputs/other.csv"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="artifact IDs or paths"):
        validate_grading_contract(case=case, specification=path, mounts=[mount])


@pytest.mark.parametrize("kind", ["absolute", "parent", "symlink", "missing"])
def test_ground_truth_escape_and_missing_rejected(tmp_path, kind):
    from responses_api_agents.bixbench3.runtime import validate_grading_contract

    case, spec, path, mount = grading_fixture(tmp_path)
    outside = mount / "leaked.csv"
    outside.write_text("gold")
    if kind == "absolute":
        spec["artifacts"][0]["ground_truth"]["path"] = str(outside)
    elif kind == "parent":
        spec["artifacts"][0]["ground_truth"]["path"] = "../mount/leaked.csv"
    elif kind == "symlink":
        (path.parent / "link.csv").symlink_to(outside)
        spec["artifacts"][0]["ground_truth"]["path"] = "link.csv"
    else:
        spec["artifacts"][0]["ground_truth"]["path"] = "missing.csv"
    path.write_text(json.dumps(spec))
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_grading_contract(case=case, specification=path, mounts=[mount])


def test_generation_settings_require_context_provenance():
    from responses_api_agents.bixbench3.runtime import generation_settings

    settings = {
        "model": "nvidia/nvidia/nemotron-3-ultra",
        "model_context_length": 32768,
        "model_context_source": "synthetic test fixture",
    }
    assert generation_settings(settings) == (32768, 16384)
    for patch in [
        {"model": "other"},
        {"model_context_length": True},
        {"model_context_source": ""},
        {"max_output_tokens": 32768},
        {"max_output_tokens": 0},
    ]:
        with pytest.raises(ValueError):
            generation_settings(settings | patch)


def test_execute_constructs_isolated_native_inspect_contract(tmp_path, monkeypatch):
    """Synthetic process-boundary test, not a Docker or model-performance result."""
    import sys

    import yaml

    from responses_api_agents.bixbench3 import runtime

    case, spec, gold, data = grading_fixture(tmp_path)
    root = tmp_path / "run"
    (root / "dataset").mkdir(parents=True)
    (root / "dataset" / "case.json").write_text(json.dumps(case))
    (root / "broker").mkdir()
    (root / "broker" / "policy.json").write_text("{}")
    checkout = tmp_path / "checkout"
    (checkout / "sandbox").mkdir(parents=True)
    (checkout / "sandbox" / "compose.yaml").write_text(
        yaml.safe_dump(
            {
                "services": {"default": {"labels": ["existing=yes"]}, "network-gateway": {"environment": {}}},
                "networks": {"proxy_egress": None, "agent_isolated": {"internal": True}},
            }
        )
    )
    key = tmp_path / "dummy-key"
    key.write_text("not-a-real-key")
    settings = {
        "upstream_checkout": str(checkout),
        "upstream_python": sys.executable,
        "run_root": str(root),
        "grading_specification": str(gold),
        "data_dir": str(data),
        "task_id": "synthetic",
        "model": "nvidia/nvidia/nemotron-3-ultra",
        "model_context_length": 32768,
        "model_context_source": "synthetic fixture",
        "sandbox_image": "image@sha256:abc",
        "gateway_image": "image@sha256:def",
        "budget_proxy_url": "http://budget-proxy:8000/v1",
        "broker_port": 18080,
        "broker_token_file": str(key),
        "budget_proxy_key_file": str(key),
        "solver_key_file": str(key),
        "solver_base_url": "http://solver:9000/v1",
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    monkeypatch.setattr(runtime, "check_revision", lambda p: None)
    calls = []

    def process(command, **kwargs):
        calls.append((command, kwargs))
        compile(command[2], "<inspect-entrypoint>", "exec")
        assert "responses_api=True" in command[2]
        assert "temperature=0" in command[2]
        assert command[5] == str(root / "dataset" / "case.json")
        assert command[-2:] == ["32768", "16384"]
        assert kwargs["env"]["NEMOTRON_API_KEY"] == "not-a-real-key"

    monkeypatch.setattr(runtime.subprocess, "run", process)
    monkeypatch.setattr(runtime, "grade_artifacts", lambda **kwargs: 0.25)
    assert runtime.execute(settings_path=path) == 0.25
    assert len(calls) == 1
    compose = yaml.safe_load((root / "compose.yaml").read_text())
    assert compose["services"]["default"]["labels"]["existing"] == "yes"
    for section in ("services", "networks"):
        for entry in compose[section].values():
            assert len(entry["labels"]["gym.bixbench3.run"]) == 24
    assert compose["networks"]["agent_isolated"]["internal"] is True
    assert compose["services"]["network-gateway"]["environment"]["OPENAI_BASE_URL"] == settings["budget_proxy_url"]
    assert "NEMOTRON_API_KEY" not in (root / "compose.yaml").read_text()
