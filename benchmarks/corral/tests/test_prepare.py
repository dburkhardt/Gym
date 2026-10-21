# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from benchmarks.corral import prepare


def fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    relative = "tasks/retrosynthesis/environments/level_3/tasks_json/make_1.json"
    path = source / relative
    path.parent.mkdir(parents=True)
    native = {"id": "make_1_lvl3", "input": {"prompt": "Exact native prompt"}, "output": [{"target": 17}]}
    path.write_text(json.dumps([native]))
    manifest = {
        "source_commit": "pinned",
        "frozen5": ["make_1_lvl3"],
        "files": {relative: hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    (tmp_path / "frozen_sources.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(prepare, "__file__", str(tmp_path / "prepare.py"))
    return source, path, native


def test_exact_native_fields_and_provenance(tmp_path, monkeypatch):
    source, path, native = fixture(tmp_path, monkeypatch)
    result = prepare.stage(source, tmp_path / "out")
    row = json.loads(result.read_text())
    assert row["verifier_metadata"]["native_task"] == native
    assert row["task_id"] == native["id"]
    assert row["verifier_metadata"]["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert row["responses_create_params"]["input"] == []


def test_source_mismatch_fails_before_publish(tmp_path, monkeypatch):
    source, path, _ = fixture(tmp_path, monkeypatch)
    path.write_text("[]")
    with pytest.raises(ValueError, match="source mismatch"):
        prepare.stage(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_prepared_row_routes_to_native_harbor_task(tmp_path, monkeypatch):
    from pathlib import Path
    import yaml
    from responses_api_agents.harbor_agent.app import HarborAgent, HarborAgentConfig, HarborRunRequest

    source, _, native = fixture(tmp_path, monkeypatch)
    destination = tmp_path / "prepared"
    row = json.loads(prepare.stage(source, destination).read_text())
    request = HarborRunRequest.model_validate(row)
    alias, task = HarborAgent._parse_instance_id(request.instance_id)
    config_path = Path(__file__).parents[1] / "config.yaml"
    settings = yaml.safe_load(config_path.read_text())["harbor_agent"]["responses_api_agents"]["harbor_agent"]
    settings["harbor_datasets"]["corral"]["local_dataset_path"] = str(destination / "tasks")
    config = HarborAgentConfig(name="harbor_agent", host="127.0.0.1", port=24000, **settings)
    server = HarborAgent.model_construct(config=config)
    job = server._build_job_config(
        dataset_alias=alias, task_name=task, model_name="fixture-model",
        api_base="http://127.0.0.1:1/v1", job_name="route-contract", jobs_dir=tmp_path / "jobs",
    )
    assert alias == "corral"
    assert task == row["task_id"] == row["uuid"] == native["id"]
    assert (destination / "tasks" / task / "instruction.md").read_text() == task + "\n"
    assert job["datasets"][0]["path"] == str(destination / "tasks")
    assert job["datasets"][0]["task_names"] == [task]


def test_legacy_routing_artifact_is_preserved(tmp_path, monkeypatch):
    source, _, _ = fixture(tmp_path, monkeypatch)
    path = prepare.stage(source, tmp_path / "old")
    row = json.loads(path.read_text())
    row["instance_id"] = row["task_id"]
    legacy = json.dumps(row) + "\n"
    path.write_text(legacy)
    with pytest.raises(ValueError, match="fresh versioned"):
        prepare.stage(source, tmp_path / "old")
    assert path.read_text() == legacy
    fresh = prepare.stage(source, tmp_path / "fresh")
    assert json.loads(fresh.read_text())["instance_id"] == "corral::" + row["task_id"]
    for original in (path.parent / "tasks").rglob("*"):
        if original.is_file() and not original.name.endswith(".stage.lock"):
            assert original.read_bytes() == (fresh.parent / "tasks" / original.relative_to(path.parent / "tasks")).read_bytes()
