import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.flyaoc import managed


def test_solver_mounts_exclude_reference_data(tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    result = managed.spec(
        image="sha256:" + "a" * 64,
        root=tmp_path / "assets",
        output=tmp_path / "output",
        phase="solver",
        lineage="trial",
        request=request,
        socket_dir="/tmp/socket",
        token="local",
    )
    volumes = result.provider_options["volumes"]
    assert not any("references.jsonl" in x for x in volumes)
    assert not any("/var/run/docker.sock" in x for x in volumes)
    assert result.env["HF_HUB_OFFLINE"] == "1"
    assert result.env["HF_DATASETS_OFFLINE"] == "1"
    assert result.resources.memory_mib == 4096
    with pytest.raises(ValueError, match="immutable"):
        managed.spec(image="python:latest", root=tmp_path, output=tmp_path, phase="solver", lineage="trial")


def test_asset_view_rejects_reference_and_symlink(tmp_path, monkeypatch):
    controller = tmp_path / "controller"
    controller.mkdir()
    root = tmp_path / "assets"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src/a.py").write_text("x=1")
    digest = hashlib.sha256((root / "src/a.py").read_bytes()).hexdigest()
    (controller / "runtime_sources.json").write_text(
        json.dumps({"source_files": {"a.py": digest}, "corpus_files": {}, "dataset_revision": "revision"})
    )
    (controller / "frozen_sources.json").write_text(json.dumps({"ontology_sha256": {}}))
    monkeypatch.setattr(managed, "CONTROLLER", controller)
    managed.validate_assets(root)
    reference = root / "reference-records.jsonl"
    reference.write_text("secret")
    with pytest.raises(ValueError, match="Unexpected solver asset"):
        managed.validate_assets(root)
    reference.unlink()
    (root / "alias").symlink_to(root / "src/a.py")
    with pytest.raises(ValueError, match="Unexpected solver asset"):
        managed.validate_assets(root)


def test_native_agent_import_and_scientific_budgets():
    from responses_api_agents.flyaoc_native.app import Config

    assert "runtime_image" in Config.model_fields
    source = (Path(managed.__file__).parent / "native_phase.py").read_text()
    assert "paper_budget=10" in source and "max_turns=50" in source and "resume=False" in source


def test_native_run_retains_collection_identity():
    from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
    from responses_api_agents.flyaoc_native.app import RunRequest

    body = RunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "{}"}]},
            "verifier_metadata": {"gene_id": "control"},
            "_ng_task_index": 3,
            "_ng_rollout_index": 2,
            "_ng_attempt_index": 1,
        }
    )
    assert maybe_rollout_id_from_run_body(body) == "3-2-a1"


def test_native_trajectory_uses_persisted_decisions_and_response_refs(tmp_path):
    from benchmarks.flyaoc.trajectory import build_trajectory
    from nemo_gym.config_types import ModelServerRef

    for i, message in enumerate(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tool1", "function": {"name": "search_corpus", "arguments": "{}"}}],
            },
            {"role": "assistant", "content": "answer"},
        ]
    ):
        (tmp_path / f"{i:06d}.request.json").write_text(
            json.dumps({"messages": [{"role": "user", "content": "public input"}]})
        )
        (tmp_path / f"{i:06d}.response.json").write_text(
            json.dumps(
                {
                    "status": 200,
                    "started_at": 10 + i,
                    "body": json.dumps({"id": f"response{i}", "choices": [{"message": message}]}),
                }
            )
        )
    trajectory = build_trajectory(
        tmp_path,
        task_id="gene",
        rollout_id="0-0",
        model_ref=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    assert len(trajectory.turns) == 2 and len(trajectory.invocations[0].model_calls) == 2
    assert trajectory.turns[0].answer[0]["type"] == "function_call"
    assert trajectory.turns[1].model_calls[0].response_id == "response1"
    assert trajectory.turns[0].question[0]["content"] == "public input"
    (tmp_path / "000001.response.json").unlink()
    with pytest.raises(FileNotFoundError):
        build_trajectory(
            tmp_path,
            task_id="gene",
            rollout_id="0-0",
            model_ref=ModelServerRef(type="responses_api_models", name="policy_model"),
        )


def test_provider_usage_is_summed_without_inventing_unknown_counts(tmp_path):
    from benchmarks.flyaoc.trajectory import summed_usage

    p = tmp_path / "000000.response.json"
    p.write_text(
        json.dumps({"body": json.dumps({"usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}})})
    )
    usage = summed_usage(tmp_path)
    assert usage.input_tokens == 3 and usage.output_tokens == 4
    assert usage.output_tokens_details.reasoning_tokens is None
    p.write_text(json.dumps({"body": "{}"}))
    assert summed_usage(tmp_path) is None
