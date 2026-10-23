from pathlib import Path

from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from responses_api_agents.metasyn_native import app as module


def test_collection_metadata_survives_validation_and_native_run_route(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "rag/203"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "prompt.txt").write_text("frozen prompt")
    (tmp_path / "report.md").write_text("test report")
    config = OmegaConf.create(
        {
            "observability_enabled": True,
            "model_call_capture_dir": str(tmp_path / "capture"),
            "provider_dispatch_guard": True,
            "policy_model_name": module.MODEL,
            "policy_model": {"responses_api_models": {"openai_model": {"host": "127.0.0.1", "port": 1}}},
        }
    )
    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=config)
    agent = module.MetaSynAgent(
        config=module.Config(
            host="127.0.0.1",
            port=2,
            name="metasyn_agent",
            entrypoint="app.py",
            native_python="/unused/python",
            runner_script=Path("/unused/runner"),
            materialized_root=tmp_path,
            output_root=tmp_path / "output",
            reference_file=tmp_path / "refs",
            inference_enabled=True,
            public_generator_image="sha256:" + "a" * 64,
        ),
        server_client=client,
    )
    seen = []

    async def evaluate(self, review_id, endpoint, lineage):
        seen.append((review_id, endpoint, lineage))
        return tmp_path, {"status": "succeeded", "metrics": {"inclusion_f1": 0.0}}

    from nemo_gym.rollout_observability import TrajectoryRecord

    monkeypatch.setattr(
        module,
        "observations",
        lambda directory, **kwargs: (
            TrajectoryRecord(task_id=kwargs["task_id"], rollout_id=kwargs["rollout_id"]),
            None,
        ),
    )
    monkeypatch.setattr(module.MetaSynAgent, "_evaluate", evaluate)
    monkeypatch.setattr(module, "get_global_config_dict", lambda: config)
    payload = {
        "responses_create_params": {"input": [{"role": "user", "content": "frozen prompt"}]},
        "verifier_metadata": {"review_id": 203},
        "_ng_task_index": 7,
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 1,
    }
    assert maybe_rollout_id_from_run_body(module.RunRequest.model_validate(payload)) == "7-2-a1"
    with TestClient(agent.setup_webserver()) as http:
        response = http.post("/run", json=payload)
    assert response.status_code == 200, response.text
    assert seen == [(203, "http://127.0.0.1:1/ng-rollout/7-2-a1/v1", "7-2-a1")]
    assert response.json()["_ng_task_index"] == 7
    assert response.json()["verifier_metadata"] == {"review_id": 203}
